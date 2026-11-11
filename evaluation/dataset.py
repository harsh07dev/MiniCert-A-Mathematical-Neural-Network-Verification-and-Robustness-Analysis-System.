"""MNIST dataset loading, preprocessing, and deterministic sample selection.

This module is the data source of the experimental evaluation layer. It loads
the MNIST test split with the same preprocessing used for training
(``transforms.ToTensor``, which maps pixel intensities to the valid range
[0, 1]), flattens every image into the feature vector
``MathematicalNetwork.forward`` expects, and selects a reproducible
evaluation population with a seeded pseudo-random generator.

The nominal prediction of every sample is computed with the extracted float64
network (``MathematicalNetwork.forward``), so the evaluation population is a
pure function of the checkpoint, the dataset, and the seed: identical
configurations always produce identical samples, target classes, and
perturbation regions for both verification methods.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import Any, Iterator

import numpy as np

from models.model_loader import MathematicalNetwork

MNIST_PIXEL_LOWER = 0.0
MNIST_PIXEL_UPPER = 1.0
DEFAULT_PREDICTION_BATCH_SIZE = 1024

__all__ = [
    "MNIST_PIXEL_LOWER",
    "MNIST_PIXEL_UPPER",
    "DEFAULT_PREDICTION_BATCH_SIZE",
    "InputSample",
    "SampleSelection",
    "flatten_image",
    "validate_input_bounds",
    "load_mnist_test_dataset",
    "collect_nominal_predictions",
    "select_samples",
]


@dataclass(frozen=True)
class InputSample:
    """One flattened, preprocessed evaluation input.

    Attributes:
        index: Position of the sample in the underlying dataset.
        image: Flattened float64 feature vector.
        label: Ground-truth class index.
        predicted: Nominal prediction of the evaluated network.
    """

    index: int
    image: np.ndarray
    label: int
    predicted: int

    @property
    def correctly_classified(self) -> bool:
        """Whether the network's nominal prediction matches the label."""
        return self.predicted == self.label


@dataclass(frozen=True)
class SampleSelection:
    """Deterministic outcome of the sample-selection stage.

    Attributes:
        samples: Selected correctly classified samples, in selection order.
        requested_count: The configured sample count.
        incorrect_excluded: Number of incorrectly classified samples encountered
            and excluded while scanning for the requested population.
        scanned: Number of dataset items examined (selected + excluded).
        seed: The random seed that produced this selection.
    """

    samples: tuple[InputSample, ...]
    requested_count: int
    incorrect_excluded: int
    scanned: int
    seed: int

    @property
    def selected_count(self) -> int:
        """Number of selected samples."""
        return len(self.samples)


def flatten_image(image: Any) -> np.ndarray:
    """Coerce any image-like object to a finite flat float64 feature vector.

    Args:
        image: Tensor, array, or nested sequence of pixel values; any shape.

    Returns:
        A 1-D float64 array (a fresh copy, never a view of the input).

    Raises:
        ValueError: If the image is empty or contains non-finite pixels.
        TypeError: If the image cannot be converted to float64.
    """
    try:
        vector = np.asarray(image, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"image must be convertible to float64: {exc}") from exc
    if vector.size == 0:
        raise ValueError("image must contain at least one pixel")
    if not np.all(np.isfinite(vector)):
        raise ValueError("image must contain only finite pixel values")
    return vector


def validate_input_bounds(input_lower: Real, input_upper: Real) -> tuple[float, float]:
    """Validate the valid input pixel range as finite ordered floats.

    Returns:
        The ``(input_lower, input_upper)`` pair as floats.

    Raises:
        TypeError: If either bound is not a real scalar.
        ValueError: If a bound is non-finite or the lower bound exceeds the
            upper bound.
    """
    if isinstance(input_lower, bool) or not isinstance(input_lower, Real):
        raise TypeError("input_lower must be a real scalar")
    if isinstance(input_upper, bool) or not isinstance(input_upper, Real):
        raise TypeError("input_upper must be a real scalar")
    lower, upper = float(input_lower), float(input_upper)
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("input bounds must be finite")
    if lower > upper:
        raise ValueError(f"input_lower {lower} must not exceed input_upper {upper}")
    return lower, upper


def _item_image(item: Any) -> Any:
    """Extract the image component of one dataset item."""
    if isinstance(item, (tuple, list)):
        if not item:
            raise ValueError("dataset item must contain an image and a label")
        return item[0]
    return item


def _item_label(item: Any) -> int:
    """Extract the integer label component of one dataset item."""
    if isinstance(item, (tuple, list)):
        if len(item) < 2:
            raise ValueError("dataset item must contain an image and a label")
        label = item[1]
    else:
        raise ValueError("dataset items must be (image, label) pairs")
    if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
        raise TypeError(f"dataset label must be an integer, got {type(label).__name__}")
    return int(label)


def _iter_images(dataset: Any) -> Iterator[np.ndarray]:
    """Yield every dataset image as a flattened float64 vector."""
    for index in range(len(dataset)):
        yield flatten_image(_item_image(dataset[index]))


def load_mnist_test_dataset(data_dir: Any) -> Any:
    """Load the MNIST test split with the training-time preprocessing.

    Args:
        data_dir: Directory holding (or receiving) the MNIST raw files.

    Returns:
        A ``torchvision.datasets.MNIST`` test dataset yielding
        ``(tensor[1, 28, 28] in [0, 1], label)`` items.

    Raises:
        RuntimeError: If torchvision is unavailable or the download fails.
    """
    from torchvision import datasets, transforms

    return datasets.MNIST(
        str(data_dir), train=False, download=True, transform=transforms.ToTensor()
    )


def collect_nominal_predictions(
    network: MathematicalNetwork,
    dataset: Any,
    batch_size: int = DEFAULT_PREDICTION_BATCH_SIZE,
) -> np.ndarray:
    """Compute the network's nominal prediction for every dataset item.

    Images are propagated in batches through the extracted float64 network, so
    the full test split is classified with a handful of vectorized forward
    passes instead of one call per sample.

    Args:
        network: An extracted float64 MathematicalNetwork.
        dataset: A sized collection of ``(image, label)`` items.
        batch_size: Number of images per forward pass (at least 1).

    Returns:
        An int64 array with the predicted class index for every dataset item,
        aligned with the dataset's natural order.

    Raises:
        TypeError: If the network is not a MathematicalNetwork or an image is
            not convertible to float64.
        ValueError: If the batch size is below 1 or a forward pass returns an
            unexpected shape.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(
            f"network must be a MathematicalNetwork, got {type(network).__name__}"
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    total = len(dataset)
    predictions = np.empty(total, dtype=np.int64)
    batch: list[np.ndarray] = []
    positions: list[int] = []

    def flush() -> None:
        if not batch:
            return
        logits = np.asarray(network.forward(np.stack(batch, axis=0)))
        if logits.ndim != 2 or logits.shape[0] != len(batch):
            raise ValueError(
                "network.forward must return one logit row per input, got shape "
                f"{logits.shape} for {len(batch)} inputs"
            )
        predictions[np.asarray(positions, dtype=np.int64)] = np.argmax(logits, axis=1)
        batch.clear()
        positions.clear()

    for index, image in enumerate(_iter_images(dataset)):
        batch.append(image)
        positions.append(index)
        if len(batch) >= batch_size:
            flush()
    flush()
    return predictions


def select_samples(
    network: MathematicalNetwork,
    dataset: Any,
    sample_count: int,
    seed: int,
    *,
    batch_size: int = DEFAULT_PREDICTION_BATCH_SIZE,
) -> SampleSelection:
    """Select a deterministic, correctly classified evaluation population.

    The dataset is scanned in an order shuffled with ``random.Random(seed)``
    (reproducible for a fixed seed); incorrectly classified samples are skipped
    and counted until ``sample_count`` correctly classified samples have been
    selected. Selection is performed once and shared by every epsilon and both
    verification methods, so all methods are evaluated on identical samples.

    Args:
        network: An extracted float64 MathematicalNetwork used for the nominal
            predictions.
        dataset: A sized collection of ``(image, label)`` items.
        sample_count: Number of correctly classified samples to select.
        seed: Random seed controlling the scan order.
        batch_size: Forward-pass batch size for the nominal predictions.

    Returns:
        A SampleSelection with the chosen samples and exclusion statistics.

    Raises:
        TypeError: If sample_count or seed is not an integer, or the network
            has the wrong type.
        ValueError: If sample_count is below 1 or a dataset item is malformed.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError(f"sample_count must be at least 1, got {sample_count}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    predictions = collect_nominal_predictions(network, dataset, batch_size)
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)

    samples: list[InputSample] = []
    incorrect_excluded = 0
    for index in order:
        item = dataset[index]
        image = flatten_image(_item_image(item))
        label = _item_label(item)
        predicted = int(predictions[index])
        if predicted != label:
            incorrect_excluded += 1
            continue
        samples.append(
            InputSample(index=index, image=image, label=label, predicted=predicted)
        )
        if len(samples) >= sample_count:
            break

    scanned = len(samples) + incorrect_excluded
    return SampleSelection(
        samples=tuple(samples),
        requested_count=sample_count,
        incorrect_excluded=incorrect_excluded,
        scanned=scanned,
        seed=seed,
    )
