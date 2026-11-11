"""Reproducible experimental evaluation orchestration.

This module assembles the complete Phase 4A experiment: it loads the trained
MNIST network and the MNIST test split once, selects a deterministic,
correctly classified evaluation population from a configurable random seed,
runs both verification methods (baseline IBP and LP-refined IBP) on identical
regions for every configured epsilon, and aggregates all metrics into an
immutable :class:`ExperimentResult`.

The verification mathematics is never reimplemented: every verdict comes from
the existing public verification APIs via :mod:`evaluation.runner`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Optional

import numpy as np

from evaluation.dataset import (
    MNIST_PIXEL_LOWER,
    MNIST_PIXEL_UPPER,
    SampleSelection,
    load_mnist_test_dataset,
    select_samples,
    validate_input_bounds,
)
from evaluation.exports import write_samples_csv, write_summary_json
from evaluation.metrics import (
    EpsilonMetrics,
    SampleRecord,
    aggregate_epsilon_metrics,
)
from evaluation.runner import evaluate_sample
from models.model_loader import MathematicalNetwork, extract_network, load_model
from models.network import MiniCertNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "PROJECT_ROOT",
    "ExperimentConfig",
    "ExperimentResult",
    "load_trained_network",
    "run_experiment",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration of one evaluation experiment.

    The defaults reproduce the suggested initial Phase 4A configuration
    (100 correctly classified samples, epsilons 0.01/0.03/0.05/0.1, seed 42);
    every value is configurable and nothing is hardcoded elsewhere.

    Attributes:
        sample_count: Number of correctly classified test samples to select.
        epsilons: L-infinity perturbation radii to evaluate, in order.
        seed: Random seed for the deterministic sample selection.
        data_dir: Directory holding (or receiving) the MNIST raw files.
        checkpoint: Path of the trained MiniCertNet checkpoint.
        output_dir: Default directory for CSV/JSON exports.
        experiment_name: Base name of the export files; empty selects the
            deterministic default ``evaluation_seed{seed}_n{sample_count}``.
        prediction_batch_size: Forward-pass batch size for nominal predictions.
        input_lower: Valid input pixel lower bound.
        input_upper: Valid input pixel upper bound.

    Raises:
        TypeError: If a numeric field has the wrong type.
        ValueError: If a numeric field is out of range or the input bounds
            are inconsistent.
    """

    sample_count: int = 100
    epsilons: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1)
    seed: int = 42
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint: Path = PROJECT_ROOT / "saved_model.pth"
    output_dir: Path = PROJECT_ROOT / "results"
    experiment_name: str = ""
    prediction_batch_size: int = 1024
    input_lower: float = MNIST_PIXEL_LOWER
    input_upper: float = MNIST_PIXEL_UPPER

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise TypeError("sample_count must be an integer")
        if self.sample_count < 1:
            raise ValueError(f"sample_count must be at least 1, got {self.sample_count}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

        epsilons = self.epsilons
        if isinstance(epsilons, (list, np.ndarray)):
            epsilons = tuple(epsilons)
        if not isinstance(epsilons, tuple) or not epsilons:
            raise ValueError("epsilons must be a non-empty sequence of radii")
        validated: list[float] = []
        for position, epsilon in enumerate(epsilons):
            if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
                raise TypeError(f"epsilons[{position}] must be a real scalar")
            value = float(epsilon)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"epsilons[{position}] must be finite and nonnegative, got {value}"
                )
            validated.append(value)
        object.__setattr__(self, "epsilons", tuple(validated))

        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if not isinstance(self.experiment_name, str):
            raise TypeError("experiment_name must be a string")
        if not self.experiment_name.strip():
            object.__setattr__(
                self,
                "experiment_name",
                f"evaluation_seed{self.seed}_n{self.sample_count}",
            )

        if isinstance(self.prediction_batch_size, bool) or not isinstance(
            self.prediction_batch_size, int
        ):
            raise TypeError("prediction_batch_size must be an integer")
        if self.prediction_batch_size < 1:
            raise ValueError(
                f"prediction_batch_size must be at least 1, got {self.prediction_batch_size}"
            )

        lower, upper = validate_input_bounds(self.input_lower, self.input_upper)
        object.__setattr__(self, "input_lower", lower)
        object.__setattr__(self, "input_upper", upper)

    def as_config_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration summary."""
        return {
            "sample_count": self.sample_count,
            "epsilons": list(self.epsilons),
            "seed": self.seed,
            "data_dir": str(self.data_dir),
            "checkpoint": str(self.checkpoint),
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "prediction_batch_size": self.prediction_batch_size,
            "input_lower": self.input_lower,
            "input_upper": self.input_upper,
        }


def load_trained_network(checkpoint: Path) -> MathematicalNetwork:
    """Load and extract the trained MiniCertNet as a float64 network.

    Args:
        checkpoint: Path of the trained MiniCertNet checkpoint.

    Returns:
        The extracted float64 MathematicalNetwork.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        CheckpointError: If the checkpoint fails strict validation.
    """
    model = load_model(MiniCertNet, Path(checkpoint), device="cpu")
    return extract_network(model)


@dataclass(frozen=True)
class ExperimentResult:
    """Immutable result of one complete evaluation experiment.

    Attributes:
        config: The configuration the experiment ran with.
        selection: The deterministic sample selection shared by every epsilon
            and both verification methods.
        metrics: Aggregated metrics, one entry per configured epsilon, in
            configuration order.
        records: All per-sample records (config order: epsilon-major, then
            selection order).
        network_source: Name of the model class the network was extracted
            from.
        dataset_size: Number of items in the evaluated test split.
    """

    config: ExperimentConfig
    selection: SampleSelection
    metrics: tuple[EpsilonMetrics, ...]
    records: tuple[SampleRecord, ...]
    network_source: str
    dataset_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the complete experiment."""
        from dataclasses import asdict

        return {
            "experiment_name": self.config.experiment_name,
            "config": self.config.as_config_dict(),
            "model": {
                "checkpoint": str(self.config.checkpoint),
                "source_model": self.network_source,
            },
            "dataset": {"name": "MNIST", "split": "test", "total_samples": self.dataset_size},
            "sample_selection": {
                "requested": self.selection.requested_count,
                "selected": self.selection.selected_count,
                "incorrect_excluded": self.selection.incorrect_excluded,
                "scanned": self.selection.scanned,
                "seed": self.selection.seed,
            },
            "results": [asdict(entry) for entry in self.metrics],
        }

    def save(self, output_dir: Optional[Path] = None) -> dict[str, Path]:
        """Export the per-sample CSV and the aggregated JSON summary.

        Args:
            output_dir: Target directory; defaults to the configured
                ``output_dir``. Created when missing.

        Returns:
            A mapping with the written ``samples_csv`` and ``summary_json``
            paths.
        """
        directory = Path(output_dir) if output_dir is not None else self.config.output_dir
        csv_path = directory / f"{self.config.experiment_name}_samples.csv"
        json_path = directory / f"{self.config.experiment_name}_summary.json"
        write_samples_csv(self.records, csv_path)
        write_summary_json(self.to_dict(), json_path)
        return {"samples_csv": csv_path, "summary_json": json_path}


def run_experiment(
    config: ExperimentConfig,
    *,
    network: Optional[MathematicalNetwork] = None,
    dataset: Optional[Any] = None,
) -> ExperimentResult:
    """Run the complete evaluation experiment for one configuration.

    The model and the dataset are loaded at most once (skipped entirely when
    supplied by the caller), the sample population is selected once and shared
    by every epsilon and both verification methods, and each
    (sample, epsilon) pair is verified with baseline IBP and with LP-refined
    IBP on identical input regions.

    Args:
        config: A validated ExperimentConfig.
        network: Optional pre-extracted network (avoids re-loading the
            checkpoint).
        dataset: Optional pre-loaded test dataset (avoids re-loading MNIST).

    Returns:
        The ExperimentResult with all per-sample records and aggregated
        metrics.

    Raises:
        TypeError: If the config has the wrong type.
        FileNotFoundError: If the checkpoint is missing and not supplied.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            f"config must be an ExperimentConfig, got {type(config).__name__}"
        )
    evaluated_network = (
        network if network is not None else load_trained_network(config.checkpoint)
    )
    evaluated_dataset = (
        dataset if dataset is not None else load_mnist_test_dataset(config.data_dir)
    )

    selection = select_samples(
        evaluated_network,
        evaluated_dataset,
        config.sample_count,
        config.seed,
        batch_size=config.prediction_batch_size,
    )

    records: list[SampleRecord] = []
    metrics: list[EpsilonMetrics] = []
    for epsilon in config.epsilons:
        epsilon_records = [
            evaluate_sample(
                evaluated_network,
                sample,
                epsilon,
                input_lower=config.input_lower,
                input_upper=config.input_upper,
            )
            for sample in selection.samples
        ]
        records.extend(epsilon_records)
        metrics.append(
            aggregate_epsilon_metrics(
                epsilon_records,
                epsilon,
                total_test_samples=len(evaluated_dataset),
                incorrect_excluded=selection.incorrect_excluded,
            )
        )

    return ExperimentResult(
        config=config,
        selection=selection,
        metrics=tuple(metrics),
        records=tuple(records),
        network_source=evaluated_network.source_model,
        dataset_size=len(evaluated_dataset),
    )
