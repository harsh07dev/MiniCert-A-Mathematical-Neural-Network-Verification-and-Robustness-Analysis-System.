"""Network-level Interval Bound Propagation (IBP).

This module provides the :func:`propagate_network` function to propagate a
multidimensional input box through an extracted
:class:`~models.model_loader.MathematicalNetwork` using the
:class:`~abstract_domain.box.Box` abstract domain.
"""

from __future__ import annotations

import numpy as np

from abstract_domain.box import Box
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification.regions import BoxProtocol


def propagate_network(network: MathematicalNetwork, input_box: BoxProtocol) -> Box:
    """Propagate an input perturbation box through an extracted mathematical network.

    Iterates through the sequential layers of the network in their exact execution
    order, applying sound interval bound propagation (IBP) transformers at each layer:
    - LinearLayer: current_box = current_box.affine(layer.weight, layer.bias)
    - ReLULayer: current_box = current_box.relu()

    Args:
        network: An extracted float64 MathematicalNetwork containing ordered layers.
        input_box: An input bound representation conforming to BoxProtocol (e.g.
            a Box or an InputRegion).

    Returns:
        The certified output Box bounding the network's output reachability set.

    Raises:
        TypeError: If network is not a MathematicalNetwork, input_box does not satisfy
            BoxProtocol, or an unsupported layer type is encountered.
        ValueError: If input_box has invalid dimensions, does not match
            network.input_features, or the final output dimension does not match
            network.output_features.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(network).__name__}")

    if (
        not isinstance(input_box, BoxProtocol)
        or callable(input_box.lower)
        or callable(input_box.upper)
    ):
        raise TypeError(f"input_box must satisfy BoxProtocol, got {type(input_box).__name__}")

    try:
        lower = np.asarray(input_box.lower, dtype=np.float64)
        upper = np.asarray(input_box.upper, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"input_box bounds must be convertible to float64 arrays: {exc}") from exc

    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError(
            f"input_box bounds must be 1-D vectors, got lower.ndim={lower.ndim}, upper.ndim={upper.ndim}"
        )

    if lower.shape != upper.shape:
        raise ValueError(
            f"input_box bound shape mismatch: lower shape {lower.shape} != upper shape {upper.shape}"
        )

    if lower.size == 0:
        raise ValueError("input_box bounds must not be empty")

    if lower.shape[0] != network.input_features:
        raise ValueError(
            f"input_box dimension {lower.shape[0]} does not match network.input_features {network.input_features}"
        )

    current_box = input_box if isinstance(input_box, Box) else Box(lower, upper)

    for position, layer in enumerate(network.layers):
        if isinstance(layer, LinearLayer):
            current_box = current_box.affine(layer.weight, layer.bias)
        elif isinstance(layer, ReLULayer):
            current_box = current_box.relu()
        else:
            layer_name = type(layer).__name__
            pos = getattr(layer, "position", position)
            raise TypeError(
                f"Unsupported layer type '{layer_name}' at sequence position {pos}; "
                "only LinearLayer and ReLULayer are supported"
            )

    if current_box.dimension != network.output_features:
        raise ValueError(
            f"Propagated output dimension {current_box.dimension} does not match "
            f"network.output_features {network.output_features}"
        )

    return current_box
