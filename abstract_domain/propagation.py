"""Network-level Interval Bound Propagation (IBP).

This module provides the :func:`propagate_network` function to propagate a
multidimensional input box through an extracted
:class:`~models.model_loader.MathematicalNetwork` using the
:class:`~abstract_domain.box.Box` abstract domain, and the
:func:`get_intermediate_bounds` function to record the bounds entering and
leaving every layer while preserving exact execution order.
"""

from __future__ import annotations

import numpy as np

from abstract_domain.bounds import IntermediateBounds, LayerBounds
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


def _coerce_input_box(network: MathematicalNetwork, input_box: BoxProtocol) -> Box:
    """Validate a network and input box, returning a Box input.

    The validation mirrors :func:`propagate_network` exactly. External
    :class:`~verification.regions.BoxProtocol` implementations are converted to
    a :class:`Box` only when necessary.
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

    return input_box if isinstance(input_box, Box) else Box(lower, upper)


def get_intermediate_bounds(
    network: MathematicalNetwork,
    input_box: BoxProtocol,
) -> IntermediateBounds:
    """Propagate an input box and record the intermediate bounds at every layer.

    Iterates through the network layers in their exact execution order, applying
    sound IBP transformers and capturing the bounds entering and leaving each
    layer. The returned :class:`~abstract_domain.bounds.IntermediateBounds`
    trace is guaranteed to match the final output of :func:`propagate_network`
    for identical inputs.

    For a :class:`~models.model_loader.LinearLayer` the affine result is stored
    as both ``pre_activation_bounds`` and ``output_bounds`` (no activation
    follows). For a :class:`~models.model_loader.ReLULayer` the incoming bounds
    are stored as ``pre_activation_bounds`` and the ReLU result as
    ``post_activation_bounds`` and ``output_bounds``.

    Args:
        network: An extracted float64 MathematicalNetwork containing ordered layers.
        input_box: An input bound representation conforming to BoxProtocol.

    Returns:
        An IntermediateBounds record containing the original input box, an
        ordered per-layer trace, and the final output box.

    Raises:
        TypeError: If network is not a MathematicalNetwork, input_box does not
            satisfy BoxProtocol, or an unsupported layer type is encountered.
        ValueError: If input_box has invalid dimensions, does not match
            network.input_features, or the final output dimension does not match
            network.output_features.
    """
    current_box = _coerce_input_box(network, input_box)
    input_bounds = current_box

    layer_bounds: list[LayerBounds] = []

    for index, layer in enumerate(network.layers):
        if isinstance(layer, LinearLayer):
            pre_activation = current_box.affine(layer.weight, layer.bias)
            layer_bounds.append(
                LayerBounds(
                    layer_index=index,
                    layer_type="linear",
                    input_bounds=current_box,
                    pre_activation_bounds=pre_activation,
                    post_activation_bounds=None,
                    output_bounds=pre_activation,
                )
            )
            current_box = pre_activation
        elif isinstance(layer, ReLULayer):
            post_activation = current_box.relu()
            layer_bounds.append(
                LayerBounds(
                    layer_index=index,
                    layer_type="relu",
                    input_bounds=current_box,
                    pre_activation_bounds=current_box,
                    post_activation_bounds=post_activation,
                    output_bounds=post_activation,
                )
            )
            current_box = post_activation
        else:
            layer_name = type(layer).__name__
            pos = getattr(layer, "position", index)
            raise TypeError(
                f"Unsupported layer type '{layer_name}' at sequence position {pos}; "
                "only LinearLayer and ReLULayer are supported"
            )

    output_bounds = current_box

    if output_bounds.dimension != network.output_features:
        raise ValueError(
            f"Propagated output dimension {output_bounds.dimension} does not match "
            f"network.output_features {network.output_features}"
        )

    return IntermediateBounds(
        input_bounds=input_bounds,
        layer_bounds=tuple(layer_bounds),
        output_bounds=output_bounds,
    )
