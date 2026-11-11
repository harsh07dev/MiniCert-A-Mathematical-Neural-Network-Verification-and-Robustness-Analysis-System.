"""Apply refined neuron bounds and re-propagate the network.

This module provides :func:`propagate_with_refinements`, which takes an
extracted network, an input perturbation region, and a collection of externally
supplied, validated :class:`~abstract_domain.bounds.RefinedBounds` proposals.
It applies each refinement to the targeted neuron and phase, then recomputes all
downstream network bounds using the existing affine and ReLU abstract
transformers.

The refinements are assumed to come from an external, sound source (for example
a Linear Programming relaxation). This module only applies them and recomputes
bounds; it does not generate constraints, solve linear programs, or establish
that a refinement is itself sound.
"""

from __future__ import annotations

from typing import Iterable

from abstract_domain.bounds import IntermediateBounds, LayerBounds, NeuronRef, RefinedBounds
from abstract_domain.box import Box
from abstract_domain.propagation import get_intermediate_bounds
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification.regions import InputRegion

# Numerical tolerance used for the containment and consistency checks. It guards
# against floating-point rounding when a refinement is compared against bounds
# computed through a different arithmetic path (e.g. an LP solver). It is small
# enough that meaningfully invalid refinements are still rejected rather than
# silently accepted or clipped.
_TOLERANCE = 1e-12


def _validate_refined_collection(
    refined_bounds: object,
) -> tuple[RefinedBounds, ...]:
    """Coerce the refinement collection to a tuple of RefinedBounds."""
    if not isinstance(refined_bounds, (tuple, list)):
        raise TypeError(
            "refined_bounds must be a tuple or list of RefinedBounds, "
            f"got {type(refined_bounds).__name__}"
        )
    items = tuple(refined_bounds)
    if not all(isinstance(item, RefinedBounds) for item in items):
        raise TypeError("every element of refined_bounds must be a RefinedBounds")
    return items


def _validate_no_duplicates(refined_bounds: tuple[RefinedBounds, ...]) -> None:
    """Reject duplicate NeuronRef entries across the refinement collection."""
    seen: set[tuple[int, str, int]] = set()
    for item in refined_bounds:
        neuron = item.neuron
        key = (neuron.layer_index, neuron.phase, neuron.neuron_index)
        if key in seen:
            raise ValueError(
                f"duplicate refinement for neuron (layer_index={neuron.layer_index}, "
                f"phase={neuron.phase!r}, neuron_index={neuron.neuron_index})"
            )
        seen.add(key)


def _validate_neuron_reference(
    network: MathematicalNetwork,
    trace: IntermediateBounds,
    neuron: NeuronRef,
) -> None:
    """Validate that a NeuronRef targets a real, refinement-compatible neuron."""
    layer_index = neuron.layer_index
    if not 0 <= layer_index < len(network.layers):
        raise ValueError(
            f"layer_index {layer_index} is out of range for a network with "
            f"{len(network.layers)} layers"
        )

    layer = network.layers[layer_index]
    if isinstance(layer, LinearLayer):
        if neuron.phase != "pre":
            raise ValueError(
                "linear layers only support 'pre' phase refinements; "
                f"got phase {neuron.phase!r} for layer {layer_index}"
            )
    elif not isinstance(layer, ReLULayer):
        raise TypeError(
            f"unsupported layer type {type(layer).__name__} at layer {layer_index}; "
            "refinement requires LinearLayer or ReLULayer"
        )

    entry = trace.layer_bounds[layer_index]
    reference_box = (
        entry.pre_activation_bounds if neuron.phase == "pre" else entry.output_bounds
    )
    if reference_box is None:
        raise ValueError(
            f"layer {layer_index} has no {neuron.phase!r} activation bounds to refine"
        )
    dimension = reference_box.dimension
    if not 0 <= neuron.neuron_index < dimension:
        raise ValueError(
            f"neuron_index {neuron.neuron_index} is out of range for layer "
            f"{layer_index} with {dimension} neurons"
        )


def _validate_containment(
    trace: IntermediateBounds,
    refined: RefinedBounds,
) -> None:
    """Reject refinements that extend outside the original IBP interval."""
    neuron = refined.neuron
    entry = trace.layer_bounds[neuron.layer_index]
    reference_box = (
        entry.pre_activation_bounds if neuron.phase == "pre" else entry.output_bounds
    )
    if reference_box is None:
        raise ValueError(
            f"layer {neuron.layer_index} has no {neuron.phase!r} activation bounds"
        )
    index = neuron.neuron_index
    ibp_lower = reference_box.lower[index]
    ibp_upper = reference_box.upper[index]

    if not (refined.lower >= ibp_lower - _TOLERANCE and refined.upper <= ibp_upper + _TOLERANCE):
        raise ValueError(
            f"refinement for (layer_index={neuron.layer_index}, phase={neuron.phase!r}, "
            f"neuron_index={index}) with bounds [{refined.lower}, {refined.upper}] extends "
            f"outside the original IBP interval [{ibp_lower}, {ibp_upper}]"
        )


def _apply_refinements(box: Box, refinements: list[RefinedBounds]) -> Box:
    """Return a copy of box with each refined neuron's bounds overridden."""
    if not refinements:
        return box
    lower = box.lower.copy()
    upper = box.upper.copy()
    for refined in refinements:
        index = refined.neuron.neuron_index
        lower[index] = refined.lower
        upper[index] = refined.upper
    return Box(lower, upper)


def _validate_relu_post_consistency(
    post_bounds: Box,
    refinements: Iterable[RefinedBounds],
) -> None:
    """Reject post-activation ReLU refinements inconsistent with the ReLU output.

    For a ReLU layer the post-activation value equals ``relu(pre)``, so a
    refinement must lie within the interval produced by the (possibly refined)
    pre-activation bounds.
    """
    for refined in refinements:
        index = refined.neuron.neuron_index
        relu_lower = post_bounds.lower[index]
        relu_upper = post_bounds.upper[index]
        if not (
            refined.lower >= relu_lower - _TOLERANCE
            and refined.upper <= relu_upper + _TOLERANCE
        ):
            raise ValueError(
                f"post-activation refinement for layer "
                f"{refined.neuron.layer_index} neuron {index} with bounds "
                f"[{refined.lower}, {refined.upper}] is inconsistent with the "
                f"ReLU output interval [{relu_lower}, {relu_upper}]"
            )


def propagate_with_refinements(
    network: MathematicalNetwork,
    input_region: InputRegion,
    refined_bounds: tuple[RefinedBounds, ...],
) -> IntermediateBounds:
    """Apply refined neuron bounds and recompute all downstream network bounds.

    The function first computes the original Interval Bound Propagation (IBP)
    intermediate bounds, then validates that every supplied refinement targets a
    real neuron, uses a valid phase for its layer type, and lies within the
    original IBP interval. It then re-propagates the network layer by layer,
    overriding the targeted neuron bounds and recomputing every subsequent layer
    with the existing affine and ReLU transformers.

    Refinements are applied deterministically and independently of the input
    collection order because duplicate ``NeuronRef`` entries are rejected, so
    each refinement targets a distinct neuron and phase.

    Args:
        network: An extracted float64 MathematicalNetwork containing ordered layers.
        input_region: The L-infinity input perturbation region.
        refined_bounds: A tuple (or list) of validated RefinedBounds proposals.

    Returns:
        A fresh IntermediateBounds trace with the refinements applied and all
        downstream bounds recomputed.

    Raises:
        TypeError: If network is not a MathematicalNetwork, input_region is not
            an InputRegion, refined_bounds is not a tuple/list of RefinedBounds,
            or a layer is unsupported.
        ValueError: If the input dimension does not match the network, a neuron
            reference is out of range or uses an invalid phase for its layer,
            refinements are duplicated or extend outside the original IBP
            interval, or a post-activation ReLU refinement is inconsistent.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(network).__name__}")

    if not isinstance(input_region, InputRegion):
        raise TypeError(
            f"input_region must be an InputRegion, got {type(input_region).__name__}"
        )
    if input_region.dimension != network.input_features:
        raise ValueError(
            f"input_region dimension {input_region.dimension} does not match "
            f"network.input_features {network.input_features}"
        )

    refined = _validate_refined_collection(refined_bounds)
    _validate_no_duplicates(refined)

    trace = get_intermediate_bounds(network, input_region)

    for item in refined:
        _validate_neuron_reference(network, trace, item.neuron)
        _validate_containment(trace, item)

    pre_refinements: dict[int, list[RefinedBounds]] = {}
    post_refinements: dict[int, list[RefinedBounds]] = {}
    for item in refined:
        layer_index = item.neuron.layer_index
        if item.neuron.phase == "pre":
            pre_refinements.setdefault(layer_index, []).append(item)
        else:
            post_refinements.setdefault(layer_index, []).append(item)

    current_box = trace.input_bounds
    new_layer_bounds: list[LayerBounds] = []

    for index, layer in enumerate(network.layers):
        layer_pre = pre_refinements.get(index, [])
        layer_post = post_refinements.get(index, [])

        if isinstance(layer, LinearLayer):
            pre_activation = current_box.affine(layer.weight, layer.bias)
            pre_activation = _apply_refinements(pre_activation, layer_pre)
            new_layer_bounds.append(
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
            pre_activation = current_box
            pre_activation = _apply_refinements(pre_activation, layer_pre)
            post_activation = pre_activation.relu()
            _validate_relu_post_consistency(post_activation, layer_post)
            post_activation = _apply_refinements(post_activation, layer_post)
            new_layer_bounds.append(
                LayerBounds(
                    layer_index=index,
                    layer_type="relu",
                    input_bounds=pre_activation,
                    pre_activation_bounds=pre_activation,
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
        input_bounds=trace.input_bounds,
        layer_bounds=tuple(new_layer_bounds),
        output_bounds=output_bounds,
    )


__all__ = ["propagate_with_refinements"]
