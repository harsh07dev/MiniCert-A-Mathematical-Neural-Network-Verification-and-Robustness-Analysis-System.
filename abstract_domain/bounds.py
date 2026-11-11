"""Intermediate bounds and shared Phase 3 refinement data contracts.

This module defines the immutable records that capture the intermediate lower
and upper bounds produced during network-level Interval Bound Propagation
(IBP), together with the shared Phase 3 interfaces consumed by Member 2's
LP-based refinement engine.

Contents:
- :class:`LayerBounds`: bounds entering and leaving a single network layer.
- :class:`IntermediateBounds`: the ordered trace across all network layers.
- :class:`NeuronRef`: a shared identifier for one neuron in a network layer.
- :class:`RefinedBounds`: a proposed refined interval for one neuron.

All records are frozen dataclasses that reuse the immutable :class:`Box`
domain. No network weights or biases are stored here; the records only carry
bound information, so no redundant parameter data is duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from abstract_domain.box import Box

ALLOWED_LAYER_TYPES = ("linear", "relu")
ALLOWED_PHASES = ("pre", "post")


def _validate_index(value: object, name: str) -> int:
    """Coerce a nonnegative integer index, rejecting booleans and negatives."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer index, not a boolean")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer index, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


def _validate_phase(value: object) -> str:
    """Coerce and validate an allowed neuron phase string."""
    if not isinstance(value, str):
        raise TypeError(f"phase must be a string, got {type(value).__name__}")
    if value not in ALLOWED_PHASES:
        raise ValueError(
            f"phase must be one of {ALLOWED_PHASES}, got {value!r}"
        )
    return value


def _validate_layer_type(value: object) -> str:
    """Coerce and validate an allowed layer type string."""
    if not isinstance(value, str):
        raise TypeError(f"layer_type must be a string, got {type(value).__name__}")
    if value not in ALLOWED_LAYER_TYPES:
        raise ValueError(
            f"layer_type must be one of {ALLOWED_LAYER_TYPES}, got {value!r}"
        )
    return value


def _require_box(value: object, name: str) -> Box:
    """Return value only if it is already a Box instance."""
    if not isinstance(value, Box):
        raise TypeError(f"{name} must be a Box, got {type(value).__name__}")
    return value


def _optional_box(value: object, name: str) -> Box | None:
    """Return value if it is None or a Box; otherwise reject it."""
    if value is None:
        return None
    if not isinstance(value, Box):
        raise TypeError(f"{name} must be a Box or None, got {type(value).__name__}")
    return value


def _finite_real(value: object, name: str) -> float:
    """Coerce a finite real scalar, rejecting NaN and infinities."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar value, got {type(value).__name__}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}")
    return scalar


@dataclass(frozen=True)
class LayerBounds:
    """Bounds entering and leaving one network layer.

    A single record captures the abstract state before and after a layer's
    transformation, stored as immutable :class:`Box` instances.

    For a :class:`~models.model_loader.LinearLayer`:
    - ``input_bounds``: the incoming neuron bounds.
    - ``pre_activation_bounds``: the affine output bounds (y = Wx + b).
    - ``post_activation_bounds``: ``None`` (no activation follows a linear layer).
    - ``output_bounds``: identical to ``pre_activation_bounds``.

    For a :class:`~models.model_loader.ReLULayer`:
    - ``input_bounds``: the incoming bounds.
    - ``pre_activation_bounds``: the ReLU argument bounds (identical to input).
    - ``post_activation_bounds``: the ReLU output bounds.
    - ``output_bounds``: identical to ``post_activation_bounds``.

    Args:
        layer_index: Zero-based index of the layer within
            ``MathematicalNetwork.layers`` (not a linear-only index).
        layer_type: ``"linear"`` or ``"relu"``.
        input_bounds: Bounds entering the layer.
        pre_activation_bounds: Bounds immediately before the (non-)activation.
        post_activation_bounds: Bounds immediately after the activation.
        output_bounds: Bounds leaving the layer.

    Raises:
        TypeError: If an index is a boolean or non-integer, an optional Box
            field is neither a Box nor None, or a required Box field is not a Box.
        ValueError: If ``layer_index`` is negative, ``layer_type`` is not an
            allowed value, or the phase-specific structure is violated.
    """

    layer_index: int
    layer_type: str
    input_bounds: Box
    pre_activation_bounds: Box | None
    post_activation_bounds: Box | None
    output_bounds: Box

    def __post_init__(self) -> None:
        index = _validate_index(self.layer_index, "layer_index")
        layer_type = _validate_layer_type(self.layer_type)
        input_bounds = _require_box(self.input_bounds, "input_bounds")
        pre_activation = _optional_box(self.pre_activation_bounds, "pre_activation_bounds")
        post_activation = _optional_box(self.post_activation_bounds, "post_activation_bounds")
        output_bounds = _require_box(self.output_bounds, "output_bounds")

        if layer_type == "linear":
            if post_activation is not None:
                raise ValueError(
                    "linear layer_type must have post_activation_bounds set to None"
                )
            if pre_activation is None:
                raise ValueError(
                    "linear layer_type requires pre_activation_bounds"
                )
        elif layer_type == "relu":
            if pre_activation is None or post_activation is None:
                raise ValueError(
                    "relu layer_type requires both pre_activation_bounds and "
                    "post_activation_bounds"
                )

        object.__setattr__(self, "layer_index", index)
        object.__setattr__(self, "layer_type", layer_type)
        object.__setattr__(self, "input_bounds", input_bounds)
        object.__setattr__(self, "pre_activation_bounds", pre_activation)
        object.__setattr__(self, "post_activation_bounds", post_activation)
        object.__setattr__(self, "output_bounds", output_bounds)


@dataclass(frozen=True)
class IntermediateBounds:
    """Ordered trace of intermediate bounds across a network.

    The trace preserves the exact execution order of the network layers. Each
    entry records the bounds entering and leaving one layer; the record also
    carries the original input bounds and the final output bounds.

    Args:
        input_bounds: The original input box used for propagation.
        layer_bounds: Ordered tuple of :class:`LayerBounds`, one per network layer.
        output_bounds: The final output box produced by propagation.

    Raises:
        TypeError: If ``layer_bounds`` is not a tuple or any record is not a
            ``LayerBounds``, or input/output bounds are not ``Box`` instances.
        ValueError: If adjacent layer dimensions are incompatible, or the final
            output dimension does not agree with ``output_bounds``.
    """

    input_bounds: Box
    layer_bounds: tuple[LayerBounds, ...]
    output_bounds: Box

    def __post_init__(self) -> None:
        input_bounds = _require_box(self.input_bounds, "input_bounds")
        output_bounds = _require_box(self.output_bounds, "output_bounds")

        if not isinstance(self.layer_bounds, tuple):
            raise TypeError(
                f"layer_bounds must be a tuple, got {type(self.layer_bounds).__name__}"
            )
        if not all(isinstance(entry, LayerBounds) for entry in self.layer_bounds):
            raise TypeError("every entry in layer_bounds must be a LayerBounds")

        if len(self.layer_bounds) == 0:
            if input_bounds.dimension != output_bounds.dimension:
                raise ValueError(
                    "empty network requires input and output dimensions to agree, "
                    f"got input {input_bounds.dimension} and output {output_bounds.dimension}"
                )
        else:
            for position, entry in enumerate(self.layer_bounds):
                if entry.layer_index != position:
                    raise ValueError(
                        "layer_bounds must be stored in original execution order: "
                        f"expected layer_index {position} at position {position}, "
                        f"got {entry.layer_index}"
                    )

            first = self.layer_bounds[0]
            if first.input_bounds.dimension != input_bounds.dimension:
                raise ValueError(
                    f"first layer input dimension {first.input_bounds.dimension} does not "
                    f"match input_bounds dimension {input_bounds.dimension}"
                )

            for previous, current in zip(
                self.layer_bounds, self.layer_bounds[1:]
            ):
                if previous.output_bounds.dimension != current.input_bounds.dimension:
                    raise ValueError(
                        f"adjacent layer dimension mismatch: layer "
                        f"{previous.layer_index} outputs dimension "
                        f"{previous.output_bounds.dimension} but layer "
                        f"{current.layer_index} expects input dimension "
                        f"{current.input_bounds.dimension}"
                    )

            last = self.layer_bounds[-1]
            if last.output_bounds.dimension != output_bounds.dimension:
                raise ValueError(
                    f"final layer output dimension {last.output_bounds.dimension} does not "
                    f"match output_bounds dimension {output_bounds.dimension}"
                )

        object.__setattr__(self, "input_bounds", input_bounds)
        object.__setattr__(self, "output_bounds", output_bounds)


@dataclass(frozen=True)
class NeuronRef:
    """Shared identifier for a single neuron in a network layer.

    ``layer_index`` always refers to the actual index within
    ``MathematicalNetwork.layers``, not a hidden-layer or linear-only index.
    The ``phase`` distinguishes the pre-activation argument bounds (``"pre"``)
    from the post-activation output bounds (``"post"``).

    Args:
        layer_index: Zero-based index into ``MathematicalNetwork.layers``.
        phase: ``"pre"`` or ``"post"``.
        neuron_index: Zero-based index of the neuron within the layer.

    Raises:
        TypeError: If either index is a boolean or non-integer, or ``phase`` is
            not a string.
        ValueError: If either index is negative or ``phase`` is not allowed.
    """

    layer_index: int
    phase: str
    neuron_index: int

    def __post_init__(self) -> None:
        layer_index = _validate_index(self.layer_index, "layer_index")
        phase = _validate_phase(self.phase)
        neuron_index = _validate_index(self.neuron_index, "neuron_index")
        object.__setattr__(self, "layer_index", layer_index)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "neuron_index", neuron_index)


@dataclass(frozen=True)
class RefinedBounds:
    """Proposed refined interval for a single neuron.

    This is a pure data record: it does not independently prove that the
    interval is sound or contained within the IBP bounds. That validation
    occurs when refinements are applied by the refinement engine.

    Args:
        neuron: The :class:`NeuronRef` identifying the neuron.
        lower: The proposed lower bound (finite real scalar).
        upper: The proposed upper bound (finite real scalar).

    Raises:
        TypeError: If ``neuron`` is not a ``NeuronRef``, or bounds are not real
            scalars.
        ValueError: If either bound is NaN or infinite, or ``lower`` exceeds
            ``upper``.
    """

    neuron: NeuronRef
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isinstance(self.neuron, NeuronRef):
            raise TypeError(
                f"neuron must be a NeuronRef, got {type(self.neuron).__name__}"
            )
        lower = _finite_real(self.lower, "lower")
        upper = _finite_real(self.upper, "upper")
        if lower > upper:
            raise ValueError(f"lower {lower} must not exceed upper {upper}")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


__all__ = [
    "ALLOWED_LAYER_TYPES",
    "ALLOWED_PHASES",
    "LayerBounds",
    "IntermediateBounds",
    "NeuronRef",
    "RefinedBounds",
]
