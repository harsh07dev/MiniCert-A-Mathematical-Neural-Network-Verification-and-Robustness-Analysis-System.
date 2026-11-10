"""Input perturbation region specifications for L-infinity robustness queries.

This module defines the concrete input sets considered by adversarial
robustness queries. It deliberately performs no neural network propagation
and no abstract-domain arithmetic: the vector Box domain (affine transforms,
ReLU, joins, splits) belongs to ``abstract_domain`` and is Developer 1's
responsibility. :class:`InputRegion` only concretizes the L-infinity ball
around one concrete input, clipped to the configured valid input domain, and
exposes the result through the same float64 ``(lower, upper)`` bound-array
contract that the propagation layer will consume.

Input shape contract: the region operates on one flattened feature vector of
shape ``(input_features,)``. Flattening image-shaped inputs (for example MNIST
``(1, 28, 28)`` samples into 784 features) is the caller's responsibility,
exactly as with ``MathematicalNetwork.forward``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


@runtime_checkable
class BoxProtocol(Protocol):
    """Consumer-side contract for the vector Box abstract domain.

    Any object exposing component-wise float64 lower and upper bound arrays
    satisfies this protocol. Structural conformance requires no inheritance
    from verification code: Developer 1's Box implementation is expected to
    expose exactly these two properties so that propagation can consume both
    Box instances and InputRegion values through one interface.
    """

    @property
    def lower(self) -> FloatArray:
        """Lower bounds of shape (n,), dtype float64."""
        ...

    @property
    def upper(self) -> FloatArray:
        """Upper bounds of shape (n,), dtype float64."""
        ...


def _validate_vector(value: object, name: str, *, finite: bool) -> np.ndarray:
    """Coerce a 1-D bound vector to an independent float64 array."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} cannot contain NaN")
    if finite and np.any(np.isinf(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _validate_input_limit(
    value: object, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Coerce one configured input bound to a shape-matching float64 array.

    Scalar limits are broadcast to the input shape; infinite limits are
    allowed and mark explicitly unbounded components.
    """
    if value is None:
        bound = -math.inf if name == "input_lower" else math.inf
        return np.full(shape, bound)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(shape, float(array))
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector or scalar, got shape {array.shape}")
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match input shape {shape}")
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} cannot contain NaN")
    return array.copy()


def _validate_radius(value: object) -> float:
    """Validate the L-infinity perturbation radius epsilon."""
    if not isinstance(value, Real):
        raise TypeError("radius must be a real scalar value")
    radius = float(value)
    if math.isnan(radius):
        raise ValueError("radius cannot be NaN")
    if math.isinf(radius):
        raise ValueError("radius must be finite")
    if radius < 0.0:
        raise ValueError(f"radius must be nonnegative, got {radius}")
    return radius


@dataclass(frozen=True, eq=False)
class InputRegion:
    """Bounded L-infinity perturbation region around a concrete input.

    For a concrete input x and perturbation radius epsilon the region is

        B_epsilon(x) = {x' : ||x' - x||_infinity <= epsilon}

    restricted to the configured valid input domain [input_lower, input_upper]:

        lower = max(x - epsilon, input_lower)
        upper = min(x + epsilon, input_upper)

    The concrete input must lie inside the configured domain, which makes the
    region nonempty by construction: ``lower <= x <= upper`` always holds for
    every component.

    This class performs no abstract-domain operations. It only stores the
    specification and concretizes the trivial initial box; affine, ReLU, join,
    and split transformers remain the Box domain's responsibility. Because the
    region exposes float64 ``lower``/``upper`` arrays, it satisfies
    :class:`BoxProtocol` and can serve as the initial box for propagation.

    Attributes:
        center: Concrete input x of shape (input_features,).
        radius: L-infinity perturbation radius epsilon; finite, nonnegative.
        input_lower: Optional per-dimension valid input lower bounds; a scalar
            is broadcast to the input shape; None means unbounded below.
        input_upper: Optional per-dimension valid input upper bounds; a scalar
            is broadcast to the input shape; None means unbounded above.
    """

    center: FloatArray
    radius: float
    input_lower: FloatArray | None = None
    input_upper: FloatArray | None = None

    def __post_init__(self) -> None:
        center = _validate_vector(self.center, "center", finite=True)
        radius = _validate_radius(self.radius)
        input_lower = _validate_input_limit(self.input_lower, "input_lower", center.shape)
        input_upper = _validate_input_limit(self.input_upper, "input_upper", center.shape)
        if np.any(input_lower > input_upper):
            raise ValueError("input_lower must never exceed input_upper")
        if np.any(center < input_lower) or np.any(center > input_upper):
            raise ValueError("center must lie within the configured input bounds")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "input_lower", input_lower)
        object.__setattr__(self, "input_upper", input_upper)

    def __eq__(self, other: object) -> bool:
        """Structural equality on all specification fields."""
        if not isinstance(other, InputRegion):
            return NotImplemented
        return (
            self.radius == other.radius
            and np.array_equal(self.center, other.center)
            and np.array_equal(self.input_lower, other.input_lower)
            and np.array_equal(self.input_upper, other.input_upper)
        )

    @property
    def lower(self) -> FloatArray:
        """Return max(center - radius, input_lower) for every component."""
        return np.maximum(self.center - self.radius, self.input_lower)

    @property
    def upper(self) -> FloatArray:
        """Return min(center + radius, input_upper) for every component."""
        return np.minimum(self.center + self.radius, self.input_upper)

    @property
    def dimension(self) -> int:
        """Return the number of input features."""
        return int(self.center.shape[0])

    def contains(self, point: FloatArray) -> bool:
        """Return whether a concrete point belongs to this region.

        Args:
            point: Concrete input of shape (input_features,).

        Returns:
            True when lower <= point <= upper component-wise.

        Raises:
            ValueError: If the point shape does not match the region shape or
                the point contains NaN.
        """
        array = np.asarray(point, dtype=np.float64)
        if array.shape != self.center.shape:
            raise ValueError(
                f"point shape {array.shape} does not match region shape {self.center.shape}"
            )
        if np.any(np.isnan(array)):
            raise ValueError("point cannot contain NaN")
        return bool(np.all((self.lower <= array) & (array <= self.upper)))
