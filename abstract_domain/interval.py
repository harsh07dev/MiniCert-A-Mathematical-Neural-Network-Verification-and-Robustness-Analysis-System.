"""Scalar interval abstract domain with explicit extended-real bound policy."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True, slots=True)
class Interval:
    """A nonempty closed interval [lower, upper] over real scalar values.

    Infinite lower bounds may only be -inf, and infinite upper bounds may only
    be +inf. Consequently, [-inf, -inf] and [+inf, +inf] are not valid real sets.
    NaN and reversed bounds are rejected.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isinstance(self.lower, Real) or not isinstance(self.upper, Real):
            raise TypeError("Interval bounds must be real scalar values")
        lower = float(self.lower)
        upper = float(self.upper)
        if math.isnan(lower) or math.isnan(upper):
            raise ValueError("Interval bounds cannot be NaN")
        if lower > upper:
            raise ValueError(f"Lower bound {lower} exceeds upper bound {upper}")
        if lower == math.inf or upper == -math.inf:
            raise ValueError("Infinite bounds must be -inf below or +inf above")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @staticmethod
    def _product(endpoint_a: float, endpoint_b: float) -> float:
        if endpoint_a == 0.0 or endpoint_b == 0.0:
            return 0.0
        return endpoint_a * endpoint_b

    @property
    def width(self) -> float:
        """Return upper - lower, which is +inf for an unbounded interval."""
        return self.upper - self.lower

    def __add__(self, other: "Interval") -> "Interval":
        if not isinstance(other, Interval):
            return NotImplemented
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "Interval") -> "Interval":
        if not isinstance(other, Interval):
            return NotImplemented
        return Interval(self.lower - other.upper, self.upper - other.lower)

    def __mul__(self, other: "Interval") -> "Interval":
        if not isinstance(other, Interval):
            return NotImplemented
        products = (
            self._product(self.lower, other.lower),
            self._product(self.lower, other.upper),
            self._product(self.upper, other.lower),
            self._product(self.upper, other.upper),
        )
        return Interval(min(products), max(products))

    def __truediv__(self, other: "Interval") -> "Interval":
        if not isinstance(other, Interval):
            return NotImplemented
        if other.contains(0.0):
            raise ZeroDivisionError("Denominator interval contains zero")
        reciprocal = Interval(1.0 / other.upper, 1.0 / other.lower)
        return self * reciprocal

    def relu(self) -> "Interval":
        """Return the exact interval image of max(0, x)."""
        return Interval(max(0.0, self.lower), max(0.0, self.upper))

    def contains(self, value: Real) -> bool:
        """Return whether a real scalar belongs to this interval."""
        if not isinstance(value, Real):
            raise TypeError("Contained value must be a real scalar")
        scalar = float(value)
        if math.isnan(scalar):
            raise ValueError("Contained value cannot be NaN")
        return self.lower <= scalar <= self.upper

    def contains_interval(self, other: "Interval") -> bool:
        """Return whether every value in other also belongs to this interval."""
        if not isinstance(other, Interval):
            raise TypeError("Contained object must be an Interval")
        return self.lower <= other.lower and other.upper <= self.upper

    def intersection(self, other: "Interval") -> "Interval | None":
        """Return the intersection, or None when the intervals are disjoint."""
        if not isinstance(other, Interval):
            raise TypeError("Intersection requires another Interval")
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        if lower > upper:
            return None
        return Interval(lower, upper)
