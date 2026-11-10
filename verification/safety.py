"""Safety checking over abstract output bounds.

This module decides whether abstract output bounds certify that a target class
strictly dominates every competitor over the queried input region:

    for every competitor j != t:  L_t > U_j

where L_t is the certified lower bound of the target logit and U_j the
certified upper bound of competitor j.

The checker performs no neural network propagation, performs no LP refinement,
depends only on the Python standard library and the verification result
contract, and never reports a violation: when the bounds overlap or the strict
condition fails for any reason, the verdict is INCONCLUSIVE.

Numerical policy: comparisons use IEEE double equality and ordering on already
finite float64 bounds with no added tolerance. A margin of exactly zero is not
certified; only a strictly positive margin certifies. NaN and infinite bounds
are rejected as invalid inputs rather than compared, so no NaN-based
comparison can occur. Certification is only as sound as the upstream bound
computation: float64 arithmetic without directed rounding does not establish a
machine-checked mathematical certificate.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable

from verification import VerificationResult, VerificationStatus


def _to_finite_bounds(values: Iterable[float], name: str) -> tuple[float, ...]:
    """Coerce an iterable of real numbers to a tuple of finite floats."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of real numbers")
    try:
        items = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real numbers") from exc
    if not items:
        raise ValueError(f"{name} must not be empty")
    bounds = []
    for item in items:
        if not isinstance(item, Real):
            raise TypeError(f"{name} entries must be real numbers, got {type(item).__name__}")
        scalar = float(item)
        if math.isnan(scalar):
            raise ValueError(f"{name} cannot contain NaN")
        if math.isinf(scalar):
            raise ValueError(f"{name} must contain only finite bounds")
        bounds.append(scalar)
    return tuple(bounds)


class SafetyChecker:
    """Certification check over abstract output bounds.

    Stateless: a single instance can safely serve any number of checks.
    """

    def check(
        self,
        lower_bounds: Iterable[float],
        upper_bounds: Iterable[float],
        target_class: int,
    ) -> VerificationResult:
        """Return CERTIFIED iff the target class strictly dominates all competitors.

        Args:
            lower_bounds: Certified lower output bounds, one per class.
            upper_bounds: Certified upper output bounds, one per class.
            target_class: Protected class index.

        Returns:
            VerificationResult carrying target_lower_bound, the maximum
            competitor upper bound, and the verification margin. A
            single-class problem is certified vacuously, with
            max_competitor_upper_bound = -inf and margin = +inf.

        Raises:
            TypeError: Non-real bounds, non-iterable vectors, or a non-integer
                target_class.
            ValueError: NaN or infinite bounds, empty vectors, mismatched
                lengths, lower > upper entries, or an out-of-range
                target_class.
        """
        lower = _to_finite_bounds(lower_bounds, "lower_bounds")
        upper = _to_finite_bounds(upper_bounds, "upper_bounds")
        if len(lower) != len(upper):
            raise ValueError(
                f"lower_bounds length {len(lower)} does not match upper_bounds length {len(upper)}"
            )
        for index, (low, high) in enumerate(zip(lower, upper, strict=True)):
            if low > high:
                raise ValueError(f"lower bound {low} exceeds upper bound {high} at class {index}")
        if isinstance(target_class, bool) or not isinstance(target_class, int):
            raise TypeError("target_class must be an integer class index")
        if not 0 <= target_class < len(lower):
            raise ValueError(f"target_class {target_class} is out of range for {len(lower)} classes")

        target_lower = lower[target_class]
        if len(lower) == 1:
            max_competitor_upper = -math.inf
            worst_class: int | None = None
        else:
            worst_class = max(
                (index for index in range(len(upper)) if index != target_class),
                key=lambda index: upper[index],
            )
            max_competitor_upper = upper[worst_class]

        margin = target_lower - max_competitor_upper
        if target_lower > max_competitor_upper:
            status = VerificationStatus.CERTIFIED
            if worst_class is None:
                explanation = (
                    f"Target class {target_class} is the only output class; "
                    f"dominance holds vacuously with lower bound {target_lower:.6g}"
                )
            else:
                explanation = (
                    f"Target class {target_class} lower bound {target_lower:.6g} strictly "
                    f"exceeds the maximum competitor upper bound {max_competitor_upper:.6g} "
                    f"(class {worst_class}); certification margin {margin:.6g}"
                )
        else:
            status = VerificationStatus.INCONCLUSIVE
            explanation = (
                f"Target class {target_class} lower bound {target_lower:.6g} does not strictly "
                f"exceed competitor class {worst_class} upper bound {max_competitor_upper:.6g}; "
                f"margin {margin:.6g}"
            )

        return VerificationResult(
            status=status,
            target_class=target_class,
            target_lower_bound=target_lower,
            max_competitor_upper_bound=max_competitor_upper,
            explanation=explanation,
        )
