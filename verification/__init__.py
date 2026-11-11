"""
MiniCert - Verification Module
==============================

This module is responsible for neural network robustness verification,
property checking, and optimization-based certification.

Responsibilities:
- Formulating robustness verification problems (e.g., L-infinity perturbation bounds).
- Computing output reachability sets using abstract interpretation.
- Formulating Linear Programming (LP) relaxations of ReLU activations (Planet relaxation).
- Solving linear programs via SciPy (scipy.optimize.linprog) or dedicated solvers.
- Returning formal verification verdicts (CERTIFIED, INCONCLUSIVE).

Assigned Developer:
- Verification, LP Optimization & Integration Engineer (Developer 2)

Note:
Full verification and LP engines will be implemented in subsequent phases.
The structures below define the integration contracts between team members.
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Optional


class VerificationStatus(str, Enum):
    """Enumeration of possible verification outcomes.

    Abstract output bounds either prove strict dominance of the target class
    over every competitor (CERTIFIED) or fail to prove it (INCONCLUSIVE).
    Overlapping bounds are never reported as a violation: the model is not
    classified as unsafe when the bounds overlap.
    """

    CERTIFIED = "CERTIFIED"        # Target lower bound strictly exceeds every competitor upper bound
    INCONCLUSIVE = "INCONCLUSIVE"  # Bounds do not prove strict dominance


def _validate_bound(
    value: object, name: str, *, allow_negative_infinity: bool = False
) -> float:
    """Coerce a bound to float, rejecting NaN and disallowed infinities."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar value")
    scalar = float(value)
    if math.isnan(scalar):
        raise ValueError(f"{name} cannot be NaN")
    if math.isinf(scalar) and not (allow_negative_infinity and scalar < 0.0):
        allowed = " or -inf" if allow_negative_infinity else ""
        raise ValueError(f"{name} must be finite{allowed}")
    return scalar


@dataclass(frozen=True)
class VerificationResult:
    """
    Immutable result returned by verification queries.

    The record is a pure value object: independent of PyTorch, SciPy, and
    Streamlit, free of solver state, and not modifiable after construction.
    ``verification_margin`` is derived from the two stored bounds instead of
    being stored again, so a margin can never disagree with the bounds it
    summarizes.

    Construction validates every field and the status/margin relationship: a
    result may claim CERTIFIED only when the derived margin is strictly
    positive, and an INCONCLUSIVE result must have a non-positive margin.
    Overlapping bounds therefore cannot be reported as a violation.

    The result does not know how many output classes exist; only the target
    class index and the two bounds are carried.

    Attributes:
        status: VerificationStatus outcome.
        target_class: Protected class index t; nonnegative.
        target_lower_bound: Certified lower bound on the target logit.
        max_competitor_upper_bound: Maximum certified upper bound over every
            competitor j != t; -inf when no competitor exists.
        explanation: Human-readable account of the decision.
        perturbation_epsilon: L-infinity radius of the queried region; None
            when the check runs without region context. The orchestration
            layer attaches it with dataclasses.replace.
    """

    status: VerificationStatus
    target_class: int
    target_lower_bound: float
    max_competitor_upper_bound: float
    explanation: str
    perturbation_epsilon: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("status must be a VerificationStatus member")
        if isinstance(self.target_class, bool) or not isinstance(self.target_class, int):
            raise TypeError("target_class must be an integer class index")
        if self.target_class < 0:
            raise ValueError(f"target_class must be nonnegative, got {self.target_class}")
        target_lower = _validate_bound(self.target_lower_bound, "target_lower_bound")
        max_competitor_upper = _validate_bound(
            self.max_competitor_upper_bound,
            "max_competitor_upper_bound",
            allow_negative_infinity=True,
        )
        if not isinstance(self.explanation, str):
            raise TypeError("explanation must be a string")
        if not self.explanation.strip():
            raise ValueError("explanation must not be empty")
        epsilon = self.perturbation_epsilon
        if epsilon is not None:
            epsilon = _validate_bound(epsilon, "perturbation_epsilon")
            if epsilon < 0.0:
                raise ValueError(f"perturbation_epsilon must be nonnegative, got {epsilon}")

        margin = target_lower - max_competitor_upper
        if self.status is VerificationStatus.CERTIFIED and margin <= 0.0:
            raise ValueError(
                f"CERTIFIED requires a strictly positive verification margin, got {margin}"
            )
        if self.status is VerificationStatus.INCONCLUSIVE and margin > 0.0:
            raise ValueError(
                f"INCONCLUSIVE requires a non-positive verification margin, got {margin}"
            )

        object.__setattr__(self, "target_lower_bound", target_lower)
        object.__setattr__(self, "max_competitor_upper_bound", max_competitor_upper)
        if epsilon is not None:
            object.__setattr__(self, "perturbation_epsilon", epsilon)

    @property
    def verification_margin(self) -> float:
        """Margin = target_lower_bound - max_competitor_upper_bound.

        CERTIFIED holds exactly when this margin is strictly positive.
        """
        return self.target_lower_bound - self.max_competitor_upper_bound

    def __str__(self) -> str:
        epsilon = (
            "none" if self.perturbation_epsilon is None else f"{self.perturbation_epsilon:.6g}"
        )
        return (
            f"{self.status.value} (target_class={self.target_class}, "
            f"target_lower_bound={self.target_lower_bound:.6g}, "
            f"max_competitor_upper_bound={self.max_competitor_upper_bound:.6g}, "
            f"margin={self.verification_margin:.6g}, epsilon={epsilon})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain serializable mapping with the status stored by value.

        The derived verification margin is included for readability. Infinite
        margins (single-class problems) require a JSON encoder that accepts
        non-finite floats.
        """
        return {
            "status": self.status.value,
            "target_class": self.target_class,
            "target_lower_bound": self.target_lower_bound,
            "max_competitor_upper_bound": self.max_competitor_upper_bound,
            "verification_margin": self.verification_margin,
            "perturbation_epsilon": self.perturbation_epsilon,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerificationResult":
        """Rebuild a result from to_dict output.

        The stored verification margin is derived and ignored on load. The
        status may be supplied as a VerificationStatus member or its string
        value; validation is otherwise identical to direct construction.
        """
        status = payload["status"]
        return cls(
            status=VerificationStatus(status) if isinstance(status, str) else status,
            target_class=payload["target_class"],
            target_lower_bound=payload["target_lower_bound"],
            max_competitor_upper_bound=payload["max_competitor_upper_bound"],
            explanation=payload["explanation"],
            perturbation_epsilon=payload.get("perturbation_epsilon"),
        )


from verification.refinement_validation import (
    evaluate_intermediate_activations,
    sample_region_inputs,
    validate_refined_bounds_empirically,
)
from verification.lp import (
    LPConstraintBuilder,
    LPProblem,
    VariableLayout,
    build_lp_problem,
)
from verification.lp_solver import (
    BoundValidationStatus,
    LPSolverResult,
    LPSolverStatus,
    solve_lp,
)
from verification.refinement import (
    NeuronRefinementOutcome,
    RefinementResult,
    RefinementStatus,
    refine_neuron_bounds,
)

__all__ = [
    "VerificationStatus",
    "VerificationResult",
    "LPConstraintBuilder",
    "LPProblem",
    "VariableLayout",
    "build_lp_problem",
    "BoundValidationStatus",
    "LPSolverResult",
    "LPSolverStatus",
    "solve_lp",
    "NeuronRefinementOutcome",
    "RefinementResult",
    "RefinementStatus",
    "refine_neuron_bounds",
]

