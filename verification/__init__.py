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

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class VerificationStatus(str, Enum):
    """Enumeration of possible verification outcomes.

    Abstract output bounds either prove strict dominance of the target class
    over every competitor (CERTIFIED) or fail to prove it (INCONCLUSIVE).
    Overlapping bounds are never reported as a violation: the model is not
    classified as unsafe when the bounds overlap.
    """

    CERTIFIED = "CERTIFIED"        # Target lower bound strictly exceeds every competitor upper bound
    INCONCLUSIVE = "INCONCLUSIVE"  # Bounds do not prove strict dominance


@dataclass(frozen=True)
class VerificationResult:
    """
    Standard result returned by verification queries.

    Attributes:
        status: VerificationStatus outcome.
        target_class: Protected class t.
        target_lower_bound: Certified lower bound on the target logit.
        max_competitor_upper_bound: Maximum certified upper bound over every
            competitor j != t; -inf when no competitor exists.
        explanation: Human-readable account of the decision.
        perturbation_radius: L-infinity radius of the queried region; None when
            the check runs without region context. The orchestration layer
            attaches it with dataclasses.replace.
    """

    status: VerificationStatus
    target_class: int
    target_lower_bound: float
    max_competitor_upper_bound: float
    explanation: str
    perturbation_radius: Optional[float] = None

    @property
    def verification_margin(self) -> float:
        """Margin = target_lower_bound - max_competitor_upper_bound.

        CERTIFIED holds exactly when this margin is strictly positive.
        """
        return self.target_lower_bound - self.max_competitor_upper_bound


__all__ = [
    "VerificationStatus",
    "VerificationResult",
]

