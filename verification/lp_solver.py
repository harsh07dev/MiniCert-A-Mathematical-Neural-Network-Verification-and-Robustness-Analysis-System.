"""
MiniCert - LP Solver with Conservative Bound Validation
========================================================

Solves the Linear Programs produced by :mod:`verification.lp` with SciPy's
HiGHS implementation of ``scipy.optimize.linprog`` and returns a structured
result that separates a *primal witness* (the objective value at the returned
solution) from a *conservative bound* (the side MiniCert needs for certified
refinement).

Interface
---------
``solve_lp(problem, objective, *, maximize=False, method="highs", options=None)``

``problem`` is an :class:`verification.lp.LPProblem` in linprog matrix form
(``min c @ x`` s.t. ``A_ub @ x <= b_ub``, ``A_eq @ x == b_eq`` and
per-variable bounds). ``objective`` replaces the stored objective and may be:

- ``None``: use the problem's stored unit-vector objective,
- a :class:`~abstract_domain.bounds.NeuronRef`: the unit vector selecting that
  neuron's LP variable (must be valid for the problem's layout),
- a raw vector of length ``n_variables``: any finite objective coefficients.

Maximization is implemented by negating the objective, solving the
minimization, and mapping every quantity back to the original direction.

Conservative-bound semantics
----------------------------
For a minimization problem the feasible-set minimum ``m`` satisfies::

    dual objective (feasible dual)  <=  m  <=  primal objective (feasible x)

so the conservative side is the **lower** bound ``dual_objective``. For
maximization the conservative side is the **upper** bound, obtained as the
negated dual objective of the negated (minimization) problem. Both directions
are exposed uniformly: ``LPSolverResult.lower_bound`` and
``LPSolverResult.upper_bound``, where exactly one of the two is the
conservative side and the other is the primal witness.

Validation method and honesty
-----------------------------
The returned HiGHS marginals provide dual multipliers whose sign
conventions were verified empirically against hand-computed duals (they
differ from the SciPy docstring phrasing for bound marginals): for
``A_ub @ x <= b_ub`` the marginals ``y`` satisfy ``y <= 0``; for variable
bounds ``l <= x <= u`` the lower marginals ``lambda_lb`` are nonnegative
and the upper marginals ``lambda_ub`` are nonpositive; the dual objective
is::

    dual = b_ub @ y + b_eq @ nu + l @ lambda_lb + u @ lambda_ub

and stationarity requires ``A_ub.T @ y + A_eq.T @ nu + lambda_lb +
lambda_ub == c``. The solver checks, at the documented relative tolerance
``_VALIDATION_TOL = 1e-8`` alongside explicit absolute safeguards:

1. primal feasibility: inequality, equality, and bound violations tracked as
   both raw absolute residuals and row-locally normalized residuals (so large
   unrelated coordinates or right-hand sides cannot hide local infeasibility,
   with normalized residual ``<= 1e-8`` and row-norm-adjusted absolute
   residual ``<= _MAX_ABS_PRIMAL_TOL = 1e-4``),
2. dual sign feasibility: checked **independently** per multiplier family
   (``y <= 0``, ``lambda_lb >= 0``, ``lambda_ub <= 0``) using each family's
   own scale ``min(max(1.0, ||m||_inf), _MAX_MULTIPLIER_SIGN_SCALE)`` rather
   than ``abs(dual)`` so large dual or primal objectives never mask
   wrong-sign multipliers exceeding ``_MAX_ABS_DUAL_SIGN_TOL = 1e-6``,
3. dual stationarity residual (normalized by gradient and objective norms),
4. duality gap in the original problem direction: checks both relative gap
   ``>= -1e-8`` and absolute gap ``>= -_MAX_ABS_NEGATIVE_GAP = -1e-4``, and
   records both absolute and relative magnitudes.

When every check passes the result is marked
``BoundValidationStatus.VALIDATED`` and ``certified_bound`` exposes the
conservative side; otherwise the result is ``UNVALIDATED`` and
``certified_bound`` is ``None``, so a numerically unvalidated objective can
never be mistaken for a certified refined bound.

This validation is **tolerance-based numerical checking**, not a rigorous
interval-arithmetic or exact-arithmetic certificate: it makes no claim about
rounding errors accumulated inside the solver. No numerical error guarantee
is invented here; the tolerances, caps, and checked quantities are documented
so downstream stages can reason about what ``VALIDATED`` means.

Status mapping
--------------
``scipy.optimize.linprog`` status codes are mapped to
:class:`LPSolverStatus` as follows: 0 -> ``OPTIMAL`` (``success=True``), 1 ->
``ITERATION_LIMIT``, 2 -> ``INFEASIBLE``, 3 -> ``UNBOUNDED``, 4 ->
``NUMERICAL_ISSUE``. Any other status is ``FAILED``. Only ``OPTIMAL`` reports
``success=True``; in particular SciPy's ``success=True`` for an iteration
limit is not propagated, because such a solution is a feasible witness only,
never an optimum. ``ITERATION_LIMIT`` still records the primal objective of a
feasible returned point (a valid one-sided bound) but never a certified
conservative bound.

The ``"pre"``/``"post"`` NeuronRef phase conventions and the constraint layout
of :mod:`verification.lp` are reused unchanged; this module does not modify
the LP builder's mathematical formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
import time
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from abstract_domain.bounds import NeuronRef
from verification.lp import FloatArray, LPProblem

# Relative validation tolerance used across primal, dual, and duality-gap
# checks. A normalized residual of 1e-8 means a violation of at most one part
# in 1e8 of the local scale. This is a numerical check, not an exact
# certificate.
_VALIDATION_TOL = 1e-8

# Multiplier-specific sign tolerance cap. Each multiplier family (y,
# lambda_lb, lambda_ub) scales its sign check only by its own infinity norm,
# floored at 1.0 and capped at _MAX_MULTIPLIER_SIGN_SCALE (100.0). That allows
# ~1e-8 relative float noise on moderately large multipliers while guaranteeing
# that no absolute wrong-sign multiplier above _MAX_ABS_DUAL_SIGN_TOL (1e-6)
# can be accepted merely because the dual or primal objective is large.
_MAX_ABS_DUAL_SIGN_TOL = 1e-6
_MAX_MULTIPLIER_SIGN_SCALE = _MAX_ABS_DUAL_SIGN_TOL / _VALIDATION_TOL

# Absolute caps on constraint-norm-adjusted primal infeasibility and negative
# duality gap. Normalized checks alone allow O(1..100) primal constraint
# violations or reversed witness gaps when problem magnitudes reach 1e10+.
# Capping both at 1e-4 preserves legitimate IEEE-754 float64 rounding while
# rejecting substantially infeasible primal points or negative gaps.
_MAX_ABS_PRIMAL_TOL = 1e-4
_MAX_ABS_NEGATIVE_GAP = 1e-4

__all__ = [
    "BoundValidationStatus",
    "LPSolverResult",
    "LPSolverStatus",
    "solve_lp",
]


class LPSolverStatus(str, Enum):
    """Outcome of a ``linprog`` call, mapped from its integer status codes."""

    OPTIMAL = "OPTIMAL"                    # optimum found and reported
    INFEASIBLE = "INFEASIBLE"              # empty feasible set
    UNBOUNDED = "UNBOUNDED"                # objective unbounded in the direction solved
    ITERATION_LIMIT = "ITERATION_LIMIT"    # time/iteration limit hit; point not optimal
    NUMERICAL_ISSUE = "NUMERICAL_ISSUE"    # solver reported numerical difficulties
    INVALID_PROBLEM = "INVALID_PROBLEM"    # the solver rejected the problem input
    FAILED = "FAILED"                      # unexpected solver error


class BoundValidationStatus(str, Enum):
    """Whether the conservative bound passed the documented numerical checks.

    ``VALIDATED`` means the tolerance-based primal and dual feasibility checks
    all passed (see the module docstring). ``UNVALIDATED`` means at least one
    check failed, no dual information was available, or the solve was not
    optimal; the conservative bound must not be used for certification.
    """

    VALIDATED = "VALIDATED"
    UNVALIDATED = "UNVALIDATED"


@dataclass(frozen=True)
class LPSolverResult:
    """Structured result of one ``solve_lp`` call.

    All bound quantities are expressed in the *original* problem direction:
    for ``maximize=True`` the objective value and bounds are those of
    ``c @ x`` (not of the internally negated objective).

    Args:
        status: Mapped solver outcome.
        success: ``True`` exactly when ``status`` is ``OPTIMAL``.
        maximize: Whether the original problem was a maximization.
        objective_neuron: The neuron the objective selects, or ``None`` when
            a raw objective vector was supplied.
        solution: Primal variable vector returned by the solver (read-only),
            or ``None`` when no usable point exists.
        objective_value: Objective value ``c @ solution`` in the original
            direction; the primal witness (a valid upper bound of the minimum
            for minimization, lower bound of the maximum for maximization)
            whenever ``solution`` is feasible.
        conservative_bound: The conservative side in the original direction:
            the dual lower bound for minimization, the negated dual objective
            for maximization. ``None`` when dual information is unavailable.
        duality_gap: ``upper_bound - lower_bound`` in the original direction;
            nonnegative for any feasible dual/primal pair, ``None`` when the
            two sides are not both available.
        max_primal_residual: Largest scaled violation of the constraints and
            variable bounds at ``solution`` (``None`` without a solution).
        max_dual_residual: Largest scaled violation of dual sign feasibility
            and stationarity (``None`` when dual information is unavailable).
        validation_status: Whether the conservative bound passed the
            documented numerical checks.
        execution_time_seconds: Wall-clock solve time including validation.
        message: Human-readable account of the outcome.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If status/success disagree (a failed solve can never be
            reported as successful), a validation-validated result lacks its
            bound quantities, execution time is negative, or a scalar field
            is not finite.
    """

    status: LPSolverStatus
    success: bool
    maximize: bool
    objective_neuron: Optional[NeuronRef]
    solution: Optional[FloatArray]
    objective_value: Optional[float]
    conservative_bound: Optional[float]
    duality_gap: Optional[float]
    max_primal_residual: Optional[float]
    max_dual_residual: Optional[float]
    validation_status: BoundValidationStatus
    execution_time_seconds: float
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, LPSolverStatus):
            raise TypeError(f"status must be a LPSolverStatus member, got {self.status!r}")
        if not isinstance(self.validation_status, BoundValidationStatus):
            raise TypeError(
                f"validation_status must be a BoundValidationStatus member, "
                f"got {self.validation_status!r}"
            )
        if not isinstance(self.success, bool):
            raise TypeError(f"success must be a bool, got {self.success!r}")
        if not isinstance(self.maximize, bool):
            raise TypeError(f"maximize must be a bool, got {self.maximize!r}")
        if self.objective_neuron is not None and not isinstance(self.objective_neuron, NeuronRef):
            raise TypeError(
                "objective_neuron must be a NeuronRef or None, "
                f"got {type(self.objective_neuron).__name__}"
            )
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")

        # A failed solve must never be reported as a successful optimization.
        expected_success = self.status is LPSolverStatus.OPTIMAL
        if self.success != expected_success:
            raise ValueError(
                f"success must equal (status is OPTIMAL): status={self.status.value}, "
                f"success={self.success}"
            )

        if isinstance(self.execution_time_seconds, bool) or not isinstance(
            self.execution_time_seconds, Real
        ):
            raise TypeError(
                "execution_time_seconds must be a real number, "
                f"got {type(self.execution_time_seconds).__name__}"
            )
        execution_time = float(self.execution_time_seconds)
        if not math.isfinite(execution_time) or execution_time < 0.0:
            raise ValueError(
                "execution_time_seconds must be finite and nonnegative, "
                f"got {execution_time}"
            )
        object.__setattr__(
            self,
            "execution_time_seconds",
            0.0 if execution_time == 0.0 else execution_time,
        )

        if self.solution is not None:
            solution = np.array(self.solution, dtype=np.float64, copy=True)
            if solution.ndim != 1:
                raise ValueError(f"solution must be 1-D, got shape {solution.shape}")
            if not np.all(np.isfinite(solution)):
                raise ValueError("solution entries must be finite")
            solution.setflags(write=False)
            object.__setattr__(self, "solution", solution)

        for name in (
            "objective_value",
            "conservative_bound",
            "duality_gap",
            "max_primal_residual",
            "max_dual_residual",
        ):
            value = getattr(self, name)
            if value is not None:
                scalar = float(value)
                if math.isnan(scalar) or math.isinf(scalar):
                    raise ValueError(f"{name} must be finite or None, got {scalar}")
                object.__setattr__(self, name, scalar)
            if name in ("max_primal_residual", "max_dual_residual") and value is not None:
                if float(value) < 0.0:
                    raise ValueError(f"{name} must be nonnegative, got {value}")

        if self.validation_status is BoundValidationStatus.VALIDATED:
            if self.conservative_bound is None or self.objective_value is None:
                raise ValueError(
                    "a VALIDATED result must carry both objective_value and "
                    "conservative_bound"
                )
            if self.max_primal_residual is None or self.max_dual_residual is None:
                raise ValueError(
                    "a VALIDATED result must carry both residual diagnostics"
                )

    @property
    def is_certified(self) -> bool:
        """Whether the conservative bound passed the numerical validation.

        Downstream stages MUST check this property before using
        :attr:`certified_bound` as a certified refined bound.
        """
        return self.validation_status is BoundValidationStatus.VALIDATED

    @property
    def lower_bound(self) -> Optional[float]:
        """Lower bound on the optimum in the original direction.

        For minimization this is the conservative (dual) side; for
        maximization it is the primal witness. ``None`` when unavailable.
        """
        if self.maximize:
            return self.objective_value
        return self.conservative_bound

    @property
    def upper_bound(self) -> Optional[float]:
        """Upper bound on the optimum in the original direction.

        For maximization this is the conservative (dual) side; for
        minimization it is the primal witness. ``None`` when unavailable.
        """
        if self.maximize:
            return self.conservative_bound
        return self.objective_value

    @property
    def certified_bound(self) -> Optional[float]:
        """The conservative bound, or ``None`` when it is not validated.

        Returns ``None`` unless :attr:`is_certified`, so an unvalidated
        numerical value cannot be mistaken for a certified refinement bound.
        """
        if self.is_certified:
            return self.conservative_bound
        return None

    def __str__(self) -> str:
        direction = "maximize" if self.maximize else "minimize"
        return (
            f"LPSolverResult(status={self.status.value}, success={self.success}, "
            f"direction={direction}, objective={self.objective_value}, "
            f"conservative_bound={self.conservative_bound}, "
            f"validation={self.validation_status.value}, gap={self.duality_gap}, "
            f"time={self.execution_time_seconds:.4f}s)"
        )


def _resolve_objective(
    problem: LPProblem, objective: object
) -> tuple[FloatArray, Optional[NeuronRef]]:
    """Coerce the objective argument into a finite length-n vector.

    Returns the vector together with the selected neuron (``None`` for raw
    vectors). Raises TypeError/ValueError for invalid inputs; dimensions are
    validated against ``problem.layout.total_variables``.
    """
    n = problem.layout.total_variables
    if objective is None:
        return problem.objective.copy(), problem.objective_neuron
    if isinstance(objective, NeuronRef):
        return problem.layout.objective_vector(objective), objective
    try:
        vector = np.array(objective, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"objective must be a NeuronRef, a length-{n} vector, or None: {exc}"
        ) from exc
    if vector.ndim != 1 or vector.shape[0] != n:
        raise ValueError(
            f"objective must be a length-{n} vector, got shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("objective coefficients must be finite")
    return vector, None


@dataclass(frozen=True)
class _PrimalResidualDiagnostics:
    """Raw and normalized primal feasibility diagnostics for one vector ``x``.

    Attributes:
        raw_residual: Maximum unscaled constraint or variable-bound violation.
        normalized_residual: Maximum violation divided by each row's local
            dot-product scale ``max(1.0, |row| @ |x|, |rhs|)`` (or bound scale
            ``max(1.0, |bound|, |value|)``).
        max_unit_row_residual: Maximum violation divided by row coefficient norm
            ``max(1.0, ||row||_1)`` (1.0 for direct variable bounds), preventing
            huge right-hand sides or bounds from masking large coordinate-space
            infeasibility while remaining invariant to uniform row scaling.
    """

    raw_residual: float
    normalized_residual: float
    max_unit_row_residual: float


@dataclass(frozen=True)
class _DualityGapDiagnostics:
    """Absolute and relative duality-gap diagnostics in the original direction."""

    absolute_gap: float
    relative_gap: float
    scale: float


@dataclass(frozen=True)
class _DualDiagnostics:
    """Internal breakdown of dual objective, stationarity, and sign checks."""

    dual_objective: float
    stationarity_violation: float
    ineq_sign_violation: float
    lb_sign_violation: float
    ub_sign_violation: float
    raw_ineq_sign_violation: float
    raw_lb_sign_violation: float
    raw_ub_sign_violation: float

    @property
    def sign_violation(self) -> float:
        """Largest scaled multiplier sign violation across all families."""
        return max(
            self.ineq_sign_violation,
            self.lb_sign_violation,
            self.ub_sign_violation,
        )

    @property
    def max_dual_residual(self) -> float:
        """Combined dual residual exposed via ``LPSolverResult.max_dual_residual``."""
        return max(self.stationarity_violation, self.sign_violation)


def _to_1d_float(array: Any) -> NDArray[np.float64]:
    """Flatten dense, matrix, or sparse-output vectors to a 1-D float64 array."""
    if hasattr(array, "A1"):
        array = array.A1
    return np.asarray(array, dtype=np.float64).ravel()


def _linear_row_violations(
    matrix: Any,
    rhs: FloatArray,
    x: FloatArray,
    *,
    equality: bool,
) -> tuple[float, float, float]:
    """Return ``(max_raw, max_normalized, max_unit_row)`` linear row violations.

    Accepts both dense ``numpy.ndarray`` and SciPy sparse matrices.  All
    per-row work is replaced by vectorised NumPy/sparse operations so only
    *violated* rows contribute to the final ``max`` reduction.
    """
    if x.size == 0:
        return 0.0, 0.0, 0.0

    diff = _to_1d_float(matrix @ x) - _to_1d_float(rhs)
    violation = np.abs(diff) if equality else np.maximum(diff, 0.0)

    mask = violation > 0.0
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    abs_matrix = np.abs(matrix)
    row_abs_dot = _to_1d_float(abs_matrix @ np.abs(x))
    row_norm = _to_1d_float(abs_matrix.sum(axis=1))
    rhs_flat = _to_1d_float(rhs)

    v = violation[mask]
    local_scale = np.maximum(1.0, np.maximum(row_abs_dot[mask], np.abs(rhs_flat[mask])))
    coeff_scale = np.maximum(1.0, row_norm[mask])

    return (
        float(np.max(v)),
        float(np.max(v / local_scale)),
        float(np.max(v / coeff_scale)),
    )


def _primal_residual_diagnostics(
    problem: LPProblem, x: FloatArray
) -> _PrimalResidualDiagnostics:
    """Compute raw and row-locally normalized primal residuals at ``x``.

    Unlike global ``x_scale = max(|x|)`` normalization, row scaling uses the
    exact inner-product magnitude ``|row| @ |x|`` so unrelated large variables
    cannot inflate the scale of rows in which they have zero coefficients.
    """
    max_raw = 0.0
    max_norm = 0.0
    max_unit_row = 0.0

    for matrix, rhs, is_eq in (
        (problem.A_ub_csc, problem.b_ub, False),
        (problem.A_eq_csc, problem.b_eq, True),
    ):
        if matrix is not None and rhs is not None:
            raw, norm, unit_row = _linear_row_violations(matrix, rhs, x, equality=is_eq)
            max_raw = max(max_raw, raw)
            max_norm = max(max_norm, norm)
            max_unit_row = max(max_unit_row, unit_row)

    # bounds_array is (n, 2) float64 with -inf / +inf for absent bounds.
    ba = problem.bounds_array
    lb, ub = ba[:, 0], ba[:, 1]
    xf = np.asarray(x, dtype=np.float64).ravel()

    for viol, bound in ((np.maximum(0.0, lb - xf), lb), (np.maximum(0.0, xf - ub), ub)):
        mask = viol > 0.0
        if np.any(mask):
            masked_viol = viol[mask]
            scale = np.maximum(1.0, np.maximum(np.abs(bound[mask]), np.abs(xf[mask])))
            raw = float(np.max(masked_viol))
            max_raw = max(max_raw, raw)
            max_norm = max(max_norm, float(np.max(masked_viol / scale)))
            max_unit_row = max(max_unit_row, raw)

    return _PrimalResidualDiagnostics(
        raw_residual=max_raw,
        normalized_residual=max_norm,
        max_unit_row_residual=max_unit_row,
    )


def _scaled_primal_residual(problem: LPProblem, x: FloatArray) -> float:
    """Largest row-locally normalized violation of the constraints at ``x``."""
    return _primal_residual_diagnostics(problem, x).normalized_residual


def _multiplier_sign_violation(
    multipliers: NDArray[np.float64],
    *,
    nonpositive: bool,
) -> tuple[float, float]:
    """Compute ``(scaled_violation, raw_violation)`` for one multiplier family.

    Each multiplier family (``y``, ``lambda_lb``, ``lambda_ub``) uses its own
    scale ``min(max(1.0, ||m||_inf), _MAX_MULTIPLIER_SIGN_SCALE)`` and never
    depends on ``abs(dual)`` or any other multiplier vector. Bounding the
    denominator at ``_MAX_MULTIPLIER_SIGN_SCALE`` prevents large objectives from
    masking substantial wrong-sign multipliers exceeding ``_MAX_ABS_DUAL_SIGN_TOL``.
    """
    if not multipliers.size:
        return 0.0, 0.0
    raw = (
        max(0.0, float(np.max(multipliers)))
        if nonpositive
        else max(0.0, -float(np.min(multipliers)))
    )
    if raw <= 0.0:
        return 0.0, 0.0
    family_norm = float(np.max(np.abs(multipliers)))
    effective_scale = min(max(1.0, family_norm), _MAX_MULTIPLIER_SIGN_SCALE)
    return raw / effective_scale, raw


def _duality_gap_diagnostics(
    primal_value: float, conservative: float, maximize: bool
) -> _DualityGapDiagnostics:
    """Compute absolute and relative duality gap in the original problem direction.

    In minimization the gap is ``primal_value - conservative`` (primal upper
    witness minus dual lower bound); in maximization it is
    ``conservative - primal_value`` (dual upper bound minus primal lower
    witness). Both share the convention ``upper_bound - lower_bound >= 0``.
    """
    absolute_gap = (
        conservative - primal_value if maximize else primal_value - conservative
    )
    scale = max(1.0, abs(primal_value), abs(conservative))
    return _DualityGapDiagnostics(
        absolute_gap=absolute_gap,
        relative_gap=absolute_gap / scale,
        scale=scale,
    )


def _dual_diagnostics(
    problem: LPProblem,
    marginals: Optional[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
    objective: FloatArray,
) -> Optional[_DualDiagnostics]:
    """Validate dual feasibility from solver marginals.

    ``marginals`` is ``(y, nu, lambda_lb, lambda_ub)`` with the
    empirically verified HiGHS conventions: ``y <= 0``, ``lambda_lb >= 0``,
    ``lambda_ub <= 0``, dual objective ``b_ub @ y + b_eq @ nu +
    l @ lambda_lb + u @ lambda_ub``, stationarity ``A_ub.T @ y +
    A_eq.T @ nu + lambda_lb + lambda_ub == objective``. Returns a
    :class:`_DualDiagnostics` record or ``None`` when the marginals are
    absent, non-finite, or wrongly shaped. All quantities are in the
    supplied (solved) direction.
    """
    if marginals is None:
        return None
    y, nu, lambda_lb, lambda_ub = marginals
    n = problem.layout.total_variables
    m_ub = problem.num_inequality_constraints
    m_eq = problem.num_equality_constraints
    expected_shapes = ((m_ub,), (m_eq,), (n,), (n,))
    for array, expected in zip(marginals, expected_shapes):
        if array is None or array.shape != expected:
            return None
        if not np.all(np.isfinite(array)):
            return None

    dual = 0.0
    if m_ub:
        dual += float(problem.b_ub @ y)
    if m_eq:
        dual += float(problem.b_eq @ nu)

    # Stationarity: A_ub.T y + A_eq.T nu + lambda_lb + lambda_ub == objective,
    # with lambda forced to zero on unbounded variable sides.
    # Use CSC matrices directly to avoid triggering dense materialisation.
    gradient = np.zeros(n, dtype=np.float64)
    if m_ub:
        g_ub = problem.A_ub_csc.T @ y
        gradient += np.asarray(g_ub, dtype=np.float64).ravel()
    if m_eq:
        g_eq = problem.A_eq_csc.T @ nu
        gradient += np.asarray(g_eq, dtype=np.float64).ravel()

    # Vectorised bounds accumulation using bounds_array (n, 2) float64.
    ba = problem.bounds_array   # (n, 2); -inf/+inf where bound is absent
    lb = ba[:, 0]
    ub = ba[:, 1]
    fin_lb = np.isfinite(lb)
    fin_ub = np.isfinite(ub)

    # Zero lambda where the corresponding bound is absent (unbounded side).
    lambda_lb_used = np.where(fin_lb, lambda_lb, 0.0)
    lambda_ub_used = np.where(fin_ub, lambda_ub, 0.0)

    # Dual objective contribution from finite bounds.
    if np.any(fin_lb):
        dual += float(np.dot(lb[fin_lb], lambda_lb[fin_lb]))
    if np.any(fin_ub):
        dual += float(np.dot(ub[fin_ub], lambda_ub[fin_ub]))

    gradient += lambda_lb_used + lambda_ub_used

    residual = gradient - objective
    scale = max(
        1.0,
        float(np.max(np.abs(gradient))) if gradient.size else 1.0,
        float(np.max(np.abs(objective))) if objective.size else 1.0,
    )
    stationarity = float(np.max(np.abs(residual))) / scale if residual.size else 0.0

    # Independent multiplier-specific sign feasibility: never scale by abs(dual).
    ineq_sign, raw_ineq_sign = _multiplier_sign_violation(y, nonpositive=True)
    lb_sign, raw_lb_sign = _multiplier_sign_violation(lambda_lb, nonpositive=False)
    ub_sign, raw_ub_sign = _multiplier_sign_violation(lambda_ub, nonpositive=True)

    return _DualDiagnostics(
        dual_objective=dual,
        stationarity_violation=stationarity,
        ineq_sign_violation=ineq_sign,
        lb_sign_violation=lb_sign,
        ub_sign_violation=ub_sign,
        raw_ineq_sign_violation=raw_ineq_sign,
        raw_lb_sign_violation=raw_lb_sign,
        raw_ub_sign_violation=raw_ub_sign,
    )


def _extract_marginals(result: Any) -> Optional[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
]:
    """Pull the four marginal arrays out of a scipy OptimizeResult, if present."""
    try:
        ineq = getattr(result, "ineqlin", None)
        eq = getattr(result, "eqlin", None)
        lower = getattr(result, "lower", None)
        upper = getattr(result, "upper", None)
        return (
            np.asarray(getattr(ineq, "marginals", None), dtype=np.float64)
            if getattr(ineq, "marginals", None) is not None
            else None,
            np.asarray(getattr(eq, "marginals", None), dtype=np.float64)
            if getattr(eq, "marginals", None) is not None
            else None,
            np.asarray(getattr(lower, "marginals", None), dtype=np.float64)
            if getattr(lower, "marginals", None) is not None
            else None,
            np.asarray(getattr(upper, "marginals", None), dtype=np.float64)
            if getattr(upper, "marginals", None) is not None
            else None,
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _map_status(code: int) -> LPSolverStatus:
    """Map a scipy linprog integer status code to LPSolverStatus."""
    if code == 0:
        return LPSolverStatus.OPTIMAL
    if code == 1:
        return LPSolverStatus.ITERATION_LIMIT
    if code == 2:
        return LPSolverStatus.INFEASIBLE
    if code == 3:
        return LPSolverStatus.UNBOUNDED
    if code == 4:
        return LPSolverStatus.NUMERICAL_ISSUE
    return LPSolverStatus.FAILED


def _make_failure(
    status: LPSolverStatus,
    maximize: bool,
    objective_neuron: Optional[NeuronRef],
    elapsed: float,
    message: str,
) -> LPSolverResult:
    """Assemble a terminal result for non-optimal outcomes."""
    return LPSolverResult(
        status=status,
        success=False,
        maximize=maximize,
        objective_neuron=objective_neuron,
        solution=None,
        objective_value=None,
        conservative_bound=None,
        duality_gap=None,
        max_primal_residual=None,
        max_dual_residual=None,
        validation_status=BoundValidationStatus.UNVALIDATED,
        execution_time_seconds=elapsed,
        message=message,
    )


def solve_lp(
    problem: LPProblem,
    objective: NeuronRef | NDArray[np.float64] | Any | None = None,
    *,
    maximize: bool = False,
    method: str = "highs",
    options: Optional[Mapping[str, Any]] = None,
) -> LPSolverResult:
    """Solve an LPProblem with ``scipy.optimize.linprog`` and validate bounds.

    The objective may be a :class:`~abstract_domain.bounds.NeuronRef` (its
    unit vector in the problem layout), a raw length-``n_variables`` vector,
    or ``None`` for the problem's stored objective. Maximization negates the
    objective internally and maps the answer back to the original direction.

    The returned :class:`LPSolverResult` separates the primal witness from
    the conservative bound and reports a tolerance-based validation verdict
    (see the module docstring for the exact checks and their tolerance).
    Only ``OPTIMAL`` solves report ``success=True``; use
    :attr:`LPSolverResult.is_certified` and
    :attr:`LPSolverResult.certified_bound` before treating a bound as
    certified.

    Args:
        problem: The LP to solve; its constraint matrices and bounds are
            passed to ``linprog`` unchanged.
        objective: Replacement objective (see above); defaults to the
            problem's stored neuron objective.
        maximize: Solve the maximization instead of the minimization.
        method: ``linprog`` method name; the default ``"highs"`` is the only
            one the validation conventions were verified against.
        options: Optional ``linprog`` options mapping (e.g. ``maxiter``,
            ``time_limit``, ``presolve``).

    Raises:
        TypeError: If ``problem`` is not an LPProblem, ``options`` is not a
            mapping, or the objective has the wrong type.
        ValueError: If the objective has invalid dimensions or non-finite
            coefficients. (Dimension errors inside the LP itself cannot occur
            here: LPProblem validates its matrices on construction.)
    """
    if not isinstance(problem, LPProblem):
        raise TypeError(
            f"problem must be an LPProblem, got {type(problem).__name__}"
        )
    if not isinstance(maximize, bool):
        raise TypeError(f"maximize must be a bool, got {maximize!r}")
    if not isinstance(method, str) or not method.strip():
        raise TypeError("method must be a non-empty string")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError(f"options must be a mapping or None, got {type(options).__name__}")

    original_objective, objective_neuron = _resolve_objective(problem, objective)
    solve_objective = -original_objective if maximize else original_objective

    # scipy is imported lazily so that importing this module (and the
    # verification package) never depends on scipy being installed.
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        return _make_failure(
            LPSolverStatus.FAILED,
            maximize,
            objective_neuron,
            0.0,
            f"scipy is required to solve LPs: {exc}",
        )

    kwargs: dict[str, Any] = {"bounds": problem.bounds_array}
    # Normalize empty matrices away: linprog prefers None over 0-row matrices.
    if problem.num_inequality_constraints:
        kwargs["A_ub"], kwargs["b_ub"] = problem.A_ub_csc, problem.b_ub
    if problem.num_equality_constraints:
        kwargs["A_eq"], kwargs["b_eq"] = problem.A_eq_csc, problem.b_eq
    if options:
        kwargs["options"] = dict(options)

    start = time.perf_counter()
    try:
        raw = linprog(solve_objective, method=method, **kwargs)
    except (TypeError, ValueError) as exc:
        elapsed = time.perf_counter() - start
        return _make_failure(
            LPSolverStatus.INVALID_PROBLEM,
            maximize,
            objective_neuron,
            elapsed,
            f"linprog rejected the problem: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive against solver bugs
        elapsed = time.perf_counter() - start
        return _make_failure(
            LPSolverStatus.FAILED,
            maximize,
            objective_neuron,
            elapsed,
            f"unexpected solver error: {type(exc).__name__}: {exc}",
        )
    elapsed = time.perf_counter() - start

    status = _map_status(int(getattr(raw, "status", -1)))
    raw_message = str(getattr(raw, "message", "")).strip() or status.value

    if status is LPSolverStatus.OPTIMAL:
        return _assemble_optimal(
            problem,
            raw,
            original_objective,
            solve_objective,
            objective_neuron,
            maximize,
            elapsed,
            raw_message,
        )

    if status is LPSolverStatus.ITERATION_LIMIT:
        # The returned point, if any, is a feasible witness but NOT an
        # optimum: never report success and never certify the conservative
        # side.
        point = _usable_point(raw)
        if point is None:
            return _make_failure(
                status,
                maximize,
                objective_neuron,
                elapsed,
                f"{raw_message} (no usable solution point)",
            )
        primal_value = float(original_objective @ point)
        primal_diag = _primal_residual_diagnostics(problem, point)
        return LPSolverResult(
            status=status,
            success=False,
            maximize=maximize,
            objective_neuron=objective_neuron,
            solution=point,
            objective_value=primal_value,
            conservative_bound=None,
            duality_gap=None,
            max_primal_residual=primal_diag.normalized_residual,
            max_dual_residual=None,
            validation_status=BoundValidationStatus.UNVALIDATED,
            execution_time_seconds=elapsed,
            message=(
                f"{raw_message}; solution is a primal witness only, "
                "never treated as optimal "
                f"(primal_norm={primal_diag.normalized_residual:.3e}, "
                f"primal_raw={primal_diag.raw_residual:.3e})"
            ),
        )

    if status is LPSolverStatus.INFEASIBLE:
        return _make_failure(
            status,
            maximize,
            objective_neuron,
            elapsed,
            f"{raw_message}; the LP relaxation has an empty feasible set",
        )
    if status is LPSolverStatus.UNBOUNDED:
        return _make_failure(
            status,
            maximize,
            objective_neuron,
            elapsed,
            f"{raw_message}; no finite optimum exists in the solved direction",
        )
    if status is LPSolverStatus.NUMERICAL_ISSUE:
        return _make_failure(
            status,
            maximize,
            objective_neuron,
            elapsed,
            f"{raw_message}; solver reported numerical difficulties",
        )
    return _make_failure(
        LPSolverStatus.FAILED,
        maximize,
        objective_neuron,
        elapsed,
        f"unrecognized solver status {getattr(raw, 'status', None)}",
    )


def _usable_point(raw: Any) -> Optional[FloatArray]:
    """Extract a finite 1-D solution point from a raw result, or None."""
    point = getattr(raw, "x", None)
    if point is None:
        return None
    try:
        array = np.asarray(point, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        return None
    array.setflags(write=False)
    return array


def _assemble_optimal(
    problem: LPProblem,
    raw: Any,
    original_objective: FloatArray,
    solve_objective: FloatArray,
    objective_neuron: Optional[NeuronRef],
    maximize: bool,
    elapsed: float,
    raw_message: str,
) -> LPSolverResult:
    """Validate and assemble the result of an optimal solve.

    The primal witness is ``original_objective @ x`` (the original direction).
    The conservative bound is the dual objective in the solved direction
    (minimization of ``solve_objective``), mapped back: for minimization it
    is the dual objective itself (a lower bound); for maximization it is its
    negation (an upper bound). Validation covers primal feasibility, dual
    sign feasibility, dual stationarity, and a nonnegative duality gap.
    """
    point = _usable_point(raw)
    if point is None:
        return _make_failure(
            LPSolverStatus.NUMERICAL_ISSUE,
            maximize,
            objective_neuron,
            elapsed,
            f"{raw_message}; solver reported optimal but returned no usable point",
        )

    primal_value = float(original_objective @ point)
    primal_diag = _primal_residual_diagnostics(problem, point)
    primal_residual = primal_diag.normalized_residual
    marginals = _extract_marginals(raw)
    dual_diag = _dual_diagnostics(problem, marginals, solve_objective)

    conservative: Optional[float] = None
    max_dual_residual: Optional[float] = None
    gap_diag: Optional[_DualityGapDiagnostics] = None
    gap: Optional[float] = None
    if dual_diag is not None:
        max_dual_residual = dual_diag.max_dual_residual
        conservative = (
            -dual_diag.dual_objective if maximize else dual_diag.dual_objective
        )
        gap_diag = _duality_gap_diagnostics(primal_value, conservative, maximize)
        gap = gap_diag.absolute_gap

    failures: list[str] = []
    if primal_diag.normalized_residual > _VALIDATION_TOL:
        failures.append(
            f"normalized primal residual {primal_diag.normalized_residual:.3e} > "
            f"{_VALIDATION_TOL:.3e} (raw {primal_diag.raw_residual:.3e})"
        )
    if primal_diag.max_unit_row_residual > _MAX_ABS_PRIMAL_TOL:
        failures.append(
            f"unit-row primal residual {primal_diag.max_unit_row_residual:.3e} > "
            f"{_MAX_ABS_PRIMAL_TOL:.3e} (raw {primal_diag.raw_residual:.3e}, "
            f"normalized {primal_diag.normalized_residual:.3e})"
        )
    if dual_diag is None:
        failures.append("dual marginals unavailable or invalid")
    else:
        if dual_diag.stationarity_violation > _VALIDATION_TOL:
            failures.append(
                f"dual stationarity residual {dual_diag.stationarity_violation:.3e} > "
                f"{_VALIDATION_TOL:.3e}"
            )
        if dual_diag.ineq_sign_violation > _VALIDATION_TOL:
            failures.append(
                f"inequality multiplier sign violation {dual_diag.ineq_sign_violation:.3e} > "
                f"{_VALIDATION_TOL:.3e} (raw {dual_diag.raw_ineq_sign_violation:.3e})"
            )
        if dual_diag.lb_sign_violation > _VALIDATION_TOL:
            failures.append(
                f"lower-bound multiplier sign violation {dual_diag.lb_sign_violation:.3e} > "
                f"{_VALIDATION_TOL:.3e} (raw {dual_diag.raw_lb_sign_violation:.3e})"
            )
        if dual_diag.ub_sign_violation > _VALIDATION_TOL:
            failures.append(
                f"upper-bound multiplier sign violation {dual_diag.ub_sign_violation:.3e} > "
                f"{_VALIDATION_TOL:.3e} (raw {dual_diag.raw_ub_sign_violation:.3e})"
            )
    if gap_diag is not None:
        if gap_diag.relative_gap < -_VALIDATION_TOL:
            failures.append(
                f"negative relative duality gap {gap_diag.relative_gap:.3e} < "
                f"-{_VALIDATION_TOL:.3e} (abs_gap {gap_diag.absolute_gap:.3e})"
            )
        if gap_diag.absolute_gap < -_MAX_ABS_NEGATIVE_GAP:
            failures.append(
                f"negative absolute duality gap {gap_diag.absolute_gap:.3e} < "
                f"-{_MAX_ABS_NEGATIVE_GAP:.3e} (rel_gap {gap_diag.relative_gap:.3e})"
            )

    if not failures:
        validation = BoundValidationStatus.VALIDATED
        assert gap_diag is not None
        detail = (
            "numerical validation PASSED ("
            f"primal_norm={primal_diag.normalized_residual:.3e}, "
            f"primal_raw={primal_diag.raw_residual:.3e}, "
            f"dual_residual={max_dual_residual:.3e}, "
            f"abs_gap={gap_diag.absolute_gap:.3e}, "
            f"rel_gap={gap_diag.relative_gap:.3e})"
        )
    else:
        validation = BoundValidationStatus.UNVALIDATED
        detail = f"numerical validation FAILED ({'; '.join(failures)})"
    message = f"{raw_message}; {detail}"

    return LPSolverResult(
        status=LPSolverStatus.OPTIMAL,
        success=True,
        maximize=maximize,
        objective_neuron=objective_neuron,
        solution=point,
        objective_value=primal_value,
        conservative_bound=conservative,
        duality_gap=gap,
        max_primal_residual=primal_residual,
        max_dual_residual=max_dual_residual,
        validation_status=validation,
        execution_time_seconds=elapsed,
        message=message,
    )
