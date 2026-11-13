"""
Tests for the MiniCert LP solver (``verification.lp_solver``).

Pure-solver tests build small two-variable LPs directly from the layout and
matrix types of ``verification.lp`` (no PyTorch involved), so they exercise
``solve_lp`` independently of the network builder. Integration tests at the
end reuse the hand-computed design network from ``tests.test_lp`` and solve
it end to end with ``build_lp_problem``.

The two-variable problems use the layout

    variables: [x (input), z0 (pre of linear layer 0)]
    objective neuron: NeuronRef(0, "pre", 0) -> column index 1

and arbitrary A_ub/A_eq rows over the two columns, giving full control over
the test LP while staying inside the agreed VariableLayout conventions.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from abstract_domain.bounds import NeuronRef
from verification.lp import LPProblem, VariableLayout
from verification.lp_solver import (
    BoundValidationStatus,
    LPSolverResult,
    LPSolverStatus,
    _duality_gap_diagnostics,
    _linear_row_violations,
    _primal_residual_diagnostics,
    solve_lp,
)

OPTIMAL = LPSolverStatus.OPTIMAL
INFEASIBLE = LPSolverStatus.INFEASIBLE
UNBOUNDED = LPSolverStatus.UNBOUNDED
ITERATION_LIMIT = LPSolverStatus.ITERATION_LIMIT
NUMERICAL_ISSUE = LPSolverStatus.NUMERICAL_ISSUE
INVALID_PROBLEM = LPSolverStatus.INVALID_PROBLEM
FAILED = LPSolverStatus.FAILED

VALIDATED = BoundValidationStatus.VALIDATED
UNVALIDATED = BoundValidationStatus.UNVALIDATED


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _two_var_layout() -> VariableLayout:
    """Layout with variables [x, z0]: one input, one linear pre group."""
    return VariableLayout(
        input_dim=1,
        pre_offsets=(1,),
        pre_dims=(1,),
        post_offsets=(None,),
        post_dims=(None,),
        total_variables=2,
    )


def _make_problem(
    *,
    a_ub: object = None,
    b_ub: object = None,
    a_eq: object = None,
    b_eq: object = None,
    bounds: object = None,
) -> LPProblem:
    """Build a two-variable LPProblem with a unit objective for column 1.

    LPProblem requires the stored objective to be the unit vector of its
    objective neuron, so the actual test objective is always passed to
    ``solve_lp`` explicitly.
    """
    layout = _two_var_layout()
    neuron = NeuronRef(0, "pre", 0)
    stored = layout.objective_vector(neuron)
    return LPProblem(
        stored,
        neuron,
        layout,
        tuple(bounds) if bounds is not None else ((None, None), (None, None)),
        A_ub=np.array(a_ub, dtype=np.float64) if a_ub is not None else None,
        b_ub=np.array(b_ub, dtype=np.float64) if b_ub is not None else None,
        A_eq=np.array(a_eq, dtype=np.float64) if a_eq is not None else None,
        b_eq=np.array(b_eq, dtype=np.float64) if b_eq is not None else None,
    )


def _fake_result(
    x: object,
    *,
    status: int = 0,
    fun: float | None = None,
    message: str = "fake",
    y: object = None,
    nu: object = None,
    lam_lb: object = None,
    lam_ub: object = None,
) -> object:
    """Build a scipy-like OptimizeResult for monkeypatching linprog."""
    from scipy.optimize import OptimizeResult

    point = np.asarray(x, dtype=np.float64)
    result = OptimizeResult(
        status=status,
        success=(status == 0),
        x=point,
        fun=float(fun) if fun is not None else 0.0,
        message=message,
    )
    n = point.shape[0]
    result.ineqlin = OptimizeResult(
        marginals=np.zeros(0, dtype=np.float64) if y is None else np.asarray(y, dtype=np.float64),
        residual=np.zeros(0, dtype=np.float64),
    )
    result.eqlin = OptimizeResult(
        marginals=np.zeros(0, dtype=np.float64) if nu is None else np.asarray(nu, dtype=np.float64),
        residual=np.zeros(0, dtype=np.float64),
    )
    result.lower = OptimizeResult(
        marginals=np.zeros(n, dtype=np.float64) if lam_lb is None else np.asarray(lam_lb, dtype=np.float64),
        residual=np.zeros(n, dtype=np.float64),
    )
    result.upper = OptimizeResult(
        marginals=np.zeros(n, dtype=np.float64) if lam_ub is None else np.asarray(lam_ub, dtype=np.float64),
        residual=np.zeros(n, dtype=np.float64),
    )
    return result


def _monkeypatch_linprog(monkeypatch: pytest.MonkeyPatch, results: list[object]) -> None:
    """Replace scipy.optimize.linprog with a function returning `results` in order."""
    import scipy.optimize

    def fake_linprog(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(scipy.optimize, "linprog", fake_linprog)


# --------------------------------------------------------------------------
# minimization / maximization with known answers
# --------------------------------------------------------------------------

def test_minimization_with_known_answer() -> None:
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    result = solve_lp(problem, [0.0, 1.0])

    assert result.status is OPTIMAL
    assert result.success is True
    assert result.objective_value == pytest.approx(1.0)
    assert result.solution == pytest.approx([0.0, 1.0])
    assert result.is_certified is True
    assert result.certified_bound == pytest.approx(1.0)


def test_maximization_with_known_answer() -> None:
    problem = _make_problem(
        a_ub=[[1.0, 0.0], [0.0, 1.0]],
        b_ub=[1.0, 2.0],
        bounds=[(0.0, None), (0.0, None)],
    )
    result = solve_lp(problem, [1.0, 2.0], maximize=True)

    assert result.status is OPTIMAL
    assert result.success is True
    assert result.objective_value == pytest.approx(5.0)
    assert result.solution == pytest.approx([1.0, 2.0])
    assert result.is_certified is True
    assert result.certified_bound == pytest.approx(5.0)
    # For maximization the conservative side is the upper bound.
    assert result.upper_bound == pytest.approx(5.0)
    assert result.lower_bound == pytest.approx(5.0)


def test_hand_computable_lp_min_and_max() -> None:
    # maximize x + z0 subject to x <= 1, z0 <= 1, x >= 0, z0 >= 0
    # answer: (1, 1) -> 2.   minimize: (0, 0) -> 0.
    problem = _make_problem(
        a_ub=[[1.0, 0.0], [0.0, 1.0]],
        b_ub=[1.0, 1.0],
        bounds=[(0.0, None), (0.0, None)],
    )
    maximum = solve_lp(problem, [1.0, 1.0], maximize=True)
    minimum = solve_lp(problem, [1.0, 1.0])

    assert maximum.objective_value == pytest.approx(2.0)
    assert maximum.solution == pytest.approx([1.0, 1.0])
    assert minimum.objective_value == pytest.approx(0.0)
    assert minimum.solution == pytest.approx([0.0, 0.0])
    assert maximum.is_certified and minimum.is_certified


# --------------------------------------------------------------------------
# constraint kinds
# --------------------------------------------------------------------------

def test_equality_constraints() -> None:
    # minimize x + z0 subject to x + z0 == 1 with both variables in [0, 1]
    problem = _make_problem(
        a_eq=[[1.0, 1.0]],
        b_eq=[1.0],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
    )
    result = solve_lp(problem, [1.0, 1.0])

    assert result.status is OPTIMAL
    assert result.objective_value == pytest.approx(1.0)
    assert result.is_certified
    assert result.certified_bound == pytest.approx(1.0)
    # equality marginals are free; stationarity must still hold
    assert result.max_dual_residual <= 1e-8


def test_inequality_constraints() -> None:
    # minimize z0 subject to z0 >= 3 (single <= row: -z0 <= -3)
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-3.0])
    result = solve_lp(problem, [0.0, 1.0])

    assert result.objective_value == pytest.approx(3.0)
    assert result.solution == pytest.approx([0.0, 3.0])
    assert result.is_certified


def test_finite_and_unbounded_variable_bounds() -> None:
    # minimize x subject to x >= -5, no upper bound on x; z0 fully free
    problem = _make_problem(
        a_ub=[[-1.0, 0.0]],
        b_ub=[5.0],
        bounds=[(None, None), (None, None)],
    )
    result = solve_lp(problem, [1.0, 0.0])

    assert result.status is OPTIMAL
    assert result.objective_value == pytest.approx(-5.0)
    assert result.solution[0] == pytest.approx(-5.0)
    assert result.is_certified
    assert result.certified_bound == pytest.approx(-5.0)


def test_bounds_only_problem_validates() -> None:
    # minimize x subject to 0 <= x <= 1 with no constraint rows at all
    problem = _make_problem(
        bounds=[(0.0, 1.0), (0.0, 0.0)],
    )
    result = solve_lp(problem, [1.0, 0.0])

    assert result.status is OPTIMAL
    assert result.objective_value == pytest.approx(0.0)
    assert result.is_certified
    assert result.certified_bound == pytest.approx(0.0)


# --------------------------------------------------------------------------
# terminal statuses
# --------------------------------------------------------------------------

def test_infeasible_problem() -> None:
    # x <= 0 but x bound to exactly 1
    problem = _make_problem(
        a_ub=[[1.0, 0.0]],
        b_ub=[0.0],
        bounds=[(1.0, 1.0), (0.0, 0.0)],
    )
    result = solve_lp(problem, [1.0, 0.0])

    assert result.status is INFEASIBLE
    assert result.success is False
    assert result.objective_value is None
    assert result.conservative_bound is None
    assert result.certified_bound is None
    assert result.validation_status is UNVALIDATED


def test_unbounded_problem() -> None:
    # minimize z0 subject to z0 <= 1 with no lower bound anywhere
    problem = _make_problem(a_ub=[[0.0, 1.0]], b_ub=[1.0])
    result = solve_lp(problem, [0.0, 1.0])

    assert result.status is UNBOUNDED
    assert result.success is False
    assert result.objective_value is None
    assert result.certified_bound is None


# --------------------------------------------------------------------------
# invalid inputs
# --------------------------------------------------------------------------

def test_invalid_objective_dimensions() -> None:
    problem = _make_problem()

    with pytest.raises(ValueError):
        solve_lp(problem, np.zeros(3))  # wrong length
    with pytest.raises(ValueError):
        solve_lp(problem, np.zeros((2, 2)))  # not 1-D
    with pytest.raises(ValueError):
        solve_lp(problem, np.array([0.0, np.nan]))  # non-finite
    with pytest.raises(TypeError):
        solve_lp(problem, "not-an-objective")


def test_invalid_objective_neuron_rejected() -> None:
    problem = _make_problem()

    with pytest.raises(ValueError):
        solve_lp(problem, NeuronRef(5, "pre", 0))  # layer out of range
    with pytest.raises(ValueError):
        solve_lp(problem, NeuronRef(0, "post", 0))  # linear layer has no post
    with pytest.raises(ValueError):
        solve_lp(problem, NeuronRef(0, "pre", 3))  # neuron out of range
    with pytest.raises(ValueError):
        solve_lp(problem, NeuronRef(0, "mid", 0))  # invalid phase


def test_invalid_constraint_dimensions_rejected_at_construction() -> None:
    layout = _two_var_layout()

    with pytest.raises(ValueError):
        LPProblem(
            layout.objective_vector(NeuronRef(0, "pre", 0)),
            NeuronRef(0, "pre", 0),
            layout,
            ((None, None), (None, None)),
            A_ub=np.zeros((1, 3)),  # wrong column count
            b_ub=np.zeros(1),
        )
    with pytest.raises(ValueError):
        LPProblem(
            layout.objective_vector(NeuronRef(0, "pre", 0)),
            NeuronRef(0, "pre", 0),
            layout,
            ((None, None), (None, None)),
            A_ub=np.zeros((1, 2)),
            b_ub=None,  # matrix without rhs
        )


def test_non_lp_problem_rejected() -> None:
    with pytest.raises(TypeError):
        solve_lp("not a problem", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        solve_lp(_make_problem(), None, maximize="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        solve_lp(_make_problem(), None, options="not-a-mapping")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# objective sign transformation
# --------------------------------------------------------------------------

def test_objective_sign_transformation() -> None:
    # x in [0, 1], z0 fixed to 0: minimize x -> 0, maximize x -> 1, and
    # minimize -x must equal -(maximize x).
    problem = _make_problem(bounds=[(0.0, 1.0), (0.0, 0.0)])

    minimum = solve_lp(problem, [1.0, 0.0])
    maximum = solve_lp(problem, [1.0, 0.0], maximize=True)
    negated = solve_lp(problem, [-1.0, 0.0])

    assert minimum.objective_value == pytest.approx(0.0)
    assert maximum.objective_value == pytest.approx(1.0)
    assert negated.objective_value == pytest.approx(-1.0)
    assert negated.objective_value == pytest.approx(-maximum.objective_value)
    assert maximum.is_certified and negated.is_certified


def test_maximization_uses_dual_upper_bound() -> None:
    # maximize z0 subject to z0 <= 2, z0 >= 0: the conservative side for a
    # maximization is the upper bound, which must come from the dual.
    problem = _make_problem(
        a_ub=[[0.0, 1.0], [0.0, -1.0]],
        b_ub=[2.0, 0.0],
    )
    result = solve_lp(problem, [0.0, 1.0], maximize=True)

    assert result.objective_value == pytest.approx(2.0)
    assert result.conservative_bound == pytest.approx(2.0)
    assert result.upper_bound == pytest.approx(2.0)
    assert result.lower_bound == pytest.approx(2.0)
    assert result.certified_bound == pytest.approx(2.0)
    assert result.duality_gap == pytest.approx(0.0, abs=1e-8)


def test_minimization_uses_dual_lower_bound() -> None:
    # minimize z0 subject to z0 >= 1: conservative side is the lower bound.
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    result = solve_lp(problem, [0.0, 1.0])

    assert result.objective_value == pytest.approx(1.0)
    assert result.conservative_bound == pytest.approx(1.0)
    assert result.lower_bound == pytest.approx(1.0)
    assert result.upper_bound == pytest.approx(1.0)
    assert result.certified_bound == pytest.approx(1.0)


# --------------------------------------------------------------------------
# numerical validation
# --------------------------------------------------------------------------

def test_validation_passes_for_clean_solve() -> None:
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    result = solve_lp(problem, [0.0, 1.0])

    assert result.validation_status is VALIDATED
    assert result.max_primal_residual <= 1e-8
    assert result.max_dual_residual <= 1e-8
    assert result.duality_gap == pytest.approx(0.0, abs=1e-8)


def test_validation_rejects_infeasible_duals(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake optimal result whose marginals violate stationarity (all zero
    # for a nonzero objective) must NOT be certified.
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    _monkeypatch_linprog(
        monkeypatch,
        [_fake_result([0.0, 1.0], status=0, fun=1.0, y=[0.0], lam_lb=[0.0, 0.0], lam_ub=[0.0, 0.0])],
    )
    result = solve_lp(problem, [0.0, 1.0])

    assert result.status is OPTIMAL
    assert result.success is True
    assert result.validation_status is UNVALIDATED
    assert result.is_certified is False
    assert result.certified_bound is None
    assert result.max_dual_residual > 1e-8


def test_validation_accepts_feasible_duals(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hand-computed dual for "minimize z0 s.t. z0 >= 1": y = -1 satisfies
    # stationarity (A_ub.T y = [0, 1]) and gives dual objective 1.
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    _monkeypatch_linprog(
        monkeypatch,
        [_fake_result([0.0, 1.0], status=0, fun=1.0, y=[-1.0], lam_lb=[0.0, 0.0], lam_ub=[0.0, 0.0])],
    )
    result = solve_lp(problem, [0.0, 1.0])

    assert result.validation_status is VALIDATED
    assert result.certified_bound == pytest.approx(1.0)
    assert result.max_dual_residual <= 1e-8


def test_validation_requires_finite_duals(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    _monkeypatch_linprog(
        monkeypatch,
        [_fake_result([0.0, 1.0], status=0, fun=1.0, y=[np.nan])],
    )
    result = solve_lp(problem, [0.0, 1.0])

    assert result.status is OPTIMAL
    assert result.validation_status is UNVALIDATED
    assert result.conservative_bound is None


def test_validation_requires_feasible_primal(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake optimal point violating the constraint z0 >= 1 must not certify.
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    _monkeypatch_linprog(
        monkeypatch,
        [_fake_result([0.0, 0.0], status=0, fun=0.0, y=[-1.0], lam_lb=[0.0, 0.0], lam_ub=[0.0, 0.0])],
    )
    result = solve_lp(problem, [0.0, 1.0])

    assert result.validation_status is UNVALIDATED
    assert result.max_primal_residual > 1e-8
    assert result.certified_bound is None


# --------------------------------------------------------------------------
# solver failure handling
# --------------------------------------------------------------------------

def test_iteration_limit_is_never_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # scipy reports success=True for an iteration limit; solve_lp must NOT
    # propagate that, and must never certify the conservative side.
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    _monkeypatch_linprog(
        monkeypatch,
        [_fake_result([0.0, 2.0], status=1, fun=2.0, message="iteration limit reached")],
    )
    result = solve_lp(problem, [0.0, 1.0])

    assert result.status is ITERATION_LIMIT
    assert result.success is False
    assert result.objective_value == pytest.approx(2.0)  # primal witness only
    assert result.conservative_bound is None
    assert result.certified_bound is None
    assert result.validation_status is UNVALIDATED


def test_unexpected_solver_error_becomes_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _make_problem()

    def explode(*args, **kwargs):
        raise RuntimeError("solver blew up")

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "linprog", explode)
    result = solve_lp(problem, [0.0, 1.0])

    assert result.status is FAILED
    assert result.success is False
    assert "solver blew up" in result.message


def test_invalid_method_name_becomes_invalid_problem() -> None:
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    result = solve_lp(problem, [0.0, 1.0], method="not-a-method")

    assert result.status is INVALID_PROBLEM
    assert result.success is False
    assert result.certified_bound is None


# --------------------------------------------------------------------------
# result structure
# --------------------------------------------------------------------------

def test_solution_array_is_read_only() -> None:
    result = solve_lp(_make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0]), [0.0, 1.0])

    with pytest.raises(ValueError):
        result.solution[0] = 99.0  # type: ignore[index]


def test_result_is_frozen() -> None:
    result = solve_lp(_make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0]), [0.0, 1.0])

    with pytest.raises(Exception):
        result.objective_value = 0.0  # type: ignore[misc]


def test_result_construction_enforces_success_consistency() -> None:
    def build(**overrides):
        defaults = dict(
            status=INFEASIBLE,
            success=False,
            maximize=False,
            objective_neuron=None,
            solution=None,
            objective_value=None,
            conservative_bound=None,
            duality_gap=None,
            max_primal_residual=None,
            max_dual_residual=None,
            validation_status=UNVALIDATED,
            execution_time_seconds=0.1,
            message="test",
        )
        defaults.update(overrides)
        return LPSolverResult(**defaults)

    assert build().success is False
    with pytest.raises(ValueError):
        build(success=True)  # INFEASIBLE can never be successful
    with pytest.raises(ValueError):
        build(status=OPTIMAL, success=False)  # OPTIMAL must be successful
    with pytest.raises(ValueError):
        build(validation_status=VALIDATED)  # VALIDATED requires bound fields
    with pytest.raises(ValueError):
        build(execution_time_seconds=-1.0)
    with pytest.raises(ValueError):
        build(message="   ")


def test_result_string_reports_direction_and_validation() -> None:
    result = solve_lp(
        _make_problem(a_ub=[[0.0, 1.0], [0.0, -1.0]], b_ub=[2.0, 0.0]),
        [0.0, 1.0],
        maximize=True,
    )

    text = str(result)
    assert "maximize" in text
    assert "VALIDATED" in text
    assert "OPTIMAL" in text


def test_options_are_passed_through() -> None:
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    result = solve_lp(problem, [0.0, 1.0], options={"presolve": False})

    assert result.status is OPTIMAL
    assert result.objective_value == pytest.approx(1.0)
    assert result.is_certified


def test_default_objective_uses_problem_neuron() -> None:
    # solve_lp with no objective argument solves the stored neuron objective.
    problem = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1.0])
    result = solve_lp(problem)

    assert result.objective_neuron == NeuronRef(0, "pre", 0)
    assert result.objective_value == pytest.approx(1.0)
    assert result.is_certified


# --------------------------------------------------------------------------
# integration with the network LP builder (PyTorch-based)
# --------------------------------------------------------------------------

def test_design_network_min_and_max_are_certified() -> None:
    from abstract_domain.propagation import get_intermediate_bounds
    from tests.test_lp import _design_net
    from verification.lp import build_lp_problem

    network = _design_net()
    region = _design_box()
    trace = get_intermediate_bounds(network, region)
    target = NeuronRef(1, "post", 0)
    problem = build_lp_problem(network, region, trace, target)

    minimum = solve_lp(problem)
    maximum = solve_lp(problem, maximize=True)

    # Hand-computed relaxation: y in [0, 4] for the design network.
    assert minimum.objective_value == pytest.approx(0.0, abs=1e-7)
    assert maximum.objective_value == pytest.approx(4.0, abs=1e-7)
    assert minimum.is_certified and maximum.is_certified
    assert minimum.certified_bound == pytest.approx(0.0, abs=1e-7)
    assert maximum.certified_bound == pytest.approx(4.0, abs=1e-7)


def test_neuron_objective_override() -> None:
    from abstract_domain.propagation import get_intermediate_bounds
    from tests.test_lp import _design_net
    from verification.lp import build_lp_problem

    network = _design_net()
    region = _design_box()
    trace = get_intermediate_bounds(network, region)
    problem = build_lp_problem(network, region, trace, NeuronRef(1, "post", 0))

    # Solve for a DIFFERENT neuron than the stored objective: the pre
    # activation z = x0 - 2x1 + 1 lies in [-2, 4].
    minimum = solve_lp(problem, NeuronRef(1, "pre", 0))
    maximum = solve_lp(problem, NeuronRef(1, "pre", 0), maximize=True)

    assert minimum.objective_value == pytest.approx(-2.0, abs=1e-7)
    assert maximum.objective_value == pytest.approx(4.0, abs=1e-7)
    assert minimum.is_certified and maximum.is_certified
    assert minimum.objective_neuron == NeuronRef(1, "pre", 0)


def test_concrete_executions_lie_between_certified_bounds() -> None:
    from abstract_domain.propagation import get_intermediate_bounds
    from tests.test_lp import _design_net
    from verification.lp import build_lp_problem

    network = _design_net()
    region = _design_box()
    trace = get_intermediate_bounds(network, region)
    target = NeuronRef(1, "post", 0)
    problem = build_lp_problem(network, region, trace, target)
    minimum = solve_lp(problem)
    maximum = solve_lp(problem, maximize=True)

    rng = np.random.default_rng(7)
    for _ in range(20):
        point = rng.uniform([-1.0, -1.0], [1.0, 1.0])
        value = float(network.forward(point)[0])
        assert minimum.certified_bound <= value + 1e-7
        assert maximum.certified_bound >= value - 1e-7


def _design_box():
    from abstract_domain.box import Box

    return Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))


# --------------------------------------------------------------------------
# hardened numerical validation (large-scale & execution-time contracts)
# --------------------------------------------------------------------------

def test_large_dual_objective_rejects_positive_inequality_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exact stationarity (0), zero primal residual, zero duality gap, and large
    # dual objective 1e10, yet y[1] = +0.01 violates y <= 0.
    problem_large_rhs = _make_problem(
        a_ub=[[0.0, -1.0], [0.0, -1.0]],
        b_ub=[-1e10, -1e10],
    )
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1e10],
                status=0,
                fun=1e10,
                y=[-1.01, 0.01],
                lam_lb=[0.0, 0.0],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    res_rhs = solve_lp(problem_large_rhs, [0.0, 1.0])
    assert res_rhs.status is OPTIMAL
    assert res_rhs.validation_status is UNVALIDATED
    assert res_rhs.is_certified is False
    assert res_rhs.certified_bound is None
    assert res_rhs.max_dual_residual > 1e-8
    assert "inequality multiplier sign violation" in res_rhs.message

    # Also verify large primal objective c = [0, 1e10] cannot mask a substantial
    # absolute sign violation (+0.01) via ||y||_inf scaling.
    problem_large_c = _make_problem(
        a_ub=[[0.0, -1.0], [0.0, -1.0]],
        b_ub=[-1.0, -1.0],
    )
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1.0],
                status=0,
                fun=1e10,
                y=[-(1e10 + 0.01), 0.01],
                lam_lb=[0.0, 0.0],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    res_c = solve_lp(problem_large_c, [0.0, 1e10])
    assert res_c.validation_status is UNVALIDATED
    assert res_c.certified_bound is None
    assert res_c.max_dual_residual > 1e-8


def test_large_dual_objective_rejects_negative_lower_bound_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exact stationarity and zero gap at dual = 1e10, with lambda_lb[1] = -0.01 < 0.
    problem = _make_problem(
        a_ub=[[0.0, -1.0]],
        b_ub=[-1e10],
        bounds=[(None, None), (1e10, None)],
    )
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1e10],
                status=0,
                fun=1e10,
                y=[-1.01],
                lam_lb=[0.0, -0.01],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    result = solve_lp(problem, [0.0, 1.0])

    assert result.status is OPTIMAL
    assert result.validation_status is UNVALIDATED
    assert result.is_certified is False
    assert result.certified_bound is None
    assert result.max_dual_residual > 1e-8
    assert "lower-bound multiplier sign violation" in result.message

    # Ensure huge y scale does not dilute an independent lambda_lb check.
    problem_huge_y = _make_problem(
        a_ub=[[0.0, -1.0]],
        b_ub=[-1.0],
        bounds=[(None, None), (1.0, None)],
    )
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1.0],
                status=0,
                fun=1e10,
                y=[-(1e10 + 0.01)],
                lam_lb=[0.0, -0.01],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    res_indep = solve_lp(problem_huge_y, [0.0, 1e10])
    assert res_indep.validation_status is UNVALIDATED
    assert res_indep.certified_bound is None
    assert res_indep.max_dual_residual > 1e-8


def test_large_dual_objective_rejects_positive_upper_bound_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Maximization of z0 <= 1e10 (dual = +1e10) with invalid lambda_ub[1] = +0.01.
    problem = _make_problem(
        a_ub=[[0.0, 1.0]],
        b_ub=[1e10],
        bounds=[(None, None), (None, 1e10)],
    )
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1e10],
                status=0,
                fun=-1e10,
                y=[-1.01],
                lam_lb=[0.0, 0.0],
                lam_ub=[0.0, 0.01],
            )
        ],
    )
    result = solve_lp(problem, [0.0, 1.0], maximize=True)

    assert result.status is OPTIMAL
    assert result.validation_status is UNVALIDATED
    assert result.is_certified is False
    assert result.certified_bound is None
    assert result.max_dual_residual > 1e-8
    assert "upper-bound multiplier sign violation" in result.message


def test_valid_dual_multipliers_accepted_at_small_and_large_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real HiGHS solves on large-scale valid LPs remain certified.
    problem = _make_problem(
        a_ub=[[0.0, -1.0], [0.0, 1.0]],
        b_ub=[-1e10, 2e10],
        bounds=[(0.0, 1e10), (None, None)],
    )
    min_res = solve_lp(problem, [0.0, 1.0])
    max_res = solve_lp(problem, [0.0, 1.0], maximize=True)

    assert min_res.validation_status is VALIDATED
    assert min_res.is_certified is True
    assert min_res.certified_bound == pytest.approx(1e10)
    assert max_res.validation_status is VALIDATED
    assert max_res.is_certified is True
    assert max_res.certified_bound == pytest.approx(2e10)

    # Moderate multiplier magnitude (||y||_inf = 50.0) tolerates tiny 4e-9
    # relative floating-point roundoff (within 1e-8 relative and 1e-6 abs cap).
    mod_problem = _make_problem(
        a_ub=[[0.0, -1.0], [1.0, 0.0]],
        b_ub=[-1.0, 0.0],
    )
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1.0],
                status=0,
                fun=50.0,
                y=[-50.0, 2e-7],
                lam_lb=[0.0, 0.0],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    mod_res = solve_lp(mod_problem, [0.0, 50.0])
    assert mod_res.validation_status is VALIDATED
    assert mod_res.certified_bound == pytest.approx(50.0)


def test_large_scale_primal_residual_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. An unrelated large coordinate (x[1] = 1e10) must not dilute a violation
    #    on row x[0] <= 1.0.
    problem_unrelated = _make_problem(
        a_ub=[[1.0, 0.0]],
        b_ub=[1.0],
        bounds=[(None, None), (1e10, 1e10)],
    )
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [2.0, 1e10],
                status=0,
                fun=1e10,
                y=[0.0],
                lam_lb=[0.0, 1.0],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    res_unrelated = solve_lp(problem_unrelated, [0.0, 1.0])
    diag_unrelated = _primal_residual_diagnostics(
        problem_unrelated, np.array([2.0, 1e10], dtype=np.float64)
    )
    assert diag_unrelated.raw_residual == pytest.approx(1.0)
    assert diag_unrelated.normalized_residual == pytest.approx(0.5)
    assert res_unrelated.max_primal_residual == pytest.approx(0.5)
    assert res_unrelated.validation_status is UNVALIDATED
    assert res_unrelated.certified_bound is None

    # 2. When a bound itself is huge (z0 >= 1e10) and primal violates it by 1.0,
    #    normalized residual is 1e-10 <= 1e-8, while raw residual is 1.0; the
    #    distinction is preserved and the infeasible point is NEVER certified.
    problem_large_bound = _make_problem(
        bounds=[(0.0, 1.0), (1e10, None)],
    )
    infeasible_x = np.array([0.0, 1e10 - 1.0], dtype=np.float64)
    diag_large = _primal_residual_diagnostics(problem_large_bound, infeasible_x)
    assert diag_large.raw_residual == pytest.approx(1.0)
    assert diag_large.normalized_residual == pytest.approx(1e-10)

    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                infeasible_x,
                status=0,
                fun=1e10 - 1.0,
                y=[],
                lam_lb=[0.0, 1.0],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    res_infeasible = solve_lp(problem_large_bound, [0.0, 1.0])
    assert res_infeasible.validation_status is UNVALIDATED
    assert res_infeasible.is_certified is False
    assert res_infeasible.certified_bound is None
    assert res_infeasible.max_primal_residual == pytest.approx(1e-10)
    assert "raw 1.000e+00" in res_infeasible.message


def test_valid_large_scale_lp_remains_certified() -> None:
    # Valid large-scale rotated constraints solved by real HiGHS still validate.
    problem_valid_large = _make_problem(
        a_ub=[[1.7, 2.3], [-3.1, 1.4], [0.7, -1.9]],
        b_ub=[3.0e10, -0.5e10, -0.1e10],
        a_eq=[[1.1, 2.9]],
        b_eq=[2.2e10],
    )
    valid_res = solve_lp(problem_valid_large, [0.3, 0.7])
    assert valid_res.status is OPTIMAL
    assert valid_res.validation_status is VALIDATED
    assert valid_res.is_certified is True
    assert "primal_raw=" in valid_res.message
    assert "primal_norm=" in valid_res.message


def test_large_scale_negative_duality_gap_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Minimization with primal = 1e10 and dual = 1e10 + 10.0: relative gap is
    # only -1e-9 (previously accepted!), yet absolute gap is -10.0.
    problem_min = _make_problem(a_ub=[[0.0, -1.0]], b_ub=[-1e10])
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1e10],
                status=0,
                fun=1e10,
                y=[-(1.0 + 1e-9)],
                lam_lb=[0.0, 0.0],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    res_min = solve_lp(problem_min, [0.0, 1.0])
    assert res_min.duality_gap == pytest.approx(-10.0)
    assert res_min.validation_status is UNVALIDATED
    assert res_min.is_certified is False
    assert res_min.certified_bound is None
    assert "negative absolute duality gap -1.000e+01" in res_min.message
    assert "rel_gap -1.000e-09" in res_min.message

    # Maximization direction: primal = 1e10, conservative dual upper = 1e10 - 10.0.
    problem_max = _make_problem(a_ub=[[0.0, 1.0]], b_ub=[1e10])
    _monkeypatch_linprog(
        monkeypatch,
        [
            _fake_result(
                [0.0, 1e10],
                status=0,
                fun=-1e10,
                y=[-(1.0 - 1e-9)],
                lam_lb=[0.0, 0.0],
                lam_ub=[0.0, 0.0],
            )
        ],
    )
    res_max = solve_lp(problem_max, [0.0, 1.0], maximize=True)
    assert res_max.duality_gap == pytest.approx(-10.0)
    assert res_max.validation_status is UNVALIDATED
    assert res_max.is_certified is False
    assert res_max.certified_bound is None
    assert "negative absolute duality gap -1.000e+01" in res_max.message
    assert "rel_gap -1.000e-09" in res_max.message

    # Verify direction helper preserves upper - lower convention in both modes.
    gap_min = _duality_gap_diagnostics(1e10, 1e10 + 10.0, maximize=False)
    gap_max = _duality_gap_diagnostics(1e10, 1e10 - 10.0, maximize=True)
    assert gap_min.absolute_gap == pytest.approx(-10.0)
    assert gap_max.absolute_gap == pytest.approx(-10.0)
    assert gap_min.relative_gap == pytest.approx(-10.0 / (1e10 + 10.0))
    assert gap_max.relative_gap == pytest.approx(-10.0 / 1e10)


def test_positive_infinite_execution_time_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        LPSolverResult(
            status=INFEASIBLE,
            success=False,
            maximize=False,
            objective_neuron=None,
            solution=None,
            objective_value=None,
            conservative_bound=None,
            duality_gap=None,
            max_primal_residual=None,
            max_dual_residual=None,
            validation_status=UNVALIDATED,
            execution_time_seconds=float("inf"),
            message="inf time",
        )


def test_negative_infinite_execution_time_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        LPSolverResult(
            status=INFEASIBLE,
            success=False,
            maximize=False,
            objective_neuron=None,
            solution=None,
            objective_value=None,
            conservative_bound=None,
            duality_gap=None,
            max_primal_residual=None,
            max_dual_residual=None,
            validation_status=UNVALIDATED,
            execution_time_seconds=float("-inf"),
            message="-inf time",
        )


def test_nan_execution_time_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        LPSolverResult(
            status=INFEASIBLE,
            success=False,
            maximize=False,
            objective_neuron=None,
            solution=None,
            objective_value=None,
            conservative_bound=None,
            duality_gap=None,
            max_primal_residual=None,
            max_dual_residual=None,
            validation_status=UNVALIDATED,
            execution_time_seconds=float("nan"),
            message="nan time",
        )


def test_minimization_and_maximization_behavior_preserved() -> None:
    problem = _make_problem(
        a_ub=[[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
        b_ub=[3.0, 4.0, -2.0],
        bounds=[(0.0, None), (0.0, None)],
    )
    minimum = solve_lp(problem, [1.0, 2.0])
    maximum = solve_lp(problem, [1.0, 2.0], maximize=True)

    assert minimum.status is OPTIMAL
    assert maximum.status is OPTIMAL
    assert minimum.is_certified and maximum.is_certified
    assert minimum.certified_bound == pytest.approx(2.0)
    assert minimum.lower_bound == pytest.approx(2.0)
    assert maximum.certified_bound == pytest.approx(11.0)
    assert maximum.upper_bound == pytest.approx(11.0)
    assert "abs_gap=" in minimum.message and "rel_gap=" in minimum.message
    assert "abs_gap=" in maximum.message and "rel_gap=" in maximum.message


def test_unvalidated_results_never_expose_certified_bound() -> None:
    optimal_but_unvalidated = LPSolverResult(
        status=OPTIMAL,
        success=True,
        maximize=False,
        objective_neuron=None,
        solution=np.array([0.0, 1.0]),
        objective_value=1.0,
        conservative_bound=0.9,
        duality_gap=0.1,
        max_primal_residual=1e-3,
        max_dual_residual=0.0,
        validation_status=UNVALIDATED,
        execution_time_seconds=0.0,
        message="unvalidated despite optimal",
    )
    assert optimal_but_unvalidated.is_certified is False
    assert optimal_but_unvalidated.certified_bound is None

    for status in (INFEASIBLE, UNBOUNDED, ITERATION_LIMIT, NUMERICAL_ISSUE, INVALID_PROBLEM, FAILED):
        res = LPSolverResult(
            status=status,
            success=False,
            maximize=False,
            objective_neuron=None,
            solution=None,
            objective_value=None,
            conservative_bound=None,
            duality_gap=None,
            max_primal_residual=None,
            max_dual_residual=None,
            validation_status=UNVALIDATED,
            execution_time_seconds=0.0,
            message=f"terminal {status.value}",
        )
        assert res.is_certified is False
        assert res.certified_bound is None


def test_repeated_solves_reuse_sparse_matrices_and_swap_objectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize
    from scipy import sparse as sp

    problem = _make_problem(
        a_ub=[[1.0, 1.0], [-1.0, 0.0]],
        b_ub=[3.0, 0.0],
        a_eq=[[1.0, -2.0]],
        b_eq=[0.0],
        bounds=[(0.0, None), (0.0, 2.0)],
    )

    seen_calls: list[dict[str, object]] = []
    real_linprog = scipy.optimize.linprog

    def spy_linprog(*args, **kwargs):
        seen_calls.append(kwargs)
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", spy_linprog)

    res_min_x = solve_lp(problem, [1.0, 0.0], maximize=False)
    res_max_x = solve_lp(problem, [1.0, 0.0], maximize=True)
    res_min_y = solve_lp(problem, [0.0, 1.0], maximize=False)
    res_max_y = solve_lp(problem, [0.0, 1.0], maximize=True)

    assert len(seen_calls) == 4
    for call_kwargs in seen_calls:
        assert sp.isspmatrix_csc(call_kwargs["A_ub"])
        assert sp.isspmatrix_csc(call_kwargs["A_eq"])
        assert call_kwargs["A_ub"] is problem.A_ub_csc
        assert call_kwargs["A_eq"] is problem.A_eq_csc
        assert call_kwargs["bounds"] is problem.bounds_array

    # x = 2y and x + y <= 3, y >= 0, y <= 2 -> x in [0, 2], y in [0, 1].
    assert res_min_x.is_certified and res_max_x.is_certified
    assert res_min_y.is_certified and res_max_y.is_certified
    assert res_min_x.certified_bound == pytest.approx(0.0, abs=1e-9)
    assert res_max_x.certified_bound == pytest.approx(2.0, abs=1e-9)
    assert res_min_y.certified_bound == pytest.approx(0.0, abs=1e-9)
    assert res_max_y.certified_bound == pytest.approx(1.0, abs=1e-9)


def test_sparse_lp_preserves_infeasible_and_unvalidated_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize

    infeasible = _make_problem(
        a_ub=[[1.0, 0.0], [-1.0, 0.0]],
        b_ub=[0.5, -1.0],
        bounds=[(0.0, 2.0), (0.0, 2.0)],
    )
    inf_res = solve_lp(infeasible, [1.0, 0.0])
    assert inf_res.status is INFEASIBLE
    assert inf_res.is_certified is False
    assert inf_res.certified_bound is None

    feasible = _make_problem(
        a_ub=[[1.0, 1.0]],
        b_ub=[2.0],
        bounds=[(0.0, 2.0), (0.0, 2.0)],
    )
    monkeypatch.setattr(
        scipy.optimize,
        "linprog",
        lambda *a, **k: _fake_result(
            [1.0, 1.0],
            fun=1.0,
            y=[-1.0],
            lam_lb=[0.0, 0.0],
            lam_ub=[0.0, 0.0],
        ),
    )
    unval = solve_lp(feasible, [1.0, 0.0])
    assert unval.status is OPTIMAL
    assert unval.validation_status is UNVALIDATED
    assert unval.is_certified is False
    assert unval.certified_bound is None


# --------------------------------------------------------------------------
# Optimization 3: vectorized row & bound validation (dense and sparse CSC)
# --------------------------------------------------------------------------

def test_linear_row_violations_sparse_matches_dense() -> None:
    """Vectorized sparse CSC row violations match dense NumPy results."""
    rng = np.random.default_rng(7)
    a_dense = rng.standard_normal((8, 4))
    b = rng.standard_normal(8)
    x = rng.standard_normal(4)
    a_csc = sp.csc_matrix(a_dense)

    for eq in (False, True):
        assert _linear_row_violations(a_dense, b, x, equality=eq) == pytest.approx(
            _linear_row_violations(a_csc, b, x, equality=eq), rel=1e-12
        )


def test_linear_row_violations_satisfied_and_empty() -> None:
    """No violated rows or empty variable vector returns zero violations."""
    a_csc = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))
    b = np.array([1.0, 1.0], dtype=np.float64)
    x = np.array([0.5, 0.5], dtype=np.float64)
    assert _linear_row_violations(a_csc, b, x, equality=False) == (0.0, 0.0, 0.0)
    assert _linear_row_violations(
        np.zeros((0, 0), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        equality=False,
    ) == (0.0, 0.0, 0.0)


def test_primal_residual_diagnostics_vectorized_bounds_and_csc() -> None:
    """Vectorized bounds_array + CSC checks detect lower, upper, and infinite bounds."""
    prob_bounds = _make_problem(bounds=[(0.5, 1.0), (0.0, 0.3)])
    assert _primal_residual_diagnostics(
        prob_bounds, np.array([0.2, 0.1], dtype=np.float64)
    ).raw_residual == pytest.approx(0.3)
    assert _primal_residual_diagnostics(
        prob_bounds, np.array([0.6, 0.8], dtype=np.float64)
    ).raw_residual == pytest.approx(0.5)

    prob_inf = _make_problem(bounds=[(None, None), (None, None)])
    assert _primal_residual_diagnostics(
        prob_inf, np.array([1e10, -1e10], dtype=np.float64)
    ).raw_residual == pytest.approx(0.0)

    prob_eq = _make_problem(a_eq=[[1.0, 1.0]], b_eq=[1.0])
    assert _primal_residual_diagnostics(
        prob_eq, np.array([1.0, 1.0], dtype=np.float64)
    ).raw_residual == pytest.approx(1.0)


def test_dual_diagnostics_vectorized_unbounded_and_equality() -> None:
    """Vectorized _dual_diagnostics handles infinite bounds and equality-only LPs."""
    prob_mixed = _make_problem(
        a_ub=[[0.0, 1.0]],
        b_ub=[2.0],
        bounds=[(None, None), (0.0, None)],
    )
    res_mixed = solve_lp(prob_mixed, [0.0, -1.0])
    assert res_mixed.status is OPTIMAL
    assert res_mixed.validation_status is VALIDATED
    assert res_mixed.certified_bound == pytest.approx(-2.0, abs=1e-6)

    prob_eq = _make_problem(
        a_eq=[[1.0, 1.0]],
        b_eq=[1.0],
        bounds=[(0.0, None), (None, None)],
    )
    res_eq = solve_lp(prob_eq, [1.0, 0.0])
    assert res_eq.status is OPTIMAL
    assert res_eq.validation_status is VALIDATED
