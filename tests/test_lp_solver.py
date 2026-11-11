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

from abstract_domain.bounds import NeuronRef
from verification.lp import LPProblem, VariableLayout
from verification.lp_solver import (
    BoundValidationStatus,
    LPSolverResult,
    LPSolverStatus,
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
