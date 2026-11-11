"""Tests for the LP-based neuron bound refinement engine.

The tests exercise :func:`verification.refinement.refine_neuron_bounds` with
small analytically tractable networks:

- ``_abs_net``: f(x) = relu(x) + relu(-x) = |x| over x in [-1, 1]. IBP on the
  output neuron is [0, 2] (dependency loss), the LP relaxation tightens it to
  the exact range [0, 1]. Used for the tightening and containment tests.
- ``_design_net`` (from tests.test_lp): z = x0 - 2 x1 + 1 over [-1, 1]^2 then
  ReLU. Every IBP interval is already exact, so refinement accepts the
  targets without improving them. Used for the no-improvement tests.
- ``_stable_inactive_net`` / ``_stable_active_net`` (from tests.test_lp):
  stable ReLU neurons, including a zero-width post interval.

Solver-failure cases monkeypatch ``verification.refinement.solve_lp`` with
hand-assembled :class:`LPSolverResult` records, mirroring the conventions of
``tests.test_lp_solver``.
"""

from __future__ import annotations

import numpy as np
import pytest

from abstract_domain.bounds import NeuronRef
from abstract_domain.propagation import get_intermediate_bounds
from models.model_loader import ReLULayer
from tests.test_lp import (
    _design_net,
    _make_linear,
    _make_network,
    _stable_active_net,
    _stable_inactive_net,
)
from verification.lp_solver import (
    BoundValidationStatus,
    LPSolverResult,
    LPSolverStatus,
)
from verification.refinement import (
    RefinementResult,
    RefinementStatus,
    refine_neuron_bounds,
)
from verification.refinement_validation import validate_refined_bounds_empirically
from verification.regions import InputRegion

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _abs_net():
    """f(x) = relu(x) + relu(-x) = |x| over x in [-1, 1].

    IBP output bounds are [0, 2]; the true (and LP-relaxed) range is [0, 1].
    """
    return _make_network(
        (
            _make_linear(0, np.array([[1.0], [-1.0]]), np.zeros(2)),
            ReLULayer(position=1),
            _make_linear(2, np.array([[1.0, 1.0]]), np.zeros(1)),
        ),
        source_model="AbsNet",
    )


def _region(center, radius: float) -> InputRegion:
    return InputRegion(np.asarray(center, dtype=np.float64), float(radius))


def _refine(network, region, trace, targets):
    return refine_neuron_bounds(network, region, trace, targets)


def _make_failed_result(neuron: NeuronRef, maximize: bool) -> LPSolverResult:
    """A non-optimal solve result (simulated infeasibility)."""
    return LPSolverResult(
        status=LPSolverStatus.INFEASIBLE,
        success=False,
        maximize=maximize,
        objective_neuron=neuron,
        solution=None,
        objective_value=None,
        conservative_bound=None,
        duality_gap=None,
        max_primal_residual=None,
        max_dual_residual=None,
        validation_status=BoundValidationStatus.UNVALIDATED,
        execution_time_seconds=0.0,
        message="simulated infeasibility",
    )


def _make_certified_result(
    neuron: NeuronRef, maximize: bool, conservative: float
) -> LPSolverResult:
    """An optimal, numerically validated solve result with a chosen certified bound."""
    return LPSolverResult(
        status=LPSolverStatus.OPTIMAL,
        success=True,
        maximize=maximize,
        objective_neuron=neuron,
        solution=np.zeros(5),
        objective_value=conservative,
        conservative_bound=conservative,
        duality_gap=0.0,
        max_primal_residual=0.0,
        max_dual_residual=0.0,
        validation_status=BoundValidationStatus.VALIDATED,
        execution_time_seconds=0.0,
        message="simulated optimal",
    )


# --------------------------------------------------------------------------
# analytically known bounds and tightening
# --------------------------------------------------------------------------


def test_small_network_with_analytically_known_bounds() -> None:
    # f(x) = |x| over [-1, 1]: true range [0, 1].
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(2, "pre", 0)])

    assert result.status is RefinementStatus.TIGHTENED
    refined = result.refined_bounds[0]
    np.testing.assert_allclose(refined.lower, 0.0, atol=1e-6)
    np.testing.assert_allclose(refined.upper, 1.0, atol=1e-6)


def test_lp_tightens_ibp_bounds() -> None:
    # IBP upper is 2 (dependency loss); LP tightens it to 1.
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(2, "pre", 0)])

    outcome = result.outcomes[0]
    np.testing.assert_allclose(outcome.ibp_lower, 0.0, atol=1e-12)
    np.testing.assert_allclose(outcome.ibp_upper, 2.0, atol=1e-12)
    assert outcome.accepted
    assert outcome.tightens
    assert outcome.refined is not None
    np.testing.assert_allclose(outcome.refined.upper, 1.0, atol=1e-6)
    assert outcome.refined.upper < outcome.ibp_upper
    assert result.num_accepted == 1
    assert result.num_rejected == 0


def test_lp_produces_no_improvement() -> None:
    # The design network's IBP is already exact: z in [-2, 4], y in [0, 4].
    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(1, "post", 0)])

    assert result.status is RefinementStatus.NO_IMPROVEMENT
    outcome = result.outcomes[0]
    assert outcome.accepted
    assert not outcome.tightens
    np.testing.assert_allclose(outcome.refined.lower, 0.0, atol=1e-6)
    np.testing.assert_allclose(outcome.refined.upper, 4.0, atol=1e-6)
    # The accepted refinement is exactly the LP-certified interval, not a
    # silently substituted IBP value: it matches the certified solve bounds.
    assert outcome.refined.lower == outcome.lower_solve.certified_bound
    assert outcome.refined.upper == outcome.upper_solve.certified_bound


def test_multiple_targets_mixed_improvement() -> None:
    # h1 = x (exact), r1 = relu(x) (exact), out = |x| (tightens).
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    targets = [NeuronRef(0, "pre", 0), NeuronRef(1, "post", 0), NeuronRef(2, "pre", 0)]
    result = _refine(network, region, trace, targets)

    assert result.status is RefinementStatus.TIGHTENED
    assert result.num_accepted == 3
    assert result.num_rejected == 0
    assert [o.tightens for o in result.outcomes] == [False, False, True]


def test_stable_relus_are_exact() -> None:
    # Stable inactive: z in [-3, -1] forces y == 0 exactly (zero width).
    network = _stable_inactive_net()
    region = _region([0.5, 0.5], 0.5)  # x in [0, 1]^2
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(1, "post", 0)])

    assert result.status is RefinementStatus.NO_IMPROVEMENT
    outcome = result.outcomes[0]
    assert outcome.accepted
    np.testing.assert_allclose(outcome.ibp_lower, 0.0, atol=1e-12)
    np.testing.assert_allclose(outcome.ibp_upper, 0.0, atol=1e-12)
    np.testing.assert_allclose(outcome.refined.lower, 0.0, atol=1e-9)
    np.testing.assert_allclose(outcome.refined.upper, 0.0, atol=1e-9)

    # Stable active: z in [1, 3] forces y == z in [1, 3] exactly.
    network = _stable_active_net()
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(1, "post", 0)])
    assert result.status is RefinementStatus.NO_IMPROVEMENT
    np.testing.assert_allclose(result.refined_bounds[0].lower, 1.0, atol=1e-9)
    np.testing.assert_allclose(result.refined_bounds[0].upper, 3.0, atol=1e-9)


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------


def test_one_optimization_direction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import verification.refinement as refinement_module

    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    target = NeuronRef(1, "post", 0)
    real_solve = refinement_module.solve_lp

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        if maximize:
            return _make_failed_result(neuron, True)
        return real_solve(problem, objective, maximize=False)

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = _refine(network, region, trace, [target])

    assert result.status is RefinementStatus.FAILED
    assert result.refined_bounds == ()
    outcome = result.outcomes[0]
    assert outcome.refined is None
    # The minimization alone succeeded, but the failed maximization rejects
    # the neuron and preserves the IBP interval.
    assert outcome.lower_solve.success is True
    assert outcome.upper_solve.success is False
    assert outcome.ibp_lower == pytest.approx(0.0)
    assert outcome.ibp_upper == pytest.approx(4.0)
    assert "upper bound not certified" in outcome.message


def test_partial_status_when_only_one_target_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import verification.refinement as refinement_module

    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    z_target = NeuronRef(0, "pre", 0)   # IBP [-2, 4]
    y_target = NeuronRef(1, "post", 0)  # IBP [0, 4]
    real_solve = refinement_module.solve_lp

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        if neuron == z_target and maximize:
            return _make_failed_result(neuron, True)
        return real_solve(problem, objective, maximize=maximize)

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = _refine(network, region, trace, [z_target, y_target])

    assert result.status is RefinementStatus.PARTIAL
    assert result.num_accepted == 1
    assert result.num_rejected == 1
    assert [o.refined.neuron for o in result.outcomes if o.refined is not None] == [y_target]


def test_unvalidated_optimal_solve_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # An OPTIMAL solve whose marginals fail validation must never become a
    # refined bound.
    import verification.refinement as refinement_module

    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    target = NeuronRef(1, "post", 0)

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        return LPSolverResult(
            status=LPSolverStatus.OPTIMAL,
            success=True,
            maximize=maximize,
            objective_neuron=neuron,
            solution=np.zeros(5),
            objective_value=0.5,
            conservative_bound=0.5,
            duality_gap=0.0,
            max_primal_residual=1e-3,  # violates the 1e-8 validation tolerance
            max_dual_residual=1e-3,
            validation_status=BoundValidationStatus.UNVALIDATED,
            execution_time_seconds=0.0,
            message="optimal but unvalidated marginals",
        )

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = _refine(network, region, trace, [target])

    assert result.status is RefinementStatus.FAILED
    assert result.refined_bounds == ()
    assert "lower bound not certified" in result.outcomes[0].message


def test_crossed_certified_bounds_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both directions "certified" but lower (3.0) exceeds upper (2.0): the
    # neuron must be rejected and never produce an invalid interval.
    import verification.refinement as refinement_module

    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    target = NeuronRef(1, "post", 0)  # IBP [0, 4]

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        conservative = 2.0 if maximize else 3.0
        return _make_certified_result(neuron, maximize, conservative)

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = _refine(network, region, trace, [target])

    assert result.status is RefinementStatus.FAILED
    assert result.refined_bounds == ()
    assert "certified bounds cross" in result.outcomes[0].message


def test_ibp_bounds_preserved_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rejected neurons keep their IBP interval: they are absent from
    # refined_bounds and their outcomes still record the preserved values.
    import verification.refinement as refinement_module

    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    target = NeuronRef(0, "pre", 0)  # IBP [-2, 4]

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        return _make_failed_result(neuron, maximize)

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = _refine(network, region, trace, [target])

    outcome = result.outcomes[0]
    assert outcome.ibp_lower == pytest.approx(-2.0)
    assert outcome.ibp_upper == pytest.approx(4.0)
    assert outcome.refined is None
    assert "IBP interval [-2, 4] preserved" in outcome.message


# --------------------------------------------------------------------------
# target list handling
# --------------------------------------------------------------------------


def test_empty_target_list() -> None:
    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [])

    assert result.status is RefinementStatus.EMPTY_TARGETS
    assert result.target_neurons == ()
    assert result.outcomes == ()
    assert result.refined_bounds == ()
    assert result.num_targets == 0


def test_duplicate_targets_are_deduplicated() -> None:
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    out_target = NeuronRef(2, "pre", 0)
    r1_target = NeuronRef(1, "post", 0)
    targets = [out_target, out_target, r1_target, out_target]

    result = _refine(network, region, trace, targets)

    assert result.target_neurons == (out_target, r1_target)
    assert result.num_targets == 2
    assert len(result.outcomes) == 2
    assert [o.neuron for o in result.outcomes] == [out_target, r1_target]


def test_invalid_neuron_references_raise() -> None:
    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)

    with pytest.raises(ValueError):
        _refine(network, region, trace, [NeuronRef(5, "pre", 0)])  # layer out of range
    with pytest.raises(ValueError):
        _refine(network, region, trace, [NeuronRef(0, "post", 0)])  # linear layer has no post
    with pytest.raises(ValueError):
        _refine(network, region, trace, [NeuronRef(0, "pre", 7)])  # neuron out of range
    with pytest.raises(TypeError):
        _refine(network, region, trace, [0, 1])  # not NeuronRefs
    with pytest.raises(TypeError):
        _refine(network, region, trace, NeuronRef(0, "pre", 0))  # not a tuple/list


def test_argument_type_and_dimension_validation() -> None:
    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    target = [NeuronRef(0, "pre", 0)]

    with pytest.raises(TypeError):
        _refine(object(), region, trace, target)
    with pytest.raises(TypeError):
        _refine(network, "not-a-region", trace, target)
    with pytest.raises(TypeError):
        _refine(network, region, object(), target)
    with pytest.raises(ValueError):
        _refine(network, _region([0.0], 1.0), trace, target)  # dimension mismatch


def test_trace_region_containment_is_enforced() -> None:
    # A trace computed for a wider region is valid for a contained query
    # region, but a query region outside the trace's input box is rejected.
    network = _design_net()
    wide = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, wide)

    # Over x in [-0.25, 0.25]^2 the design network has z = x0 - 2 x1 + 1 in
    # [0.25, 1.75] and y = relu(z) = z. The LP is built for the query region
    # with the wide trace's relaxation parameters (z in [-2, 4]), so the
    # triangle relaxation certifies y in [0.25, 2.5]: tighter than the wide
    # IBP envelope [0, 4] while still enclosing the concrete range.
    small = _region([0.0, 0.0], 0.25)
    result = _refine(network, small, trace, [NeuronRef(1, "post", 0)])
    assert result.status is RefinementStatus.TIGHTENED
    refined = result.refined_bounds[0]
    np.testing.assert_allclose(refined.lower, 0.25, atol=1e-6)
    np.testing.assert_allclose(refined.upper, 2.5, atol=1e-6)
    assert refined.lower >= 0.0 - 1e-12  # contained in the trace IBP envelope
    assert refined.upper <= 4.0 + 1e-12

    rng = np.random.default_rng(3)
    for _ in range(20):
        point = rng.uniform([-0.25, -0.25], [0.25, 0.25])
        value = float(network.forward(point)[0])
        assert refined.lower <= value + 1e-7
        assert refined.upper >= value - 1e-7

    outside = _region([3.0, 3.0], 0.5)
    with pytest.raises(ValueError):
        _refine(network, outside, trace, [NeuronRef(1, "post", 0)])


# --------------------------------------------------------------------------
# soundness and structural invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "neuron",
    [
        NeuronRef(0, "pre", 0),
        NeuronRef(1, "post", 0),
        NeuronRef(2, "pre", 0),
    ],
    ids=["h1", "r1", "out"],
)
def test_refinements_are_contained_in_ibp_intervals(neuron: NeuronRef) -> None:
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [neuron])

    outcome = result.outcomes[0]
    assert outcome.accepted
    assert outcome.refined is not None
    # Never widen the IBP interval (up to the documented 1e-12 tolerance).
    assert outcome.refined.lower >= outcome.ibp_lower - 1e-12
    assert outcome.refined.upper <= outcome.ibp_upper + 1e-12
    # Never an invalid interval.
    assert outcome.refined.lower <= outcome.refined.upper
    # Only certified conservative bounds are used.
    assert outcome.lower_solve.is_certified and outcome.upper_solve.is_certified
    assert outcome.refined.lower == outcome.lower_solve.certified_bound
    assert outcome.refined.upper == outcome.upper_solve.certified_bound


def test_concrete_executions_lie_inside_refined_bounds() -> None:
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(2, "pre", 0)])

    refined = result.refined_bounds[0]
    rng = np.random.default_rng(11)
    for _ in range(50):
        point = rng.uniform([-1.0], [1.0])
        value = float(network.forward(point)[0])  # |x|
        assert refined.lower <= value + 1e-7
        assert refined.upper >= value - 1e-7


def test_empirical_validation_of_accepted_refinements() -> None:
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(
        network, region, trace, [NeuronRef(1, "post", 0), NeuronRef(2, "pre", 0)]
    )

    assert result.status is RefinementStatus.TIGHTENED
    report = validate_refined_bounds_empirically(
        network, region, result.refined_bounds, num_samples=300, seed=7
    )
    assert report["all_refinements_passed"]


def test_result_construction_enforces_status_consistency() -> None:
    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(1, "post", 0)])
    assert result.status is RefinementStatus.NO_IMPROVEMENT

    # The same outcomes must never be re-labeled with an inconsistent status.
    with pytest.raises(ValueError):
        RefinementResult(
            status=RefinementStatus.FAILED,
            target_neurons=result.target_neurons,
            outcomes=result.outcomes,
            execution_time_seconds=result.execution_time_seconds,
            message="inconsistent label",
        )


def test_result_construction_rejects_duplicate_targets() -> None:
    neuron = NeuronRef(0, "pre", 0)
    network = _design_net()
    region = _region([0.0, 0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [neuron])

    with pytest.raises(ValueError):
        RefinementResult(
            status=result.status,
            target_neurons=(neuron, neuron),
            outcomes=result.outcomes,
            execution_time_seconds=0.0,
            message="duplicated",
        )


def test_result_is_frozen_and_records_solver_outcomes() -> None:
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(2, "pre", 0)])

    outcome = result.outcomes[0]
    # Individual lower/upper solver outcomes are recorded verbatim.
    assert outcome.lower_solve.maximize is False
    assert outcome.upper_solve.maximize is True
    assert outcome.lower_solve.objective_neuron == NeuronRef(2, "pre", 0)
    assert outcome.upper_solve.objective_neuron == NeuronRef(2, "pre", 0)
    assert outcome.lower_solve.status is LPSolverStatus.OPTIMAL
    assert outcome.upper_solve.status is LPSolverStatus.OPTIMAL
    assert result.execution_time_seconds >= 0.0
    assert result.message

    with pytest.raises(Exception):
        result.status = RefinementStatus.FAILED  # type: ignore[misc]


def test_result_string_reports_status() -> None:
    network = _abs_net()
    region = _region([0.0], 1.0)
    trace = get_intermediate_bounds(network, region)
    result = _refine(network, region, trace, [NeuronRef(2, "pre", 0)])

    text = str(result)
    assert "TIGHTENED" in text
    assert "accepted=1" in text
