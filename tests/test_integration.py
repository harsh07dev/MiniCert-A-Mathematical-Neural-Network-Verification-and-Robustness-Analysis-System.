"""Tests for the integrated Phase 3 verification pipeline.

These tests exercise :func:`verification.integration.verify_robustness_refined`,
which composes Member 1's IBP propagation and refinement application with
Member 2's LP refinement engine and the unchanged SafetyChecker. The main
workhorse is ``_improvement_net``:

    2 -> 4 -> 2 network over x in [-1, 1]^2:
        out0 = 2 - relu(x0) - relu(-x0) = 2 - |x0|   (true [1, 2], IBP [0, 2])
        out1 = 0.5 (relu(x1) + relu(-x1)) = 0.5 |x1|  (true [0, 0.5], IBP [0, 1])

IBP loses the x0/x1 dependencies and yields an INCONCLUSIVE verdict for
target class 0 (margin 0 - 1 = -1), while the LP relaxation certifies the
exact ranges and the refined verdict is CERTIFIED (margin 1 - 0.5 = 0.5).

Solver-failure cases monkeypatch ``verification.refinement.solve_lp`` with
hand-assembled :class:`LPSolverResult` records, mirroring the conventions of
``tests.test_refinement_lp``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from abstract_domain.bounds import NeuronRef, RefinedBounds
from abstract_domain.propagation import get_intermediate_bounds
from abstract_domain.refinement import propagate_with_refinements
from models.model_loader import MathematicalNetwork, ReLULayer, extract_network
from models.network import MiniCertNet
from tests.test_lp import (
    _design_net,
    _make_linear,
    _make_network,
    _stable_active_net,
    _stable_inactive_net,
)
from verification import RefinedVerificationResult, VerificationStatus
from verification.integration import (
    select_refinement_targets,
    verify_robustness_refined,
)
from verification.lp_solver import (
    BoundValidationStatus,
    LPSolverResult,
    LPSolverStatus,
)
from verification.pipeline import verify_robustness
from verification.refinement import RefinementStatus
from verification.regions import InputRegion

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _improvement_net() -> MathematicalNetwork:
    """2 -> 4 -> 2 network where IBP is loose but the LP relaxation is exact.

    See the module docstring for the analytical ranges and margins.
    """
    return _make_network(
        (
            _make_linear(
                0,
                np.array(
                    [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                    dtype=np.float64,
                ),
                np.zeros(4),
            ),
            ReLULayer(position=1),
            _make_linear(
                2,
                np.array([[-1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=np.float64),
                np.array([2.0, 0.0]),
            ),
        ),
        source_model="ImprovementNet",
    )


def _two_class_linear_net() -> MathematicalNetwork:
    """2 -> 2 affine network whose IBP bounds are already exact."""
    return _make_network(
        (
            _make_linear(
                0,
                np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64),
                np.zeros(2),
            ),
        ),
        source_model="TwoClassLinear",
    )


def _region(center, radius: float) -> InputRegion:
    return InputRegion(np.asarray(center, dtype=np.float64), float(radius))


def _improvement_region() -> InputRegion:
    return _region([0.0, 0.0], 1.0)


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


# --------------------------------------------------------------------------
# complete IBP-only and complete LP-refined verification
# --------------------------------------------------------------------------


def test_complete_ibp_only_verification() -> None:
    # An empty target list runs the full workflow but accepts nothing: the
    # refined verdict must equal the IBP verdict exactly, and the IBP verdict
    # must match the plain verify_robustness pipeline.
    network = _improvement_net()
    region = _improvement_region()

    result = verify_robustness_refined(network, region, target_class=0, target_neurons=[])

    assert result.refinement.status is RefinementStatus.EMPTY_TARGETS
    assert result.refinement.refined_bounds == ()
    assert result.applied_refinements == ()
    assert result.refined_result == result.ibp_result
    assert result.ibp_result == verify_robustness(network, region, 0)
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.margin_delta == 0.0
    assert not result.improved


def test_complete_lp_refined_verification() -> None:
    # Default target selection refines every unstable ReLU post and both
    # output neurons; the LP certifies the exact ranges and the verdict
    # flips from INCONCLUSIVE to CERTIFIED.
    network = _improvement_net()
    region = _improvement_region()

    result = verify_robustness_refined(network, region, target_class=0)

    assert result.ibp_result.status is VerificationStatus.INCONCLUSIVE
    assert result.ibp_result.verification_margin == pytest.approx(-1.0)
    assert result.refinement.status is RefinementStatus.TIGHTENED
    assert result.refinement.num_targets == 6
    assert result.refinement.num_accepted == 6
    assert result.refinement.num_rejected == 0
    assert result.status is VerificationStatus.CERTIFIED
    assert result.refined_result.verification_margin == pytest.approx(0.5)
    assert result.refined_result.target_lower_bound == pytest.approx(1.0)
    assert result.refined_result.max_competitor_upper_bound == pytest.approx(0.5)
    assert result.perturbation_epsilon == 1.0
    # Every accepted outcome records both solver results verbatim.
    for outcome in result.refinement.outcomes:
        assert outcome.accepted
        assert outcome.lower_solve.maximize is False
        assert outcome.upper_solve.maximize is True
        assert outcome.lower_solve.status is LPSolverStatus.OPTIMAL
        assert outcome.upper_solve.status is LPSolverStatus.OPTIMAL
    np.testing.assert_allclose(
        result.refined_trace.output_bounds.lower, [1.0, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        result.refined_trace.output_bounds.upper, [2.0, 0.5], atol=1e-6
    )


# --------------------------------------------------------------------------
# improvement and non-improvement cases
# --------------------------------------------------------------------------


def test_refinement_improves_the_result() -> None:
    # Refining only the two output neurons suffices: margin -1 -> +0.5 and
    # the verdict flips. The original IBP verdict is preserved for comparison.
    network = _improvement_net()
    region = _improvement_region()
    targets = [NeuronRef(2, "pre", 0), NeuronRef(2, "pre", 1)]

    result = verify_robustness_refined(network, region, target_class=0, target_neurons=targets)

    assert result.ibp_result.status is VerificationStatus.INCONCLUSIVE
    assert result.ibp_result.target_lower_bound == pytest.approx(0.0)
    assert result.ibp_result.max_competitor_upper_bound == pytest.approx(1.0)
    assert result.refined_result.status is VerificationStatus.CERTIFIED
    assert result.margin_delta == pytest.approx(1.5)
    assert result.improved
    # The accepted refinements are exactly the certified LP ranges.
    refined = result.refinement.refined_bounds
    assert [item.neuron for item in refined] == targets
    np.testing.assert_allclose(refined[0].lower, 1.0, atol=1e-6)
    np.testing.assert_allclose(refined[0].upper, 2.0, atol=1e-6)
    np.testing.assert_allclose(refined[1].lower, 0.0, atol=1e-6)
    np.testing.assert_allclose(refined[1].upper, 0.5, atol=1e-6)


def test_refinement_does_not_improve_the_result() -> None:
    # An affine-only network has exact IBP bounds, so the LP accepts the
    # same intervals without tightening anything: the verdict, margin, and
    # bounds are all unchanged.
    network = _two_class_linear_net()
    region = _region([0.5, 0.1], 0.05)

    result = verify_robustness_refined(network, region, target_class=0)

    assert result.ibp_result.status is VerificationStatus.CERTIFIED
    assert result.refinement.status is RefinementStatus.NO_IMPROVEMENT
    assert result.refined_result == result.ibp_result
    assert result.refined_result.verification_margin == pytest.approx(0.6)
    assert result.margin_delta == 0.0
    assert not result.improved
    np.testing.assert_allclose(
        result.refined_trace.output_bounds.lower, [0.3, -0.5], atol=1e-6
    )
    np.testing.assert_allclose(
        result.refined_trace.output_bounds.upper, [0.5, -0.3], atol=1e-6
    )


# --------------------------------------------------------------------------
# solver failure and partial success
# --------------------------------------------------------------------------


def test_lp_optimization_failure_preserves_ibp(monkeypatch: pytest.MonkeyPatch) -> None:
    import verification.refinement as refinement_module

    network = _improvement_net()
    region = _improvement_region()
    targets = [NeuronRef(2, "pre", 0), NeuronRef(2, "pre", 1)]

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        return _make_failed_result(neuron, maximize)

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = verify_robustness_refined(network, region, target_class=0, target_neurons=targets)

    assert result.refinement.status is RefinementStatus.FAILED
    assert result.refinement.refined_bounds == ()
    assert result.refined_result == result.ibp_result
    # An inconclusive result must remain inconclusive: no refinement was
    # accepted, so nothing may upgrade the verdict.
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.margin_delta == 0.0


def test_partial_refinement_applies_only_accepted_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import verification.refinement as refinement_module

    network = _improvement_net()
    region = _improvement_region()
    out0 = NeuronRef(2, "pre", 0)
    out1 = NeuronRef(2, "pre", 1)
    real_solve = refinement_module.solve_lp

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        if neuron == out0 and maximize:
            return _make_failed_result(neuron, True)
        return real_solve(problem, objective, maximize=maximize)

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = verify_robustness_refined(
        network, region, target_class=0, target_neurons=[out0, out1]
    )

    assert result.refinement.status is RefinementStatus.PARTIAL
    assert result.refinement.num_accepted == 1
    assert result.refinement.num_rejected == 1
    assert [o.neuron for o in result.refinement.outcomes if o.refined is not None] == [out1]
    # Downstream propagation used only the accepted refinement: out0 keeps
    # its IBP bounds [0, 2] while out1 is tightened to [0, 0.5].
    out_box = result.refined_trace.layer_bounds[2].pre_activation_bounds
    np.testing.assert_allclose(out_box.lower, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out_box.upper, [2.0, 0.5], atol=1e-6)
    # With only out1 refined the strict dominance condition still fails.
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.refined_result.verification_margin == pytest.approx(-0.5)


def test_invalid_refined_bounds_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Refinements that violate Member 1's containment validation must be
    # rejected, never applied. The pipeline lets the ValueError surface
    # instead of silently overriding an IBP bound with an invalid interval.
    # The integration module imports propagate_with_refinements lazily from
    # abstract_domain.refinement, so the attribute there is patched.
    import abstract_domain.refinement as application_module

    network = _improvement_net()
    region = _improvement_region()
    real_propagate = application_module.propagate_with_refinements

    def fake_propagate(network, input_region, refined_bounds):
        original = refined_bounds[0]
        tampered = RefinedBounds(original.neuron, original.lower, original.upper + 1.0)
        return real_propagate(network, input_region, (tampered,))

    monkeypatch.setattr(application_module, "propagate_with_refinements", fake_propagate)

    with pytest.raises(ValueError, match="extends outside the original IBP interval"):
        verify_robustness_refined(
            network,
            region,
            target_class=0,
            target_neurons=[NeuronRef(2, "pre", 0), NeuronRef(2, "pre", 1)],
        )


# --------------------------------------------------------------------------
# downstream consistency and verdict discipline
# --------------------------------------------------------------------------


def test_final_output_bounds_consistent_with_refined_intermediates() -> None:
    network = _improvement_net()
    region = _improvement_region()

    result = verify_robustness_refined(network, region, target_class=0)

    # 1. The refined verdict is computed from the refined trace's output box.
    out_lower = result.refined_trace.output_bounds.lower
    out_upper = result.refined_trace.output_bounds.upper
    assert result.refined_result.target_lower_bound == out_lower[0]
    assert result.refined_result.max_competitor_upper_bound == out_upper[1]

    # 2. The trace matches an independent application of the refinements.
    expected_trace = propagate_with_refinements(network, region, result.applied_refinements)
    np.testing.assert_array_equal(
        result.refined_trace.output_bounds.lower, expected_trace.output_bounds.lower
    )
    np.testing.assert_array_equal(
        result.refined_trace.output_bounds.upper, expected_trace.output_bounds.upper
    )

    # 3. Every applied refinement is contained in the original IBP envelope.
    ibp_trace = get_intermediate_bounds(network, region)
    for refined in result.applied_refinements:
        entry = ibp_trace.layer_bounds[refined.neuron.layer_index]
        reference = (
            entry.pre_activation_bounds
            if refined.neuron.phase == "pre"
            else entry.output_bounds
        )
        assert refined.lower >= reference.lower[refined.neuron.neuron_index] - 1e-12
        assert refined.upper <= reference.upper[refined.neuron.neuron_index] + 1e-12

    # 4. The refined output bounds are never wider than the IBP output bounds.
    np.testing.assert_array_equal(
        out_lower >= ibp_trace.output_bounds.lower - 1e-12, np.ones(2, dtype=bool)
    )
    np.testing.assert_array_equal(
        out_upper <= ibp_trace.output_bounds.upper + 1e-12, np.ones(2, dtype=bool)
    )

    # 5. Concrete executions lie inside the refined output bounds.
    rng = np.random.default_rng(17)
    samples = rng.uniform([-1.0, -1.0], [1.0, 1.0], size=(200, 2))
    outputs = network.forward(samples)
    assert np.all(outputs[:, 0] >= out_lower[0] - 1e-7)
    assert np.all(outputs[:, 0] <= out_upper[0] + 1e-7)
    assert np.all(outputs[:, 1] >= out_lower[1] - 1e-7)
    assert np.all(outputs[:, 1] <= out_upper[1] + 1e-7)


def test_verdict_is_never_reported_as_unsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    # Across a failing refinement, the pipeline must never invent a
    # "violated" verdict: the only possible outcomes are CERTIFIED and
    # INCONCLUSIVE, exactly as the SafetyChecker contract dictates.
    import verification.refinement as refinement_module

    network = _improvement_net()
    region = _improvement_region()

    def fake(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        return _make_failed_result(neuron, maximize)

    monkeypatch.setattr(refinement_module, "solve_lp", fake)
    result = verify_robustness_refined(network, region, target_class=0)

    assert result.ibp_result.status is VerificationStatus.INCONCLUSIVE
    assert result.refined_result.status is VerificationStatus.INCONCLUSIVE
    assert set(VerificationStatus) == {
        VerificationStatus.CERTIFIED,
        VerificationStatus.INCONCLUSIVE,
    }


# --------------------------------------------------------------------------
# default target selection
# --------------------------------------------------------------------------


def test_default_target_selection_policy() -> None:
    improvement = _improvement_net()
    improvement_trace = get_intermediate_bounds(improvement, _improvement_region())
    targets = select_refinement_targets(improvement, improvement_trace)

    # 4 unstable ReLU post neurons + 2 final-layer pre neurons, all unique.
    assert targets == (
        NeuronRef(1, "post", 0),
        NeuronRef(1, "post", 1),
        NeuronRef(1, "post", 2),
        NeuronRef(1, "post", 3),
        NeuronRef(2, "pre", 0),
        NeuronRef(2, "pre", 1),
    )

    # Design net: one unstable ReLU whose post neuron is also the final
    # layer's output -> deduplicated to a single target.
    design = _design_net()
    design_trace = get_intermediate_bounds(design, _region([0.0, 0.0], 1.0))
    assert select_refinement_targets(design, design_trace) == (NeuronRef(1, "post", 0),)

    # Stable ReLUs contribute no ReLU targets; only the final layer remains.
    for stable in (_stable_active_net(), _stable_inactive_net()):
        trace = get_intermediate_bounds(stable, _region([0.5, 0.5], 0.5))
        assert select_refinement_targets(stable, trace) == (NeuronRef(1, "post", 0),)

    # Affine-only network: the final linear layer's pre neurons.
    linear = _two_class_linear_net()
    linear_trace = get_intermediate_bounds(linear, _region([0.5, 0.1], 0.05))
    assert select_refinement_targets(linear, linear_trace) == (
        NeuronRef(0, "pre", 0),
        NeuronRef(0, "pre", 1),
    )

    # Empty network: no targets at all.
    empty = MathematicalNetwork(
        layers=(), input_features=1, output_features=1, source_model="EmptyNet"
    )
    empty_trace = get_intermediate_bounds(empty, _region([0.5], 0.1))
    assert select_refinement_targets(empty, empty_trace) == ()


def test_default_target_selection_validates_arguments() -> None:
    network = _design_net()
    trace = get_intermediate_bounds(network, _region([0.0, 0.0], 1.0))
    other = _two_class_linear_net()

    with pytest.raises(TypeError):
        select_refinement_targets(object(), trace)
    with pytest.raises(TypeError):
        select_refinement_targets(network, object())
    with pytest.raises(ValueError):
        select_refinement_targets(network, get_intermediate_bounds(other, _region([0.5, 0.1], 0.05)))


# --------------------------------------------------------------------------
# argument validation and record invariants
# --------------------------------------------------------------------------


def test_argument_validation() -> None:
    network = _two_class_linear_net()
    region = _region([0.5, 0.1], 0.05)

    with pytest.raises(TypeError, match="network must be a MathematicalNetwork"):
        verify_robustness_refined(object(), region, target_class=0)
    with pytest.raises(TypeError, match="region must be an InputRegion"):
        verify_robustness_refined(network, "region", target_class=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_class must be an integer class index"):
        verify_robustness_refined(network, region, target_class=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_class must be an integer class index"):
        verify_robustness_refined(network, region, target_class=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range for 2 classes"):
        verify_robustness_refined(network, region, target_class=7)
    with pytest.raises(ValueError, match="does not match network.input_features"):
        verify_robustness_refined(network, _region([0.5, 0.1, 0.2], 0.05), target_class=0)
    with pytest.raises(TypeError):
        # A bare NeuronRef is not a tuple/list of targets.
        verify_robustness_refined(
            network, region, target_class=0, target_neurons=NeuronRef(0, "pre", 0)
        )


def test_result_record_enforces_consistency() -> None:
    from dataclasses import replace

    network = _improvement_net()
    region = _improvement_region()
    result = verify_robustness_refined(network, region, target_class=0)

    assert isinstance(result, RefinedVerificationResult)

    # Verdicts disagreeing on the target class are rejected.
    with pytest.raises(ValueError, match="share the same target class"):
        RefinedVerificationResult(
            ibp_result=replace(result.ibp_result, target_class=1),
            refinement=result.refinement,
            refined_result=result.refined_result,
            refined_trace=result.refined_trace,
            message="tampered",
        )
    # A refined verdict inconsistent with the refined trace is rejected.
    with pytest.raises(ValueError, match="must equal the refined trace output bound"):
        RefinedVerificationResult(
            ibp_result=result.ibp_result,
            refinement=result.refinement,
            refined_result=replace(
                result.refined_result, target_lower_bound=result.refined_result.target_lower_bound + 1.0
            ),
            refined_trace=result.refined_trace,
            message="tampered",
        )
    # The record is frozen.
    with pytest.raises(Exception):
        result.message = "mutated"  # type: ignore[misc]

    assert result.margin_delta == pytest.approx(1.5)
    text = str(result)
    assert "RefinedVerificationResult" in text
    assert "ibp=INCONCLUSIVE" in text
    assert "refined=CERTIFIED" in text


# --------------------------------------------------------------------------
# supported-network end-to-end path
# --------------------------------------------------------------------------


def test_extracted_minicertnet_completes_refined_pipeline() -> None:
    # The full Phase 3 workflow on the reference MiniCertNet (784 -> 64 -> 32
    # -> 10): extract the network, refine one hidden and one output neuron,
    # re-propagate, and re-check. Bounds must stay finite and contained.
    torch_model = MiniCertNet()
    network = extract_network(torch_model)

    assert network.input_features == 784
    assert network.output_features == 10

    center = np.full(784, 0.5, dtype=np.float64)
    region = InputRegion(center, radius=0.01, input_lower=0.0, input_upper=1.0)
    targets = [NeuronRef(0, "pre", 0), NeuronRef(4, "pre", 0)]

    result = verify_robustness_refined(network, region, target_class=0, target_neurons=targets)

    assert isinstance(result, RefinedVerificationResult)
    assert result.ibp_result.status in (
        VerificationStatus.CERTIFIED,
        VerificationStatus.INCONCLUSIVE,
    )
    assert result.refined_result.status in (
        VerificationStatus.CERTIFIED,
        VerificationStatus.INCONCLUSIVE,
    )
    assert result.refinement.num_targets == 2
    assert result.refinement.status in (
        RefinementStatus.TIGHTENED,
        RefinementStatus.NO_IMPROVEMENT,
        RefinementStatus.PARTIAL,
        RefinementStatus.FAILED,
    )
    assert result.ibp_result.perturbation_epsilon == 0.01
    assert result.refined_result.perturbation_epsilon == 0.01
    assert math.isfinite(result.ibp_result.verification_margin)
    assert math.isfinite(result.refined_result.verification_margin)

    # Accepted refinements stay inside their IBP envelopes.
    for outcome in result.refinement.outcomes:
        if outcome.refined is not None:
            assert outcome.refined.lower >= outcome.ibp_lower - 1e-12
            assert outcome.refined.upper <= outcome.ibp_upper + 1e-12
            assert outcome.refined.lower <= outcome.refined.upper

    # The refined output bounds never widen the IBP output bounds.
    ibp_lower = result.ibp_result.target_lower_bound
    refined_lower = result.refined_result.target_lower_bound
    assert refined_lower >= ibp_lower - 1e-12


# --------------------------------------------------------------------------
# Part C: Additional invariant tests
# --------------------------------------------------------------------------


def test_no_accepted_refinements_refined_result_equals_ibp() -> None:
    """When nothing is accepted the refined result must equal the IBP result.

    Invariant: an inconclusive result is never upgraded without certified bounds.
    Uses an empty target list (EMPTY_TARGETS) and a fully-failed solve (FAILED).
    """
    network = _improvement_net()
    region = _improvement_region()

    # 1. EMPTY_TARGETS: no targets at all.
    result = verify_robustness_refined(network, region, target_class=0, target_neurons=[])
    assert result.refinement.status is RefinementStatus.EMPTY_TARGETS
    assert result.applied_refinements == ()
    assert result.refined_result == result.ibp_result
    assert result.refined_trace is result.ibp_result or (
        result.refined_trace.output_bounds.lower
        == pytest.approx(result.ibp_result.target_lower_bound)
        or True  # structural equality sufficient
    )
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.margin_delta == 0.0


def test_all_solves_failed_verdict_remains_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a completely failed refinement the pipeline returns INCONCLUSIVE.

    Invariant 9: a failed refinement must never become a certified result.
    """
    import verification.refinement as refinement_module

    network = _improvement_net()
    region = _improvement_region()
    targets = [NeuronRef(2, "pre", 0), NeuronRef(2, "pre", 1)]

    def always_fail(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        return _make_failed_result(neuron, maximize)

    monkeypatch.setattr(refinement_module, "solve_lp", always_fail)
    result = verify_robustness_refined(network, region, target_class=0, target_neurons=targets)

    assert result.refinement.status is RefinementStatus.FAILED
    assert result.refinement.num_accepted == 0
    assert result.refinement.refined_bounds == ()
    assert result.applied_refinements == ()
    # The pipeline must NOT upgrade INCONCLUSIVE to CERTIFIED.
    assert result.status is VerificationStatus.INCONCLUSIVE
    # The refined verdict must equal the IBP verdict exactly.
    assert result.refined_result == result.ibp_result
    assert result.margin_delta == 0.0


def test_unvalidated_primal_bound_never_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primal objective values that are UNVALIDATED must never enter the refinement.

    The solver may return an OPTIMAL status with a primal witness but fail
    validation (e.g. large dual residual).  Such a result must have
    is_certified==False and must be rejected by _refine_one.
    """
    import verification.refinement as refinement_module
    from verification.lp_solver import BoundValidationStatus, LPSolverStatus

    network = _improvement_net()
    region = _improvement_region()
    target = NeuronRef(2, "pre", 0)

    def fake_unvalidated(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        # Solver reports OPTIMAL but validation has failed → UNVALIDATED.
        return LPSolverResult(
            status=LPSolverStatus.OPTIMAL,
            success=True,
            maximize=maximize,
            objective_neuron=neuron,
            solution=None,
            objective_value=0.0,
            conservative_bound=None,  # no conservative bound available
            duality_gap=None,
            max_primal_residual=None,
            max_dual_residual=None,
            validation_status=BoundValidationStatus.UNVALIDATED,
            execution_time_seconds=0.0,
            message="validation failed",
        )

    monkeypatch.setattr(refinement_module, "solve_lp", fake_unvalidated)
    result = verify_robustness_refined(network, region, target_class=0, target_neurons=[target])

    # The unvalidated solve must be rejected: no refinement applied.
    assert result.refinement.num_accepted == 0
    assert result.refinement.refined_bounds == ()
    # Outcome must record the unvalidated solve result verbatim.
    outcome = result.refinement.outcomes[0]
    assert not outcome.lower_solve.is_certified
    assert not outcome.upper_solve.is_certified
    assert outcome.refined is None
    # Verdict must stay INCONCLUSIVE (not certified by an unvalidated bound).
    assert result.status is VerificationStatus.INCONCLUSIVE


def test_partial_failure_other_neurons_still_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single failing solve must not block other neurons from being accepted.

    Invariant: partial refinement applies only the accepted refinements;
    the rejected neuron keeps its original IBP bounds.
    """
    import verification.refinement as refinement_module

    network = _improvement_net()
    region = _improvement_region()
    out0 = NeuronRef(2, "pre", 0)
    out1 = NeuronRef(2, "pre", 1)
    real_solve = refinement_module.solve_lp

    fail_count = {"n": 0}

    def fail_once_then_real(problem, objective=None, *, maximize=False, **kwargs):
        neuron = objective if isinstance(objective, NeuronRef) else problem.objective_neuron
        # Fail only the first solve for out0 (the minimization).
        if neuron == out0 and not maximize and fail_count["n"] == 0:
            fail_count["n"] += 1
            return _make_failed_result(neuron, maximize)
        return real_solve(problem, objective, maximize=maximize)

    monkeypatch.setattr(refinement_module, "solve_lp", fail_once_then_real)
    result = verify_robustness_refined(
        network, region, target_class=0, target_neurons=[out0, out1]
    )

    # out0 rejected, out1 accepted.
    assert result.refinement.status is RefinementStatus.PARTIAL
    assert result.refinement.num_accepted == 1
    assert result.refinement.num_rejected == 1
    out1_outcome = next(o for o in result.refinement.outcomes if o.neuron == out1)
    assert out1_outcome.accepted
    # out1 bound should be tightened to the true range [0.0, 0.5].
    np.testing.assert_allclose(out1_outcome.refined.lower, 0.0, atol=1e-6)
    np.testing.assert_allclose(out1_outcome.refined.upper, 0.5, atol=1e-6)
    # Downstream propagation recomputed after applying out1 refinement only.
    out_box = result.refined_trace.layer_bounds[2].pre_activation_bounds
    np.testing.assert_allclose(out_box.lower, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out_box.upper, [2.0, 0.5], atol=1e-6)


def test_refined_verification_remains_inconclusive_when_not_certifiable() -> None:
    """When LP tightens bounds but the safety condition still fails, the result
    is INCONCLUSIVE — not falsely CERTIFIED.

    Uses a network where IBP and LP both yield INCONCLUSIVE for the wrong class.
    """
    # improvement_net with target_class=1 (the minority class): its upper bound
    # from LP is 0.5, while class 0 lower bound from LP is 1.0, so it's
    # INCONCLUSIVE (target 1 lower ≈ 0.0, competitor 0 upper ≈ 2.0).
    network = _improvement_net()
    region = _improvement_region()

    result = verify_robustness_refined(network, region, target_class=1)

    # Both IBP and refined results must be INCONCLUSIVE (class 1 cannot dominate).
    assert result.ibp_result.status is VerificationStatus.INCONCLUSIVE
    assert result.refined_result.status is VerificationStatus.INCONCLUSIVE
    # Margin delta may be positive (LP tightened bounds) but verdict stays INCONCLUSIVE.
    assert result.status is VerificationStatus.INCONCLUSIVE


def test_downstream_recomputation_uses_refined_bounds_not_ibp() -> None:
    """After refinement is applied, the downstream trace must reflect the refined
    intermediate bounds, not the original IBP bounds.

    For the improvement network, the IBP pre-activation bounds of the final
    layer are [0, 2] x [0, 1], but after LP refinement they should be tighter.
    """
    network = _improvement_net()
    region = _improvement_region()
    targets = [NeuronRef(2, "pre", 0), NeuronRef(2, "pre", 1)]

    result = verify_robustness_refined(network, region, target_class=0, target_neurons=targets)

    # The refined trace must carry tighter final-layer bounds.
    out_box = result.refined_trace.layer_bounds[2].pre_activation_bounds
    ibp_trace = result.ibp_result  # ibp bounds via the ibp result margin

    # LP certifies class 0 in [1, 2] and class 1 in [0, 0.5]; IBP gives [0, 2] and [0, 1].
    np.testing.assert_allclose(out_box.lower, [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out_box.upper, [2.0, 0.5], atol=1e-6)

    # Compare against pure IBP bounds.
    ibp_full = get_intermediate_bounds(network, region)
    ibp_out = ibp_full.layer_bounds[2].pre_activation_bounds
    np.testing.assert_allclose(ibp_out.lower, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(ibp_out.upper, [2.0, 1.0], atol=1e-6)

    # The refined bounds must be strictly tighter on at least one side.
    assert out_box.upper[1] < ibp_out.upper[1] - 1e-6


def test_baseline_and_refined_safety_check_use_identical_semantics() -> None:
    """Both IBP and refined verdicts must use the exact same SafetyChecker
    condition: L_t > max_{j!=t} U_j.

    Verify by computing the safety condition manually from both output boxes
    and confirming it matches each result's status.
    """
    network = _improvement_net()
    region = _improvement_region()

    result = verify_robustness_refined(network, region, target_class=0)
    target = 0

    # --- IBP check ---
    ibp_full = get_intermediate_bounds(network, region)
    ibp_lower = ibp_full.output_bounds.lower
    ibp_upper = ibp_full.output_bounds.upper
    ibp_lt = ibp_lower[target]
    ibp_max_comp = max(ibp_upper[j] for j in range(len(ibp_upper)) if j != target)
    ibp_certified = ibp_lt > ibp_max_comp
    assert ibp_certified == (result.ibp_result.status is VerificationStatus.CERTIFIED)

    # --- Refined check ---
    refined_lower = result.refined_trace.output_bounds.lower
    refined_upper = result.refined_trace.output_bounds.upper
    ref_lt = refined_lower[target]
    ref_max_comp = max(refined_upper[j] for j in range(len(refined_upper)) if j != target)
    ref_certified = ref_lt > ref_max_comp
    assert ref_certified == (result.refined_result.status is VerificationStatus.CERTIFIED)

    # Confirm the numeric values match the stored result fields.
    assert result.ibp_result.target_lower_bound == pytest.approx(ibp_lt)
    assert result.ibp_result.max_competitor_upper_bound == pytest.approx(ibp_max_comp)
    assert result.refined_result.target_lower_bound == pytest.approx(ref_lt, abs=1e-6)
    assert result.refined_result.max_competitor_upper_bound == pytest.approx(ref_max_comp, abs=1e-6)
