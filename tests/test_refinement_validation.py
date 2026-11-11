"""Tests for empirical soundness validation of refined bounds."""

from __future__ import annotations

import json

import numpy as np
import pytest

from abstract_domain import (
    NeuronRef,
    RefinedBounds,
    get_intermediate_bounds,
    propagate_with_refinements,
)
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer, extract_network
from models.network import MiniCertNet
from verification import (
    evaluate_intermediate_activations,
    sample_region_inputs,
    validate_refined_bounds_empirically,
)
from verification.regions import InputRegion


# ==============================================================================
# Helper Factories
# ==============================================================================


def _make_linear(
    position: int,
    weight: np.ndarray,
    bias: np.ndarray,
) -> LinearLayer:
    """Helper to build a LinearLayer with float64 arrays."""
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    return LinearLayer(
        position=position,
        weight=w,
        bias=b,
        input_features=int(w.shape[1]),
        output_features=int(w.shape[0]),
    )


def _simple_linear_relu_net() -> MathematicalNetwork:
    """Deterministic 2 -> 2 -> 2 network for activation testing."""
    # Linear 0: y0 = 2*x0, y1 = -x1 + 1
    w0 = np.array([[2.0, 0.0], [0.0, -1.0]])
    b0 = np.array([0.0, 1.0])
    l0 = _make_linear(0, w0, b0)

    # ReLU 1
    r1 = ReLULayer(position=1)

    # Linear 2: z0 = y0 + y1, z1 = y0 - y1
    w2 = np.array([[1.0, 1.0], [1.0, -1.0]])
    b2 = np.array([0.5, -0.5])
    l2 = _make_linear(2, w2, b2)

    return MathematicalNetwork(
        layers=(l0, r1, l2),
        input_features=2,
        output_features=2,
        source_model="SimpleLinearReLUNet",
    )


# ==============================================================================
# 1. Input Sampling Tests
# ==============================================================================


def test_reproducible_input_sampling_same_seed() -> None:
    region = InputRegion(np.array([0.5, 0.5]), 0.2, 0.0, 1.0)
    samples1 = sample_region_inputs(region, num_samples=100, seed=123)
    samples2 = sample_region_inputs(region, num_samples=100, seed=123)

    np.testing.assert_array_equal(samples1, samples2)


def test_different_seeds_produce_different_samples() -> None:
    region = InputRegion(np.array([0.5, 0.5]), 0.2, 0.0, 1.0)
    samples1 = sample_region_inputs(region, num_samples=100, seed=123)
    samples2 = sample_region_inputs(region, num_samples=100, seed=456)

    assert not np.array_equal(samples1, samples2)


def test_samples_stay_inside_input_region() -> None:
    region = InputRegion(np.array([0.4, 0.6]), 0.15, 0.0, 1.0)
    samples = sample_region_inputs(region, num_samples=500, seed=42)

    assert np.all(samples >= region.lower)
    assert np.all(samples <= region.upper)
    for sample in samples:
        assert region.contains(sample)


def test_correct_sample_shape() -> None:
    region = InputRegion(np.zeros(5), 0.1)
    samples = sample_region_inputs(region, num_samples=37, seed=7)

    assert samples.shape == (37, 5)
    assert samples.dtype == np.float64


@pytest.mark.parametrize("invalid_count", [0, -1, -50])
def test_invalid_sample_count_rejected(invalid_count: int) -> None:
    region = InputRegion(np.zeros(2), 0.1)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        sample_region_inputs(region, num_samples=invalid_count)


@pytest.mark.parametrize("invalid_type", [1.5, "100", None, True])
def test_non_integer_sample_count_rejected(invalid_type: object) -> None:
    region = InputRegion(np.zeros(2), 0.1)
    with pytest.raises(TypeError, match="num_samples must be an integer"):
        sample_region_inputs(region, num_samples=invalid_type)  # type: ignore[arg-type]


def test_invalid_region_type_rejected() -> None:
    with pytest.raises(TypeError, match="input_region must be an InputRegion"):
        sample_region_inputs("not a region", num_samples=10)  # type: ignore[arg-type]


# ==============================================================================
# 2. Intermediate Activation Evaluation Tests
# ==============================================================================


def test_intermediate_activations_extraction_linear_and_relu() -> None:
    network = _simple_linear_relu_net()
    # Test on known inputs: x0 = [0.5, 0.2], x1 = [1.0, 3.0]
    inputs = np.array([
        [0.5, 0.2],
        [1.0, 3.0],
    ], dtype=np.float64)

    trace = evaluate_intermediate_activations(network, inputs)

    assert isinstance(trace, tuple)
    assert len(trace) == 3

    # Layer 0 (Linear): y0 = 2*x0, y1 = -x1 + 1
    # Sample 0: [2*0.5, -0.2 + 1] = [1.0, 0.8]
    # Sample 1: [2*1.0, -3.0 + 1] = [2.0, -2.0]
    n0_0 = NeuronRef(0, "pre", 0)
    n0_1 = NeuronRef(0, "pre", 1)
    np.testing.assert_allclose(trace[0][n0_0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(trace[0][n0_1], np.array([0.8, -2.0]))
    # Direct NeuronRef indexing on trace
    np.testing.assert_allclose(trace[n0_0], np.array([1.0, 2.0]))

    # Layer 1 (ReLU): pre matches linear output, post applies ReLU
    # Pre:
    n1_pre0 = NeuronRef(1, "pre", 0)
    n1_pre1 = NeuronRef(1, "pre", 1)
    np.testing.assert_allclose(trace[1][n1_pre0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(trace[1][n1_pre1], np.array([0.8, -2.0]))

    # Post: [1.0, 0.8] -> [1.0, 0.8]; [2.0, -2.0] -> [2.0, 0.0]
    n1_post0 = NeuronRef(1, "post", 0)
    n1_post1 = NeuronRef(1, "post", 1)
    np.testing.assert_allclose(trace[1][n1_post0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(trace[1][n1_post1], np.array([0.8, 0.0]))

    # Layer 2 (Linear): z0 = y0 + y1 + 0.5, z1 = y0 - y1 - 0.5
    # Sample 0: [1.0 + 0.8 + 0.5, 1.0 - 0.8 - 0.5] = [2.3, -0.3]
    # Sample 1: [2.0 + 0.0 + 0.5, 2.0 - 0.0 - 0.5] = [2.5, 1.5]
    n2_0 = NeuronRef(2, "pre", 0)
    n2_1 = NeuronRef(2, "pre", 1)
    np.testing.assert_allclose(trace[2][n2_0], np.array([2.3, 2.5]))
    np.testing.assert_allclose(trace[2][n2_1], np.array([-0.3, 1.5]))


def test_single_vector_input_supported() -> None:
    network = _simple_linear_relu_net()
    single_input = np.array([0.5, 0.2])

    trace = evaluate_intermediate_activations(network, single_input)
    assert trace[NeuronRef(0, "pre", 0)].shape == (1,)
    assert trace[NeuronRef(0, "pre", 0)][0] == pytest.approx(1.0)


def test_invalid_input_dimension_rejected() -> None:
    network = _simple_linear_relu_net()  # input_features = 2
    wrong_inputs = np.zeros((10, 5))

    with pytest.raises(ValueError, match="does not match network.input_features"):
        evaluate_intermediate_activations(network, wrong_inputs)


def test_invalid_input_shapes_and_nan_rejected() -> None:
    network = _simple_linear_relu_net()

    with pytest.raises(ValueError, match="inputs must be 1-D or 2-D"):
        evaluate_intermediate_activations(network, np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="inputs must not be empty"):
        evaluate_intermediate_activations(network, np.empty((0, 2)))

    with pytest.raises(ValueError, match="inputs cannot contain NaN"):
        evaluate_intermediate_activations(network, np.array([[0.5, np.nan]]))


# ==============================================================================
# 3. Empirical Refined Bounds Validation
# ==============================================================================


def test_valid_refinements_pass_empirical_checks() -> None:
    network = _simple_linear_relu_net()
    region = InputRegion(np.array([0.5, 0.5]), 0.1, 0.0, 1.0)
    # Range of x0 in [0.4, 0.6] -> 2*x0 in [0.8, 1.2]
    # Refinement [0.75, 1.25] soundly encloses [0.8, 1.2]
    refined = RefinedBounds(NeuronRef(0, "pre", 0), 0.75, 1.25)

    report = validate_refined_bounds_empirically(
        network, region, (refined,), num_samples=300, seed=42
    )

    assert report["num_samples"] == 300
    assert report["seed"] == 42
    assert report["all_refinements_passed"] is True
    assert len(report["results"]) == 1

    r0 = report["results"][0]
    assert r0["neuron"]["layer_index"] == 0
    assert r0["neuron"]["phase"] == "pre"
    assert r0["neuron"]["neuron_index"] == 0
    assert r0["violations"] == 0
    assert r0["max_lower_violation"] == 0.0
    assert r0["max_upper_violation"] == 0.0
    assert r0["passed"] is True

    # JSON serializability test
    serialized = json.dumps(report)
    assert "all_refinements_passed" in serialized


def test_violations_correctly_counted_and_measured() -> None:
    network = _simple_linear_relu_net()
    region = InputRegion(np.array([0.5, 0.5]), 0.1, 0.0, 1.0)
    # x0 in [0.4, 0.6] -> 2*x0 in [0.8, 1.2].
    # Too tight interval: [0.95, 1.05] excludes reachable samples!
    too_tight = RefinedBounds(NeuronRef(0, "pre", 0), 0.95, 1.05)

    report = validate_refined_bounds_empirically(
        network, region, (too_tight,), num_samples=500, seed=42
    )

    assert report["all_refinements_passed"] is False
    r0 = report["results"][0]
    assert r0["passed"] is False
    assert r0["violations"] > 0
    # Both lower and upper violations should occur because samples reach ~0.8 and ~1.2
    assert r0["max_lower_violation"] > 0.10  # 0.95 - ~0.8 >= 0.15
    assert r0["max_upper_violation"] > 0.10  # ~1.2 - 1.05 >= 0.15


def test_empty_refinements_collection() -> None:
    network = _simple_linear_relu_net()
    region = InputRegion(np.array([0.5, 0.5]), 0.1)

    report = validate_refined_bounds_empirically(network, region, (), num_samples=100)

    assert report["num_samples"] == 100
    assert report["all_refinements_passed"] is True
    assert report["results"] == []


# ==============================================================================
# 4. Critical Scenario: Containment Alone Does NOT Prove Soundness
# ==============================================================================


def test_containment_accepted_but_empirical_violation_detected() -> None:
    """Demonstrates that containment is a consistency check, not mathematical proof of soundness.

    Network:
        Layer 0 (Linear): y0 = x0 + x1, y1 = x0 - x1
        Layer 1 (ReLU): y0, y1
        Layer 2 (Linear): z0 = y0 + y1 = (x0 + x1) + (x0 - x1) = 2*x0

    Region:
        x0 in [0, 1], x1 in [0, 1] (center [0.5, 0.5], radius 0.5)

    Behavior:
        - IBP computes z0 in [0, 3] due to independent interval over-approximation.
        - True concrete range of z0 = 2*x0 is [0, 2].
        - An unsound refinement proposes z0 in [0.0, 1.5].
        - Since [0.0, 1.5] is strictly contained inside [0.0, 3.0], propagate_with_refinements()
          accepts it based on containment.
        - However, for inputs where x0 > 0.75 (e.g. x0=0.9, x1=0.5 -> z0=1.8), z0 > 1.5!
        - validate_refined_bounds_empirically() successfully catches the violation!
    """
    w0 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    b0 = np.zeros(2, dtype=np.float64)
    l0 = _make_linear(0, w0, b0)
    r1 = ReLULayer(1)
    w2 = np.array([[1.0, 1.0]], dtype=np.float64)
    b2 = np.zeros(1, dtype=np.float64)
    l2 = _make_linear(2, w2, b2)

    network = MathematicalNetwork((l0, r1, l2), 2, 1, "CancellationNet")
    region = InputRegion(np.array([0.5, 0.5]), 0.5, 0.0, 1.0)

    # Verify original IBP bounds
    ibp_trace = get_intermediate_bounds(network, region)
    assert ibp_trace.output_bounds.lower[0] == pytest.approx(0.0)
    assert ibp_trace.output_bounds.upper[0] == pytest.approx(3.0)

    # 1. Unsound refinement: z0 in [0.0, 1.5]
    unsound_ref = RefinedBounds(NeuronRef(2, "pre", 0), 0.0, 1.5)

    # 2. propagate_with_refinements ACCEPTS it because [0.0, 1.5] is within [0.0, 3.0]
    refined_trace = propagate_with_refinements(network, region, (unsound_ref,))
    assert refined_trace.output_bounds.upper[0] == pytest.approx(1.5)

    # 3. validate_refined_bounds_empirically DETECTS that it is unsound!
    report = validate_refined_bounds_empirically(
        network, region, (unsound_ref,), num_samples=500, seed=42
    )

    assert report["all_refinements_passed"] is False
    res = report["results"][0]
    assert res["passed"] is False
    assert res["violations"] > 0
    assert res["max_upper_violation"] > 0.2  # Max sampled activation approaches ~1.95 > 1.5


# ==============================================================================
# 5. MiniCertNet Architecture Integration & Immutability
# ==============================================================================


def test_minicertnet_empirical_validation() -> None:
    torch_model = MiniCertNet()
    network = extract_network(torch_model)
    region = InputRegion(np.full(784, 0.5), 0.01, 0.0, 1.0)

    ibp_trace = get_intermediate_bounds(network, region)
    l0_low = float(ibp_trace.layer_bounds[0].output_bounds.lower[0])
    l0_upp = float(ibp_trace.layer_bounds[0].output_bounds.upper[0])

    # Valid refinement covering the full IBP bounds
    valid_ref = RefinedBounds(NeuronRef(0, "pre", 0), l0_low, l0_upp)

    report = validate_refined_bounds_empirically(
        network, region, (valid_ref,), num_samples=50, seed=123
    )

    assert report["num_samples"] == 50
    assert report["all_refinements_passed"] is True
    assert report["results"][0]["passed"] is True


def test_network_and_region_remain_unchanged() -> None:
    network = _simple_linear_relu_net()
    center = np.array([0.5, 0.5])
    region = InputRegion(center, 0.1, 0.0, 1.0)
    initial_lower = region.lower.copy()
    initial_upper = region.upper.copy()

    refined = RefinedBounds(NeuronRef(0, "pre", 0), 0.5, 1.5)
    _ = validate_refined_bounds_empirically(network, region, (refined,), num_samples=100)

    assert len(network.layers) == 3
    np.testing.assert_array_equal(region.center, center)
    np.testing.assert_array_equal(region.lower, initial_lower)
    np.testing.assert_array_equal(region.upper, initial_upper)

