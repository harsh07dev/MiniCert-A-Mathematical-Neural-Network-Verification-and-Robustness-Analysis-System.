"""Comprehensive tests for the end-to-end verification orchestration pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from abstract_domain.propagation import propagate_network
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer, extract_network
from models.network import MiniCertNet
from verification import VerificationResult, VerificationStatus
from verification.pipeline import verify_robustness
from verification.regions import InputRegion


# ==============================================================================
# Helper Fixtures & Builders
# ==============================================================================


def _build_two_class_network() -> MathematicalNetwork:
    """Build a deterministic 2 -> 2 network separating [0.5, 0.1]."""
    # Class 0: x0 - x1; Class 1: -x0 + x1
    w = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    linear = LinearLayer(
        position=0,
        weight=w,
        bias=b,
        input_features=2,
        output_features=2,
    )
    return MathematicalNetwork(
        layers=(linear,),
        input_features=2,
        output_features=2,
        source_model="TwoClassLinear",
    )


def _build_deep_network() -> MathematicalNetwork:
    """Build a deterministic 2 -> 4 -> 3 network with ReLU activation."""
    w1 = np.array([
        [1.0, 0.5],
        [-0.5, 1.0],
        [0.0, -1.0],
        [2.0, -1.5],
    ], dtype=np.float64)
    b1 = np.array([0.1, -0.2, 0.5, 0.0], dtype=np.float64)

    w2 = np.array([
        [1.0, -1.0, 0.5, 2.0],
        [-0.5, 0.0, 1.0, -1.0],
        [0.0, 0.5, -0.5, 0.0],
    ], dtype=np.float64)
    b2 = np.array([1.0, 0.0, -0.5], dtype=np.float64)

    l1 = LinearLayer(position=0, weight=w1, bias=b1, input_features=2, output_features=4)
    r1 = ReLULayer(position=1)
    l2 = LinearLayer(position=2, weight=w2, bias=b2, input_features=4, output_features=3)

    return MathematicalNetwork(
        layers=(l1, r1, l2),
        input_features=2,
        output_features=3,
        source_model="DeepNet",
    )


# ==============================================================================
# 1. Basic Verification Outcomes (Certified vs Inconclusive)
# ==============================================================================


def test_simple_network_returns_certified() -> None:
    network = _build_two_class_network()
    # At center [0.5, 0.1] with radius 0.05:
    # Class 0: [0.30, 0.50]; Class 1: [-0.50, -0.30]
    region = InputRegion(np.array([0.5, 0.1]), 0.05)

    result = verify_robustness(network, region, target_class=0)

    assert isinstance(result, VerificationResult)
    assert result.status is VerificationStatus.CERTIFIED
    assert result.target_class == 0
    assert result.target_lower_bound == pytest.approx(0.3)
    assert result.max_competitor_upper_bound == pytest.approx(-0.3)
    assert result.verification_margin == pytest.approx(0.6)
    assert result.perturbation_epsilon == 0.05
    assert "strictly exceeds" in result.explanation


def test_wide_region_returns_inconclusive() -> None:
    network = _build_two_class_network()
    # At center [0.5, 0.1] with radius 0.3:
    # Class 0: [-0.20, 0.80]; Class 1: [-0.80, 0.20]
    # Margin = -0.20 - 0.20 = -0.40 <= 0 -> Inconclusive
    region = InputRegion(np.array([0.5, 0.1]), 0.3)

    result = verify_robustness(network, region, target_class=0)

    assert isinstance(result, VerificationResult)
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.target_class == 0
    assert result.target_lower_bound == pytest.approx(-0.2)
    assert result.max_competitor_upper_bound == pytest.approx(0.2)
    assert result.verification_margin == pytest.approx(-0.4)
    assert result.perturbation_epsilon == 0.3
    assert "does not strictly exceed" in result.explanation


# ==============================================================================
# 2. Target Class Handling and Rejection
# ==============================================================================


def test_correct_target_class_selection() -> None:
    network = _build_two_class_network()
    # Region where class 1 strongly dominates: center [-0.5, 0.5]
    region = InputRegion(np.array([-0.5, 0.5]), 0.05)

    result_c1 = verify_robustness(network, region, target_class=1)
    assert result_c1.status is VerificationStatus.CERTIFIED
    assert result_c1.target_class == 1
    assert result_c1.verification_margin > 0.0

    result_c0 = verify_robustness(network, region, target_class=0)
    assert result_c0.status is VerificationStatus.INCONCLUSIVE
    assert result_c0.target_class == 0
    assert result_c0.verification_margin < 0.0


@pytest.mark.parametrize(
    "invalid_target",
    [-1, 2, 5, -100],
)
def test_out_of_range_target_class_rejected(invalid_target: int) -> None:
    network = _build_two_class_network()  # output_features = 2
    region = InputRegion(np.array([0.5, 0.1]), 0.05)

    with pytest.raises(ValueError, match="out of range for 2 classes"):
        verify_robustness(network, region, target_class=invalid_target)


@pytest.mark.parametrize(
    "invalid_type_target",
    [True, False, 1.0, "0", None, [0]],
)
def test_non_integer_target_class_rejected(invalid_type_target: object) -> None:
    network = _build_two_class_network()
    region = InputRegion(np.array([0.5, 0.1]), 0.05)

    with pytest.raises(TypeError, match="target_class must be an integer class index"):
        verify_robustness(network, region, target_class=invalid_type_target)  # type: ignore[arg-type]


# ==============================================================================
# 3. Validation: Dimensions and Types
# ==============================================================================


def test_input_dimension_mismatch_rejected() -> None:
    network = _build_two_class_network()  # input_features = 2
    region_dim3 = InputRegion(np.array([0.1, 0.2, 0.3]), 0.05)  # dimension = 3

    with pytest.raises(ValueError, match="dimension 3 does not match network.input_features 2"):
        verify_robustness(network, region_dim3, target_class=0)


def test_non_mathematical_network_rejected() -> None:
    region = InputRegion(np.array([0.5, 0.1]), 0.05)

    with pytest.raises(TypeError, match="network must be a MathematicalNetwork"):
        verify_robustness("not a network", region, target_class=0)  # type: ignore[arg-type]


def test_non_input_region_rejected() -> None:
    network = _build_two_class_network()

    with pytest.raises(TypeError, match="region must be an InputRegion"):
        verify_robustness(network, "not an input region", target_class=0)  # type: ignore[arg-type]


# ==============================================================================
# 4. Invariants, Epsilon Attachment, and Immutability
# ==============================================================================


def test_perturbation_epsilon_attached_and_serializable() -> None:
    network = _build_two_class_network()
    region = InputRegion(np.array([0.5, 0.1]), 0.08)

    result = verify_robustness(network, region, target_class=0)

    assert result.perturbation_epsilon == 0.08
    assert "epsilon=0.08" in str(result)

    payload = result.to_dict()
    assert payload["perturbation_epsilon"] == 0.08
    restored = VerificationResult.from_dict(payload)
    assert restored == result


def test_input_region_remains_unchanged() -> None:
    network = _build_two_class_network()
    center_copy = np.array([0.5, 0.1], dtype=np.float64)
    region = InputRegion(center_copy, 0.05, 0.0, 1.0)

    initial_lower = region.lower.copy()
    initial_upper = region.upper.copy()
    initial_radius = region.radius

    _ = verify_robustness(network, region, target_class=0)

    np.testing.assert_array_equal(region.center, center_copy)
    np.testing.assert_array_equal(region.lower, initial_lower)
    np.testing.assert_array_equal(region.upper, initial_upper)
    assert region.radius == initial_radius


def test_output_bounds_agree_with_direct_propagation() -> None:
    network = _build_deep_network()
    region = InputRegion(np.array([0.5, -0.2]), 0.1)

    result = verify_robustness(network, region, target_class=1)
    direct_box = propagate_network(network, region)

    assert result.target_lower_bound == direct_box.lower[1]
    competitor_uppers = [direct_box.upper[i] for i in range(network.output_features) if i != 1]
    assert result.max_competitor_upper_bound == pytest.approx(max(competitor_uppers))
    assert result.verification_margin == pytest.approx(direct_box.lower[1] - max(competitor_uppers))


# ==============================================================================
# 5. Extracted MiniCertNet Full Pipeline
# ==============================================================================


def test_extracted_minicertnet_completes_pipeline() -> None:
    torch_model = MiniCertNet()
    network = extract_network(torch_model)

    assert network.input_features == 784
    assert network.output_features == 10

    # Clean centered region in [0, 1] pixel space
    center = np.full(784, 0.5, dtype=np.float64)
    region = InputRegion(center, radius=0.01, input_lower=0.0, input_upper=1.0)

    result = verify_robustness(network, region, target_class=0)

    assert isinstance(result, VerificationResult)
    assert result.target_class == 0
    assert result.status in (VerificationStatus.CERTIFIED, VerificationStatus.INCONCLUSIVE)
    assert result.perturbation_epsilon == 0.01
    assert math.isfinite(result.target_lower_bound)
    assert math.isfinite(result.max_competitor_upper_bound)
    assert math.isfinite(result.verification_margin)


# ==============================================================================
# 6. Empirical Sampling Sanity Check
# ==============================================================================


def test_deterministic_empirical_sampling_sanity_check() -> None:
    rng = np.random.default_rng(2026)
    network = _build_two_class_network()
    region = InputRegion(np.array([0.5, 0.1]), 0.05)

    result = verify_robustness(network, region, target_class=0)
    assert result.status is VerificationStatus.CERTIFIED

    # Uniformly draw 250 empirical sample points inside the InputRegion
    samples = rng.uniform(region.lower, region.upper, size=(250, 2))

    # Evaluate concrete predictions through network forward pass
    outputs = network.forward(samples)
    predictions = np.argmax(outputs, axis=1)

    # When CERTIFIED, 100% of sampled points MUST predict the target class
    assert np.all(predictions == 0)
    assert np.all(outputs[:, 0] > outputs[:, 1])


def test_empirical_sampling_inconclusive_does_not_falsely_certify() -> None:
    # For a wide region where bounds overlap, verify status is INCONCLUSIVE
    network = _build_two_class_network()
    region = InputRegion(np.array([0.5, 0.1]), 0.3)

    result = verify_robustness(network, region, target_class=0)
    assert result.status is VerificationStatus.INCONCLUSIVE

    # Sample points to confirm an actual violation exists inside this wide box
    # Witness: [lower[0], upper[1]] = [0.2, 0.4] -> outputs: [-0.2, 0.2] (class 1 wins)
    witness = np.array([region.lower[0], region.upper[1]])
    assert region.contains(witness)
    witness_out = network.forward(witness)
    assert np.argmax(witness_out) == 1  # Misclassified into class 1


def test_vacuous_single_class_certification() -> None:
    # Network with 1 output feature
    linear = LinearLayer(
        position=0,
        weight=np.array([[2.0, 1.0]]),
        bias=np.array([0.5]),
        input_features=2,
        output_features=1,
    )
    single_class_net = MathematicalNetwork(
        layers=(linear,),
        input_features=2,
        output_features=1,
        source_model="SingleClassNet",
    )

    region = InputRegion(np.array([1.0, 2.0]), 0.1)
    result = verify_robustness(single_class_net, region, target_class=0)

    assert result.status is VerificationStatus.CERTIFIED
    assert result.target_class == 0
    assert result.max_competitor_upper_bound == -math.inf
    assert result.verification_margin == math.inf
    assert "only output class" in result.explanation

