"""Comprehensive unit and property tests for network-level Interval Bound Propagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from abstract_domain.box import Box
from abstract_domain.propagation import propagate_network
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer, extract_network
from models.network import MiniCertNet
from tests.test_verification_pipeline import _ibp_linear
from verification.regions import BoxProtocol, InputRegion


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


# ==============================================================================
# 1. Single and Multi-Layer Propagation
# ==============================================================================


def test_single_linear_layer_propagation() -> None:
    weight = np.array([[2.0, -1.0], [0.5, 3.0]])
    bias = np.array([0.1, -0.2])
    linear = _make_linear(0, weight, bias)
    network = MathematicalNetwork(
        layers=(linear,),
        input_features=2,
        output_features=2,
        source_model="SingleLinearNet",
    )

    box = Box([1.0, 2.0], [3.0, 4.0])
    out_box = propagate_network(network, box)

    assert isinstance(out_box, Box)
    assert out_box.dimension == 2

    oracle_lower, oracle_upper = _ibp_linear(box.lower, box.upper, weight, bias)
    np.testing.assert_allclose(out_box.lower, oracle_lower)
    np.testing.assert_allclose(out_box.upper, oracle_upper)


def test_linear_relu_propagation() -> None:
    # Layer 0: w = [[-1, 0], [1, 2]], b = [0, -5]
    # Input box: [1, 1] to [2, 2]
    # Linear bounds:
    # row 0: lower = -1*2 + 0 = -2, upper = -1*1 + 0 = -1
    # row 1: lower = 1*1 + 2*1 - 5 = -2, upper = 1*2 + 2*2 - 5 = 1
    # After ReLU:
    # row 0: max(0, -2) = 0, max(0, -1) = 0
    # row 1: max(0, -2) = 0, max(0, 1) = 1
    linear = _make_linear(0, [[-1.0, 0.0], [1.0, 2.0]], [0.0, -5.0])
    relu = ReLULayer(position=1)
    network = MathematicalNetwork(
        layers=(linear, relu),
        input_features=2,
        output_features=2,
        source_model="LinearReLUNet",
    )

    box = Box([1.0, 1.0], [2.0, 2.0])
    out_box = propagate_network(network, box)

    np.testing.assert_allclose(out_box.lower, np.array([0.0, 0.0]))
    np.testing.assert_allclose(out_box.upper, np.array([0.0, 1.0]))


def test_multiple_linear_and_relu_layers() -> None:
    # Architecture: 2 -> 3 -> 2 with ReLU in between
    rng = np.random.default_rng(100)
    w1 = rng.normal(size=(3, 2))
    b1 = rng.normal(size=3)
    w2 = rng.normal(size=(2, 3))
    b2 = rng.normal(size=2)

    l1 = _make_linear(0, w1, b1)
    r1 = ReLULayer(position=1)
    l2 = _make_linear(2, w2, b2)
    network = MathematicalNetwork(
        layers=(l1, r1, l2),
        input_features=2,
        output_features=2,
        source_model="TwoLayerNet",
    )

    box = Box([-1.0, 0.5], [1.5, 2.0])
    out_box = propagate_network(network, box)

    # Step-by-step oracle comparison
    low1, upp1 = _ibp_linear(box.lower, box.upper, w1, b1)
    low1_r, upp1_r = np.maximum(low1, 0.0), np.maximum(upp1, 0.0)
    expected_lower, expected_upper = _ibp_linear(low1_r, upp1_r, w2, b2)

    np.testing.assert_allclose(out_box.lower, expected_lower)
    np.testing.assert_allclose(out_box.upper, expected_upper)


# ==============================================================================
# 2. InputRegion Integration
# ==============================================================================


def test_input_region_passed_directly_as_input() -> None:
    linear = _make_linear(0, [[1.0, 2.0], [-1.0, 1.0]], [0.0, 1.0])
    network = MathematicalNetwork(
        layers=(linear,),
        input_features=2,
        output_features=2,
        source_model="TestNet",
    )

    region = InputRegion(np.array([0.5, 0.5]), 0.1, 0.0, 1.0)
    assert isinstance(region, BoxProtocol)

    out_box = propagate_network(network, region)
    assert isinstance(out_box, Box)
    assert out_box.dimension == 2

    expected_low, expected_upp = _ibp_linear(region.lower, region.upper, linear.weight, linear.bias)
    np.testing.assert_allclose(out_box.lower, expected_low)
    np.testing.assert_allclose(out_box.upper, expected_upp)


# ==============================================================================
# 3. Validation and Error Handling
# ==============================================================================


def test_input_dimension_mismatch_rejected() -> None:
    linear = _make_linear(0, np.ones((2, 3)), np.zeros(2))
    network = MathematicalNetwork(
        layers=(linear,),
        input_features=3,
        output_features=2,
        source_model="TestNet",
    )

    box_wrong_dim = Box([1.0, 2.0], [3.0, 4.0])  # Dim 2 instead of 3
    with pytest.raises(ValueError, match="does not match network.input_features"):
        propagate_network(network, box_wrong_dim)


def test_incorrect_output_dimension_rejected() -> None:
    linear = _make_linear(0, np.ones((2, 3)), np.zeros(2))
    # Claim network outputs 5, but final layer outputs 2
    network = MathematicalNetwork(
        layers=(linear,),
        input_features=3,
        output_features=5,
        source_model="MismatchedOutputNet",
    )

    box = Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="does not match network.output_features"):
        propagate_network(network, box)


def test_unsupported_layer_rejected() -> None:
    @dataclass(frozen=True)
    class UnsupportedDummyLayer:
        position: int = 1

    linear = _make_linear(0, np.ones((2, 2)), np.zeros(2))
    network = MathematicalNetwork(
        layers=(linear, UnsupportedDummyLayer()),  # type: ignore[arg-type]
        input_features=2,
        output_features=2,
        source_model="UnsupportedNet",
    )

    box = Box([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(TypeError, match="Unsupported layer type 'UnsupportedDummyLayer'"):
        propagate_network(network, box)


def test_non_box_protocol_input_rejected() -> None:
    linear = _make_linear(0, np.ones((2, 2)), np.zeros(2))
    network = MathematicalNetwork(
        layers=(linear,),
        input_features=2,
        output_features=2,
        source_model="TestNet",
    )

    with pytest.raises(TypeError, match="must satisfy BoxProtocol"):
        propagate_network(network, "not a box protocol")  # type: ignore[arg-type]


def test_non_mathematical_network_rejected() -> None:
    box = Box([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(TypeError, match="must be a MathematicalNetwork"):
        propagate_network("not a network", box)  # type: ignore[arg-type]


# ==============================================================================
# 4. Layer Order Preservation
# ==============================================================================


def test_layer_order_preservation() -> None:
    # W1 scales by 2, W2 adds 10 via bias
    w1 = np.array([[2.0]])
    b1 = np.array([0.0])
    w2 = np.array([[1.0]])
    b2 = np.array([10.0])

    l1 = _make_linear(0, w1, b1)
    l2 = _make_linear(1, w2, b2)

    # Order 1: x -> *2 -> +10: [1, 2] -> [2, 4] -> [12, 14]
    net_1 = MathematicalNetwork(layers=(l1, l2), input_features=1, output_features=1, source_model="Order1")
    out_1 = propagate_network(net_1, Box([1.0], [2.0]))
    assert out_1.lower[0] == pytest.approx(12.0)
    assert out_1.upper[0] == pytest.approx(14.0)

    # Order 2: x -> +10 -> *2: [1, 2] -> [11, 12] -> [22, 24]
    l2_first = _make_linear(0, w2, b2)
    l1_second = _make_linear(1, w1, b1)
    net_2 = MathematicalNetwork(layers=(l2_first, l1_second), input_features=1, output_features=1, source_model="Order2")
    out_2 = propagate_network(net_2, Box([1.0], [2.0]))
    assert out_2.lower[0] == pytest.approx(22.0)
    assert out_2.upper[0] == pytest.approx(24.0)


# ==============================================================================
# 5. Randomized Soundness & Comparison against Oracle
# ==============================================================================


def test_randomized_soundness_and_oracle_comparison() -> None:
    rng = np.random.default_rng(2026)

    for trial in range(5):
        in_dim = int(rng.integers(2, 6))
        hidden_dim = int(rng.integers(3, 8))
        out_dim = int(rng.integers(2, 5))

        w1 = rng.normal(0.0, 1.0, size=(hidden_dim, in_dim))
        b1 = rng.normal(0.0, 0.5, size=hidden_dim)
        w2 = rng.normal(0.0, 1.0, size=(out_dim, hidden_dim))
        b2 = rng.normal(0.0, 0.5, size=out_dim)

        l1 = _make_linear(0, w1, b1)
        r1 = ReLULayer(position=1)
        l2 = _make_linear(2, w2, b2)

        network = MathematicalNetwork(
            layers=(l1, r1, l2),
            input_features=in_dim,
            output_features=out_dim,
            source_model=f"RandomNet_{trial}",
        )

        center = rng.uniform(-2.0, 2.0, size=in_dim)
        radius = rng.uniform(0.1, 0.5, size=in_dim)
        box = Box(center - radius, center + radius)

        out_box = propagate_network(network, box)

        # Oracle comparison
        low1, upp1 = _ibp_linear(box.lower, box.upper, w1, b1)
        low1_r, upp1_r = np.maximum(low1, 0.0), np.maximum(upp1, 0.0)
        exp_low, exp_upp = _ibp_linear(low1_r, upp1_r, w2, b2)
        np.testing.assert_allclose(out_box.lower, exp_low)
        np.testing.assert_allclose(out_box.upper, exp_upp)

        # Soundness check: 100 concrete samples must fall inside the propagated output box
        samples = rng.uniform(box.lower, box.upper, size=(100, in_dim))
        concrete_outputs = network.forward(samples)

        assert np.all(concrete_outputs >= out_box.lower - 1e-12)
        assert np.all(concrete_outputs <= out_box.upper + 1e-12)


# ==============================================================================
# 6. MiniCertNet Reference Architecture
# ==============================================================================


def test_minicertnet_architecture_propagation() -> None:
    torch_model = MiniCertNet()
    network = extract_network(torch_model)

    assert network.input_features == 784
    assert network.output_features == 10
    assert len(network.layers) == 5

    # 1. Propagate Box
    box = Box(np.zeros(784), np.ones(784) * 0.1)
    out_box = propagate_network(network, box)

    assert isinstance(out_box, Box)
    assert out_box.dimension == 10
    assert np.all(out_box.lower <= out_box.upper)
    assert np.all(np.isfinite(out_box.lower))
    assert np.all(np.isfinite(out_box.upper))

    # 2. Propagate InputRegion directly
    region = InputRegion(np.full(784, 0.5), 0.02, 0.0, 1.0)
    out_region_box = propagate_network(network, region)

    assert isinstance(out_region_box, Box)
    assert out_region_box.dimension == 10
    assert np.all(out_region_box.lower <= out_region_box.upper)


# ==============================================================================
# 7. Immutability of Original Input Bounds
# ==============================================================================


def test_original_input_bounds_remain_unchanged() -> None:
    linear = _make_linear(0, np.array([[2.0, -1.0]]), np.array([0.5]))
    network = MathematicalNetwork(
        layers=(linear,),
        input_features=2,
        output_features=1,
        source_model="ImmutabilityNet",
    )

    init_lower = np.array([1.0, 2.0])
    init_upper = np.array([3.0, 4.0])
    box = Box(init_lower, init_upper)

    out = propagate_network(network, box)

    assert out is not box
    np.testing.assert_array_equal(box.lower, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(box.upper, np.array([3.0, 4.0]))
    np.testing.assert_array_equal(init_lower, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(init_upper, np.array([3.0, 4.0]))

