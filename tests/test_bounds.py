"""Comprehensive tests for intermediate-bounds extraction and shared Phase 3 contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from abstract_domain import (
    Box,
    IntermediateBounds,
    LayerBounds,
    NeuronRef,
    RefinedBounds,
    get_intermediate_bounds,
    propagate_network,
)
from abstract_domain.bounds import ALLOWED_LAYER_TYPES, ALLOWED_PHASES
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer, extract_network
from models.network import MiniCertNet
from tests.test_verification_pipeline import _ibp_linear
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


BOX2 = Box([0.0, 0.0], [1.0, 1.0])
BOX3 = Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])


def _linear_bounds(layer_index: int, in_box: Box, out_box: Box) -> LayerBounds:
    """Build a valid linear LayerBounds record."""
    return LayerBounds(
        layer_index=layer_index,
        layer_type="linear",
        input_bounds=in_box,
        pre_activation_bounds=out_box,
        post_activation_bounds=None,
        output_bounds=out_box,
    )


def _relu_bounds(layer_index: int, in_box: Box, out_box: Box) -> LayerBounds:
    """Build a valid relu LayerBounds record."""
    return LayerBounds(
        layer_index=layer_index,
        layer_type="relu",
        input_bounds=in_box,
        pre_activation_bounds=in_box,
        post_activation_bounds=out_box,
        output_bounds=out_box,
    )


# ==============================================================================
# A. LayerBounds
# ==============================================================================


def test_valid_linear_layer_bounds() -> None:
    input_bounds = Box([0.0, 0.0], [1.0, 1.0])
    pre = Box([-1.0, 0.5], [2.0, 3.0])
    lb = LayerBounds(
        layer_index=0,
        layer_type="linear",
        input_bounds=input_bounds,
        pre_activation_bounds=pre,
        post_activation_bounds=None,
        output_bounds=pre,
    )

    assert lb.layer_index == 0
    assert lb.layer_type == "linear"
    assert lb.input_bounds is input_bounds
    assert lb.pre_activation_bounds is pre
    assert lb.post_activation_bounds is None
    assert lb.output_bounds is pre


def test_valid_relu_layer_bounds() -> None:
    in_b = Box([-1.0, 2.0], [3.0, 5.0])
    post = Box([0.0, 2.0], [3.0, 5.0])
    lb = LayerBounds(
        layer_index=1,
        layer_type="relu",
        input_bounds=in_b,
        pre_activation_bounds=in_b,
        post_activation_bounds=post,
        output_bounds=post,
    )

    assert lb.layer_index == 1
    assert lb.layer_type == "relu"
    assert lb.pre_activation_bounds is in_b
    assert lb.post_activation_bounds is post
    assert lb.output_bounds is post


def test_negative_layer_index_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        LayerBounds(-1, "linear", BOX2, BOX2, None, BOX2)


def test_boolean_layer_index_rejected() -> None:
    with pytest.raises(TypeError, match="not a boolean"):
        LayerBounds(True, "linear", BOX2, BOX2, None, BOX2)  # type: ignore[arg-type]


def test_invalid_layer_type_rejected() -> None:
    with pytest.raises(ValueError, match="layer_type must be one of"):
        LayerBounds(0, "conv", BOX2, BOX2, None, BOX2)


def test_invalid_box_fields_rejected() -> None:
    with pytest.raises(TypeError, match="input_bounds must be a Box"):
        LayerBounds(0, "linear", [0.0, 0.0], BOX2, None, BOX2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="output_bounds must be a Box"):
        LayerBounds(0, "linear", BOX2, BOX2, None, [0.0, 0.0])  # type: ignore[arg-type]


def test_linear_layer_requires_none_post_activation() -> None:
    with pytest.raises(ValueError, match="post_activation_bounds set to None"):
        LayerBounds(0, "linear", BOX2, BOX2, BOX2, BOX2)


def test_linear_layer_requires_pre_activation() -> None:
    with pytest.raises(ValueError, match="requires pre_activation_bounds"):
        LayerBounds(0, "linear", BOX2, None, None, BOX2)


def test_relu_layer_requires_pre_and_post_activation() -> None:
    with pytest.raises(ValueError, match="requires both pre_activation_bounds"):
        LayerBounds(0, "relu", BOX2, None, None, BOX2)


def test_layer_bounds_frozen() -> None:
    lb = LayerBounds(0, "linear", BOX2, BOX2, None, BOX2)

    with pytest.raises(FrozenInstanceError):
        lb.layer_index = 1  # type: ignore[misc]


# ==============================================================================
# B. IntermediateBounds
# ==============================================================================


def test_valid_single_layer_trace() -> None:
    ib = IntermediateBounds(
        input_bounds=BOX2,
        layer_bounds=(_linear_bounds(0, BOX2, BOX2),),
        output_bounds=BOX2,
    )

    assert len(ib.layer_bounds) == 1
    assert ib.layer_bounds[0].layer_index == 0
    assert ib.input_bounds == BOX2
    assert ib.output_bounds == BOX2


def test_valid_multi_layer_trace() -> None:
    l0 = _linear_bounds(0, BOX2, BOX3)
    l1 = _relu_bounds(1, BOX3, BOX3)
    l2 = _linear_bounds(2, BOX3, BOX2)
    ib = IntermediateBounds(
        input_bounds=BOX2,
        layer_bounds=(l0, l1, l2),
        output_bounds=BOX2,
    )

    assert [entry.layer_index for entry in ib.layer_bounds] == [0, 1, 2]
    assert [entry.layer_type for entry in ib.layer_bounds] == ["linear", "relu", "linear"]


def test_layer_bounds_must_be_tuple() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        IntermediateBounds(
            input_bounds=BOX2,
            layer_bounds=[_linear_bounds(0, BOX2, BOX2)],  # type: ignore[arg-type]
            output_bounds=BOX2,
        )


def test_layer_bounds_must_be_layer_bounds_instances() -> None:
    with pytest.raises(TypeError, match="must be a LayerBounds"):
        IntermediateBounds(
            input_bounds=BOX2,
            layer_bounds=(BOX2,),  # type: ignore[arg-type]
            output_bounds=BOX2,
        )


def test_layer_order_validation() -> None:
    # Out-of-order layer indices must be rejected
    l0 = _linear_bounds(1, BOX2, BOX2)
    l1 = _linear_bounds(0, BOX2, BOX2)
    with pytest.raises(ValueError, match="original execution order"):
        IntermediateBounds(
            input_bounds=BOX2,
            layer_bounds=(l0, l1),
            output_bounds=BOX2,
        )


def test_adjacent_dimension_mismatch_rejected() -> None:
    l0 = _linear_bounds(0, BOX2, BOX2)  # outputs dim 2
    l1 = _linear_bounds(1, BOX3, BOX2)  # expects input dim 3
    with pytest.raises(ValueError, match="adjacent layer dimension mismatch"):
        IntermediateBounds(
            input_bounds=BOX2,
            layer_bounds=(l0, l1),
            output_bounds=BOX2,
        )


def test_first_layer_input_dimension_consistency() -> None:
    l0 = _linear_bounds(0, BOX3, BOX2)
    with pytest.raises(ValueError, match="first layer input dimension"):
        IntermediateBounds(
            input_bounds=BOX2,
            layer_bounds=(l0,),
            output_bounds=BOX2,
        )


def test_final_layer_output_dimension_consistency() -> None:
    l0 = _linear_bounds(0, BOX2, BOX3)
    with pytest.raises(ValueError, match="final layer output dimension"):
        IntermediateBounds(
            input_bounds=BOX2,
            layer_bounds=(l0,),
            output_bounds=BOX2,
        )


def test_empty_network_consistent_with_network_invariants() -> None:
    # A network with no layers maps input directly to output (identity)
    ib = IntermediateBounds(input_bounds=BOX2, layer_bounds=(), output_bounds=BOX2)
    assert ib.layer_bounds == ()
    assert ib.output_bounds == ib.input_bounds


def test_empty_network_dimension_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="empty network"):
        IntermediateBounds(input_bounds=BOX2, layer_bounds=(), output_bounds=BOX3)


def test_intermediate_bounds_frozen() -> None:
    ib = IntermediateBounds(
        input_bounds=BOX2,
        layer_bounds=(_linear_bounds(0, BOX2, BOX2),),
        output_bounds=BOX2,
    )

    with pytest.raises(FrozenInstanceError):
        ib.input_bounds = BOX3  # type: ignore[misc]


# ==============================================================================
# C. NeuronRef
# ==============================================================================


def test_valid_pre_and_post_neuron_refs() -> None:
    pre = NeuronRef(layer_index=2, phase="pre", neuron_index=0)
    post = NeuronRef(layer_index=0, phase="post", neuron_index=5)

    assert pre.phase == "pre"
    assert post.phase == "post"
    assert pre.layer_index == 2
    assert post.neuron_index == 5


def test_invalid_phase_rejected() -> None:
    with pytest.raises(ValueError, match="phase must be one of"):
        NeuronRef(0, "mid", 0)  # type: ignore[arg-type]


def test_negative_indices_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        NeuronRef(-1, "pre", 0)
    with pytest.raises(ValueError, match="nonnegative"):
        NeuronRef(0, "pre", -1)


def test_boolean_index_rejected() -> None:
    with pytest.raises(TypeError, match="not a boolean"):
        NeuronRef(True, "pre", 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a boolean"):
        NeuronRef(0, "pre", False)  # type: ignore[arg-type]


def test_phase_constants_are_exported() -> None:
    assert ALLOWED_PHASES == ("pre", "post")
    assert ALLOWED_LAYER_TYPES == ("linear", "relu")


# ==============================================================================
# D. RefinedBounds
# ==============================================================================


def test_valid_refined_bounds() -> None:
    neuron = NeuronRef(0, "pre", 0)
    rb = RefinedBounds(neuron=neuron, lower=-0.5, upper=0.5)

    assert rb.neuron is neuron
    assert rb.lower == -0.5
    assert rb.upper == 0.5


def test_invalid_lower_upper_order_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed upper"):
        RefinedBounds(NeuronRef(0, "pre", 0), 0.5, -0.5)


def test_nan_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        RefinedBounds(NeuronRef(0, "pre", 0), float("nan"), 0.5)


def test_infinite_bound_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        RefinedBounds(NeuronRef(0, "pre", 0), -float("inf"), 0.5)
    with pytest.raises(ValueError, match="must be finite"):
        RefinedBounds(NeuronRef(0, "pre", 0), 0.0, float("inf"))


def test_invalid_neuron_reference_rejected() -> None:
    with pytest.raises(TypeError, match="neuron must be a NeuronRef"):
        RefinedBounds("not a neuron", 0.0, 1.0)  # type: ignore[arg-type]


def test_refined_bounds_frozen() -> None:
    rb = RefinedBounds(NeuronRef(0, "pre", 0), -0.5, 0.5)

    with pytest.raises(FrozenInstanceError):
        rb.lower = 0.0  # type: ignore[misc]


# ==============================================================================
# E. Intermediate Extraction
# ==============================================================================


def test_single_linear_layer_extraction() -> None:
    weight = np.array([[2.0, -1.0], [0.5, 3.0]])
    bias = np.array([0.1, -0.2])
    network = MathematicalNetwork(
        layers=(_make_linear(0, weight, bias),),
        input_features=2,
        output_features=2,
        source_model="SingleLinearNet",
    )
    box = Box([1.0, 2.0], [3.0, 4.0])

    ib = get_intermediate_bounds(network, box)

    assert len(ib.layer_bounds) == 1
    lb = ib.layer_bounds[0]
    assert lb.layer_index == 0
    assert lb.layer_type == "linear"
    assert lb.input_bounds == box
    assert lb.post_activation_bounds is None
    assert lb.pre_activation_bounds == lb.output_bounds

    expected_lower, expected_upper = _ibp_linear(box.lower, box.upper, weight, bias)
    np.testing.assert_allclose(lb.output_bounds.lower, expected_lower)
    np.testing.assert_allclose(lb.output_bounds.upper, expected_upper)
    np.testing.assert_allclose(ib.output_bounds.lower, expected_lower)
    np.testing.assert_allclose(ib.output_bounds.upper, expected_upper)


def test_linear_then_relu_extraction() -> None:
    weight = np.array([[-1.0, 0.0], [1.0, 2.0]])
    bias = np.array([0.0, -5.0])
    network = MathematicalNetwork(
        layers=(_make_linear(0, weight, bias), ReLULayer(position=1)),
        input_features=2,
        output_features=2,
        source_model="LinearReLUNet",
    )
    box = Box([1.0, 1.0], [2.0, 2.0])

    ib = get_intermediate_bounds(network, box)

    assert len(ib.layer_bounds) == 2
    lin, relu = ib.layer_bounds
    assert lin.layer_type == "linear"
    assert relu.layer_type == "relu"
    assert relu.pre_activation_bounds == relu.input_bounds
    assert relu.post_activation_bounds == relu.output_bounds
    # ReLU output is max(0, linear pre-activation)
    np.testing.assert_allclose(
        relu.output_bounds.lower, np.maximum(lin.output_bounds.lower, 0.0)
    )
    np.testing.assert_allclose(
        relu.output_bounds.upper, np.maximum(lin.output_bounds.upper, 0.0)
    )
    np.testing.assert_allclose(ib.output_bounds.lower, np.array([0.0, 0.0]))
    np.testing.assert_allclose(ib.output_bounds.upper, np.array([0.0, 1.0]))


def test_multiple_linear_and_relu_layers_extraction() -> None:
    rng = np.random.default_rng(100)
    w1 = rng.normal(size=(3, 2))
    b1 = rng.normal(size=3)
    w2 = rng.normal(size=(2, 3))
    b2 = rng.normal(size=2)
    network = MathematicalNetwork(
        layers=(_make_linear(0, w1, b1), ReLULayer(position=1), _make_linear(2, w2, b2)),
        input_features=2,
        output_features=2,
        source_model="TwoLayerNet",
    )
    box = Box([-1.0, 0.5], [1.5, 2.0])

    ib = get_intermediate_bounds(network, box)

    assert [entry.layer_type for entry in ib.layer_bounds] == ["linear", "relu", "linear"]
    assert ib.input_bounds == box
    # Adjacent dimensions
    assert ib.layer_bounds[0].output_bounds.dimension == 3
    assert ib.layer_bounds[1].input_bounds.dimension == 3
    assert ib.layer_bounds[1].output_bounds.dimension == 3
    assert ib.layer_bounds[2].input_bounds.dimension == 3
    assert ib.output_bounds.dimension == 2


def test_minicertnet_extraction() -> None:
    torch_model = MiniCertNet()
    network = extract_network(torch_model)

    assert network.input_features == 784
    assert network.output_features == 10

    box = Box(np.zeros(784), np.ones(784) * 0.1)
    ib = get_intermediate_bounds(network, box)

    assert ib.input_bounds.dimension == 784
    assert ib.output_bounds.dimension == 10
    assert len(ib.layer_bounds) == 5
    assert [entry.layer_type for entry in ib.layer_bounds] == [
        "linear", "relu", "linear", "relu", "linear",
    ]
    assert all(np.isfinite(entry.output_bounds.lower).all() for entry in ib.layer_bounds)
    assert all(np.isfinite(entry.output_bounds.upper).all() for entry in ib.layer_bounds)


def test_direct_input_region_compatibility() -> None:
    network = MathematicalNetwork(
        layers=(_make_linear(0, [[1.0, 2.0], [-1.0, 1.0]], [0.0, 1.0]),),
        input_features=2,
        output_features=2,
        source_model="RegionNet",
    )
    region = InputRegion(np.array([0.5, 0.5]), 0.1, 0.0, 1.0)

    ib = get_intermediate_bounds(network, region)

    assert ib.input_bounds == Box(region.lower, region.upper)
    assert isinstance(ib.input_bounds, Box)


def test_layer_execution_order_preserved() -> None:
    # W1 scales by 2, W2 adds 10 via bias
    l1 = _make_linear(0, [[2.0]], [0.0])
    l2 = _make_linear(1, [[1.0]], [10.0])
    network = MathematicalNetwork(
        layers=(l1, l2), input_features=1, output_features=1, source_model="OrderNet"
    )
    box = Box([1.0], [2.0])

    ib = get_intermediate_bounds(network, box)

    assert [entry.layer_index for entry in ib.layer_bounds] == [0, 1]
    assert [entry.layer_type for entry in ib.layer_bounds] == ["linear", "linear"]
    # Layer 0 output is [2, 4]; layer 1 output is [12, 14]
    np.testing.assert_allclose(ib.layer_bounds[0].output_bounds.lower, np.array([2.0]))
    np.testing.assert_allclose(ib.layer_bounds[0].output_bounds.upper, np.array([4.0]))
    np.testing.assert_allclose(ib.layer_bounds[1].output_bounds.lower, np.array([12.0]))
    np.testing.assert_allclose(ib.layer_bounds[1].output_bounds.upper, np.array([14.0]))
    np.testing.assert_allclose(ib.output_bounds.lower, np.array([12.0]))
    np.testing.assert_allclose(ib.output_bounds.upper, np.array([14.0]))


def test_output_matches_propagate_network() -> None:
    rng = np.random.default_rng(2026)
    for trial in range(5):
        in_dim = int(rng.integers(2, 6))
        hidden_dim = int(rng.integers(3, 8))
        out_dim = int(rng.integers(2, 5))
        w1 = rng.normal(0.0, 1.0, size=(hidden_dim, in_dim))
        b1 = rng.normal(0.0, 0.5, size=hidden_dim)
        w2 = rng.normal(0.0, 1.0, size=(out_dim, hidden_dim))
        b2 = rng.normal(0.0, 0.5, size=out_dim)
        network = MathematicalNetwork(
            layers=(
                _make_linear(0, w1, b1),
                ReLULayer(position=1),
                _make_linear(2, w2, b2),
            ),
            input_features=in_dim,
            output_features=out_dim,
            source_model=f"RandomNet_{trial}",
        )
        center = rng.uniform(-2.0, 2.0, size=in_dim)
        radius = rng.uniform(0.1, 0.5, size=in_dim)
        box = Box(center - radius, center + radius)

        ib = get_intermediate_bounds(network, box)
        propagated = propagate_network(network, box)

        np.testing.assert_array_equal(ib.output_bounds.lower, propagated.lower)
        np.testing.assert_array_equal(ib.output_bounds.upper, propagated.upper)


def test_original_input_region_remains_unchanged() -> None:
    network = MathematicalNetwork(
        layers=(_make_linear(0, [[2.0, -1.0]], [0.5]),),
        input_features=2,
        output_features=1,
        source_model="ImmutabilityNet",
    )
    center = np.array([0.5, 0.5])
    region = InputRegion(center, 0.1, 0.0, 1.0)
    initial_lower = region.lower.copy()
    initial_upper = region.upper.copy()

    _ = get_intermediate_bounds(network, region)

    np.testing.assert_array_equal(region.center, center)
    np.testing.assert_array_equal(region.lower, initial_lower)
    np.testing.assert_array_equal(region.upper, initial_upper)


def test_extraction_dimension_mismatch_rejected() -> None:
    network = MathematicalNetwork(
        layers=(_make_linear(0, np.ones((2, 3)), np.zeros(2)),),
        input_features=3,
        output_features=2,
        source_model="TestNet",
    )
    with pytest.raises(ValueError, match="does not match network.input_features"):
        get_intermediate_bounds(network, BOX2)


def test_extraction_non_box_protocol_rejected() -> None:
    network = MathematicalNetwork(
        layers=(_make_linear(0, np.ones((2, 2)), np.zeros(2)),),
        input_features=2,
        output_features=2,
        source_model="TestNet",
    )
    with pytest.raises(TypeError, match="must satisfy BoxProtocol"):
        get_intermediate_bounds(network, "not a box")  # type: ignore[arg-type]
