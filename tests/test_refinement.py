"""Comprehensive tests for applying refined bounds and re-propagating the network."""

from __future__ import annotations

import numpy as np
import pytest

from abstract_domain import (
    Box,
    NeuronRef,
    RefinedBounds,
    get_intermediate_bounds,
    propagate_with_refinements,
)
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
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


def _two_class_linear_net() -> MathematicalNetwork:
    """2 -> 2 linear network separating class 0 and class 1."""
    weight = np.array([[1.0, -1.0], [-1.0, 1.0]])
    return MathematicalNetwork(
        layers=(_make_linear(0, weight, np.zeros(2)),),
        input_features=2,
        output_features=2,
        source_model="TwoClassLinear",
    )


def _linear_relu_net() -> MathematicalNetwork:
    """2 -> 2 network with a Linear then ReLU layer."""
    weight = np.array([[-1.0, 0.0], [1.0, 2.0]])
    bias = np.array([0.0, -5.0])
    return MathematicalNetwork(
        layers=(_make_linear(0, weight, bias), ReLULayer(position=1)),
        input_features=2,
        output_features=2,
        source_model="LinearReLUNet",
    )


def _deep_net() -> MathematicalNetwork:
    """2 -> 3 -> 2 network with Linear, ReLU, Linear layers."""
    w0 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    w1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return MathematicalNetwork(
        layers=(
            _make_linear(0, w0, np.zeros(3)),
            ReLULayer(position=1),
            _make_linear(2, w1, np.zeros(2)),
        ),
        input_features=2,
        output_features=2,
        source_model="DeepNet",
    )


REGION = InputRegion(np.array([0.5, 0.1]), 0.05)


# ==============================================================================
# 1. Empty refinement matches ordinary propagation
# ==============================================================================


def test_empty_refinement_matches_ordinary_propagation() -> None:
    network = _two_class_linear_net()

    result = propagate_with_refinements(network, REGION, ())

    assert result == get_intermediate_bounds(network, REGION)


# ==============================================================================
# 2. Valid linear-layer pre-activation refinement
# ==============================================================================


def test_valid_linear_pre_activation_refinement() -> None:
    network = _two_class_linear_net()
    # IBP output bounds: [0.30, -0.50] to [0.50, -0.30]
    refined = RefinedBounds(NeuronRef(0, "pre", 0), 0.35, 0.45)

    result = propagate_with_refinements(network, REGION, (refined,))

    assert result.output_bounds.lower[0] == pytest.approx(0.35)
    assert result.output_bounds.upper[0] == pytest.approx(0.45)
    # Neuron 1 is unrefined and preserved
    assert result.output_bounds.lower[1] == pytest.approx(-0.50)
    assert result.output_bounds.upper[1] == pytest.approx(-0.30)
    assert result.layer_bounds[0].pre_activation_bounds == result.output_bounds


def test_valid_linear_pre_refinement_as_list() -> None:
    network = _two_class_linear_net()
    refined = RefinedBounds(NeuronRef(0, "pre", 1), -0.45, -0.35)

    result = propagate_with_refinements(network, REGION, [refined])

    assert result.output_bounds.lower[1] == pytest.approx(-0.45)
    assert result.output_bounds.upper[1] == pytest.approx(-0.35)


# ==============================================================================
# 3. Valid ReLU pre-activation refinement
# ==============================================================================


def test_valid_relu_pre_activation_refinement() -> None:
    network = _linear_relu_net()
    # Input box [1, 1] to [2, 2] maps to linear output pre-activation bounds [-2, -2] to [-1, 1]
    region = InputRegion(np.array([1.5, 1.5]), 0.5)
    # Refine ReLU layer (index 1) pre-activation neuron 1 to [0.0, 0.5]
    refined = RefinedBounds(NeuronRef(1, "pre", 1), 0.0, 0.5)

    result = propagate_with_refinements(network, region, (refined,))

    relu_bounds = result.layer_bounds[1]
    # Refined pre-activation is the ReLU argument (== input for relu)
    np.testing.assert_allclose(relu_bounds.input_bounds.lower, np.array([-2.0, 0.0]))
    np.testing.assert_allclose(relu_bounds.input_bounds.upper, np.array([-1.0, 0.5]))
    np.testing.assert_allclose(relu_bounds.pre_activation_bounds.lower, np.array([-2.0, 0.0]))
    np.testing.assert_allclose(relu_bounds.pre_activation_bounds.upper, np.array([-1.0, 0.5]))
    # ReLU output recomputed from the refined argument
    np.testing.assert_allclose(result.output_bounds.lower, np.array([0.0, 0.0]))
    np.testing.assert_allclose(result.output_bounds.upper, np.array([0.0, 0.5]))


# ==============================================================================
# 4. Valid ReLU post-activation refinement
# ==============================================================================


def test_valid_relu_post_activation_refinement() -> None:
    network = _linear_relu_net()
    # Input box [1, 1] to [2, 2] -> ReLU output bounds [0, 0] to [0, 1]
    region = InputRegion(np.array([1.5, 1.5]), 0.5)
    # Refine neuron 1 post to [0.0, 0.5]
    refined = RefinedBounds(NeuronRef(1, "post", 1), 0.0, 0.5)

    result = propagate_with_refinements(network, region, (refined,))

    relu_bounds = result.layer_bounds[1]
    np.testing.assert_allclose(relu_bounds.post_activation_bounds.lower, np.array([0.0, 0.0]))
    np.testing.assert_allclose(relu_bounds.post_activation_bounds.upper, np.array([0.0, 0.5]))
    np.testing.assert_allclose(result.output_bounds.lower, np.array([0.0, 0.0]))
    np.testing.assert_allclose(result.output_bounds.upper, np.array([0.0, 0.5]))


# ==============================================================================
# 5. Multiple refinements across different layers
# ==============================================================================


def test_multiple_refinements_across_layers() -> None:
    network = _deep_net()
    refine_layer0 = RefinedBounds(NeuronRef(0, "pre", 2), 0.55, 0.60)
    refine_layer2 = RefinedBounds(NeuronRef(2, "pre", 1), 0.10, 0.12)

    result = propagate_with_refinements(network, REGION, (refine_layer0, refine_layer2))

    np.testing.assert_allclose(
        result.layer_bounds[0].output_bounds.lower, np.array([0.45, 0.05, 0.55])
    )
    np.testing.assert_allclose(
        result.layer_bounds[0].output_bounds.upper, np.array([0.55, 0.15, 0.60])
    )
    np.testing.assert_allclose(
        result.layer_bounds[2].output_bounds.lower, np.array([0.45, 0.10])
    )
    np.testing.assert_allclose(
        result.layer_bounds[2].output_bounds.upper, np.array([0.55, 0.12])
    )
    np.testing.assert_allclose(result.output_bounds.lower, np.array([0.45, 0.10]))
    np.testing.assert_allclose(result.output_bounds.upper, np.array([0.55, 0.12]))


# ==============================================================================
# 6. Downstream bounds recomputed after refinement
# ==============================================================================


def test_downstream_bounds_recomputed_after_refinement() -> None:
    # 2 -> 2 -> 2 network: layer 0 separates classes, layer 1 scales x0 by 2
    w0 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    w1 = np.array([[2.0, 0.0], [0.0, 1.0]])
    network = MathematicalNetwork(
        layers=(
            _make_linear(0, w0, np.zeros(2)),
            _make_linear(1, w1, np.zeros(2)),
        ),
        input_features=2,
        output_features=2,
        source_model="TwoLayerLinear",
    )
    refined = RefinedBounds(NeuronRef(0, "pre", 0), 0.35, 0.45)

    result = propagate_with_refinements(network, REGION, (refined,))

    # Layer 0 output refined: neuron 0 -> [0.35, 0.45]
    np.testing.assert_allclose(result.layer_bounds[0].output_bounds.lower, np.array([0.35, -0.50]))
    np.testing.assert_allclose(result.layer_bounds[0].output_bounds.upper, np.array([0.45, -0.30]))
    # Layer 1 input reflects the refined layer 0 output
    np.testing.assert_allclose(result.layer_bounds[1].input_bounds.lower, np.array([0.35, -0.50]))
    np.testing.assert_allclose(result.layer_bounds[1].input_bounds.upper, np.array([0.45, -0.30]))
    # Layer 1 recomputed: output0 = 2 * input0 -> [0.70, 0.90]
    np.testing.assert_allclose(result.output_bounds.lower, np.array([0.70, -0.50]))
    np.testing.assert_allclose(result.output_bounds.upper, np.array([0.90, -0.30]))


# ==============================================================================
# 7. Unrefined neuron bounds remain unchanged
# ==============================================================================


def test_unrefined_neuron_bounds_unchanged() -> None:
    network = _two_class_linear_net()
    ibp = get_intermediate_bounds(network, REGION)
    refined = RefinedBounds(NeuronRef(0, "pre", 0), 0.35, 0.45)

    result = propagate_with_refinements(network, REGION, (refined,))

    # Neuron 1 (index 1) keeps its original IBP bounds at every layer
    assert result.layer_bounds[0].output_bounds.lower[1] == ibp.layer_bounds[0].output_bounds.lower[1]
    assert result.layer_bounds[0].output_bounds.upper[1] == ibp.layer_bounds[0].output_bounds.upper[1]
    assert result.output_bounds.lower[1] == ibp.output_bounds.lower[1]
    assert result.output_bounds.upper[1] == ibp.output_bounds.upper[1]


# ==============================================================================
# 8. Validation rejections
# ==============================================================================


def test_invalid_layer_index_rejected() -> None:
    network = _two_class_linear_net()  # 1 layer
    refined = RefinedBounds(NeuronRef(5, "pre", 0), 0.35, 0.45)

    with pytest.raises(ValueError, match="out of range for a network"):
        propagate_with_refinements(network, REGION, (refined,))


def test_invalid_neuron_index_rejected() -> None:
    network = _two_class_linear_net()  # output_features = 2
    refined = RefinedBounds(NeuronRef(0, "pre", 5), 0.35, 0.45)

    with pytest.raises(ValueError, match="out of range for layer"):
        propagate_with_refinements(network, REGION, (refined,))


def test_invalid_phase_rejected() -> None:
    with pytest.raises(ValueError, match="phase must be one of"):
        NeuronRef(0, "mid", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="phase must be one of"):
        RefinedBounds(NeuronRef(0, "mid", 0), 0.0, 1.0)  # type: ignore[arg-type]


def test_linear_layer_post_activation_rejected() -> None:
    network = _two_class_linear_net()
    refined = RefinedBounds(NeuronRef(0, "post", 0), 0.35, 0.45)

    with pytest.raises(ValueError, match="only support 'pre' phase"):
        propagate_with_refinements(network, REGION, (refined,))


def test_duplicate_neuron_references_rejected() -> None:
    network = _two_class_linear_net()
    first = RefinedBounds(NeuronRef(0, "pre", 0), 0.35, 0.45)
    second = RefinedBounds(NeuronRef(0, "pre", 0), 0.32, 0.48)

    with pytest.raises(ValueError, match="duplicate refinement"):
        propagate_with_refinements(network, REGION, (first, second))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (0.20, 0.60),  # extends below AND above [0.30, 0.50]
        (0.20, 0.45),  # extends below only
        (0.35, 0.60),  # extends above only
    ],
)
def test_refinement_outside_ibp_rejected(lower: float, upper: float) -> None:
    network = _two_class_linear_net()
    refined = RefinedBounds(NeuronRef(0, "pre", 0), lower, upper)

    with pytest.raises(ValueError, match="extends outside the original IBP"):
        propagate_with_refinements(network, REGION, (refined,))


def test_lower_greater_than_upper_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed upper"):
        RefinedBounds(NeuronRef(0, "pre", 0), 0.60, 0.20)


# ==============================================================================
# 9. Original objects remain unchanged
# ==============================================================================


def test_network_and_region_remain_unchanged() -> None:
    network = _two_class_linear_net()
    center = np.array([0.5, 0.1])
    region = InputRegion(center, 0.05)
    initial_lower = region.lower.copy()
    initial_upper = region.upper.copy()
    refined = RefinedBounds(NeuronRef(0, "pre", 0), 0.35, 0.45)

    _ = propagate_with_refinements(network, region, (refined,))

    assert len(network.layers) == 1
    assert isinstance(network.layers[0], LinearLayer)
    np.testing.assert_array_equal(region.center, center)
    np.testing.assert_array_equal(region.lower, initial_lower)
    np.testing.assert_array_equal(region.upper, initial_upper)


# ==============================================================================
# 10. Trace correctness
# ==============================================================================


def test_trace_has_correct_layer_indices_and_dimensions() -> None:
    network = _deep_net()
    result = propagate_with_refinements(network, REGION, ())

    assert [entry.layer_index for entry in result.layer_bounds] == [0, 1, 2]
    assert [entry.layer_type for entry in result.layer_bounds] == ["linear", "relu", "linear"]
    assert result.input_bounds.dimension == 2
    assert result.layer_bounds[0].output_bounds.dimension == 3
    assert result.layer_bounds[1].output_bounds.dimension == 3
    assert result.layer_bounds[2].output_bounds.dimension == 2
    assert result.output_bounds.dimension == 2


def test_refined_final_output_consistent_with_trace() -> None:
    network = _deep_net()
    refined = RefinedBounds(NeuronRef(2, "pre", 1), 0.10, 0.12)

    result = propagate_with_refinements(network, REGION, (refined,))

    assert result.output_bounds == result.layer_bounds[-1].output_bounds
    np.testing.assert_array_equal(
        result.output_bounds.lower, result.layer_bounds[-1].output_bounds.lower
    )
    np.testing.assert_array_equal(
        result.output_bounds.upper, result.layer_bounds[-1].output_bounds.upper
    )


# ==============================================================================
# 11. Edge cases and consistency
# ==============================================================================


def test_refined_bounds_must_be_collection() -> None:
    network = _two_class_linear_net()
    refined = RefinedBounds(NeuronRef(0, "pre", 0), 0.35, 0.45)

    with pytest.raises(TypeError, match="must be a tuple or list"):
        propagate_with_refinements(network, REGION, refined)  # type: ignore[arg-type]


def test_refined_bounds_items_must_be_refined_bounds() -> None:
    network = _two_class_linear_net()

    with pytest.raises(TypeError, match="must be a RefinedBounds"):
        propagate_with_refinements(network, REGION, (NeuronRef(0, "pre", 0),))  # type: ignore[arg-type]


def test_input_region_must_be_input_region() -> None:
    network = _two_class_linear_net()
    box = Box([0.45, 0.05], [0.55, 0.15])

    with pytest.raises(TypeError, match="must be an InputRegion"):
        propagate_with_refinements(network, box, ())  # type: ignore[arg-type]


def test_input_dimension_mismatch_rejected() -> None:
    network = _two_class_linear_net()
    wrong_region = InputRegion(np.array([0.1, 0.2, 0.3]), 0.05)

    with pytest.raises(ValueError, match="does not match network.input_features"):
        propagate_with_refinements(network, wrong_region, ())


def test_relu_post_inconsistent_with_refined_pre_rejected() -> None:
    network = _linear_relu_net()
    # Input region [1, 1] to [2, 2] -> pre bounds [-2, -2] to [-1, 1], post bounds [0, 0] to [0, 1]
    region = InputRegion(np.array([1.5, 1.5]), 0.5)
    # Refine the ReLU pre-activation neuron 1 to [0.0, 0.5], so the ReLU output
    # for that neuron is [0.0, 0.5]. A post refinement to [0.0, 0.9] is within
    # the original IBP post bounds [0.0, 1.0] but inconsistent with the refined
    # ReLU output, so it must be rejected.
    pre = RefinedBounds(NeuronRef(1, "pre", 1), 0.0, 0.5)
    post = RefinedBounds(NeuronRef(1, "post", 1), 0.0, 0.9)

    with pytest.raises(ValueError, match="inconsistent with the ReLU output"):
        propagate_with_refinements(network, region, (pre, post))


def test_non_mathematical_network_rejected() -> None:
    with pytest.raises(TypeError, match="must be a MathematicalNetwork"):
        propagate_with_refinements("not a network", REGION, ())  # type: ignore[arg-type]


def test_minicertnet_architecture_refinement() -> None:
    from models.network import MiniCertNet
    from models.model_loader import extract_network

    net = extract_network(MiniCertNet())
    region = InputRegion(np.full(784, 0.5), 0.01, 0.0, 1.0)
    trace = get_intermediate_bounds(net, region)

    l0_bounds = trace.layer_bounds[0].output_bounds
    n0_l, n0_u = l0_bounds.lower[0], l0_bounds.upper[0]
    mid = (n0_l + n0_u) / 2.0
    refined = RefinedBounds(NeuronRef(0, "pre", 0), mid, n0_u)

    result = propagate_with_refinements(net, region, (refined,))

    assert result.output_bounds.dimension == 10
    assert result.layer_bounds[0].output_bounds.lower[0] == pytest.approx(mid)
    assert result.layer_bounds[0].output_bounds.upper[0] == pytest.approx(n0_u)
