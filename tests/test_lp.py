"""Tests for the Phase 3 LP constraint generation (verification.lp).

These tests verify the sound outer-approximation encoding:

- Variable layout and neuron-to-column indexing.
- Exact affine equality rows for Linear layers.
- Exact constraints for stable ReLUs and the triangle relaxation for
  unstable ReLUs, classified from baseline IBP pre-activation bounds.
- Input clipping and per-variable bounds (including infinite-bound handling).
- Validation of dimensions, traces, layer types, and neuron references.
- Containment of known concrete executions inside the LP feasible set.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from abstract_domain import (
    Box,
    IntermediateBounds,
    LayerBounds,
    NeuronRef,
    get_intermediate_bounds,
)
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification.lp import (
    LPConstraintBuilder,
    LPProblem,
    VariableLayout,
    build_lp_problem,
)
from verification.refinement_validation import sample_region_inputs
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


def _make_network(
    layers: tuple[object, ...],
    source_model: str = "LPTestNet",
) -> MathematicalNetwork:
    """Build a MathematicalNetwork from layer objects, deriving the dimensions."""
    linears = [layer for layer in layers if isinstance(layer, LinearLayer)]
    return MathematicalNetwork(
        layers=layers,
        input_features=int(linears[0].weight.shape[1]),
        output_features=int(linears[-1].weight.shape[0]),
        source_model=source_model,
    )


def _design_net() -> MathematicalNetwork:
    """The hand-computed design example: 2 -> 1 Linear then ReLU.

    ``z0 = x0 - 2 x1 + 1`` with ``x`` in ``[-1, 1]^2``, so the IBP pre bounds
    are ``[-2, 4]`` and the ReLU is unstable.
    """
    return _make_network(
        (
            _make_linear(0, np.array([[1.0, -2.0]]), np.array([1.0])),
            ReLULayer(position=1),
        ),
        source_model="DesignNet",
    )


def _stable_active_net() -> MathematicalNetwork:
    """2 -> 1 Linear then ReLU with ``z = x0 + x1 + 1`` over ``[0, 1]^2``."""
    return _make_network(
        (
            _make_linear(0, np.array([[1.0, 1.0]]), np.array([1.0])),
            ReLULayer(position=1),
        ),
        source_model="StableActiveNet",
    )


def _stable_inactive_net() -> MathematicalNetwork:
    """2 -> 1 Linear then ReLU with ``z = -x0 - x1 - 1`` over ``[0, 1]^2``."""
    return _make_network(
        (
            _make_linear(0, np.array([[-1.0, -1.0]]), np.array([-1.0])),
            ReLULayer(position=1),
        ),
        source_model="StableInactiveNet",
    )


def _identity_net() -> MathematicalNetwork:
    """2 -> 1 Linear then ReLU with ``z = x0`` (x1 is ignored)."""
    return _make_network(
        (
            _make_linear(0, np.array([[1.0, 0.0]]), np.array([0.0])),
            ReLULayer(position=1),
        ),
        source_model="IdentityNet",
    )


def _linear_only_net() -> MathematicalNetwork:
    """2 -> 2 pure linear network from the refinement test suite."""
    weight = np.array([[1.0, -1.0], [-1.0, 1.0]])
    return _make_network(
        (_make_linear(0, weight, np.zeros(2)),),
        source_model="TwoClassLinear",
    )


UNIT_BOX = Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
UNIT_REGION = InputRegion(center=np.array([0.0, 0.0]), radius=1.0)


def _concrete_variables(network: MathematicalNetwork, x: np.ndarray) -> np.ndarray:
    """Return the LP variable vector of one concrete execution of ``x``.

    The assembly order mirrors the VariableLayout: inputs, then per sequence
    layer the pre-activation group, followed by the post-activation group for
    ReLU layers.
    """
    current = np.asarray(x, dtype=np.float64)
    parts = [current]
    for layer in network.layers:
        if isinstance(layer, LinearLayer):
            current = layer.apply(current)
            parts.append(current)
        else:
            parts.append(current.copy())
            current = layer.apply(current)
            parts.append(current)
    return np.concatenate(parts)


def _assert_concrete_execution_satisfies(
    problem: LPProblem,
    network: MathematicalNetwork,
    x: np.ndarray,
) -> None:
    """Assert that one concrete execution satisfies every LP constraint."""
    vector = _concrete_variables(network, x)
    assert vector.shape == (problem.layout.total_variables,)
    for index in range(vector.shape[0]):
        lower, upper = problem.variable_bounds[index]
        if lower is not None:
            assert vector[index] >= lower - 1e-9, (index, lower, vector[index])
        if upper is not None:
            assert vector[index] <= upper + 1e-9, (index, upper, vector[index])
    if problem.A_eq is not None:
        np.testing.assert_allclose(
            problem.A_eq @ vector, problem.b_eq, rtol=1e-5, atol=1e-9
        )
    if problem.A_ub is not None:
        assert np.all(problem.A_ub @ vector <= problem.b_ub + 1e-9)


# ==============================================================================
# 1. VariableLayout
# ==============================================================================


def _deep_layout() -> VariableLayout:
    """Layout of a 2 -> 3 ReLU 3 -> 2 network (2 + 3 + 3 + 3 + 2 = 13)."""
    return VariableLayout(
        input_dim=2,
        pre_offsets=(2, 5, 11),
        pre_dims=(3, 3, 2),
        post_offsets=(None, 8, None),
        post_dims=(None, 3, None),
        total_variables=13,
    )


def test_variable_layout_slices() -> None:
    layout = _deep_layout()

    assert layout.variable_slice(0, "pre") == slice(2, 5)
    assert layout.variable_slice(1, "pre") == slice(5, 8)
    assert layout.variable_slice(1, "post") == slice(8, 11)
    assert layout.variable_slice(2, "pre") == slice(11, 13)


def test_variable_layout_objective_vector() -> None:
    layout = _deep_layout()

    expected = np.zeros(13)
    expected[9] = 1.0
    np.testing.assert_array_equal(
        layout.objective_vector(NeuronRef(1, "post", 1)), expected
    )

    expected = np.zeros(13)
    expected[12] = 1.0
    np.testing.assert_array_equal(
        layout.objective_vector(NeuronRef(2, "pre", 1)), expected
    )


def test_variable_layout_variable_index_errors() -> None:
    layout = _deep_layout()

    with pytest.raises(ValueError, match="only ReLU layers support phase 'post'"):
        layout.variable_slice(0, "post")
    with pytest.raises(ValueError, match="only ReLU layers support phase 'post'"):
        layout.variable_index(NeuronRef(0, "post", 0))
    with pytest.raises(ValueError, match="phase must be one of"):
        layout.variable_slice(0, "bogus")
    with pytest.raises(ValueError, match="layer_index"):
        layout.variable_slice(3, "pre")
    with pytest.raises(ValueError, match="neuron_index"):
        layout.variable_index(NeuronRef(2, "pre", 5))
    with pytest.raises(TypeError, match="phase must be a string"):
        layout.variable_slice(0, 5)
    with pytest.raises(TypeError, match="must be a NeuronRef"):
        layout.variable_index(5)


def test_variable_layout_contiguity_validation() -> None:
    with pytest.raises(ValueError, match="expected pre_offsets"):
        VariableLayout(2, (3, 5, 11), (3, 3, 2), (None, 8, None), (None, 3, None), 13)
    with pytest.raises(ValueError, match="expected post_offsets"):
        VariableLayout(2, (2, 5, 11), (3, 3, 2), (None, 9, None), (None, 3, None), 13)
    with pytest.raises(ValueError, match="end exactly at total_variables"):
        VariableLayout(2, (2, 5, 11), (3, 3, 2), (None, 8, None), (None, 3, None), 12)
    with pytest.raises(ValueError, match="both be set for ReLU layers"):
        VariableLayout(2, (2, 5, 11), (3, 3, 2), (8, None, None), (None, 3, None), 13)
    with pytest.raises(ValueError, match="positive"):
        VariableLayout(2, (2, 5, 11), (3, 0, 2), (None, 8, None), (None, 3, None), 13)
    with pytest.raises(ValueError, match="equal lengths"):
        VariableLayout(2, (2, 5), (3, 3, 2), (None, 8, None), (None, 3, None), 13)
    with pytest.raises(TypeError, match="must be a tuple"):
        VariableLayout(2, [2, 5, 11], (3, 3, 2), (None, 8, None), (None, 3, None), 13)
    with pytest.raises(TypeError, match="must be an integer"):
        VariableLayout(True, (2, 5, 11), (3, 3, 2), (None, 8, None), (None, 3, None), 13)


# ==============================================================================
# 2. LPProblem record
# ==============================================================================


def _design_layout() -> VariableLayout:
    """Layout of the design example: 2 inputs, 1-1 Linear and ReLU layers."""
    return VariableLayout(
        input_dim=2,
        pre_offsets=(2, 3),
        pre_dims=(1, 1),
        post_offsets=(None, 4),
        post_dims=(None, 1),
        total_variables=5,
    )


def _design_problem() -> LPProblem:
    network = _design_net()
    trace = get_intermediate_bounds(network, UNIT_BOX)
    return build_lp_problem(network, UNIT_BOX, trace, NeuronRef(1, "post", 0))


def test_lp_problem_shape_and_properties() -> None:
    problem = _design_problem()

    assert problem.layout.total_variables == 5
    assert problem.num_inequality_constraints == 3
    assert problem.num_equality_constraints == 2
    assert problem.A_eq.shape == (2, 5)
    assert problem.A_ub.shape == (3, 5)
    assert problem.objective_neuron == NeuronRef(1, "post", 0)
    np.testing.assert_array_equal(problem.objective, [0.0, 0.0, 0.0, 0.0, 1.0])
    assert problem.variable_bounds == (
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-2.0, 4.0),
        (-2.0, 4.0),
        (0.0, 4.0),
    )


def test_lp_problem_arrays_are_read_only_and_frozen() -> None:
    problem = _design_problem()

    assert not problem.A_eq.flags.writeable
    assert not problem.b_eq.flags.writeable
    assert not problem.objective.flags.writeable
    with pytest.raises(FrozenInstanceError):
        problem.objective = np.zeros(5)


def test_lp_problem_validation_errors() -> None:
    layout = _design_layout()
    neuron = NeuronRef(1, "post", 0)
    objective = layout.objective_vector(neuron)
    bounds = ((-1.0, 1.0),) * 5
    A_eq = np.array([[-1.0, 2.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0]])
    b_eq = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="objective must have length"):
        LPProblem(np.zeros(4), neuron, layout, bounds, A_eq=A_eq, b_eq=b_eq)
    with pytest.raises(ValueError, match="unit vector selecting objective_neuron"):
        LPProblem(np.ones(5), neuron, layout, bounds, A_eq=A_eq, b_eq=b_eq)
    with pytest.raises(ValueError, match="variable_bounds must have"):
        LPProblem(objective, neuron, layout, bounds[:4], A_eq=A_eq, b_eq=b_eq)
    with pytest.raises(TypeError, match="variable_bounds must be a tuple"):
        LPProblem(objective, neuron, layout, list(bounds), A_eq=A_eq, b_eq=b_eq)
    with pytest.raises(ValueError, match="must both be provided or both be None"):
        LPProblem(objective, neuron, layout, bounds, A_eq=A_eq, b_eq=None)
    with pytest.raises(ValueError, match="must have 5 columns"):
        LPProblem(
            objective,
            neuron,
            layout,
            bounds,
            A_eq=np.ones((1, 4)),
            b_eq=np.zeros(1),
        )
    with pytest.raises(ValueError, match="must be finite"):
        LPProblem(
            objective,
            neuron,
            layout,
            bounds,
            A_eq=A_eq,
            b_eq=np.array([np.inf, 0.0]),
        )
    with pytest.raises(ValueError, match="must contain only finite values"):
        LPProblem(np.full(5, np.inf), neuron, layout, bounds)
    with pytest.raises(TypeError, match="must be a NeuronRef"):
        LPProblem(objective, 5, layout, bounds)
    with pytest.raises(TypeError, match="must be a VariableLayout"):
        LPProblem(objective, neuron, 5, bounds)


# ==============================================================================
# 3. LPConstraintBuilder
# ==============================================================================


def test_builder_linear_rows() -> None:
    layout = VariableLayout(2, (2,), (1,), (None,), (None,), 3)
    builder = LPConstraintBuilder(3)
    builder.add_linear_layer(
        np.array([[1.0, -2.0]]), np.array([1.0]), slice(0, 2), slice(2, 3)
    )

    problem = builder.build(
        np.array([0.0, 0.0, 1.0]), NeuronRef(0, "pre", 0), layout
    )

    assert problem.A_ub is None
    assert problem.num_equality_constraints == 1
    np.testing.assert_allclose(problem.A_eq, [[-1.0, 2.0, 1.0]], rtol=0, atol=0)
    np.testing.assert_allclose(problem.b_eq, [1.0], rtol=0, atol=0)


def test_builder_relu_stable_inactive_rows() -> None:
    layout = VariableLayout(1, (1,), (1,), (2,), (1,), 3)
    builder = LPConstraintBuilder(3)
    builder.add_relu_layer(
        slice(0, 1), slice(1, 2), slice(2, 3), np.array([-3.0]), np.array([-1.0])
    )

    problem = builder.build(
        np.array([0.0, 0.0, 1.0]), NeuronRef(0, "post", 0), layout
    )

    assert problem.A_ub is None
    np.testing.assert_allclose(
        problem.A_eq, [[-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rtol=0, atol=0
    )


def test_builder_relu_stable_active_rows() -> None:
    layout = VariableLayout(1, (1,), (1,), (2,), (1,), 3)
    builder = LPConstraintBuilder(3)
    builder.add_relu_layer(
        slice(0, 1), slice(1, 2), slice(2, 3), np.array([1.0]), np.array([3.0])
    )

    problem = builder.build(
        np.array([0.0, 0.0, 1.0]), NeuronRef(0, "post", 0), layout
    )

    assert problem.A_ub is None
    np.testing.assert_allclose(
        problem.A_eq, [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], rtol=0, atol=0
    )


def test_builder_relu_unstable_rows() -> None:
    layout = VariableLayout(1, (1,), (1,), (2,), (1,), 3)
    builder = LPConstraintBuilder(3)
    builder.add_relu_layer(
        slice(0, 1), slice(1, 2), slice(2, 3), np.array([-3.0]), np.array([2.0])
    )

    problem = builder.build(
        np.array([0.0, 0.0, 1.0]), NeuronRef(0, "post", 0), layout
    )

    assert problem.num_equality_constraints == 1
    assert problem.num_inequality_constraints == 3
    np.testing.assert_allclose(problem.A_eq, [[-1.0, 1.0, 0.0]], rtol=0, atol=0)
    np.testing.assert_allclose(
        problem.A_ub, [[0.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, -2.0, 5.0]], rtol=0, atol=0
    )
    np.testing.assert_allclose(problem.b_ub, [0.0, 0.0, 6.0], rtol=0, atol=0)


def test_builder_relu_validation_errors() -> None:
    layout = VariableLayout(1, (1,), (1,), (2,), (1,), 3)

    builder = LPConstraintBuilder(3)
    with pytest.raises(ValueError, match="must not exceed"):
        builder.add_relu_layer(
            slice(0, 1), slice(1, 2), slice(2, 3), np.array([2.0]), np.array([-1.0])
        )

    builder = LPConstraintBuilder(3)
    with pytest.raises(ValueError, match="same number of neurons"):
        builder.add_relu_layer(
            slice(0, 2), slice(1, 2), slice(2, 3), np.array([0.0]), np.array([1.0])
        )

    builder = LPConstraintBuilder(3)
    with pytest.raises(ValueError, match="requires finite bounds"):
        builder.add_relu_layer(
            slice(0, 1),
            slice(1, 2),
            slice(2, 3),
            np.array([-np.inf]),
            np.array([np.inf]),
        )


def test_builder_box_bounds_infinite_mapping() -> None:
    layout = VariableLayout(2, (2,), (1,), (None,), (None,), 3)
    builder = LPConstraintBuilder(3)
    builder.add_box_bounds(slice(0, 2), [-np.inf, -1.0], [0.0, np.inf])

    problem = builder.build(
        np.array([0.0, 0.0, 1.0]), NeuronRef(0, "pre", 0), layout
    )

    assert problem.variable_bounds[0] == (None, 0.0)
    assert problem.variable_bounds[1] == (-1.0, None)


def test_builder_box_bounds_validation_errors() -> None:
    builder = LPConstraintBuilder(4)
    builder.add_box_bounds(slice(0, 2), [-1.0, -1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="already assigned"):
        builder.add_box_bounds(slice(1, 3), [-1.0, -1.0], [1.0, 1.0])

    with pytest.raises(ValueError, match="cannot contain NaN"):
        builder.add_box_bounds(slice(2, 3), [np.nan], [1.0])
    with pytest.raises(ValueError, match="must have 1 entries"):
        builder.add_box_bounds(slice(2, 3), [-1.0, -1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="infinite bounds must be"):
        builder.add_box_bounds(slice(2, 3), [np.inf], [np.inf])
    with pytest.raises(ValueError, match="step 1"):
        builder.add_box_bounds(slice(2, 4, 2), [-1.0], [1.0])
    with pytest.raises(ValueError, match="out of bounds"):
        builder.add_box_bounds(slice(2, 5), [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])


def test_builder_build_column_mismatch() -> None:
    builder = LPConstraintBuilder(3)
    with pytest.raises(ValueError, match="builder was created for"):
        builder.build(np.zeros(5), NeuronRef(1, "post", 0), _design_layout())


# ==============================================================================
# 4. build_lp_problem: hand-computed Linear-ReLU example
# ==============================================================================


def test_design_example_full_problem() -> None:
    network = _design_net()
    trace = get_intermediate_bounds(network, UNIT_BOX)
    problem = build_lp_problem(network, UNIT_BOX, trace, NeuronRef(1, "post", 0))

    np.testing.assert_allclose(
        problem.A_eq, [[-1.0, 2.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0]],
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(problem.b_eq, [1.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(
        problem.A_ub,
        [
            [0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, -4.0, 6.0],
        ],
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(problem.b_ub, [0.0, 0.0, 8.0], rtol=0, atol=0)
    assert problem.variable_bounds == (
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-2.0, 4.0),
        (-2.0, 4.0),
        (0.0, 4.0),
    )
    np.testing.assert_array_equal(problem.objective, [0.0, 0.0, 0.0, 0.0, 1.0])


def test_design_example_concrete_executions_satisfy() -> None:
    network = _design_net()
    trace = get_intermediate_bounds(network, UNIT_BOX)
    problem = build_lp_problem(network, UNIT_BOX, trace, NeuronRef(1, "post", 0))

    # Hand-computed corner case: x = (0.5, -0.25) -> z = 2.0 -> y = 2.0.
    np.testing.assert_allclose(
        _concrete_variables(network, np.array([0.5, -0.25])),
        [0.5, -0.25, 2.0, 2.0, 2.0],
        rtol=0, atol=1e-12,
    )

    corners = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.5, -0.25],
        ]
    )
    for x in corners:
        _assert_concrete_execution_satisfies(problem, network, x)

    for x in sample_region_inputs(UNIT_REGION, 50, seed=11):
        _assert_concrete_execution_satisfies(problem, network, x)


# ==============================================================================
# 5. Stable ReLU encodings
# ==============================================================================


def test_stable_active_relu_is_exact() -> None:
    network = _stable_active_net()
    region = InputRegion(center=np.array([0.5, 0.5]), radius=0.5)
    trace = get_intermediate_bounds(network, region)
    problem = build_lp_problem(network, region, trace, NeuronRef(1, "post", 0))

    assert problem.A_ub is None
    np.testing.assert_allclose(
        problem.A_eq,
        [
            [-1.0, -1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 1.0],
        ],
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(problem.b_eq, [1.0, 0.0, 0.0], rtol=0, atol=0)
    assert problem.variable_bounds[4] == (1.0, 3.0)

    for x in sample_region_inputs(region, 30, seed=5):
        _assert_concrete_execution_satisfies(problem, network, x)


def test_stable_inactive_relu_is_exact() -> None:
    network = _stable_inactive_net()
    region = InputRegion(center=np.array([0.5, 0.5]), radius=0.5)
    trace = get_intermediate_bounds(network, region)
    problem = build_lp_problem(network, region, trace, NeuronRef(1, "post", 0))

    assert problem.A_ub is None
    np.testing.assert_allclose(
        problem.A_eq,
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(problem.b_eq, [-1.0, 0.0, 0.0], rtol=0, atol=0)
    assert problem.variable_bounds[2] == (-3.0, -1.0)
    assert problem.variable_bounds[4] == (0.0, 0.0)

    for x in sample_region_inputs(region, 30, seed=6):
        _assert_concrete_execution_satisfies(problem, network, x)


@pytest.mark.parametrize(
    ("box", "expected_y_row", "expected_z_bounds", "expected_y_bounds"),
    [
        # Zero-width positive pre bounds -> stable active.
        (
            Box(np.array([0.5, -1.0]), np.array([0.5, 1.0])),
            [0.0, 0.0, 0.0, -1.0, 1.0],
            (0.5, 0.5),
            (0.5, 0.5),
        ),
        # Zero-width negative pre bounds -> stable inactive.
        (
            Box(np.array([-0.5, -1.0]), np.array([-0.5, 1.0])),
            [0.0, 0.0, 0.0, 0.0, 1.0],
            (-0.5, -0.5),
            (0.0, 0.0),
        ),
        # Zero-width zero pre bounds -> inactive branch pins y = 0.
        (
            Box(np.array([0.0, -1.0]), np.array([0.0, 1.0])),
            [0.0, 0.0, 0.0, 0.0, 1.0],
            (0.0, 0.0),
            (0.0, 0.0),
        ),
    ],
)
def test_zero_width_interval_classification(
    box: Box,
    expected_y_row: list[float],
    expected_z_bounds: tuple[float, float],
    expected_y_bounds: tuple[float, float],
) -> None:
    network = _identity_net()
    trace = get_intermediate_bounds(network, box)
    problem = build_lp_problem(network, box, trace, NeuronRef(1, "post", 0))

    assert problem.A_ub is None
    np.testing.assert_allclose(
        problem.A_eq,
        [[-1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0], expected_y_row],
        rtol=0, atol=0,
    )
    assert problem.variable_bounds[2] == expected_z_bounds
    assert problem.variable_bounds[4] == expected_y_bounds

    concrete_x = np.array([box.lower[0], 0.0])
    _assert_concrete_execution_satisfies(problem, network, concrete_x)


# ==============================================================================
# 6. Unstable ReLU relaxation
# ==============================================================================


def test_unstable_relu_triangle_rows() -> None:
    network = _identity_net()
    box = Box(np.array([-3.0, -1.0]), np.array([2.0, 1.0]))
    trace = get_intermediate_bounds(network, box)
    problem = build_lp_problem(network, box, trace, NeuronRef(1, "post", 0))

    # l = -3, u = 2: y <= (2/5)(z + 3)  <=>  -2 z + 5 y <= 6.
    np.testing.assert_allclose(
        problem.A_eq,
        [[-1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0]],
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        problem.A_ub,
        [
            [0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, -2.0, 5.0],
        ],
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(problem.b_ub, [0.0, 0.0, 6.0], rtol=0, atol=0)
    assert problem.variable_bounds[2] == (-3.0, 2.0)
    assert problem.variable_bounds[4] == (0.0, 2.0)

    # The chord endpoints (l, 0) and (u, u) are themselves concrete executions.
    for x in (np.array([-3.0, 0.0]), np.array([2.0, 0.0]), np.array([0.5, 0.0])):
        _assert_concrete_execution_satisfies(problem, network, x)


# ==============================================================================
# 7. Input clipping and variable bounds
# ==============================================================================


def test_input_region_clipping_applies_to_variable_bounds() -> None:
    network = _identity_net()
    region = InputRegion(
        center=np.array([0.0, 0.0]),
        radius=10.0,
        input_lower=[-1.0, -1.0],
        input_upper=[1.0, 1.0],
    )
    trace = get_intermediate_bounds(network, region)
    problem = build_lp_problem(network, region, trace, NeuronRef(0, "pre", 0))

    assert problem.variable_bounds[0] == (-1.0, 1.0)
    assert problem.variable_bounds[1] == (-1.0, 1.0)


def test_input_region_without_limits() -> None:
    network = _identity_net()
    region = InputRegion(center=np.array([0.5, 0.1]), radius=0.05)
    trace = get_intermediate_bounds(network, region)
    problem = build_lp_problem(network, region, trace, NeuronRef(0, "pre", 0))

    assert problem.variable_bounds[0] == pytest.approx((0.45, 0.55))
    assert problem.variable_bounds[1] == pytest.approx((0.05, 0.15))


def test_box_input_bounds_accepted() -> None:
    network = _identity_net()
    box = Box(np.array([0.0, -1.0]), np.array([2.0, 1.0]))
    trace = get_intermediate_bounds(network, box)
    problem = build_lp_problem(network, box, trace, NeuronRef(1, "post", 0))

    assert problem.variable_bounds[0] == (0.0, 2.0)
    for x in (np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, -1.0])):
        _assert_concrete_execution_satisfies(problem, network, x)


def test_infinite_stable_bounds_accepted() -> None:
    network = _identity_net()
    box = Box(np.array([-np.inf, -1.0]), np.array([0.0, 1.0]))
    trace = get_intermediate_bounds(network, box)
    problem = build_lp_problem(network, box, trace, NeuronRef(1, "post", 0))

    assert problem.variable_bounds[0] == (None, 0.0)
    assert problem.variable_bounds[2] == (None, 0.0)
    assert problem.variable_bounds[4] == (0.0, 0.0)
    for x in (np.array([-5.0, 0.0]), np.array([0.0, 0.0]), np.array([-0.5, 1.0])):
        _assert_concrete_execution_satisfies(problem, network, x)


def test_infinite_unstable_bounds_rejected() -> None:
    network = _identity_net()
    box = Box(np.array([-np.inf, -1.0]), np.array([np.inf, 1.0]))
    trace = get_intermediate_bounds(network, box)
    with pytest.raises(ValueError, match="triangle relaxation requires finite"):
        build_lp_problem(network, box, trace, NeuronRef(1, "post", 0))


# ==============================================================================
# 8. Neuron indexing and objectives
# ==============================================================================


def test_objective_positions_follow_phase_convention() -> None:
    network = _design_net()
    trace = get_intermediate_bounds(network, UNIT_BOX)

    pre_linear = build_lp_problem(network, UNIT_BOX, trace, NeuronRef(0, "pre", 0))
    np.testing.assert_array_equal(pre_linear.objective, [0.0, 0.0, 1.0, 0.0, 0.0])

    pre_relu = build_lp_problem(network, UNIT_BOX, trace, NeuronRef(1, "pre", 0))
    np.testing.assert_array_equal(pre_relu.objective, [0.0, 0.0, 0.0, 1.0, 0.0])

    post_relu = build_lp_problem(network, UNIT_BOX, trace, NeuronRef(1, "post", 0))
    np.testing.assert_array_equal(post_relu.objective, [0.0, 0.0, 0.0, 0.0, 1.0])


def test_target_neuron_validation() -> None:
    network = _design_net()
    trace = get_intermediate_bounds(network, UNIT_BOX)

    with pytest.raises(ValueError, match="only support 'pre' phase"):
        build_lp_problem(network, UNIT_BOX, trace, NeuronRef(0, "post", 0))
    with pytest.raises(ValueError, match="neuron_index"):
        build_lp_problem(network, UNIT_BOX, trace, NeuronRef(0, "pre", 7))
    with pytest.raises(ValueError, match="layer_index"):
        build_lp_problem(network, UNIT_BOX, trace, NeuronRef(5, "pre", 0))
    with pytest.raises(TypeError, match="must be a NeuronRef"):
        build_lp_problem(network, UNIT_BOX, trace, "neuron")


def test_relu_first_layer_couples_to_input() -> None:
    network = _make_network(
        (ReLULayer(position=0), _make_linear(1, np.array([[1.0]]), np.zeros(1))),
        source_model="ReluFirst",
    )
    region = InputRegion(center=np.array([0.0]), radius=1.0)
    trace = get_intermediate_bounds(network, region)
    problem = build_lp_problem(network, region, trace, NeuronRef(1, "pre", 0))

    # Variables: x0, z0, y0, z1. The ReLU input is x0 itself, so the coupling
    # row is z0 - x0 == 0 and the linear row is z1 - y0 == 0.
    np.testing.assert_allclose(
        problem.A_eq, [[-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0]], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        problem.A_ub,
        [[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, -1.0, 2.0, 0.0]],
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(problem.b_ub, [0.0, 0.0, 1.0], rtol=0, atol=0)
    for x in (np.array([-1.0]), np.array([0.0]), np.array([1.0]), np.array([0.25])):
        _assert_concrete_execution_satisfies(problem, network, x)


def test_linear_only_network_has_no_inequalities() -> None:
    network = _linear_only_net()
    region = InputRegion(center=np.array([0.5, 0.1]), radius=0.05)
    trace = get_intermediate_bounds(network, region)
    problem = build_lp_problem(network, region, trace, NeuronRef(0, "pre", 0))

    assert problem.A_ub is None and problem.b_ub is None
    assert problem.layout.total_variables == 4
    np.testing.assert_allclose(
        problem.A_eq, [[-1.0, 1.0, 1.0, 0.0], [1.0, -1.0, 0.0, 1.0]], rtol=0, atol=0
    )
    np.testing.assert_allclose(problem.b_eq, [0.0, 0.0], rtol=0, atol=0)
    for x in sample_region_inputs(region, 30, seed=2):
        _assert_concrete_execution_satisfies(problem, network, x)


# ==============================================================================
# 9. Validation: dimensions, traces, and unsupported layers
# ==============================================================================


def test_input_dimension_mismatch() -> None:
    network = _design_net()
    region = InputRegion(center=np.array([0.0]), radius=0.1)
    with pytest.raises(ValueError, match="does not match network.input_features"):
        build_lp_problem(network, region, None, NeuronRef(0, "pre", 0))


def test_trace_layer_count_mismatch() -> None:
    design_net = _design_net()
    single_layer_net = _make_network(
        (_make_linear(0, np.array([[1.0, -2.0]]), np.array([1.0])),)
    )
    trace = get_intermediate_bounds(single_layer_net, UNIT_REGION)
    with pytest.raises(ValueError, match="layer entries"):
        build_lp_problem(design_net, UNIT_REGION, trace, NeuronRef(0, "pre", 0))


def test_trace_layer_type_mismatch() -> None:
    design_net = _design_net()
    double_linear_net = _make_network(
        (
            _make_linear(0, np.array([[1.0, -2.0]]), np.array([1.0])),
            _make_linear(1, np.array([[1.0]]), np.array([0.0])),
        )
    )
    trace = get_intermediate_bounds(double_linear_net, UNIT_REGION)
    with pytest.raises(ValueError, match="layer_type"):
        build_lp_problem(design_net, UNIT_REGION, trace, NeuronRef(0, "pre", 0))


def test_trace_dimension_mismatch() -> None:
    network = _make_network(
        (
            _make_linear(0, np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2)),
            ReLULayer(position=1),
        )
    )
    trace = IntermediateBounds(
        input_bounds=UNIT_BOX,
        layer_bounds=(
            LayerBounds(
                layer_index=0,
                layer_type="linear",
                input_bounds=UNIT_BOX,
                pre_activation_bounds=Box(np.array([-1.0]), np.array([1.0])),
                post_activation_bounds=None,
                output_bounds=UNIT_BOX,
            ),
            LayerBounds(
                layer_index=1,
                layer_type="relu",
                input_bounds=UNIT_BOX,
                pre_activation_bounds=UNIT_BOX,
                post_activation_bounds=Box(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
                output_bounds=Box(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            ),
        ),
        output_bounds=Box(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    )
    with pytest.raises(ValueError, match="pre-activation dimension"):
        build_lp_problem(network, UNIT_REGION, trace, NeuronRef(0, "pre", 0))


def test_trace_region_containment_enforced() -> None:
    network = _identity_net()
    trace = get_intermediate_bounds(network, UNIT_REGION)
    wide_region = InputRegion(center=np.array([0.0, 0.0]), radius=2.0)
    with pytest.raises(ValueError, match="must contain the input region"):
        build_lp_problem(network, wide_region, trace, NeuronRef(0, "pre", 0))


class _UnsupportedLayer:
    position = 1


def test_unsupported_layer_rejected() -> None:
    network = _make_network(
        (
            _make_linear(0, np.array([[1.0, -2.0]]), np.array([1.0])),
            _UnsupportedLayer(),
        ),
        source_model="UnsupportedNet",
    )
    trace = IntermediateBounds(
        input_bounds=UNIT_BOX,
        layer_bounds=(
            LayerBounds(
                layer_index=0,
                layer_type="linear",
                input_bounds=UNIT_BOX,
                pre_activation_bounds=Box(np.array([-2.0]), np.array([4.0])),
                post_activation_bounds=None,
                output_bounds=Box(np.array([-2.0]), np.array([4.0])),
            ),
            LayerBounds(
                layer_index=1,
                layer_type="linear",
                input_bounds=Box(np.array([-2.0]), np.array([4.0])),
                pre_activation_bounds=Box(np.array([-2.0]), np.array([4.0])),
                post_activation_bounds=None,
                output_bounds=Box(np.array([-2.0]), np.array([4.0])),
            ),
        ),
        output_bounds=Box(np.array([-2.0]), np.array([4.0])),
    )
    with pytest.raises(TypeError, match="Unsupported layer type '_UnsupportedLayer'"):
        build_lp_problem(network, UNIT_REGION, trace, NeuronRef(0, "pre", 0))


def test_type_validation_errors() -> None:
    network = _design_net()
    trace = get_intermediate_bounds(network, UNIT_REGION)

    with pytest.raises(TypeError, match="network must be a MathematicalNetwork"):
        build_lp_problem(object(), UNIT_REGION, trace, NeuronRef(0, "pre", 0))
    with pytest.raises(TypeError, match="must satisfy BoxProtocol"):
        build_lp_problem(network, "not a box", trace, NeuronRef(0, "pre", 0))
    with pytest.raises(TypeError, match="must be an IntermediateBounds"):
        build_lp_problem(network, UNIT_REGION, object(), NeuronRef(0, "pre", 0))
    with pytest.raises(TypeError, match="must be a NeuronRef"):
        build_lp_problem(network, UNIT_REGION, trace, object())


# ==============================================================================
# 10. Soundness sweep over a deeper randomized network
# ==============================================================================


def test_deep_network_soundness_sweep() -> None:
    rng = np.random.default_rng(7)
    network = _make_network(
        (
            _make_linear(0, rng.standard_normal((4, 2)), rng.standard_normal(4)),
            ReLULayer(position=1),
            _make_linear(2, rng.standard_normal((3, 4)), rng.standard_normal(3)),
            ReLULayer(position=3),
            _make_linear(4, rng.standard_normal((2, 3)), rng.standard_normal(2)),
        ),
        source_model="DeepRandomNet",
    )
    region = UNIT_REGION
    trace = get_intermediate_bounds(network, region)
    samples = sample_region_inputs(region, 100, seed=3)

    assert trace.layer_bounds[1].layer_type == "relu"
    assert trace.layer_bounds[3].layer_type == "relu"

    for layer_index in (2, 4):
        width = network.layers[layer_index].output_features
        for neuron_index in range(width):
            problem = build_lp_problem(
                network, region, trace, NeuronRef(layer_index, "pre", neuron_index)
            )
            assert problem.layout.total_variables == 25
            for x in samples:
                _assert_concrete_execution_satisfies(problem, network, x)
