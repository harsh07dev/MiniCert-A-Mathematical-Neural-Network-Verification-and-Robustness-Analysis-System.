"""Comprehensive unit and property tests for the vector Box abstract domain."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from abstract_domain.box import Box
from tests.test_verification_pipeline import _ibp_linear
from verification.regions import BoxProtocol


# ==============================================================================
# 1. Construction, Types, and Coercion
# ==============================================================================


def test_normal_vector_box_construction() -> None:
    box = Box(np.array([1.0, 2.0]), np.array([3.0, 4.0]))

    assert isinstance(box.lower, np.ndarray)
    assert isinstance(box.upper, np.ndarray)
    assert box.dimension == 2
    assert len(box) == 2
    np.testing.assert_array_equal(box.lower, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(box.upper, np.array([3.0, 4.0]))


def test_python_list_and_tuple_inputs() -> None:
    box_list = Box([0.0, -1.0], [1.0, 2.0])
    box_tuple = Box((0.0, -1.0), (1.0, 2.0))

    assert box_list.lower.dtype == np.float64
    assert box_list.upper.dtype == np.float64
    assert box_list == box_tuple


def test_float64_conversion_from_integer_and_float32() -> None:
    int_box = Box(np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int64))
    float32_box = Box(np.array([0.5, 1.5], dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32))

    assert int_box.lower.dtype == np.float64
    assert int_box.upper.dtype == np.float64
    assert float32_box.lower.dtype == np.float64
    assert float32_box.upper.dtype == np.float64


def test_non_convertible_types_rejected() -> None:
    with pytest.raises(TypeError, match="convertible to float64"):
        Box(["not", "numbers"], [1.0, 2.0])  # type: ignore[arg-type]


# ==============================================================================
# 2. Validation: Dimensions, Emptiness, NaN, and Ordering
# ==============================================================================


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (np.zeros((2, 2)), np.ones((2, 2))),  # 2-D arrays
        (0.5, 1.0),                            # 0-D scalars
        (np.zeros((1, 2, 3)), np.ones((1, 2, 3))),  # 3-D arrays
    ],
)
def test_invalid_dimensions_rejected(lower: object, upper: object) -> None:
    with pytest.raises(ValueError, match="1-D vector"):
        Box(lower, upper)  # type: ignore[arg-type]


def test_empty_vectors_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Box(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="must not be empty"):
        Box([], [])


def test_mismatched_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match="Bound shape mismatch"):
        Box([1.0, 2.0], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Bound shape mismatch"):
        Box(np.zeros(5), np.ones(3))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([np.nan, 0.0], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, np.nan]),
        ([np.nan], [np.nan]),
    ],
)
def test_nan_rejection(lower: list[float], upper: list[float]) -> None:
    with pytest.raises(ValueError, match="cannot contain NaN"):
        Box(lower, upper)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([2.0, 1.0], [1.0, 2.0]),        # first component inverted
        ([0.0, 5.0], [1.0, 4.0]),        # second component inverted
        ([1.0000001], [1.0]),             # slight violation
    ],
)
def test_invalid_lower_upper_ordering_rejected(lower: list[float], upper: list[float]) -> None:
    with pytest.raises(ValueError, match="Lower bound cannot exceed upper bound"):
        Box(lower, upper)


# ==============================================================================
# 3. Defensive Copying and Immutability
# ==============================================================================


def test_independent_defensive_copies() -> None:
    input_lower = np.array([1.0, 2.0])
    input_upper = np.array([3.0, 4.0])
    box = Box(input_lower, input_upper)

    # Mutate original caller arrays
    input_lower[0] = 999.0
    input_upper[1] = -999.0

    # Stored box bounds must remain unaltered
    assert box.lower[0] == 1.0
    assert box.upper[1] == 4.0


def test_read_only_array_behavior() -> None:
    box = Box([1.0, 2.0], [3.0, 4.0])

    with pytest.raises(ValueError, match="read-only"):
        box.lower[0] = 99.0

    with pytest.raises(ValueError, match="read-only"):
        box.upper[1] = 0.0


def test_frozen_dataclass_attributes() -> None:
    box = Box([1.0], [2.0])

    with pytest.raises(FrozenInstanceError):
        box.lower = np.array([0.0])  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        box.upper = np.array([5.0])  # type: ignore[misc]


def test_structural_equality() -> None:
    box1 = Box([1.0, 2.0], [3.0, 4.0])
    box2 = Box(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    box_diff = Box([1.0, 2.0], [3.0, 4.1])

    assert box1 == box2
    assert box1 != box_diff
    assert (box1 == "not a box") is False


# ==============================================================================
# 4. BoxProtocol Structural Conformance
# ==============================================================================


def test_box_protocol_structural_compatibility() -> None:
    box = Box([0.0, 1.0], [2.0, 3.0])

    assert isinstance(box, BoxProtocol)
    assert hasattr(box, "lower")
    assert hasattr(box, "upper")
    assert isinstance(box.lower, np.ndarray)
    assert isinstance(box.upper, np.ndarray)
    assert box.lower.dtype == np.float64
    assert box.upper.dtype == np.float64
    assert box.lower.ndim == 1
    assert box.upper.ndim == 1


# ==============================================================================
# 5. Affine Transformation (IBP)
# ==============================================================================


def test_affine_all_positive_weights() -> None:
    box = Box([1.0, 2.0], [3.0, 4.0])
    weight = np.array([[2.0, 1.0], [0.5, 3.0]])
    bias = np.array([0.1, -0.2])

    result = box.affine(weight, bias)
    oracle_lower, oracle_upper = _ibp_linear(box.lower, box.upper, weight, bias)

    assert isinstance(result, Box)
    assert result is not box
    np.testing.assert_allclose(result.lower, oracle_lower)
    np.testing.assert_allclose(result.upper, oracle_upper)
    # Analytical: lower = 2*1 + 1*2 + 0.1 = 4.1, upper = 2*3 + 1*4 + 0.1 = 10.1
    assert result.lower[0] == pytest.approx(4.1)
    assert result.upper[0] == pytest.approx(10.1)


def test_affine_all_negative_weights() -> None:
    box = Box([1.0, 2.0], [3.0, 4.0])
    weight = np.array([[-2.0, -1.0], [-0.5, -3.0]])
    bias = np.array([1.0, 2.0])

    result = box.affine(weight, bias)
    oracle_lower, oracle_upper = _ibp_linear(box.lower, box.upper, weight, bias)

    assert isinstance(result, Box)
    np.testing.assert_allclose(result.lower, oracle_lower)
    np.testing.assert_allclose(result.upper, oracle_upper)
    # Analytical: for W <= 0, lower uses upper: -2*3 + -1*4 + 1.0 = -9.0
    # upper uses lower: -2*1 + -1*2 + 1.0 = -3.0
    assert result.lower[0] == pytest.approx(-9.0)
    assert result.upper[0] == pytest.approx(-3.0)


def test_affine_mixed_weights() -> None:
    box = Box([-1.0, 2.0], [3.0, 5.0])
    weight = np.array([[1.5, -2.0], [-0.5, 3.0]])
    bias = np.array([0.5, -1.0])

    result = box.affine(weight, bias)
    oracle_lower, oracle_upper = _ibp_linear(box.lower, box.upper, weight, bias)

    np.testing.assert_allclose(result.lower, oracle_lower)
    np.testing.assert_allclose(result.upper, oracle_upper)


def test_affine_bias_handling() -> None:
    box = Box([0.0, 1.0], [2.0, 3.0])
    weight = np.eye(2)

    zero_bias = box.affine(weight, [0.0, 0.0])
    pos_bias = box.affine(weight, [5.0, 10.0])
    neg_bias = box.affine(weight, [-2.0, -4.0])

    np.testing.assert_array_equal(zero_bias.lower, box.lower)
    np.testing.assert_array_equal(zero_bias.upper, box.upper)

    np.testing.assert_array_equal(pos_bias.lower, box.lower + np.array([5.0, 10.0]))
    np.testing.assert_array_equal(pos_bias.upper, box.upper + np.array([5.0, 10.0]))

    np.testing.assert_array_equal(neg_bias.lower, box.lower - np.array([2.0, 4.0]))
    np.testing.assert_array_equal(neg_bias.upper, box.upper - np.array([2.0, 4.0]))


def test_affine_randomized_comparison_with_oracle() -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        in_dim = int(rng.integers(1, 15))
        out_dim = int(rng.integers(1, 10))

        c = rng.uniform(-5.0, 5.0, size=in_dim)
        r = rng.uniform(0.01, 2.0, size=in_dim)
        box = Box(c - r, c + r)

        weight = rng.normal(0.0, 2.0, size=(out_dim, in_dim))
        bias = rng.normal(0.0, 1.0, size=out_dim)

        result = box.affine(weight, bias)
        oracle_lower, oracle_upper = _ibp_linear(box.lower, box.upper, weight, bias)

        np.testing.assert_allclose(result.lower, oracle_lower)
        np.testing.assert_allclose(result.upper, oracle_upper)


@pytest.mark.parametrize(
    ("weight", "bias", "match"),
    [
        (np.ones(3), np.zeros(3), "2-D"),                            # 1D weight
        (np.ones((2, 2, 2)), np.zeros(2), "2-D"),                   # 3D weight
        (np.ones((2, 4)), np.zeros(2), "does not match Box dimension"),  # input dim mismatch (4 != 3)
        (np.ones((3, 3)), np.zeros((3, 1)), "1-D"),                 # 2D bias
        (np.ones((3, 3)), np.zeros(2), "does not match weight output"),  # bias length mismatch (2 != 3)
        (np.array([[1.0, np.nan], [0.0, 1.0]]), np.zeros(2), "cannot contain NaN"),
        (np.ones((2, 2)), np.array([np.nan, 0.0]), "cannot contain NaN"),
        (np.array([[1.0, np.inf], [0.0, 1.0]]), np.zeros(2), "must be finite"),
        (np.ones((2, 2)), np.array([0.0, -np.inf]), "must be finite"),
    ],
)
def test_affine_invalid_inputs_rejected(weight: np.ndarray, bias: np.ndarray, match: str) -> None:
    box = Box([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])  # dim = 3
    if weight.shape == (2, 2) or weight.shape == (2, 4):
        box = Box([0.0, 1.0], [1.0, 2.0])  # dim = 2
    with pytest.raises(ValueError, match=match):
        box.affine(weight, bias)


# ==============================================================================
# 6. ReLU Transformation
# ==============================================================================


def test_relu_on_strictly_positive_intervals() -> None:
    box = Box([1.0, 2.5, 0.1], [3.0, 4.0, 5.0])
    result = box.relu()

    assert isinstance(result, Box)
    assert result is not box
    np.testing.assert_array_equal(result.lower, box.lower)
    np.testing.assert_array_equal(result.upper, box.upper)


def test_relu_on_strictly_negative_intervals() -> None:
    box = Box([-5.0, -2.0, -0.5], [-1.0, -0.1, -0.01])
    result = box.relu()

    np.testing.assert_array_equal(result.lower, np.zeros(3))
    np.testing.assert_array_equal(result.upper, np.zeros(3))


def test_relu_on_zero_crossing_intervals() -> None:
    box = Box([-3.0, -0.5, 2.0], [2.0, 0.0, 4.0])
    result = box.relu()

    # Lower is clamped to 0.0; upper >= 0 is preserved
    expected_lower = np.array([0.0, 0.0, 2.0])
    expected_upper = np.array([2.0, 0.0, 4.0])

    np.testing.assert_array_equal(result.lower, expected_lower)
    np.testing.assert_array_equal(result.upper, expected_upper)


def test_relu_exact_zero_bounds() -> None:
    box = Box([0.0, 0.0], [0.0, 0.0])
    result = box.relu()

    np.testing.assert_array_equal(result.lower, np.zeros(2))
    np.testing.assert_array_equal(result.upper, np.zeros(2))


# ==============================================================================
# 7. Infinite Bounds and Numerical Policy
# ==============================================================================


def test_infinite_lower_and_upper_bounds_supported() -> None:
    box = Box([-math.inf, 0.0, -math.inf], [1.0, math.inf, math.inf])

    assert box.lower[0] == -np.inf
    assert box.upper[1] == np.inf
    assert box.lower[2] == -np.inf
    assert box.upper[2] == np.inf


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([math.inf], [math.inf]),       # [+inf, +inf] rejected
        ([-math.inf], [-math.inf]),     # [-inf, -inf] rejected
        ([math.inf], [0.0]),            # lower is +inf
        ([0.0], [-math.inf]),           # upper is -inf
    ],
)
def test_disallowed_infinite_bounds_rejected(lower: list[float], upper: list[float]) -> None:
    with pytest.raises(ValueError, match=r"Infinite bounds must be -inf below or \+inf above"):
        Box(lower, upper)


def test_affine_with_infinite_bounds_and_zero_weight_convention() -> None:
    # Test project convention 0 * inf = 0:
    # y = [1.0, 0.0] @ [-inf, 2.0]^T + 0.0 = -inf + 0 = -inf (not NaN!)
    box = Box([-np.inf, 1.0], [2.0, np.inf])
    weight = np.array([[1.0, 0.0], [0.0, 2.0]])
    bias = np.array([0.0, 0.0])

    result = box.affine(weight, bias)

    assert result.lower[0] == -np.inf
    assert result.upper[0] == 2.0
    assert result.lower[1] == 2.0
    assert result.upper[1] == np.inf


def test_affine_flips_infinite_bounds_with_negative_weight() -> None:
    # y = -2 * [-inf, 3] + 1 = lower is -2 * 3 + 1 = -5, upper is -2 * (-inf) + 1 = +inf
    box = Box([-np.inf], [3.0])
    result = box.affine(np.array([[-2.0]]), np.array([1.0]))

    assert result.lower[0] == -5.0
    assert result.upper[0] == np.inf


def test_relu_with_infinite_bounds() -> None:
    box = Box([-np.inf, -np.inf, 2.0], [5.0, np.inf, np.inf])
    result = box.relu()

    # max(0, -inf) = 0.0; max(0, +inf) = +inf
    np.testing.assert_array_equal(result.lower, np.array([0.0, 0.0, 2.0]))
    np.testing.assert_array_equal(result.upper, np.array([5.0, np.inf, np.inf]))


# ==============================================================================
# 8. Point Containment and Helpers
# ==============================================================================


def test_contains_valid_and_invalid_points() -> None:
    box = Box([0.0, 1.0], [2.0, 3.0])

    assert box.contains([1.0, 2.0]) is True
    assert box.contains([0.0, 1.0]) is True  # corner
    assert box.contains([2.0, 3.0]) is True  # corner
    assert box.contains([-0.1, 2.0]) is False
    assert box.contains([1.0, 3.1]) is False

    with pytest.raises(ValueError, match="does not match Box shape"):
        box.contains([1.0])

    with pytest.raises(ValueError, match="cannot contain NaN"):
        box.contains([np.nan, 2.0])
