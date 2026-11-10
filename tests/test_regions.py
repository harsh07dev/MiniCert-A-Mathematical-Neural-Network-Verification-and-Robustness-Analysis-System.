import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from verification.regions import BoxProtocol, InputRegion


def _region(center=(0.4, 0.5), radius=0.2, lower=None, upper=None) -> InputRegion:
    return InputRegion(np.asarray(center, dtype=np.float64), radius, lower, upper)


def test_epsilon_zero_produces_singleton_region() -> None:
    center = np.array([0.4, 0.5], dtype=np.float64)
    region = InputRegion(center, 0.0, 0.0, 1.0)

    np.testing.assert_array_equal(region.lower, center)
    np.testing.assert_array_equal(region.upper, center)
    assert region.dimension == 2


def test_positive_epsilon_produces_valid_bounds() -> None:
    region = _region(center=(0.4, 0.5), radius=0.2)

    np.testing.assert_allclose(region.lower, np.array([0.2, 0.3]))
    np.testing.assert_allclose(region.upper, np.array([0.6, 0.7]))


def test_bounds_are_clipped_to_unit_domain() -> None:
    region = _region(center=(0.05, 0.95), radius=0.2, lower=0.0, upper=1.0)

    np.testing.assert_allclose(region.lower, np.array([0.0, 0.75]))
    np.testing.assert_allclose(region.upper, np.array([0.25, 1.0]))


def test_bounds_are_clipped_to_custom_domain() -> None:
    region = _region(center=(-0.9, 2.1), radius=0.5, lower=(-1.0, -1.0), upper=(1.0, 3.0))

    np.testing.assert_allclose(region.lower, np.array([-1.0, 1.6]))
    np.testing.assert_allclose(region.upper, np.array([-0.4, 2.6]))


def test_default_domain_is_unbounded() -> None:
    region = _region(center=(0.4,), radius=0.2)

    assert np.all(np.isneginf(region.input_lower))
    assert np.all(np.isposinf(region.input_upper))
    np.testing.assert_allclose(region.lower, np.array([0.2]))
    np.testing.assert_allclose(region.upper, np.array([0.6]))


@pytest.mark.parametrize("radius", [-1.0, -1e-12])
def test_negative_epsilon_is_rejected(radius: float) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        InputRegion(np.array([0.4]), radius)


@pytest.mark.parametrize("radius", [math.nan, float("nan")])
def test_nan_epsilon_is_rejected(radius: float) -> None:
    with pytest.raises(ValueError, match="NaN"):
        InputRegion(np.array([0.4]), radius)


@pytest.mark.parametrize("radius", [math.inf, -math.inf])
def test_infinite_epsilon_is_rejected(radius: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        InputRegion(np.array([0.4]), radius)


def test_non_scalar_epsilon_is_rejected() -> None:
    with pytest.raises(TypeError, match="real scalar"):
        InputRegion(np.array([0.4]), "0.1")


@pytest.mark.parametrize("center", [np.array([0.0, math.nan]), np.array([math.nan])])
def test_nan_center_is_rejected(center: np.ndarray) -> None:
    with pytest.raises(ValueError, match="NaN"):
        InputRegion(center, 0.1)


@pytest.mark.parametrize("center", [np.array([0.0, math.inf]), np.array([-math.inf])])
def test_infinite_center_is_rejected(center: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        InputRegion(center, 0.1)


def test_multidimensional_center_is_rejected() -> None:
    with pytest.raises(ValueError, match="1-D"):
        InputRegion(np.zeros((2, 2)), 0.1)


def test_empty_center_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        InputRegion(np.array([]), 0.1)


@pytest.mark.parametrize(
    ("center", "lower", "upper"),
    [
        (np.array([0.4]), 0.5, None),
        (np.array([0.6]), None, 0.5),
        (np.array([0.4, 0.4]), (0.0, 0.6), None),
        (np.array([0.4, 0.4]), None, (1.0, 0.2)),
    ],
)
def test_center_outside_configured_domain_is_rejected(
    center: np.ndarray, lower: object, upper: object
) -> None:
    with pytest.raises(ValueError, match="within the configured input bounds"):
        InputRegion(center, 0.1, lower, upper)


def test_reversed_input_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="never exceed"):
        InputRegion(np.array([0.4]), 0.1, 0.9, 0.1)


def test_nan_input_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="NaN"):
        InputRegion(np.array([0.4]), 0.1, np.array([math.nan]), None)


def test_mismatched_input_bound_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match input shape"):
        InputRegion(np.array([0.4, 0.4]), 0.1, np.array([0.0]), None)


def test_infinite_input_bounds_are_allowed() -> None:
    region = InputRegion(
        np.array([0.4]), 0.1, np.array([-math.inf]), np.array([math.inf])
    )

    assert region.lower[0] == pytest.approx(0.3)
    assert region.upper[0] == 0.5


def test_shape_is_preserved() -> None:
    center = np.full(784, 0.4, dtype=np.float64)
    region = InputRegion(center, 0.2, 0.0, 1.0)

    assert region.center.shape == (784,)
    assert region.lower.shape == (784,)
    assert region.upper.shape == (784,)
    assert region.input_lower.shape == (784,)
    assert region.input_upper.shape == (784,)
    assert region.dimension == 784


def test_bound_dtype_is_float64() -> None:
    region = _region()

    assert region.center.dtype == np.float64
    assert region.lower.dtype == np.float64
    assert region.upper.dtype == np.float64
    assert region.input_lower.dtype == np.float64
    assert region.input_upper.dtype == np.float64


def test_lower_bounds_never_exceed_upper_bounds() -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        center = rng.uniform(0.0, 1.0, size=32)
        radius = float(rng.uniform(0.0, 0.5))
        region = InputRegion(center, radius, 0.0, 1.0)

        assert np.all(region.lower <= region.upper)


def test_region_satisfies_box_protocol() -> None:
    region = _region()

    assert isinstance(region, BoxProtocol)
    assert isinstance(region.lower, np.ndarray)
    assert isinstance(region.upper, np.ndarray)


def test_contains_inside_boundary_and_outside() -> None:
    region = _region(center=(0.5,), radius=0.1, lower=0.0, upper=1.0)

    assert region.contains(np.array([0.55]))
    assert region.contains(np.array([0.4]))
    assert region.contains(np.array([0.6]))
    assert not region.contains(np.array([0.39]))
    assert not region.contains(np.array([0.61]))


def test_contains_uses_clipped_bounds() -> None:
    region = _region(center=(0.05,), radius=0.2, lower=0.0, upper=1.0)

    assert region.contains(np.array([0.0]))
    assert not region.contains(np.array([-0.01]))


def test_contains_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match region shape"):
        _region(center=(0.5, 0.5)).contains(np.array([0.5]))


def test_contains_rejects_nan_point() -> None:
    with pytest.raises(ValueError, match="NaN"):
        _region().contains(np.array([math.nan, 0.5]))


def test_center_is_copied_from_caller_array() -> None:
    center = np.array([0.4], dtype=np.float64)
    region = InputRegion(center, 0.1)
    center[0] = 0.9

    assert region.center[0] == 0.4
    assert region.lower[0] == pytest.approx(0.3)


def test_scalar_input_bounds_are_broadcast() -> None:
    region = InputRegion(np.array([0.4, 0.4]), 0.2, 0.0, 1.0)

    np.testing.assert_array_equal(region.input_lower, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(region.input_upper, np.array([1.0, 1.0]))


def test_structural_equality() -> None:
    first = _region(center=(0.4, 0.5), radius=0.2)
    second = _region(center=(0.4, 0.5), radius=0.2)
    different = _region(center=(0.4, 0.5), radius=0.3)

    assert first == second
    assert first != different
    assert (first == "not a region") is False


def test_radius_accepts_integer_scalar() -> None:
    region = InputRegion(np.array([0.5]), 1)

    assert isinstance(region.radius, float)
    assert region.radius == 1.0
    np.testing.assert_allclose(region.lower, np.array([-0.5]))
    np.testing.assert_allclose(region.upper, np.array([1.5]))


def test_radius_accepts_numpy_scalar() -> None:
    region = InputRegion(np.array([0.5]), np.float64(0.25))

    assert isinstance(region.radius, float)
    assert region.radius == 0.25


def test_list_center_is_coerced_to_float64() -> None:
    region = InputRegion([0.4, 0.5], 0.1)

    assert region.center.dtype == np.float64
    assert region.dimension == 2


def test_integer_center_is_coerced_to_float64() -> None:
    region = InputRegion(np.array([1, 2]), 0.5)

    assert region.center.dtype == np.float64
    np.testing.assert_allclose(region.lower, np.array([0.5, 1.5]))


def test_large_radius_is_clipped_to_the_configured_domain() -> None:
    region = _region(center=(0.5,), radius=1e6, lower=0.0, upper=1.0)

    np.testing.assert_array_equal(region.lower, np.array([0.0]))
    np.testing.assert_array_equal(region.upper, np.array([1.0]))


def test_tiny_radius_is_retained() -> None:
    region = _region(center=(0.5,), radius=1e-12)

    assert region.lower[0] < 0.5
    assert region.upper[0] > 0.5
    assert region.upper[0] - region.lower[0] == pytest.approx(2e-12, rel=1e-6)


def test_lower_and_upper_are_recomputed_not_cached() -> None:
    region = _region()

    first_lower = region.lower
    first_lower[:] = 99.0

    np.testing.assert_allclose(region.lower, np.array([0.2, 0.3]))


def test_region_is_immutable() -> None:
    region = _region()

    with pytest.raises(FrozenInstanceError):
        region.radius = 0.5
    with pytest.raises(FrozenInstanceError):
        region.center = np.zeros(2, dtype=np.float64)


def test_default_domain_equals_explicit_infinite_domain() -> None:
    implicit = _region(center=(0.4,), radius=0.1)
    explicit = InputRegion(
        np.asarray([0.4]), 0.1, np.asarray([-np.inf]), np.asarray([np.inf])
    )

    assert implicit == explicit


@pytest.mark.parametrize(
    "other",
    [
        InputRegion(np.asarray([0.4, 0.6]), 0.2),
        InputRegion(np.asarray([0.4, 0.5]), 0.25),
        InputRegion(np.asarray([0.4, 0.5]), 0.2, 0.0, 1.0),
    ],
)
def test_structural_inequality_covers_center_radius_and_domain(
    other: InputRegion,
) -> None:
    assert _region(center=(0.4, 0.5), radius=0.2) != other


def test_contains_accepts_a_list_point() -> None:
    assert _region(center=(0.5, 0.5), radius=0.1).contains([0.55, 0.45])


@pytest.mark.parametrize("point", [np.zeros((1, 2)), np.float64(0.5), np.array([])])
def test_contains_rejects_non_matching_point_shape(point: object) -> None:
    with pytest.raises(ValueError, match="does not match region shape"):
        _region(center=(0.5, 0.5)).contains(point)


def test_singleton_region_contains_only_its_center() -> None:
    region = InputRegion(np.array([0.4, 0.5]), 0.0)

    assert region.contains(np.array([0.4, 0.5]))
    assert not region.contains(np.array([0.4, 0.5 + 1e-15]))


def test_center_always_belongs_to_its_region() -> None:
    rng = np.random.default_rng(11)
    for _ in range(25):
        center = rng.uniform(0.0, 1.0, size=16)
        region = InputRegion(center, float(rng.uniform(0.0, 0.5)), 0.0, 1.0)

        assert region.contains(center)


def test_region_grows_monotonically_with_radius() -> None:
    rng = np.random.default_rng(3)
    center = rng.uniform(0.0, 1.0, size=8)
    previous = InputRegion(center, 0.0, 0.0, 1.0)

    for radius in (0.01, 0.05, 0.1, 0.3):
        region = InputRegion(center, radius, 0.0, 1.0)

        assert np.all(region.lower <= previous.lower)
        assert np.all(region.upper >= previous.upper)
        previous = region


class _DuckBox:
    """Structurally conforming box that inherits from no verification type."""

    def __init__(self) -> None:
        self._lower = np.zeros(3, dtype=np.float64)
        self._upper = np.ones(3, dtype=np.float64)

    @property
    def lower(self) -> np.ndarray:
        return self._lower

    @property
    def upper(self) -> np.ndarray:
        return self._upper


def test_structural_box_conformance_without_inheritance() -> None:
    assert isinstance(_DuckBox(), BoxProtocol)


@pytest.mark.parametrize("candidate", [object(), {}, 0.5])
def test_non_conforming_objects_are_not_boxes(candidate: object) -> None:
    assert not isinstance(candidate, BoxProtocol)


class _HalfBox:
    """Exposes only a lower bound, so the contract is incomplete."""

    lower = np.zeros(3, dtype=np.float64)


def test_partial_box_conformance_is_rejected() -> None:
    assert not isinstance(_HalfBox(), BoxProtocol)


def test_randomized_perturbation_membership_matches_definition() -> None:
    # contains() must agree with the mathematical definition on random
    # samples: within L-infinity radius of the center AND inside the domain.
    rng = np.random.default_rng(5)
    center = rng.uniform(0.0, 1.0, size=6)
    radius = 0.2
    region = InputRegion(center, radius, 0.0, 1.0)

    perturbations = rng.uniform(-radius, radius, size=(300, 6))
    for point in center + perturbations:
        expected = bool(
            np.all(np.abs(point - center) <= radius)
            and np.all(point >= region.input_lower)
            and np.all(point <= region.input_upper)
        )

        assert region.contains(point) is expected


def test_randomized_bounds_never_violate_configured_domain() -> None:
    rng = np.random.default_rng(9)
    for _ in range(30):
        dimension = int(rng.integers(1, 32))
        center = rng.uniform(-2.0, 2.0, size=dimension)
        lower_domain = rng.uniform(-4.0, -2.0, size=dimension)
        upper_domain = rng.uniform(2.0, 4.0, size=dimension)
        radius = float(rng.uniform(0.0, 2.0))

        region = InputRegion(center, radius, lower_domain, upper_domain)

        assert np.all(region.lower >= lower_domain)
        assert np.all(region.upper <= upper_domain)
        assert np.all(region.lower <= region.upper)
        assert np.all(region.lower <= center)
        assert np.all(center <= region.upper)
        assert np.all(region.upper - region.lower <= 2.0 * radius + 1e-12)


def test_randomized_region_nesting_under_increasing_radius() -> None:
    # A larger epsilon must produce a superset region for the same center
    # and domain, including when clipping is active.
    rng = np.random.default_rng(13)
    for _ in range(20):
        center = rng.uniform(-1.0, 1.0, size=8)
        small_radius = float(rng.uniform(0.0, 0.5))
        large_radius = float(rng.uniform(0.5, 2.0))

        small = InputRegion(center, small_radius, -1.0, 1.0)
        large = InputRegion(center, large_radius, -1.0, 1.0)

        assert np.all(large.lower <= small.lower)
        assert np.all(small.upper <= large.upper)
