import math

import pytest

from abstract_domain.interval import Interval


def test_valid_interval_construction() -> None:
    interval = Interval(-2, 3)

    assert interval.lower == -2.0
    assert interval.upper == 3.0
    assert isinstance(interval.lower, float)
    assert isinstance(interval.upper, float)


def test_degenerate_interval_is_valid() -> None:
    assert Interval(2, 2) == Interval(2.0, 2.0)


@pytest.mark.parametrize(("lower", "upper"), [(2, 1), (1, -1), (math.inf, math.inf), (-math.inf, -math.inf)])
def test_invalid_interval_construction(lower: float, upper: float) -> None:
    with pytest.raises(ValueError):
        Interval(lower, upper)


def test_non_scalar_bounds_are_rejected() -> None:
    with pytest.raises(TypeError, match="real scalar"):
        Interval([0.0], [1.0])


def test_addition() -> None:
    assert Interval(1, 3) + Interval(-2, 4) == Interval(-1, 7)


def test_subtraction() -> None:
    assert Interval(1, 3) - Interval(-2, 4) == Interval(-3, 5)


def test_multiplication_of_positive_intervals() -> None:
    assert Interval(1, 2) * Interval(3, 4) == Interval(3, 8)


def test_multiplication_of_negative_intervals() -> None:
    assert Interval(-4, -2) * Interval(-3, -1) == Interval(2, 12)


def test_multiplication_of_mixed_sign_intervals() -> None:
    assert Interval(-2, 3) * Interval(-4, 5) == Interval(-12, 15)


def test_division_of_valid_intervals() -> None:
    assert Interval(6, 8) / Interval(2, 4) == Interval(1.5, 4)


def test_division_by_negative_interval() -> None:
    assert Interval(2, 4) / Interval(-4, -2) == Interval(-2, -0.5)


@pytest.mark.parametrize("denominator", [Interval(-1, 1), Interval(0, 2), Interval(-2, 0), Interval(0, 0)])
def test_division_by_interval_containing_zero(denominator: Interval) -> None:
    with pytest.raises(ZeroDivisionError, match="contains zero"):
        Interval(1, 2) / denominator


def test_relu_on_positive_interval() -> None:
    assert Interval(2, 5).relu() == Interval(2, 5)


def test_relu_on_negative_interval() -> None:
    assert Interval(-5, -2).relu() == Interval(0, 0)


def test_relu_on_interval_crossing_zero() -> None:
    assert Interval(-3, 4).relu() == Interval(0, 4)


def test_scalar_containment() -> None:
    interval = Interval(-1, 2)

    assert interval.contains(-1)
    assert interval.contains(0)
    assert interval.contains(2)
    assert not interval.contains(3)
    assert not interval.contains(-2)


def test_interval_containment() -> None:
    interval = Interval(-1, 3)

    assert interval.contains_interval(Interval(0, 2))
    assert interval.contains_interval(Interval(-1, 3))
    assert not interval.contains_interval(Interval(-2, 0))
    assert not interval.contains_interval(Interval(2, 4))


def test_intersection() -> None:
    assert Interval(0, 4).intersection(Interval(2, 6)) == Interval(2, 4)


def test_touching_intersection_is_a_point() -> None:
    assert Interval(0, 2).intersection(Interval(2, 5)) == Interval(2, 2)


def test_disjoint_intervals_have_no_intersection() -> None:
    assert Interval(0, 1).intersection(Interval(2, 3)) is None


def test_interval_width() -> None:
    assert Interval(-2, 5).width == 7
    assert Interval(3, 3).width == 0


def test_infinite_bounds_and_width() -> None:
    left_unbounded = Interval(-math.inf, 0)
    right_unbounded = Interval(0, math.inf)

    assert left_unbounded.width == math.inf
    assert right_unbounded.width == math.inf
    assert (left_unbounded + Interval(1, 2)) == Interval(-math.inf, 2)
    assert right_unbounded.relu() == right_unbounded


def test_zero_times_unbounded_interval() -> None:
    assert Interval(0, 0) * Interval(-math.inf, math.inf) == Interval(0, 0)


def test_mixed_zero_and_unbounded_multiplication() -> None:
    assert Interval(0, 1) * Interval(0, math.inf) == Interval(0, math.inf)


def test_unbounded_denominator() -> None:
    result = Interval(1, 2) / Interval(1, math.inf)

    assert result == Interval(0, 2)


@pytest.mark.parametrize("bound", [math.nan, float("nan")])
def test_nan_bounds_are_rejected(bound: float) -> None:
    with pytest.raises(ValueError, match="cannot be NaN"):
        Interval(bound, 1)
    with pytest.raises(ValueError, match="cannot be NaN"):
        Interval(0, bound)


def test_nan_containment_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be NaN"):
        Interval(0, 1).contains(math.nan)
