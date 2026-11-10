import math

import pytest

from verification import VerificationResult, VerificationStatus
from verification.safety import SafetyChecker


def test_certified_when_target_lower_exceeds_all_competitor_uppers() -> None:
    result = SafetyChecker().check([0.5, -1.0, -1.0], [0.9, 0.2, 0.1], 0)

    assert isinstance(result, VerificationResult)
    assert result.status is VerificationStatus.CERTIFIED
    assert result.target_class == 0
    assert result.target_lower_bound == 0.5
    assert result.max_competitor_upper_bound == 0.2
    assert result.verification_margin == pytest.approx(0.3)
    assert result.perturbation_radius is None
    assert "strictly exceeds" in result.explanation


def test_certification_ignores_targets_own_upper_bound() -> None:
    # The target's own upper bound may exceed its lower bound; only
    # competitor bounds participate in the condition.
    result = SafetyChecker().check([0.5, -1.0], [0.9, 0.1], 0)

    assert result.status is VerificationStatus.CERTIFIED


def test_overlapping_bounds_are_inconclusive() -> None:
    result = SafetyChecker().check([0.3, -1.0], [0.9, 0.4], 0)

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.verification_margin == pytest.approx(-0.1)


def test_competitor_upper_exceeding_target_lower_is_inconclusive() -> None:
    result = SafetyChecker().check([0.5, 0.5], [0.6, 0.9], 1)

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.max_competitor_upper_bound == 0.6
    assert "class 0" in result.explanation


def test_single_output_class_is_certified_vacuously() -> None:
    result = SafetyChecker().check([0.7], [0.9], 0)

    assert result.status is VerificationStatus.CERTIFIED
    assert result.max_competitor_upper_bound == -math.inf
    assert result.verification_margin == math.inf
    assert "only output class" in result.explanation


@pytest.mark.parametrize("target", [-1, 3])
def test_out_of_range_target_class_is_rejected(target: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        SafetyChecker().check([0.1, 0.1, 0.1], [0.2, 0.2, 0.2], target)


@pytest.mark.parametrize("target", [1.5, "1", True, None])
def test_non_integer_target_class_is_rejected(target: object) -> None:
    with pytest.raises(TypeError, match="integer class index"):
        SafetyChecker().check([0.1, 0.1], [0.2, 0.2], target)


def test_mismatched_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        SafetyChecker().check([0.1, 0.2], [0.3], 0)


def test_empty_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        SafetyChecker().check([], [], 0)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([math.nan, 0.1], [0.2, 0.2]),
        ([0.1, 0.1], [math.nan, 0.2]),
    ],
)
def test_nan_bounds_are_rejected(lower: list, upper: list) -> None:
    with pytest.raises(ValueError, match="NaN"):
        SafetyChecker().check(lower, upper, 0)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([math.inf, 0.1], [0.2, 0.2]),
        ([0.1, 0.1], [-math.inf, 0.2]),
    ],
)
def test_infinite_bounds_are_rejected(lower: list, upper: list) -> None:
    with pytest.raises(ValueError, match="finite"):
        SafetyChecker().check(lower, upper, 0)


def test_non_numeric_bound_is_rejected() -> None:
    with pytest.raises(TypeError, match="real numbers"):
        SafetyChecker().check([0.1, "x"], [0.2, 0.2], 0)


def test_string_bound_vector_is_rejected() -> None:
    with pytest.raises(TypeError, match="iterable"):
        SafetyChecker().check("0.1", [0.2], 0)


def test_lower_bound_exceeding_upper_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        SafetyChecker().check([0.5], [0.4], 0)


def test_exact_tie_is_inconclusive() -> None:
    result = SafetyChecker().check([0.4, 0.0], [0.9, 0.4], 0)

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.verification_margin == 0.0


def test_tiny_positive_margin_is_certified() -> None:
    # No tolerance floor: any strictly positive margin certifies.
    just_below = math.nextafter(0.4, 0.0)
    result = SafetyChecker().check([0.4, 0.0], [0.9, just_below], 0)

    assert result.status is VerificationStatus.CERTIFIED
    assert result.verification_margin > 0.0


def test_competitor_tie_selects_lowest_class_index() -> None:
    result = SafetyChecker().check([0.4, 0.0, 0.0], [0.9, 0.4, 0.4], 0)

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.max_competitor_upper_bound == 0.4
    assert "class 1" in result.explanation


def test_margin_is_target_lower_minus_max_competitor_upper() -> None:
    result = SafetyChecker().check([2.0, 1.0], [2.5, 1.5], 0)

    assert result.verification_margin == pytest.approx(2.0 - 1.5)
