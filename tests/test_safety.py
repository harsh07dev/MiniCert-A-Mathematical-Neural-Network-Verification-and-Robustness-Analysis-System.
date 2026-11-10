import math

import numpy as np
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
    assert result.perturbation_epsilon is None
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


def test_numpy_array_bounds_are_accepted() -> None:
    result = SafetyChecker().check(
        np.array([0.5, -1.0, -1.0]), np.array([0.9, 0.2, 0.1]), 0
    )

    assert result.status is VerificationStatus.CERTIFIED
    assert result.max_competitor_upper_bound == 0.2


def test_generator_bounds_are_accepted() -> None:
    result = SafetyChecker().check(
        (value for value in [0.5, -1.0]), (value for value in [0.9, 0.1]), 0
    )

    assert result.status is VerificationStatus.CERTIFIED


@pytest.mark.parametrize("bounds", [None, 5, 1.5])
def test_non_iterable_bounds_are_rejected(bounds: object) -> None:
    with pytest.raises(TypeError, match="iterable"):
        SafetyChecker().check(bounds, [0.2], 0)


def test_checker_is_stateless_across_calls() -> None:
    checker = SafetyChecker()

    first = checker.check([0.5, -1.0], [0.9, 0.1], 0)
    second = checker.check([0.5, -1.0], [0.9, 0.1], 0)
    third = checker.check([0.1, 0.1], [0.2, 0.2], 0)

    assert first == second
    assert third.status is VerificationStatus.INCONCLUSIVE


@pytest.mark.parametrize("bounds", [None, 5, 1.5])
def test_non_iterable_upper_bounds_are_rejected(bounds: object) -> None:
    with pytest.raises(TypeError, match="iterable"):
        SafetyChecker().check([0.2], bounds, 0)


def test_target_class_participates_as_competitor_for_other_targets() -> None:
    # Class 0 has the largest upper bound; selecting class 1 as target makes
    # class 0 the decisive competitor.
    result = SafetyChecker().check([0.5, -0.1, -0.1], [0.9, 0.0, 0.0], 1)

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.max_competitor_upper_bound == 0.9
    assert "class 0" in result.explanation


def test_last_class_can_be_the_target() -> None:
    result = SafetyChecker().check([-1.0, -1.0, 0.5], [0.0, 0.0, 0.9], 2)

    assert result.status is VerificationStatus.CERTIFIED
    assert result.max_competitor_upper_bound == 0.0


def test_ten_class_mnist_shaped_verdict() -> None:
    lower = [0.8] + [0.1] * 9
    upper = [0.95] + [0.6] * 9

    result = SafetyChecker().check(lower, upper, 0)

    assert result.status is VerificationStatus.CERTIFIED
    assert result.target_lower_bound == 0.8
    assert result.max_competitor_upper_bound == 0.6
    assert result.verification_margin == pytest.approx(0.2)


def test_all_negative_logits_are_certified() -> None:
    result = SafetyChecker().check([-0.5, -1.0, -2.0], [-0.1, -0.6, -1.5], 0)

    assert result.status is VerificationStatus.CERTIFIED
    assert result.verification_margin == pytest.approx(0.1)


def test_tiny_negative_margin_is_inconclusive() -> None:
    just_above = math.nextafter(0.4, 1.0)

    result = SafetyChecker().check([0.4, 0.0], [0.9, just_above], 0)

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.verification_margin < 0.0


def test_lower_exceeding_upper_reports_the_offending_class() -> None:
    with pytest.raises(ValueError, match="class 1"):
        SafetyChecker().check([0.1, 0.9], [0.2, 0.8], 0)


def test_explanation_reports_the_margin() -> None:
    result = SafetyChecker().check([0.5, -1.0], [0.9, 0.2], 0)

    assert "margin 0.3" in result.explanation


def test_explanation_reports_target_and_competitor_bounds() -> None:
    result = SafetyChecker().check([0.5, -1.0], [0.9, 0.2], 0)

    assert "0.5" in result.explanation
    assert "0.2" in result.explanation


@pytest.mark.parametrize("target", [np.int64(0), np.int32(0)])
def test_numpy_integer_target_is_rejected(target: object) -> None:
    # Callers must cast a numpy index to int before calling check().
    with pytest.raises(TypeError, match="integer class index"):
        SafetyChecker().check([0.5, -1.0], [0.9, 0.1], target)


def test_returned_result_is_immutable_and_consistent() -> None:
    result = SafetyChecker().check([0.5, -1.0], [0.9, 0.2], 0)

    assert result.verification_margin == pytest.approx(
        result.target_lower_bound - result.max_competitor_upper_bound
    )
    assert result.verification_margin > 0.0
    assert result.perturbation_epsilon is None


def test_randomized_certification_matches_dominance_definition() -> None:
    # CERTIFIED holds exactly when the target lower bound strictly exceeds
    # every competitor upper bound, for randomly generated valid inputs.
    rng = np.random.default_rng(17)
    checker = SafetyChecker()

    for _ in range(200):
        classes = int(rng.integers(1, 12))
        target = int(rng.integers(0, classes))
        centers = rng.uniform(-2.0, 2.0, size=classes)
        widths = rng.uniform(0.0, 1.0, size=classes)
        lower = centers - widths
        upper = centers + widths

        result = checker.check(lower, upper, target)

        competitors = [j for j in range(classes) if j != target]
        expected_margin = lower[target] - max(
            (upper[j] for j in competitors), default=-math.inf
        )
        if expected_margin > 0.0:
            assert result.status is VerificationStatus.CERTIFIED
        else:
            assert result.status is VerificationStatus.INCONCLUSIVE
        assert result.target_lower_bound == lower[target]
        assert result.max_competitor_upper_bound == max(
            (upper[j] for j in competitors), default=-math.inf
        )
        if classes == 1:
            assert result.verification_margin == math.inf
        else:
            assert result.verification_margin == pytest.approx(expected_margin)


def test_randomized_overlapping_bounds_never_certify() -> None:
    # Construct overlaps explicitly: a competitor upper bound reaching the
    # target lower bound must force INCONCLUSIVE.
    rng = np.random.default_rng(19)
    checker = SafetyChecker()

    for _ in range(100):
        classes = int(rng.integers(2, 8))
        target = int(rng.integers(0, classes))
        lower = rng.uniform(-2.0, 2.0, size=classes)
        upper = lower + rng.uniform(0.0, 2.0, size=classes)
        competitor = int(rng.choice([j for j in range(classes) if j != target]))
        upper[competitor] = max(upper[competitor], lower[competitor], lower[target])

        result = checker.check(lower, upper, target)

        assert result.status is VerificationStatus.INCONCLUSIVE
        assert result.verification_margin <= 0.0


def test_increasing_a_competitor_upper_bound_never_improves_certification() -> None:
    rng = np.random.default_rng(23)
    checker = SafetyChecker()

    for _ in range(150):
        classes = int(rng.integers(2, 10))
        target = int(rng.integers(0, classes))
        lower = rng.uniform(-2.0, 2.0, size=classes)
        upper = lower + rng.uniform(0.0, 2.0, size=classes)
        competitor = int(rng.choice([j for j in range(classes) if j != target]))

        original = checker.check(lower, upper, target)

        boosted = upper.copy()
        boosted[competitor] += float(rng.uniform(1e-9, 3.0))
        strengthened = checker.check(lower, boosted, target)

        assert strengthened.verification_margin <= original.verification_margin + 1e-12
        if strengthened.status is VerificationStatus.CERTIFIED:
            assert original.status is VerificationStatus.CERTIFIED


def test_lowering_the_target_lower_bound_never_improves_certification() -> None:
    rng = np.random.default_rng(31)
    checker = SafetyChecker()

    for _ in range(150):
        classes = int(rng.integers(2, 10))
        target = int(rng.integers(0, classes))
        lower = rng.uniform(-2.0, 2.0, size=classes)
        upper = lower + rng.uniform(0.0, 2.0, size=classes)

        original = checker.check(lower, upper, target)

        weakened = lower.copy()
        weakened[target] -= float(rng.uniform(1e-9, 3.0))
        weakened_result = checker.check(weakened, upper, target)

        assert weakened_result.verification_margin <= original.verification_margin + 1e-12
        if weakened_result.status is VerificationStatus.CERTIFIED:
            assert original.status is VerificationStatus.CERTIFIED
