"""End-to-end tests across Member 2's verification components.

The verification layer is built from three cooperating values:
:class:`~verification.regions.InputRegion` (the concrete L-infinity query
region), :class:`~verification.safety.SafetyChecker` (the verdict over abstract
output bounds), and :class:`~verification.VerificationResult` (the immutable
record). These tests exercise them together rather than in isolation.

Soundness cross-checks use :func:`_ibp_linear`, a deliberately small
interval-bound-propagation oracle local to this module. Production propagation
belongs to the Box domain and is not part of the verification layer, so the
oracle exists only to give the verdict an independent reference.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from verification import VerificationResult, VerificationStatus
from verification.regions import BoxProtocol, InputRegion
from verification.safety import SafetyChecker

WEIGHT = np.array([[1.0, -1.0], [-1.0, 1.0]])
BIAS = np.zeros(2)
CENTER = np.array([0.5, 0.1])


def _ibp_linear(
    lower: np.ndarray, upper: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a box through one dense layer with an interval oracle."""
    positive = np.maximum(weight, 0.0)
    negative = np.minimum(weight, 0.0)
    out_lower = positive @ lower + negative @ upper + bias
    out_upper = positive @ upper + negative @ lower + bias
    return out_lower, out_upper


def _propagate(box: BoxProtocol) -> tuple[np.ndarray, np.ndarray]:
    """Consume any conforming box through the propagation contract."""
    return _ibp_linear(box.lower, box.upper, WEIGHT, BIAS)


def _check(radius: float, target: int = 0) -> VerificationResult:
    region = InputRegion(CENTER, radius)
    lower, upper = _propagate(region)
    return SafetyChecker().check(lower, upper, target)


def test_region_bound_propagation_and_verdict_flow() -> None:
    # Region [0.45, 0.55] x [0.05, 0.15] maps to class 0 in [0.30, 0.50] and
    # class 1 in [-0.50, -0.30], so the classes are separated.
    result = _check(0.05)

    assert result.status is VerificationStatus.CERTIFIED
    assert result.target_class == 0
    assert result.target_lower_bound == pytest.approx(0.3)
    assert result.max_competitor_upper_bound == pytest.approx(-0.3)
    assert result.verification_margin == pytest.approx(0.6)
    assert result.perturbation_epsilon is None


def test_wide_region_is_inconclusive() -> None:
    result = _check(0.3)

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.target_lower_bound == pytest.approx(-0.2)
    assert result.max_competitor_upper_bound == pytest.approx(0.2)
    assert result.verification_margin == pytest.approx(-0.4)


def test_certified_verdict_holds_for_every_region_corner() -> None:
    region = InputRegion(CENTER, 0.05)
    result = _check(0.05)

    assert result.status is VerificationStatus.CERTIFIED
    for first in (region.lower[0], region.upper[0]):
        for second in (region.lower[1], region.upper[1]):
            logits = WEIGHT @ np.array([first, second]) + BIAS

            assert logits[0] > logits[1]


def test_certified_verdict_holds_for_sampled_region_points() -> None:
    region = InputRegion(CENTER, 0.05)
    result = _check(0.05)
    rng = np.random.default_rng(0)

    samples = rng.uniform(region.lower, region.upper, size=(200, 2))
    logits = samples @ WEIGHT.T + BIAS

    assert result.status is VerificationStatus.CERTIFIED
    assert np.all(logits[:, 0] > logits[:, 1])


def test_inconclusive_verdict_admits_a_concrete_counterexample() -> None:
    # An INCONCLUSIVE verdict claims nothing; a witness attack must exist.
    region = InputRegion(CENTER, 0.3)
    result = _check(0.3)

    witness = np.array([region.lower[0], region.upper[1]])
    logits = WEIGHT @ witness + BIAS

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert region.contains(witness)
    assert logits[1] > logits[0]


def test_verdict_is_monotone_in_epsilon() -> None:
    radii = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = [_check(radius) for radius in radii]
    margins = [result.verification_margin for result in results]
    statuses = [result.status for result in results]

    assert all(
        later <= earlier + 1e-12
        for earlier, later in zip(margins, margins[1:], strict=False)
    )
    first_failure = statuses.index(VerificationStatus.INCONCLUSIVE)
    assert all(
        status is VerificationStatus.INCONCLUSIVE for status in statuses[first_failure:]
    )


def test_smaller_region_is_nested_in_larger_region() -> None:
    small = InputRegion(CENTER, 0.05)
    large = InputRegion(CENTER, 0.2)

    assert np.all(large.lower <= small.lower)
    assert np.all(small.upper <= large.upper)
    assert small.contains(CENTER)
    assert large.contains(CENTER)


def test_input_region_is_accepted_where_a_box_is_required() -> None:
    lower, upper = _propagate(InputRegion(CENTER, 0.05))

    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.all(lower <= upper)


def test_orchestration_attaches_epsilon_and_round_trips() -> None:
    region = InputRegion(CENTER, 0.05)
    lower, upper = _propagate(region)
    result = SafetyChecker().check(lower, upper, 0)

    attached = replace(result, perturbation_epsilon=region.radius)

    assert attached.status is result.status
    assert attached.verification_margin == result.verification_margin
    assert attached.perturbation_epsilon == 0.05
    assert VerificationResult.from_dict(attached.to_dict()) == attached
    assert "epsilon=0.05" in str(attached)


def test_verdict_status_and_margin_never_disagree() -> None:
    for radius in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
        result = _check(radius)

        if result.status is VerificationStatus.CERTIFIED:
            assert result.verification_margin > 0.0
        else:
            assert result.status is VerificationStatus.INCONCLUSIVE
            assert result.verification_margin <= 0.0


def test_ten_class_flow_reports_propagated_bounds() -> None:
    rng = np.random.default_rng(21)
    weight = rng.normal(size=(10, 4))
    bias = np.zeros(10)
    center = rng.uniform(0.5, 1.0, size=4)
    target = int(np.argmax(weight @ center + bias))

    region = InputRegion(center, 0.01, 0.0, 1.0)
    lower, upper = _ibp_linear(region.lower, region.upper, weight, bias)
    result = SafetyChecker().check(lower, upper, target)

    assert result.target_class == target
    assert result.target_lower_bound == lower[target]
    assert result.max_competitor_upper_bound == pytest.approx(
        float(np.max(np.delete(upper, target)))
    )
    assert result.status in (
        VerificationStatus.CERTIFIED,
        VerificationStatus.INCONCLUSIVE,
    )
