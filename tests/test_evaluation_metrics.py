"""Deterministic unit tests for the evaluation metrics layer.

All tests operate on fabricated :class:`~evaluation.metrics.SampleRecord`
values and pure numeric helpers; no MNIST data, model loading, LP solving, or
verification execution is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    SampleRecord,
    aggregate_epsilon_metrics,
    bound_tightening_metrics,
    certification_metrics,
    interval_widths,
    runtime_metrics,
    width_statistics,
)
from verification import VerificationStatus

CERTIFIED = VerificationStatus.CERTIFIED.value
INCONCLUSIVE = VerificationStatus.INCONCLUSIVE.value


def _record(
    *,
    sample_index: int = 0,
    epsilon: float = 0.03,
    baseline_status: str = CERTIFIED,
    refined_status: str = CERTIFIED,
    baseline_margin: float | None = 1.0,
    refined_margin: float | None = 1.5,
    baseline_time: float | None = 0.1,
    refined_time: float | None = 0.5,
    lp_time: float | None = 0.3,
    widths: tuple[float, float, float, float, int] | None = (1.0, 0.5, 0.5, 0.5, 0),
    error: str | None = None,
) -> SampleRecord:
    """Build one SampleRecord with metric-friendly defaults."""
    refinement_status: str | None
    if refined_status == "ERROR":
        refinement_status = "ERROR"
        lp_time = None
        widths = None
    else:
        refinement_status = "TIGHTENED"
    return SampleRecord(
        sample_index=sample_index,
        label=7,
        predicted=7,
        target_class=7,
        epsilon=epsilon,
        baseline_status=baseline_status,
        baseline_margin=baseline_margin,
        baseline_time_seconds=baseline_time,
        refined_status=refined_status,
        refined_margin=refined_margin,
        refined_time_seconds=refined_time,
        lp_refinement_time_seconds=lp_time,
        refinement_status=refinement_status,
        refinement_targets=10,
        refinement_accepted=8,
        refinement_rejected=2,
        baseline_width_mean=None if widths is None else widths[0],
        refined_width_mean=None if widths is None else widths[1],
        abs_width_reduction_mean=None if widths is None else widths[2],
        rel_width_reduction_mean=None if widths is None else widths[3],
        zero_width_baseline_intervals=None if widths is None else widths[4],
        error=error,
    )


class TestCertificationMetrics:
    def test_certification_rates_are_computed_over_evaluated_population(self) -> None:
        records = [
            _record(baseline_status=CERTIFIED, refined_status=CERTIFIED),
            _record(baseline_status=CERTIFIED, refined_status=INCONCLUSIVE),
            _record(baseline_status=INCONCLUSIVE, refined_status=CERTIFIED),
            _record(baseline_status=INCONCLUSIVE, refined_status=INCONCLUSIVE),
        ]
        metrics = certification_metrics(records, 0.03, 10_000, 3)

        assert metrics.baseline_certified == 2
        assert metrics.refined_certified == 2
        assert metrics.baseline_rate == pytest.approx(0.5)
        assert metrics.refined_rate == pytest.approx(0.5)

    def test_percentage_point_improvement(self) -> None:
        records = [
            _record(baseline_status=INCONCLUSIVE, refined_status=CERTIFIED),
            _record(baseline_status=INCONCLUSIVE, refined_status=INCONCLUSIVE),
            _record(baseline_status=CERTIFIED, refined_status=CERTIFIED),
            _record(baseline_status=CERTIFIED, refined_status=CERTIFIED),
        ]
        metrics = certification_metrics(records, 0.05, 10_000, 0)

        assert metrics.baseline_rate == pytest.approx(0.5)
        assert metrics.refined_rate == pytest.approx(0.75)
        assert metrics.improvement_pp == pytest.approx(25.0)

    def test_improvement_can_be_negative(self) -> None:
        records = [
            _record(baseline_status=CERTIFIED, refined_status=INCONCLUSIVE),
            _record(baseline_status=CERTIFIED, refined_status=CERTIFIED),
        ]
        metrics = certification_metrics(records, 0.1, 10_000, 0)

        assert metrics.improvement_pp == pytest.approx(-50.0)

    def test_inconclusive_and_error_statuses_are_counted_separately(self) -> None:
        records = [
            _record(baseline_status=INCONCLUSIVE, refined_status=INCONCLUSIVE),
            _record(baseline_status=INCONCLUSIVE, refined_status="ERROR", error="RuntimeError: boom"),
            _record(baseline_status="ERROR", refined_status=INCONCLUSIVE, error="ValueError: bad"),
        ]
        metrics = certification_metrics(records, 0.01, 10_000, 0)

        assert metrics.baseline_inconclusive == 2
        assert metrics.refined_inconclusive == 2
        assert metrics.baseline_errors == 1
        assert metrics.refined_errors == 1
        assert metrics.baseline_certified == 0
        assert metrics.refined_certified == 0

    def test_empty_population_yields_none_rates_not_zero(self) -> None:
        metrics = certification_metrics([], 0.03, 10_000, 5)

        assert metrics.evaluated_samples == 0
        assert metrics.baseline_rate is None
        assert metrics.refined_rate is None
        assert metrics.improvement_pp is None

    def test_excluded_incorrect_samples_are_reported(self) -> None:
        metrics = certification_metrics([_record()], 0.01, 10_000, 12)

        assert metrics.total_test_samples == 10_000
        assert metrics.evaluated_samples == 1
        assert metrics.incorrect_excluded == 12


class TestWidthStatistics:
    def test_interval_widths(self) -> None:
        widths = interval_widths(np.array([0.0, 1.0]), np.array([2.0, 1.5]))

        assert widths == pytest.approx(np.array([2.0, 0.5]))

    def test_interval_widths_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            interval_widths(np.array([1.0]), np.array([0.5]))

    def test_per_sample_width_statistics(self) -> None:
        stats = width_statistics(
            np.array([0.0, 1.0]), np.array([1.0, 3.0]),
            np.array([0.5, 1.0]), np.array([0.75, 2.0]),
        )

        assert stats["baseline_width_mean"] == pytest.approx(1.5)
        assert stats["refined_width_mean"] == pytest.approx(0.625)
        assert stats["abs_width_reduction_mean"] == pytest.approx(0.875)
        assert stats["rel_width_reduction_mean"] == pytest.approx(0.625)
        assert stats["zero_width_baseline_intervals"] == 0

    def test_relative_reduction_definition(self) -> None:
        stats = width_statistics(
            np.array([0.0, 2.0]), np.array([1.0, 4.0]),
            np.array([0.25, 2.5]), np.array([0.75, 3.5]),
        )

        assert stats["rel_width_reduction_mean"] == pytest.approx(0.5)

    def test_zero_width_baseline_intervals_are_excluded_and_counted(self) -> None:
        stats = width_statistics(
            np.array([0.5, 0.5, 1.0]), np.array([0.5, 1.5, 3.0]),
            np.array([0.5, 0.75, 2.0]), np.array([0.5, 1.25, 2.5]),
        )

        assert stats["zero_width_baseline_intervals"] == 1
        assert stats["rel_width_reduction_mean"] == pytest.approx(0.625)

    def test_all_zero_width_baseline_yields_none_relative_reduction(self) -> None:
        stats = width_statistics(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]),
            np.array([1.0, 2.0]), np.array([1.0, 2.0]),
        )

        assert stats["rel_width_reduction_mean"] is None
        assert stats["zero_width_baseline_intervals"] == 2
        assert stats["abs_width_reduction_mean"] == pytest.approx(0.0)

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            width_statistics(
                np.zeros(3), np.ones(3), np.zeros(2), np.ones(2),
            )


class TestAggregatedWidthMetrics:
    def test_mean_width_aggregation_over_records(self) -> None:
        records = [
            _record(widths=(2.0, 1.0, 1.0, 0.5, 0)),
            _record(widths=(1.0, 0.5, 0.5, 0.5, 1)),
        ]
        metrics = bound_tightening_metrics(records)

        assert metrics.mean_baseline_width == pytest.approx(1.5)
        assert metrics.mean_refined_width == pytest.approx(0.75)
        assert metrics.mean_abs_width_reduction == pytest.approx(0.75)
        assert metrics.mean_relative_width_reduction == pytest.approx(0.5)
        assert metrics.zero_width_baseline_intervals == 1
        assert metrics.measured_samples == 2

    def test_failed_refinement_records_are_excluded_not_zero_filled(self) -> None:
        records = [_record(widths=(2.0, 1.0, 1.0, 0.5, 0)), _record(refined_status="ERROR")]
        metrics = bound_tightening_metrics(records)

        assert metrics.measured_samples == 1
        assert metrics.mean_baseline_width == pytest.approx(2.0)
        assert metrics.mean_refined_width == pytest.approx(1.0)

    def test_no_measured_samples_yields_none(self) -> None:
        metrics = bound_tightening_metrics([_record(refined_status="ERROR")])

        assert metrics.mean_baseline_width is None
        assert metrics.mean_refined_width is None
        assert metrics.mean_abs_width_reduction is None
        assert metrics.mean_relative_width_reduction is None
        assert metrics.measured_samples == 0


class TestRuntimeMetrics:
    def test_mean_and_median_over_measured_values(self) -> None:
        records = [
            _record(baseline_time=0.1, refined_time=1.0, lp_time=0.5),
            _record(baseline_time=0.3, refined_time=2.0, lp_time=1.5),
            _record(baseline_time=0.2, refined_time=3.0, lp_time=2.5),
        ]
        metrics = runtime_metrics(records)

        assert metrics.mean_baseline_time == pytest.approx(0.2)
        assert metrics.median_baseline_time == pytest.approx(0.2)
        assert metrics.mean_lp_refinement_time == pytest.approx(1.5)
        assert metrics.median_lp_refinement_time == pytest.approx(1.5)
        assert metrics.mean_total_refined_time == pytest.approx(2.0)
        assert metrics.median_total_refined_time == pytest.approx(2.0)
        assert metrics.baseline_times_measured == 3
        assert metrics.lp_times_measured == 3
        assert metrics.refined_times_measured == 3

    def test_even_count_median_averages_middle_pair(self) -> None:
        records = [
            _record(baseline_time=0.1, refined_time=1.0, lp_time=0.0),
            _record(baseline_time=0.3, refined_time=2.0, lp_time=1.0),
        ]
        metrics = runtime_metrics(records)

        assert metrics.median_baseline_time == pytest.approx(0.2)

    def test_missing_measurements_are_skipped_never_zero(self) -> None:
        records = [
            _record(baseline_time=0.1, refined_time=1.0, lp_time=None),
            _record(baseline_time=None, refined_time=3.0, lp_time=2.0),
            _record(baseline_time=0.5, refined_time=None, lp_time=4.0),
        ]
        metrics = runtime_metrics(records)

        assert metrics.mean_baseline_time == pytest.approx(0.3)
        assert metrics.baseline_times_measured == 2
        assert metrics.mean_lp_refinement_time == pytest.approx(3.0)
        assert metrics.lp_times_measured == 2
        assert metrics.mean_total_refined_time == pytest.approx(2.0)
        assert metrics.refined_times_measured == 2

    def test_no_measurements_yields_none_never_zero(self) -> None:
        metrics = runtime_metrics([])

        assert metrics.mean_baseline_time is None
        assert metrics.median_baseline_time is None
        assert metrics.mean_lp_refinement_time is None
        assert metrics.mean_total_refined_time is None
        assert metrics.baseline_times_measured == 0


class TestAggregateEpsilonMetrics:
    def test_aggregation_combines_all_metric_groups(self) -> None:
        records = [
            _record(sample_index=1, epsilon=0.05, baseline_status=CERTIFIED, refined_status=CERTIFIED),
            _record(sample_index=2, epsilon=0.05, baseline_status=INCONCLUSIVE, refined_status=CERTIFIED),
        ]
        metrics = aggregate_epsilon_metrics(records, 0.05, 10_000, 4)

        assert metrics.epsilon == pytest.approx(0.05)
        assert metrics.certification.baseline_certified == 1
        assert metrics.certification.refined_certified == 2
        assert metrics.certification.improvement_pp == pytest.approx(50.0)
        assert metrics.certification.incorrect_excluded == 4
        assert metrics.bounds.measured_samples == 2
        assert metrics.runtime.baseline_times_measured == 2

    def test_records_with_wrong_epsilon_are_rejected(self) -> None:
        records = [_record(epsilon=0.01)]

        with pytest.raises(ValueError):
            aggregate_epsilon_metrics(records, 0.03, 10_000, 0)


class TestSampleRecordSerialization:
    def test_csv_round_trip_preserves_all_fields(self) -> None:
        record = _record(
            sample_index=17,
            epsilon=0.07,
            baseline_status=INCONCLUSIVE,
            refined_status=CERTIFIED,
            baseline_margin=-0.25,
            refined_margin=0.5,
            baseline_time=0.123,
            refined_time=1.234,
            lp_time=0.567,
        )
        rebuilt = SampleRecord.from_row(record.to_row())

        assert rebuilt == record

    def test_csv_round_trip_preserves_none_fields(self) -> None:
        record = _record(refined_status="ERROR", error="RuntimeError: boom")
        row = record.to_row()

        assert row["lp_refinement_time_seconds"] == ""
        assert row["error"] == "RuntimeError: boom"

        rebuilt = SampleRecord.from_row(row)

        assert rebuilt == record
        assert rebuilt.lp_refinement_time_seconds is None
