"""Aggregated experimental metrics and the per-sample evaluation record.

This module defines the value objects of the evaluation layer:

- :class:`SampleRecord`: the immutable per-sample outcome of one
  (sample, epsilon) verification query under both methods, including a CSV
  row serialization contract.
- :class:`CertificationMetrics`, :class:`BoundTighteningMetrics`,
  :class:`RuntimeMetrics`, :class:`EpsilonMetrics`: aggregated metrics for one
  epsilon value.

Metric conventions
------------------
- Certification rates are computed over the evaluated (correctly classified)
  population only; the number of excluded incorrectly classified samples is
  reported separately.
- The absolute certification improvement is expressed in percentage points:
  ``(refined_rate - baseline_rate) * 100``.
- Interval-width statistics operate on the output bounds each method produced.
  The relative width reduction of one output coordinate is defined as

      (baseline_width - refined_width) / baseline_width

  and is computed only over coordinates whose *baseline* width is strictly
  positive; zero-width baseline intervals cannot be tightened in relative
  terms, are never divided by, and are counted separately. A per-sample
  relative reduction is ``None`` when every output coordinate is zero-width.
- Runtime statistics are computed only over measurements that are actually
  present. A missing measurement (for example an LP time after a failed
  refined verification) contributes nothing and is never reported as zero;
  an aggregate over an empty measurement set is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, ClassVar, Mapping, Optional, Sequence

import numpy as np

from verification import VerificationStatus

__all__ = [
    "SampleRecord",
    "CertificationMetrics",
    "BoundTighteningMetrics",
    "RuntimeMetrics",
    "EpsilonMetrics",
    "interval_widths",
    "width_statistics",
    "certification_metrics",
    "bound_tightening_metrics",
    "runtime_metrics",
    "aggregate_epsilon_metrics",
]

_CERTIFIED = VerificationStatus.CERTIFIED.value
_INCONCLUSIVE = VerificationStatus.INCONCLUSIVE.value
_ERROR_STATUS = "ERROR"

_CSV_INT_FIELDS = frozenset(
    {
        "sample_index",
        "label",
        "predicted",
        "target_class",
        "refinement_targets",
        "refinement_accepted",
        "refinement_rejected",
        "zero_width_baseline_intervals",
    }
)
_CSV_FLOAT_FIELDS = frozenset(
    {
        "epsilon",
        "baseline_margin",
        "baseline_time_seconds",
        "refined_margin",
        "refined_time_seconds",
        "lp_refinement_time_seconds",
        "baseline_width_mean",
        "refined_width_mean",
        "abs_width_reduction_mean",
        "rel_width_reduction_mean",
    }
)


def _finite_float(value: object, name: str) -> float:
    """Coerce a finite real scalar, rejecting NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}")
    return scalar


def _optional_finite_float(value: object, name: str) -> Optional[float]:
    """Coerce None or a finite real scalar."""
    if value is None:
        return None
    return _finite_float(value, name)


def _optional_int(value: object, name: str) -> Optional[int]:
    """Coerce None or a nonnegative integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer or None, got {type(value).__name__}")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative, got {integer}")
    return integer


@dataclass(frozen=True)
class SampleRecord:
    """Immutable per-sample outcome of one (sample, epsilon) evaluation.

    Statuses use the string values of :class:`~verification.VerificationStatus`
    (``"CERTIFIED"`` / ``"INCONCLUSIVE"``). A method whose execution raised is
    recorded with status ``"ERROR"`` and a message in ``error``; its
    unavailable measurements are ``None`` and never zero.

    Attributes:
        sample_index: Position of the sample in the underlying dataset.
        label: Ground-truth class index.
        predicted: Nominal prediction (used as the target class).
        target_class: The class both methods verified.
        epsilon: L-infinity perturbation radius of the query.
        baseline_status: Baseline IBP verdict string.
        baseline_margin: Baseline verification margin (target lower bound minus
            maximum competitor upper bound).
        baseline_time_seconds: Wall-clock time of the baseline verification.
        refined_status: Refined verification verdict string.
        refined_margin: Refined verification margin.
        refined_time_seconds: Wall-clock time of the complete refined
            verification (IBP + LP refinement + re-propagation + check).
        lp_refinement_time_seconds: Wall-clock time of the LP refinement stage
            alone, when the refinement stage completed.
        refinement_status: Aggregate refinement outcome string
            (:class:`~verification.RefinementStatus` value), or ``"ERROR"``.
        refinement_targets: Number of deduplicated refinement targets.
        refinement_accepted: Number of accepted refined intervals.
        refinement_rejected: Number of rejected targets.
        baseline_width_mean: Mean output interval width of the baseline IBP
            bounds over all output classes.
        refined_width_mean: Mean output interval width of the refined bounds.
        abs_width_reduction_mean: Mean absolute width reduction
            (baseline minus refined) over all output classes.
        rel_width_reduction_mean: Mean relative width reduction over output
            classes with strictly positive baseline width; ``None`` when no
            such class exists.
        zero_width_baseline_intervals: Number of output classes whose baseline
            interval width is exactly zero.
        error: First error message encountered during the sample evaluation,
            or ``None``.
    """

    sample_index: int
    label: int
    predicted: int
    target_class: int
    epsilon: float
    baseline_status: str
    baseline_margin: Optional[float]
    baseline_time_seconds: Optional[float]
    refined_status: str
    refined_margin: Optional[float]
    refined_time_seconds: Optional[float]
    lp_refinement_time_seconds: Optional[float]
    refinement_status: Optional[str]
    refinement_targets: Optional[int]
    refinement_accepted: Optional[int]
    refinement_rejected: Optional[int]
    baseline_width_mean: Optional[float]
    refined_width_mean: Optional[float]
    abs_width_reduction_mean: Optional[float]
    rel_width_reduction_mean: Optional[float]
    zero_width_baseline_intervals: Optional[int]
    error: Optional[str]

    CSV_FIELDNAMES: ClassVar[tuple[str, ...]] = (
        "sample_index",
        "label",
        "predicted",
        "target_class",
        "epsilon",
        "baseline_status",
        "baseline_margin",
        "baseline_time_seconds",
        "refined_status",
        "refined_margin",
        "refined_time_seconds",
        "lp_refinement_time_seconds",
        "refinement_status",
        "refinement_targets",
        "refinement_accepted",
        "refinement_rejected",
        "baseline_width_mean",
        "refined_width_mean",
        "abs_width_reduction_mean",
        "rel_width_reduction_mean",
        "zero_width_baseline_intervals",
        "error",
    )

    def __post_init__(self) -> None:
        for name in ("sample_index", "label", "predicted", "target_class"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
        object.__setattr__(self, "epsilon", _finite_float(self.epsilon, "epsilon"))
        object.__setattr__(
            self, "baseline_margin", _optional_finite_float(self.baseline_margin, "baseline_margin")
        )
        object.__setattr__(
            self,
            "baseline_time_seconds",
            _optional_finite_float(self.baseline_time_seconds, "baseline_time_seconds"),
        )
        object.__setattr__(
            self, "refined_margin", _optional_finite_float(self.refined_margin, "refined_margin")
        )
        object.__setattr__(
            self,
            "refined_time_seconds",
            _optional_finite_float(self.refined_time_seconds, "refined_time_seconds"),
        )
        object.__setattr__(
            self,
            "lp_refinement_time_seconds",
            _optional_finite_float(
                self.lp_refinement_time_seconds, "lp_refinement_time_seconds"
            ),
        )
        for name in (
            "baseline_width_mean",
            "refined_width_mean",
            "abs_width_reduction_mean",
            "rel_width_reduction_mean",
        ):
            object.__setattr__(self, name, _optional_finite_float(getattr(self, name), name))
        for name in (
            "refinement_targets",
            "refinement_accepted",
            "refinement_rejected",
            "zero_width_baseline_intervals",
        ):
            object.__setattr__(self, name, _optional_int(getattr(self, name), name))
        for name in ("baseline_status", "refined_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty status string")
        if self.refinement_status is not None and (
            not isinstance(self.refinement_status, str) or not self.refinement_status.strip()
        ):
            raise ValueError("refinement_status must be a non-empty string or None")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise ValueError("error must be a non-empty string or None")

    @property
    def baseline_certified(self) -> bool:
        """Whether the baseline verdict is CERTIFIED."""
        return self.baseline_status == _CERTIFIED

    @property
    def refined_certified(self) -> bool:
        """Whether the refined verdict is CERTIFIED."""
        return self.refined_status == _CERTIFIED

    @property
    def baseline_inconclusive(self) -> bool:
        """Whether the baseline verdict is INCONCLUSIVE."""
        return self.baseline_status == _INCONCLUSIVE

    @property
    def refined_inconclusive(self) -> bool:
        """Whether the refined verdict is INCONCLUSIVE."""
        return self.refined_status == _INCONCLUSIVE

    @property
    def refined_error(self) -> bool:
        """Whether the refined verification raised before producing a verdict."""
        return self.refined_status == _ERROR_STATUS

    def to_row(self) -> dict[str, str]:
        """Serialize the record to one CSV row (missing values become "")."""
        row: dict[str, str] = {}
        for name in self.CSV_FIELDNAMES:
            value = getattr(self, name)
            row[name] = "" if value is None else str(value)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "SampleRecord":
        """Rebuild a record from a CSV row produced by :meth:`to_row`.

        Empty cells become ``None`` for optional fields; integer and float
        columns are parsed back to their numeric types.

        Raises:
            KeyError: If a required CSV column is missing.
            ValueError: If a numeric cell cannot be parsed.
        """
        kwargs: dict[str, Any] = {}
        for name in cls.CSV_FIELDNAMES:
            raw = row[name]
            if raw == "":
                kwargs[name] = None
            elif name in _CSV_INT_FIELDS:
                kwargs[name] = int(raw)
            elif name in _CSV_FLOAT_FIELDS:
                kwargs[name] = float(raw)
            else:
                kwargs[name] = raw
        return cls(**kwargs)


def interval_widths(lower: Any, upper: Any) -> np.ndarray:
    """Compute component-wise interval widths ``upper - lower``.

    Args:
        lower: Lower bound array of shape (n,).
        upper: Upper bound array of shape (n,).

    Returns:
        A nonnegative float64 width array of shape (n,).

    Raises:
        TypeError: If the bounds are not array-convertible.
        ValueError: If the shapes differ, a width is negative, or a width is
            non-finite.
    """
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if lower_array.shape != upper_array.shape:
        raise ValueError(
            f"lower shape {lower_array.shape} does not match upper shape {upper_array.shape}"
        )
    widths = upper_array - lower_array
    if np.any(np.isnan(widths)) or np.any(np.isinf(widths)):
        raise ValueError("interval widths must be finite")
    if np.any(widths < 0.0):
        raise ValueError(f"interval widths must be nonnegative, got {widths}")
    return widths


def width_statistics(
    baseline_lower: Any,
    baseline_upper: Any,
    refined_lower: Any,
    refined_upper: Any,
) -> dict[str, Any]:
    """Compute per-sample output interval-width statistics.

    For every output coordinate the baseline and refined widths are compared.
    The per-sample statistics are means over all output coordinates; the
    relative reduction excludes zero-width baseline intervals (see the module
    docstring for the exact definition).

    Args:
        baseline_lower: Baseline lower output bounds, shape (n,).
        baseline_upper: Baseline upper output bounds, shape (n,).
        refined_lower: Refined lower output bounds, shape (n,).
        refined_upper: Refined upper output bounds, shape (n,).

    Returns:
        A dictionary with ``baseline_width_mean``, ``refined_width_mean``,
        ``abs_width_reduction_mean``, ``rel_width_reduction_mean`` (``None``
        when no baseline width is positive), and
        ``zero_width_baseline_intervals``.

    Raises:
        ValueError: If the bound shapes are inconsistent or a width is
            negative or non-finite.
    """
    baseline_widths = interval_widths(baseline_lower, baseline_upper)
    refined_widths = interval_widths(refined_lower, refined_upper)
    if baseline_widths.shape != refined_widths.shape:
        raise ValueError(
            f"baseline width shape {baseline_widths.shape} does not match "
            f"refined width shape {refined_widths.shape}"
        )
    reduction = baseline_widths - refined_widths
    positive = baseline_widths > 0.0
    if np.any(positive):
        rel_mean = float(np.mean(reduction[positive] / baseline_widths[positive]))
    else:
        rel_mean = None
    return {
        "baseline_width_mean": float(np.mean(baseline_widths)),
        "refined_width_mean": float(np.mean(refined_widths)),
        "abs_width_reduction_mean": float(np.mean(reduction)),
        "rel_width_reduction_mean": rel_mean,
        "zero_width_baseline_intervals": int(np.count_nonzero(~positive)),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    """Mean of a non-empty sequence, or None."""
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> Optional[float]:
    """Median of a non-empty sequence, or None."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _count_status(records: Sequence[SampleRecord], attribute: str) -> int:
    """Count records whose boolean status property is true."""
    return sum(1 for record in records if getattr(record, attribute))


@dataclass(frozen=True)
class CertificationMetrics:
    """Certification metrics for one epsilon value.

    Attributes:
        epsilon: The perturbation radius these metrics describe.
        total_test_samples: Size of the full test split.
        evaluated_samples: Correctly classified samples evaluated by both
            methods at this epsilon.
        incorrect_excluded: Incorrectly classified samples excluded from the
            evaluation population.
        baseline_certified: Number of baseline CERTIFIED verdicts.
        refined_certified: Number of refined CERTIFIED verdicts.
        baseline_rate: ``baseline_certified / evaluated_samples`` (None when
            no samples were evaluated).
        refined_rate: ``refined_certified / evaluated_samples``.
        improvement_pp: Absolute certification improvement in percentage
            points, ``100 * (refined_rate - baseline_rate)``.
        baseline_inconclusive: Number of baseline INCONCLUSIVE verdicts.
        refined_inconclusive: Number of refined INCONCLUSIVE verdicts.
        baseline_errors: Number of baseline executions that raised.
        refined_errors: Number of refined executions that raised (recorded,
            never silently skipped).
    """

    epsilon: float
    total_test_samples: int
    evaluated_samples: int
    incorrect_excluded: int
    baseline_certified: int
    refined_certified: int
    baseline_rate: Optional[float]
    refined_rate: Optional[float]
    improvement_pp: Optional[float]
    baseline_inconclusive: int
    refined_inconclusive: int
    baseline_errors: int
    refined_errors: int


@dataclass(frozen=True)
class BoundTighteningMetrics:
    """Output interval-width metrics for one epsilon value.

    Attributes:
        mean_baseline_width: Mean over samples of the per-sample mean baseline
            output interval width.
        mean_refined_width: Mean over samples of the per-sample mean refined
            output interval width.
        mean_abs_width_reduction: Mean absolute width reduction
            (baseline minus refined).
        mean_relative_width_reduction: Mean relative width reduction over
            samples with a defined per-sample relative reduction (see the
            module docstring).
        zero_width_baseline_intervals: Total number of output coordinates whose
            baseline interval width is exactly zero.
        measured_samples: Number of samples whose width statistics were
            measured (both methods produced output bounds).
    """

    mean_baseline_width: Optional[float]
    mean_refined_width: Optional[float]
    mean_abs_width_reduction: Optional[float]
    mean_relative_width_reduction: Optional[float]
    zero_width_baseline_intervals: int
    measured_samples: int


@dataclass(frozen=True)
class RuntimeMetrics:
    """Runtime metrics for one epsilon value.

    Every aggregate is computed over the measurements that are present; an
    aggregate over an empty measurement set is ``None`` and never zero.

    Attributes:
        mean_baseline_time: Mean baseline verification time.
        median_baseline_time: Median baseline verification time.
        mean_lp_refinement_time: Mean LP refinement stage time.
        median_lp_refinement_time: Median LP refinement stage time.
        mean_total_refined_time: Mean total refined verification time.
        median_total_refined_time: Median total refined verification time.
        baseline_times_measured: Number of baseline time measurements.
        lp_times_measured: Number of LP refinement time measurements.
        refined_times_measured: Number of total refined time measurements.
    """

    mean_baseline_time: Optional[float]
    median_baseline_time: Optional[float]
    mean_lp_refinement_time: Optional[float]
    median_lp_refinement_time: Optional[float]
    mean_total_refined_time: Optional[float]
    median_total_refined_time: Optional[float]
    baseline_times_measured: int
    lp_times_measured: int
    refined_times_measured: int


@dataclass(frozen=True)
class EpsilonMetrics:
    """All aggregated metrics for one epsilon value."""

    epsilon: float
    certification: CertificationMetrics
    bounds: BoundTighteningMetrics
    runtime: RuntimeMetrics


def certification_metrics(
    records: Sequence[SampleRecord],
    epsilon: float,
    total_test_samples: int,
    incorrect_excluded: int,
) -> CertificationMetrics:
    """Aggregate certification metrics over one epsilon's records.

    Args:
        records: Per-sample records of a single epsilon value.
        epsilon: The epsilon value the records belong to.
        total_test_samples: Size of the full test split.
        incorrect_excluded: Number of incorrectly classified samples excluded
            from the evaluation population during selection.

    Returns:
        The aggregated CertificationMetrics.

    Raises:
        ValueError: If total_test_samples or incorrect_excluded is negative.
    """
    if total_test_samples < 0:
        raise ValueError("total_test_samples must be nonnegative")
    if incorrect_excluded < 0:
        raise ValueError("incorrect_excluded must be nonnegative")

    evaluated = len(records)
    baseline_certified = _count_status(records, "baseline_certified")
    refined_certified = _count_status(records, "refined_certified")
    baseline_rate = (
        baseline_certified / evaluated if evaluated > 0 else None
    )
    refined_rate = refined_certified / evaluated if evaluated > 0 else None
    improvement = (
        100.0 * (refined_rate - baseline_rate)
        if baseline_rate is not None and refined_rate is not None
        else None
    )
    return CertificationMetrics(
        epsilon=float(epsilon),
        total_test_samples=total_test_samples,
        evaluated_samples=evaluated,
        incorrect_excluded=incorrect_excluded,
        baseline_certified=baseline_certified,
        refined_certified=refined_certified,
        baseline_rate=baseline_rate,
        refined_rate=refined_rate,
        improvement_pp=improvement,
        baseline_inconclusive=_count_status(records, "baseline_inconclusive"),
        refined_inconclusive=_count_status(records, "refined_inconclusive"),
        baseline_errors=sum(1 for record in records if record.baseline_status == _ERROR_STATUS),
        refined_errors=_count_status(records, "refined_error"),
    )


def bound_tightening_metrics(records: Sequence[SampleRecord]) -> BoundTighteningMetrics:
    """Aggregate output interval-width metrics over one epsilon's records.

    Records without measured width statistics (for example when the refined
    verification failed) contribute nothing; the means are never zero-fill.
    """
    measured = [
        record
        for record in records
        if record.baseline_width_mean is not None and record.refined_width_mean is not None
    ]
    relative = [
        record.rel_width_reduction_mean
        for record in records
        if record.rel_width_reduction_mean is not None
    ]
    zero_width = sum(
        record.zero_width_baseline_intervals or 0 for record in measured
    )
    return BoundTighteningMetrics(
        mean_baseline_width=_mean(
            [record.baseline_width_mean for record in measured]  # type: ignore[misc]
        ),
        mean_refined_width=_mean(
            [record.refined_width_mean for record in measured]  # type: ignore[misc]
        ),
        mean_abs_width_reduction=_mean(
            [record.abs_width_reduction_mean for record in measured]  # type: ignore[misc]
        ),
        mean_relative_width_reduction=_mean(relative),  # type: ignore[arg-type]
        zero_width_baseline_intervals=zero_width,
        measured_samples=len(measured),
    )


def runtime_metrics(records: Sequence[SampleRecord]) -> RuntimeMetrics:
    """Aggregate runtime metrics over one epsilon's records.

    Missing measurements (``None``) are excluded from every statistic; an
    aggregate with no measurements is ``None``, never zero.
    """
    baseline = _time_stats([record.baseline_time_seconds for record in records])
    lp = _time_stats([record.lp_refinement_time_seconds for record in records])
    total = _time_stats([record.refined_time_seconds for record in records])
    return RuntimeMetrics(
        mean_baseline_time=baseline[0],
        median_baseline_time=baseline[1],
        mean_lp_refinement_time=lp[0],
        median_lp_refinement_time=lp[1],
        mean_total_refined_time=total[0],
        median_total_refined_time=total[1],
        baseline_times_measured=baseline[2],
        lp_times_measured=lp[2],
        refined_times_measured=total[2],
    )


def _time_stats(
    values: Sequence[Optional[float]],
) -> tuple[Optional[float], Optional[float], int]:
    """Return ``(mean, median, measured_count)`` over the present values."""
    present = [float(value) for value in values if value is not None]
    return _mean(present), _median(present), len(present)


def aggregate_epsilon_metrics(
    records: Sequence[SampleRecord],
    epsilon: float,
    total_test_samples: int,
    incorrect_excluded: int,
) -> EpsilonMetrics:
    """Aggregate certification, bound-tightening, and runtime metrics.

    Args:
        records: Per-sample records belonging to exactly one epsilon value.
        epsilon: The epsilon value the records belong to.
        total_test_samples: Size of the full test split.
        incorrect_excluded: Incorrectly classified samples excluded from the
            evaluation population.

    Returns:
        The aggregated EpsilonMetrics for this epsilon.

    Raises:
        ValueError: If a record carries a different epsilon value.
    """
    for record in records:
        if record.epsilon != float(epsilon):
            raise ValueError(
                f"record for sample {record.sample_index} carries epsilon "
                f"{record.epsilon}, expected {epsilon}"
            )
    return EpsilonMetrics(
        epsilon=float(epsilon),
        certification=certification_metrics(
            records, epsilon, total_test_samples, incorrect_excluded
        ),
        bounds=bound_tightening_metrics(records),
        runtime=runtime_metrics(records),
    )
