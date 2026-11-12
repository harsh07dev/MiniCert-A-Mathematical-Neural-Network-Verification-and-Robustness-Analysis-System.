"""Pure data transformations between evaluation results and dashboard views.

This module must stay Streamlit-free so the helpers are unit-testable outside
the dashboard runtime. Every chart consumes real ExperimentResult data; no
demonstration numbers are hardcoded anywhere.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from evaluation.experiment import ExperimentResult
from evaluation.metrics import EpsilonMetrics, SampleRecord

STATUS_DISPLAY = {
    "CERTIFIED": "CERTIFIED",
    "INCONCLUSIVE": "INCONCLUSIVE",
    "ERROR": "ERROR",
    "COMPLETED": "Completed",
    "Completed": "Completed",
    "FAILED": "Failed",
    "Failed": "Failed",
    "INCOMPLETE": "Incomplete",
    "Incomplete": "Incomplete",
    "CORRUPTED": "Corrupted",
    "Corrupted": "Corrupted",
    "TIGHTENED": "TIGHTENED",
    "UNCHANGED": "UNCHANGED",
}

PER_SAMPLE_COLUMNS = (
    "sample_index",
    "label",
    "predicted",
    "epsilon",
    "baseline_status",
    "baseline_margin",
    "baseline_time_seconds",
    "refined_status",
    "refined_margin",
    "refined_time_seconds",
    "lp_refinement_time_seconds",
    "refinement_targets",
    "refinement_accepted",
    "refinement_rejected",
    "baseline_width_mean",
    "refined_width_mean",
    "error",
)

MISSING = "\u2014"


def status_display(status: str) -> str:
    """Return the display label for a verification or experiment status string."""
    if not status:
        return "UNKNOWN"
    if status in STATUS_DISPLAY:
        return STATUS_DISPLAY[status]
    if status.upper() in STATUS_DISPLAY:
        return STATUS_DISPLAY[status.upper()]
    return "UNKNOWN"


def format_optional(value: Any, digits: int = 4) -> str:
    """Format an optional measurement; missing values stay visibly missing."""
    if value is None:
        return MISSING
    return f"{float(value):.{digits}g}"


def certification_rate_series(
    metrics: Sequence[EpsilonMetrics],
) -> list[dict[str, Any]]:
    """Return per-epsilon certification percentages for both methods.

    Rates are stored as fractions; the chart series carries percentages.
    A missing rate (no evaluated samples) stays None, never zero.
    """
    series: list[dict[str, Any]] = []
    for entry in metrics:
        certification = entry.certification
        baseline_rate = certification.baseline_rate
        refined_rate = certification.refined_rate
        series.append(
            {
                "epsilon": entry.epsilon,
                "baseline_percent": None
                if baseline_rate is None
                else baseline_rate * 100.0,
                "refined_percent": None
                if refined_rate is None
                else refined_rate * 100.0,
            }
        )
    return series


def width_series(metrics: Sequence[EpsilonMetrics]) -> list[dict[str, Any]]:
    """Return per-epsilon mean output interval widths for both methods."""
    return [
        {
            "epsilon": entry.epsilon,
            "baseline_width": entry.bounds.mean_baseline_width,
            "refined_width": entry.bounds.mean_refined_width,
        }
        for entry in metrics
    ]


def runtime_series(metrics: Sequence[EpsilonMetrics]) -> list[dict[str, Any]]:
    """Return per-epsilon mean runtimes in seconds.

    All three series use the aggregated mean measurements; a missing mean
    (no measurement at all) stays None, never zero.
    """
    return [
        {
            "epsilon": entry.epsilon,
            "baseline_time": entry.runtime.mean_baseline_time,
            "lp_refinement_time": entry.runtime.mean_lp_refinement_time,
            "refined_total_time": entry.runtime.mean_total_refined_time,
        }
        for entry in metrics
    ]


def improvement_series(metrics: Sequence[EpsilonMetrics]) -> list[dict[str, Any]]:
    """Return per-epsilon absolute certification improvement in percentage points."""
    return [
        {"epsilon": entry.epsilon, "improvement_pp": entry.certification.improvement_pp}
        for entry in metrics
    ]


def parse_epsilon_input(text: str) -> tuple[float, ...]:
    """Parse a comma-separated epsilon text field into an ordered tuple.

    Raises:
        ValueError: If the text is empty or an entry is not a float.
    """
    if not isinstance(text, str):
        raise TypeError("epsilon input must be a string")
    entries = [part.strip() for part in text.split(",") if part.strip()]
    if not entries:
        raise ValueError("At least one epsilon value is required")
    try:
        values = tuple(float(entry) for entry in entries)
    except ValueError as exc:
        raise ValueError(f"Invalid epsilon value: {exc}") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Epsilon values must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("Epsilon values must be nonnegative")
    return values


def records_table_rows(result: ExperimentResult) -> list[dict[str, str]]:
    """Build per-sample display rows with missing values kept distinct from zero."""
    return _rows_from_records(result.records)


def _rows_from_records(records: Sequence[SampleRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "sample_index": str(record.sample_index),
                "label": str(record.label),
                "predicted": str(record.predicted),
                "epsilon": f"{record.epsilon:.6g}",
                "baseline_status": status_display(record.baseline_status),
                "baseline_margin": format_optional(record.baseline_margin),
                "baseline_time_seconds": format_optional(record.baseline_time_seconds),
                "refined_status": status_display(record.refined_status),
                "refined_margin": format_optional(record.refined_margin),
                "refined_time_seconds": format_optional(record.refined_time_seconds),
                "lp_refinement_time_seconds": format_optional(
                    record.lp_refinement_time_seconds
                ),
                "refinement_targets": format_optional(record.refinement_targets, 0),
                "refinement_accepted": format_optional(record.refinement_accepted, 0),
                "refinement_rejected": format_optional(record.refinement_rejected, 0),
                "baseline_width_mean": format_optional(record.baseline_width_mean),
                "refined_width_mean": format_optional(record.refined_width_mean),
                "error": record.error if record.error else "",
            }
        )
    return rows


HISTORY_TABLE_COLUMNS = (
    "Experiment ID",
    "Type",
    "Date",
    "Samples",
    "Epsilon values",
    "Seed",
    "Status",
)


def history_table_rows(metadata_list: Sequence[Any]) -> list[dict[str, Any]]:
    """Format experiment metadata for the Experiment History summary table."""
    rows: list[dict[str, Any]] = []
    for meta in metadata_list:
        ts_str = meta.timestamp
        if "T" in ts_str:
            date_display = ts_str[:19].replace("T", " ")
        else:
            date_display = ts_str if ts_str else MISSING
        epsilons_str = ", ".join(f"{float(e):g}" for e in meta.epsilons) if meta.epsilons else MISSING
        rows.append(
            {
                "Experiment ID": meta.experiment_id,
                "Type": meta.experiment_type.capitalize(),
                "Date": date_display,
                "Samples": meta.sample_count,
                "Epsilon values": epsilons_str,
                "Seed": meta.seed,
                "Status": meta.status,
            }
        )
    return rows

