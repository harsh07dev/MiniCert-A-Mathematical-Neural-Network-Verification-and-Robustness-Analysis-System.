"""Pure comparison logic for MiniCert evaluation experiments.

This module provides data models, matching epsilon resolution, and metric
extraction across multiple evaluation experiments. It stays Streamlit-free
so it is 100% unit-testable.

Rules enforced:
- If epsilon values differ, compare only matching epsilon values.
- If sample counts differ, clearly display and flag the sample counts.
- If random seeds differ, display that information and flag distinct samples.
- Never imply that two experiments use identical samples unless verified identical.
- Never treat missing metrics as zero.
- Clearly distinguish development experiments from configured experiments.
- Do not combine different experiment records into one aggregated result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

from dashboard.components import MISSING, format_optional
from dashboard.storage import ExperimentMetadata
from evaluation.experiment import ExperimentResult
from evaluation.metrics import EpsilonMetrics

__all__ = [
    "ComparisonItem",
    "ComparisonAnalysis",
    "build_comparison_analysis",
    "build_comparison_metric_table",
    "get_comparison_series_for_plotting",
]


@dataclass(frozen=True)
class ComparisonItem:
    """An experiment bundled with its metadata and full result."""

    metadata: ExperimentMetadata
    result: ExperimentResult

    @property
    def display_label(self) -> str:
        """Formatted human-readable experiment identifier."""
        type_str = self.metadata.experiment_type.capitalize()
        return f"{self.metadata.experiment_id} ({type_str}, n={self.metadata.sample_count}, seed={self.metadata.seed})"


@dataclass(frozen=True)
class ComparisonAnalysis:
    """Multi-experiment comparison analysis and matching resolution.

    Attributes:
        items: The compared experiment items in order.
        matching_epsilons: Perturbation radii common to ALL compared experiments.
        all_epsilons: Union of all perturbation radii across compared experiments.
        differing_epsilons: Mapping of experiment_id to epsilons not shared by all.
        sample_counts_differ: True if any experiments evaluated different sample counts.
        seeds_differ: True if any experiments used different random seeds.
        identical_samples: True ONLY if all experiments share the same seed, sample count,
            and dataset split, confirming identical sample populations.
        notices: Informative notes and caveats regarding comparability.
    """

    items: tuple[ComparisonItem, ...]
    matching_epsilons: tuple[float, ...]
    all_epsilons: tuple[float, ...]
    differing_epsilons: dict[str, tuple[float, ...]]
    sample_counts_differ: bool
    seeds_differ: bool
    identical_samples: bool
    notices: tuple[str, ...]

    @property
    def experiment_count(self) -> int:
        return len(self.items)


def build_comparison_analysis(
    items: Sequence[ComparisonItem],
) -> ComparisonAnalysis:
    """Analyze and resolve comparability constraints across experiments.

    Args:
        items: Sequence of ComparisonItems to compare (at least two).

    Returns:
        ComparisonAnalysis with matching epsilons, warnings, and comparability flags.

    Raises:
        ValueError: If fewer than two experiments are provided.
    """
    if len(items) < 2:
        raise ValueError("Comparison requires at least two experiments")

    # Find matching and all epsilons
    epsilon_sets = [set(item.metadata.epsilons) for item in items]
    common_set = set.intersection(*epsilon_sets)
    matching_epsilons = tuple(sorted(common_set))

    union_set = set.union(*epsilon_sets)
    all_epsilons = tuple(sorted(union_set))

    differing_epsilons: dict[str, tuple[float, ...]] = {}
    for item in items:
        unmatched = sorted(set(item.metadata.epsilons) - common_set)
        if unmatched:
            differing_epsilons[item.metadata.experiment_id] = tuple(unmatched)

    # Check sample counts and seeds
    sample_counts = {item.metadata.sample_count for item in items}
    sample_counts_differ = len(sample_counts) > 1

    seeds = {item.metadata.seed for item in items}
    seeds_differ = len(seeds) > 1

    # Check sample identity
    # Even if count and seed match, verify sample indices match if possible
    first_records = {r.sample_index for r in items[0].result.records}
    all_same_indices = all(
        {r.sample_index for r in item.result.records} == first_records
        for item in items
    )
    identical_samples = not sample_counts_differ and not seeds_differ and all_same_indices

    # Generate notices
    notices: list[str] = []

    if sample_counts_differ:
        counts_str = ", ".join(
            f"'{item.metadata.experiment_id}': n={item.metadata.sample_count}"
            for item in items
        )
        notices.append(
            f"Sample counts differ ({counts_str}). Certification rates and width statistics "
            "are computed over different sample population sizes."
        )

    if seeds_differ:
        seeds_str = ", ".join(
            f"'{item.metadata.experiment_id}': seed={item.metadata.seed}"
            for item in items
        )
        notices.append(
            f"Random seeds differ ({seeds_str}). Experiments evaluated different subsets of test samples."
        )

    if not identical_samples:
        notices.append(
            "Selected experiments do not evaluate identical sample sets. Do not assume sample-level equivalence."
        )
    else:
        notices.append(
            "Experiments evaluated the exact same test samples (matching sample count, random seed, and dataset split)."
        )

    if differing_epsilons:
        unmatched_details = "; ".join(
            f"'{k}' has extra epsilons {list(v)}" for k, v in differing_epsilons.items()
        )
        notices.append(
            f"Epsilon values differ across experiments. Only common epsilons {list(matching_epsilons)} "
            f"are compared. ({unmatched_details})"
        )

    types = {item.metadata.experiment_type for item in items}
    if len(types) > 1:
        notices.append(
            "Comparison includes both Development and Configured experiments. Development runs typically use smaller exploratory configurations."
        )

    return ComparisonAnalysis(
        items=tuple(items),
        matching_epsilons=matching_epsilons,
        all_epsilons=all_epsilons,
        differing_epsilons=differing_epsilons,
        sample_counts_differ=sample_counts_differ,
        seeds_differ=seeds_differ,
        identical_samples=identical_samples,
        notices=tuple(notices),
    )


def _find_epsilon_metrics(result: ExperimentResult, epsilon: float) -> Optional[EpsilonMetrics]:
    """Find the EpsilonMetrics entry matching epsilon, or None."""
    for entry in result.metrics:
        if math.isclose(entry.epsilon, epsilon, abs_tol=1e-9):
            return entry
    return None


def build_comparison_metric_table(
    analysis: ComparisonAnalysis, epsilon: float
) -> list[dict[str, Any]]:
    """Produce structured rows comparing the 10 core metrics for a matching epsilon.

    Every row has ``"metric"`` and one column per experiment (keyed by display_label).
    Missing values stay visibly missing (MISSING), never zero.

    Args:
        analysis: ComparisonAnalysis of the selected experiments.
        epsilon: The matching epsilon value to compare.

    Returns:
        List of row dicts for tabular display.
    """
    labels = [item.display_label for item in analysis.items]
    entries = [_find_epsilon_metrics(item.result, epsilon) for item in analysis.items]

    def _row(metric_name: str, values: Sequence[Any]) -> dict[str, Any]:
        row: dict[str, Any] = {"Metric": metric_name}
        for label, val in zip(labels, values, strict=True):
            row[label] = val
        return row

    def _pct(val: Optional[float]) -> str:
        if val is None:
            return MISSING
        return f"{val * 100:.2f}%"

    def _pp(val: Optional[float]) -> str:
        if val is None:
            return MISSING
        prefix = "+" if val > 0.0 else ""
        return f"{prefix}{val:.2f} pp"

    def _num(val: Optional[float], digits: int = 4) -> str:
        return format_optional(val, digits=digits)

    rows: list[dict[str, Any]] = []

    # 1. Baseline certification rate
    rows.append(
        _row(
            "1. Baseline certification rate",
            [_pct(e.certification.baseline_rate if e else None) for e in entries],
        )
    )

    # 2. Refined IBP+LP certification rate
    rows.append(
        _row(
            "2. Refined IBP+LP certification rate",
            [_pct(e.certification.refined_rate if e else None) for e in entries],
        )
    )

    # 3. Certification improvement (percentage points)
    rows.append(
        _row(
            "3. Certification improvement (pp)",
            [_pp(e.certification.improvement_pp if e else None) for e in entries],
        )
    )

    # 4. Baseline mean output interval width
    rows.append(
        _row(
            "4. Baseline mean output interval width",
            [_num(e.bounds.mean_baseline_width if e else None) for e in entries],
        )
    )

    # 5. Refined mean output interval width
    rows.append(
        _row(
            "5. Refined mean output interval width",
            [_num(e.bounds.mean_refined_width if e else None) for e in entries],
        )
    )

    # 6. Interval-width reduction
    def _reduction_str(e: Optional[EpsilonMetrics]) -> str:
        if e is None or e.bounds.mean_abs_width_reduction is None:
            return MISSING
        abs_val = e.bounds.mean_abs_width_reduction
        rel_val = e.bounds.mean_relative_width_reduction
        rel_str = f" ({rel_val * 100:.1f}%)" if rel_val is not None else ""
        return f"{abs_val:.4g}{rel_str}"

    rows.append(
        _row(
            "6. Interval-width reduction (abs & rel)",
            [_reduction_str(e) for e in entries],
        )
    )

    # 7. Baseline runtime
    rows.append(
        _row(
            "7. Baseline runtime (mean s)",
            [_num(e.runtime.mean_baseline_time if e else None) for e in entries],
        )
    )

    # 8. LP refinement runtime
    rows.append(
        _row(
            "8. LP refinement runtime (mean s)",
            [_num(e.runtime.mean_lp_refinement_time if e else None) for e in entries],
        )
    )

    # 9. Total refined verification runtime
    rows.append(
        _row(
            "9. Total refined verification runtime (mean s)",
            [_num(e.runtime.mean_total_refined_time if e else None) for e in entries],
        )
    )

    # 10. Number of errors and inconclusive outcomes
    def _outcomes_str(e: Optional[EpsilonMetrics]) -> str:
        if e is None:
            return MISSING
        cert = e.certification
        return (
            f"Base inconcl: {cert.baseline_inconclusive}, "
            f"Ref inconcl: {cert.refined_inconclusive}, "
            f"Errors: {cert.baseline_errors + cert.refined_errors}"
        )

    rows.append(
        _row(
            "10. Inconclusive & error counts",
            [_outcomes_str(e) for e in entries],
        )
    )

    return rows


def get_comparison_series_for_plotting(
    analysis: ComparisonAnalysis,
) -> dict[str, Any]:
    """Extract plot-ready data structures across matching epsilons.

    Returns:
        Dict with ``matching_epsilons`` and a list of experiment dictionaries
        suitable for feeding to Matplotlib comparison functions.
    """
    exp_series: list[dict[str, Any]] = []

    for item in analysis.items:
        data: dict[str, Any] = {
            "id": item.metadata.experiment_id,
            "label": item.display_label,
            "type": item.metadata.experiment_type,
            "sample_count": item.metadata.sample_count,
            "seed": item.metadata.seed,
            "per_epsilon": {},
        }
        for eps in analysis.matching_epsilons:
            m = _find_epsilon_metrics(item.result, eps)
            if m is not None:
                data["per_epsilon"][eps] = {
                    "baseline_rate_pct": (
                        m.certification.baseline_rate * 100.0
                        if m.certification.baseline_rate is not None
                        else None
                    ),
                    "refined_rate_pct": (
                        m.certification.refined_rate * 100.0
                        if m.certification.refined_rate is not None
                        else None
                    ),
                    "improvement_pp": m.certification.improvement_pp,
                    "baseline_width": m.bounds.mean_baseline_width,
                    "refined_width": m.bounds.mean_refined_width,
                    "abs_width_reduction": m.bounds.mean_abs_width_reduction,
                    "rel_width_reduction_pct": (
                        m.bounds.mean_relative_width_reduction * 100.0
                        if m.bounds.mean_relative_width_reduction is not None
                        else None
                    ),
                    "baseline_time": m.runtime.mean_baseline_time,
                    "lp_time": m.runtime.mean_lp_refinement_time,
                    "refined_total_time": m.runtime.mean_total_refined_time,
                }
            else:
                data["per_epsilon"][eps] = None
        exp_series.append(data)

    return {
        "matching_epsilons": list(analysis.matching_epsilons),
        "experiments": exp_series,
    }

