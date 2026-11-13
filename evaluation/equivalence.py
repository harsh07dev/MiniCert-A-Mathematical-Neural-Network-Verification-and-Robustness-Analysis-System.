"""100-sample equivalence validation for L4-only vs. all-target LP refinement.

This module compares the two Phase 3 target-selection strategies over identical
MNIST test samples, target classes, and L-infinity input perturbation regions:

- **Mode A (Original all-target refinement)**:
  ``verify_robustness_refined(..., include_hidden_relu=True)``, which selects
  unstable hidden ReLU post-activation neurons plus the final output neurons.
- **Mode B (Optimized L4-only refinement)**:
  ``verify_robustness_refined(..., include_hidden_relu=False)`` (the
  production default), which selects only the final output neurons.

The module reuses MiniCert's existing model loading, deterministic dataset
selection, public verification pipeline, tolerance conventions, and JSON/CSV
export helpers without altering production defaults or historical experiments.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence
import uuid

import numpy as np

from evaluation.dataset import (
    MNIST_PIXEL_LOWER,
    MNIST_PIXEL_UPPER,
    InputSample,
    SampleSelection,
    load_mnist_test_dataset,
    select_samples,
    validate_input_bounds,
)
from evaluation.experiment import PROJECT_ROOT, load_trained_network
from evaluation.exports import write_summary_json
from models.model_loader import MathematicalNetwork
from verification.integration import RefinedVerificationResult, verify_robustness_refined
from verification.refinement import _TOLERANCE as REFINEMENT_TOLERANCE
from verification.regions import InputRegion

DEFAULT_EQUIVALENCE_TOLERANCE = float(REFINEMENT_TOLERANCE)

__all__ = [
    "DEFAULT_EQUIVALENCE_TOLERANCE",
    "EquivalenceConfig",
    "EquivalenceEpsilonSummary",
    "EquivalencePairRecord",
    "EquivalenceValidationResult",
    "aggregate_equivalence_metrics",
    "compare_sample_modes",
    "generate_equivalence_experiment_id",
    "run_equivalence_validation",
    "write_equivalence_csv",
]


def generate_equivalence_experiment_id(prefix: str = "equivalence") -> str:
    """Generate a unique experiment identifier with UTC timestamp and hex salt."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    salt = uuid.uuid4().hex[:6]
    clean_prefix = prefix.strip() or "equivalence"
    return f"{clean_prefix}_{stamp}_{salt}"


def _finite_real(value: object, name: str) -> float:
    """Validate and coerce one finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}")
    return scalar


def _format_error(exc: BaseException) -> str:
    """Format an exception cleanly as ``Type: message``."""
    return f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class EquivalenceConfig:
    """Immutable configuration for an L4-only vs. all-target equivalence study."""

    sample_count: int = 100
    epsilons: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1)
    seed: int = 39
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint: Path = PROJECT_ROOT / "saved_model.pth"
    output_dir: Path = PROJECT_ROOT / "results"
    experiment_name: str = ""
    tolerance: float = DEFAULT_EQUIVALENCE_TOLERANCE
    prediction_batch_size: int = 1024
    input_lower: float = MNIST_PIXEL_LOWER
    input_upper: float = MNIST_PIXEL_UPPER
    workers: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise TypeError("sample_count must be an integer")
        if self.sample_count < 1:
            raise ValueError(f"sample_count must be at least 1, got {self.sample_count}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

        epsilons = self.epsilons
        if isinstance(epsilons, (list, np.ndarray)):
            epsilons = tuple(epsilons)
        if not isinstance(epsilons, tuple) or not epsilons:
            raise ValueError("epsilons must be a non-empty sequence of radii")
        cleaned_eps: list[float] = []
        for idx, eps in enumerate(epsilons):
            val = _finite_real(eps, f"epsilons[{idx}]")
            if val < 0.0:
                raise ValueError(f"epsilons[{idx}] must be nonnegative, got {val}")
            cleaned_eps.append(val)
        object.__setattr__(self, "epsilons", tuple(cleaned_eps))

        tol = _finite_real(self.tolerance, "tolerance")
        if tol < 0.0:
            raise ValueError(f"tolerance must be nonnegative, got {tol}")
        object.__setattr__(self, "tolerance", tol)

        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if not isinstance(self.experiment_name, str):
            raise TypeError("experiment_name must be a string")
        if not self.experiment_name.strip():
            object.__setattr__(
                self,
                "experiment_name",
                f"equivalence_seed{self.seed}_n{self.sample_count}",
            )

        if isinstance(self.prediction_batch_size, bool) or not isinstance(
            self.prediction_batch_size, int
        ):
            raise TypeError("prediction_batch_size must be an integer")
        if self.prediction_batch_size < 1:
            raise ValueError(
                f"prediction_batch_size must be at least 1, got {self.prediction_batch_size}"
            )

        lower, upper = validate_input_bounds(self.input_lower, self.input_upper)
        object.__setattr__(self, "input_lower", lower)
        object.__setattr__(self, "input_upper", upper)

        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise TypeError("workers must be an integer")
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")
        object.__setattr__(self, "workers", self.workers)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        return {
            "sample_count": self.sample_count,
            "epsilons": list(self.epsilons),
            "seed": self.seed,
            "data_dir": str(self.data_dir),
            "checkpoint": str(self.checkpoint),
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "tolerance": self.tolerance,
            "prediction_batch_size": self.prediction_batch_size,
            "input_lower": self.input_lower,
            "input_upper": self.input_upper,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class _ModeOutcome:
    """Internal snapshot of executing one refinement mode on a single query."""

    targets: Optional[int]
    lp_solves: Optional[int]
    accepted: Optional[int]
    rejected: Optional[int]
    refinement_status: Optional[str]
    verdict: str
    margin: Optional[float]
    time_seconds: Optional[float]
    lower: Optional[tuple[float, ...]]
    upper: Optional[tuple[float, ...]]
    error: Optional[str]


def _run_single_mode(
    network: MathematicalNetwork,
    region: InputRegion,
    target_class: int,
    *,
    include_hidden_relu: bool,
) -> _ModeOutcome:
    """Execute one verification mode and capture all diagnostics or errors."""
    start = time.perf_counter()
    try:
        res: RefinedVerificationResult = (
            verify_robustness_refined(
                network,
                region,
                target_class,
                include_hidden_relu=True,
            )
            if include_hidden_relu
            else verify_robustness_refined(network, region, target_class)
        )
        elapsed = time.perf_counter() - start
        refinement = res.refinement
        lp_solves = 2 * len(refinement.outcomes)
        out_bounds = res.refined_trace.output_bounds
        lower = tuple(float(v) for v in np.asarray(out_bounds.lower, dtype=np.float64))
        upper = tuple(float(v) for v in np.asarray(out_bounds.upper, dtype=np.float64))
        return _ModeOutcome(
            targets=int(refinement.num_targets),
            lp_solves=int(lp_solves),
            accepted=int(refinement.num_accepted),
            rejected=int(refinement.num_rejected),
            refinement_status=refinement.status.value,
            verdict=res.refined_result.status.value,
            margin=float(res.refined_result.verification_margin),
            time_seconds=float(elapsed),
            lower=lower,
            upper=upper,
            error=None,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return _ModeOutcome(
            targets=None,
            lp_solves=None,
            accepted=None,
            rejected=None,
            refinement_status="ERROR",
            verdict="ERROR",
            margin=None,
            time_seconds=float(elapsed),
            lower=None,
            upper=None,
            error=_format_error(exc),
        )


@dataclass(frozen=True)
class EquivalencePairRecord:
    """Per-(sample, epsilon) comparison between Mode A (all) and Mode B (L4)."""

    sample_index: int
    label: int
    predicted: int
    target_class: int
    epsilon: float
    tolerance: float

    mode_a_targets: Optional[int]
    mode_a_lp_solves: Optional[int]
    mode_a_accepted: Optional[int]
    mode_a_rejected: Optional[int]
    mode_a_refinement_status: Optional[str]
    mode_a_verdict: str
    mode_a_margin: Optional[float]
    mode_a_time_seconds: Optional[float]
    mode_a_lower: Optional[tuple[float, ...]]
    mode_a_upper: Optional[tuple[float, ...]]
    mode_a_error: Optional[str]

    mode_b_targets: Optional[int]
    mode_b_lp_solves: Optional[int]
    mode_b_accepted: Optional[int]
    mode_b_rejected: Optional[int]
    mode_b_refinement_status: Optional[str]
    mode_b_verdict: str
    mode_b_margin: Optional[float]
    mode_b_time_seconds: Optional[float]
    mode_b_lower: Optional[tuple[float, ...]]
    mode_b_upper: Optional[tuple[float, ...]]
    mode_b_error: Optional[str]

    comparison_complete: bool
    max_lower_diff: Optional[float]
    max_upper_diff: Optional[float]
    worst_lower_class: Optional[int]
    worst_upper_class: Optional[int]
    margin_diff: Optional[float]
    verdict_mismatch: bool
    exceeds_tolerance: bool
    discrepancy_reason: Optional[str]

    CSV_FIELDNAMES: ClassVar[tuple[str, ...]] = (
        "sample_index",
        "label",
        "predicted",
        "target_class",
        "epsilon",
        "tolerance",
        "mode_a_targets",
        "mode_a_lp_solves",
        "mode_a_accepted",
        "mode_a_rejected",
        "mode_a_refinement_status",
        "mode_a_verdict",
        "mode_a_margin",
        "mode_a_time_seconds",
        "mode_a_lower",
        "mode_a_upper",
        "mode_a_error",
        "mode_b_targets",
        "mode_b_lp_solves",
        "mode_b_accepted",
        "mode_b_rejected",
        "mode_b_refinement_status",
        "mode_b_verdict",
        "mode_b_margin",
        "mode_b_time_seconds",
        "mode_b_lower",
        "mode_b_upper",
        "mode_b_error",
        "comparison_complete",
        "max_lower_diff",
        "max_upper_diff",
        "worst_lower_class",
        "worst_upper_class",
        "margin_diff",
        "verdict_mismatch",
        "exceeds_tolerance",
        "discrepancy_reason",
    )

    def to_row(self) -> dict[str, Any]:
        """Serialize this comparison record to a flat CSV-safe mapping."""
        return {
            "sample_index": self.sample_index,
            "label": self.label,
            "predicted": self.predicted,
            "target_class": self.target_class,
            "epsilon": self.epsilon,
            "tolerance": self.tolerance,
            "mode_a_targets": "" if self.mode_a_targets is None else self.mode_a_targets,
            "mode_a_lp_solves": "" if self.mode_a_lp_solves is None else self.mode_a_lp_solves,
            "mode_a_accepted": "" if self.mode_a_accepted is None else self.mode_a_accepted,
            "mode_a_rejected": "" if self.mode_a_rejected is None else self.mode_a_rejected,
            "mode_a_refinement_status": self.mode_a_refinement_status or "",
            "mode_a_verdict": self.mode_a_verdict,
            "mode_a_margin": "" if self.mode_a_margin is None else repr(self.mode_a_margin),
            "mode_a_time_seconds": (
                "" if self.mode_a_time_seconds is None else repr(self.mode_a_time_seconds)
            ),
            "mode_a_lower": "" if self.mode_a_lower is None else json.dumps(list(self.mode_a_lower)),
            "mode_a_upper": "" if self.mode_a_upper is None else json.dumps(list(self.mode_a_upper)),
            "mode_a_error": self.mode_a_error or "",
            "mode_b_targets": "" if self.mode_b_targets is None else self.mode_b_targets,
            "mode_b_lp_solves": "" if self.mode_b_lp_solves is None else self.mode_b_lp_solves,
            "mode_b_accepted": "" if self.mode_b_accepted is None else self.mode_b_accepted,
            "mode_b_rejected": "" if self.mode_b_rejected is None else self.mode_b_rejected,
            "mode_b_refinement_status": self.mode_b_refinement_status or "",
            "mode_b_verdict": self.mode_b_verdict,
            "mode_b_margin": "" if self.mode_b_margin is None else repr(self.mode_b_margin),
            "mode_b_time_seconds": (
                "" if self.mode_b_time_seconds is None else repr(self.mode_b_time_seconds)
            ),
            "mode_b_lower": "" if self.mode_b_lower is None else json.dumps(list(self.mode_b_lower)),
            "mode_b_upper": "" if self.mode_b_upper is None else json.dumps(list(self.mode_b_upper)),
            "mode_b_error": self.mode_b_error or "",
            "comparison_complete": self.comparison_complete,
            "max_lower_diff": "" if self.max_lower_diff is None else repr(self.max_lower_diff),
            "max_upper_diff": "" if self.max_upper_diff is None else repr(self.max_upper_diff),
            "worst_lower_class": "" if self.worst_lower_class is None else self.worst_lower_class,
            "worst_upper_class": "" if self.worst_upper_class is None else self.worst_upper_class,
            "margin_diff": "" if self.margin_diff is None else repr(self.margin_diff),
            "verdict_mismatch": self.verdict_mismatch,
            "exceeds_tolerance": self.exceeds_tolerance,
            "discrepancy_reason": self.discrepancy_reason or "",
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> EquivalencePairRecord:
        """Reconstruct an EquivalencePairRecord from its serialized dictionary row."""
        def _parse_opt_int(k: str) -> Optional[int]:
            v = row.get(k)
            return int(v) if v is not None and str(v).strip() != "" else None

        def _parse_opt_float(k: str) -> Optional[float]:
            v = row.get(k)
            return float(v) if v is not None and str(v).strip() != "" else None

        def _parse_tuple(k: str) -> Optional[tuple[float, ...]]:
            v = row.get(k)
            if v is None or str(v).strip() == "":
                return None
            return tuple(float(x) for x in json.loads(str(v)))

        return cls(
            sample_index=int(row["sample_index"]),
            label=int(row["label"]),
            predicted=int(row["predicted"]),
            target_class=int(row["target_class"]),
            epsilon=float(row["epsilon"]),
            tolerance=float(row["tolerance"]),
            mode_a_targets=_parse_opt_int("mode_a_targets"),
            mode_a_lp_solves=_parse_opt_int("mode_a_lp_solves"),
            mode_a_accepted=_parse_opt_int("mode_a_accepted"),
            mode_a_rejected=_parse_opt_int("mode_a_rejected"),
            mode_a_refinement_status=row.get("mode_a_refinement_status") or None,
            mode_a_verdict=str(row["mode_a_verdict"]),
            mode_a_margin=_parse_opt_float("mode_a_margin"),
            mode_a_time_seconds=_parse_opt_float("mode_a_time_seconds"),
            mode_a_lower=_parse_tuple("mode_a_lower"),
            mode_a_upper=_parse_tuple("mode_a_upper"),
            mode_a_error=row.get("mode_a_error") or None,
            mode_b_targets=_parse_opt_int("mode_b_targets"),
            mode_b_lp_solves=_parse_opt_int("mode_b_lp_solves"),
            mode_b_accepted=_parse_opt_int("mode_b_accepted"),
            mode_b_rejected=_parse_opt_int("mode_b_rejected"),
            mode_b_refinement_status=row.get("mode_b_refinement_status") or None,
            mode_b_verdict=str(row["mode_b_verdict"]),
            mode_b_margin=_parse_opt_float("mode_b_margin"),
            mode_b_time_seconds=_parse_opt_float("mode_b_time_seconds"),
            mode_b_lower=_parse_tuple("mode_b_lower"),
            mode_b_upper=_parse_tuple("mode_b_upper"),
            mode_b_error=row.get("mode_b_error") or None,
            comparison_complete=str(row.get("comparison_complete")).lower() in ("true", "1"),
            max_lower_diff=_parse_opt_float("max_lower_diff"),
            max_upper_diff=_parse_opt_float("max_upper_diff"),
            worst_lower_class=_parse_opt_int("worst_lower_class"),
            worst_upper_class=_parse_opt_int("worst_upper_class"),
            margin_diff=_parse_opt_float("margin_diff"),
            verdict_mismatch=str(row.get("verdict_mismatch")).lower() in ("true", "1"),
            exceeds_tolerance=str(row.get("exceeds_tolerance")).lower() in ("true", "1"),
            discrepancy_reason=row.get("discrepancy_reason") or None,
        )


def build_pair_record(
    sample: InputSample,
    epsilon: float,
    mode_a: _ModeOutcome,
    mode_b: _ModeOutcome,
    *,
    tolerance: float,
) -> EquivalencePairRecord:
    """Compare two mode outcomes explicitly without suppressing discrepancies."""
    reasons: list[str] = []
    if mode_a.error is not None:
        reasons.append(f"mode_a error ({mode_a.error})")
    if mode_b.error is not None:
        reasons.append(f"mode_b error ({mode_b.error})")

    has_bounds = (
        mode_a.lower is not None
        and mode_a.upper is not None
        and mode_b.lower is not None
        and mode_b.upper is not None
        and mode_a.margin is not None
        and mode_b.margin is not None
    )

    max_lower_diff: Optional[float] = None
    max_upper_diff: Optional[float] = None
    worst_lower_class: Optional[int] = None
    worst_upper_class: Optional[int] = None
    margin_diff: Optional[float] = None
    complete = False

    if mode_a.error is None and mode_b.error is None and has_bounds:
        a_l = np.asarray(mode_a.lower, dtype=np.float64)
        b_l = np.asarray(mode_b.lower, dtype=np.float64)
        a_u = np.asarray(mode_a.upper, dtype=np.float64)
        b_u = np.asarray(mode_b.upper, dtype=np.float64)
        if (
            a_l.shape == b_l.shape == a_u.shape == b_u.shape
            and a_l.ndim == 1
            and a_l.size > 0
            and np.all(np.isfinite(a_l))
            and np.all(np.isfinite(b_l))
            and np.all(np.isfinite(a_u))
            and np.all(np.isfinite(b_u))
            and math.isfinite(mode_a.margin)
            and math.isfinite(mode_b.margin)
        ):
            lower_diffs = np.abs(a_l - b_l)
            upper_diffs = np.abs(a_u - b_u)
            worst_lower_class = int(np.argmax(lower_diffs))
            worst_upper_class = int(np.argmax(upper_diffs))
            max_lower_diff = float(lower_diffs[worst_lower_class])
            max_upper_diff = float(upper_diffs[worst_upper_class])
            margin_diff = abs(float(mode_a.margin) - float(mode_b.margin))
            complete = True
        else:
            reasons.append("incomplete or non-finite output bounds")
    elif not reasons:
        reasons.append("incomplete comparison outcome")

    verdict_mismatch = mode_a.verdict != mode_b.verdict
    if verdict_mismatch:
        reasons.append(f"verdict mismatch ({mode_a.verdict} vs {mode_b.verdict})")

    exceeds_tol = False
    if not complete:
        exceeds_tol = True
    else:
        assert max_lower_diff is not None and max_upper_diff is not None and margin_diff is not None
        if max_lower_diff > tolerance:
            exceeds_tol = True
            reasons.append(
                f"lower bound diff {max_lower_diff:.3e} > tol {tolerance:.3e} at class {worst_lower_class}"
            )
        if max_upper_diff > tolerance:
            exceeds_tol = True
            reasons.append(
                f"upper bound diff {max_upper_diff:.3e} > tol {tolerance:.3e} at class {worst_upper_class}"
            )
        if margin_diff > tolerance:
            exceeds_tol = True
            reasons.append(f"margin diff {margin_diff:.3e} > tol {tolerance:.3e}")

    return EquivalencePairRecord(
        sample_index=sample.index,
        label=sample.label,
        predicted=sample.predicted,
        target_class=sample.predicted,
        epsilon=float(epsilon),
        tolerance=float(tolerance),
        mode_a_targets=mode_a.targets,
        mode_a_lp_solves=mode_a.lp_solves,
        mode_a_accepted=mode_a.accepted,
        mode_a_rejected=mode_a.rejected,
        mode_a_refinement_status=mode_a.refinement_status,
        mode_a_verdict=mode_a.verdict,
        mode_a_margin=mode_a.margin,
        mode_a_time_seconds=mode_a.time_seconds,
        mode_a_lower=mode_a.lower,
        mode_a_upper=mode_a.upper,
        mode_a_error=mode_a.error,
        mode_b_targets=mode_b.targets,
        mode_b_lp_solves=mode_b.lp_solves,
        mode_b_accepted=mode_b.accepted,
        mode_b_rejected=mode_b.rejected,
        mode_b_refinement_status=mode_b.refinement_status,
        mode_b_verdict=mode_b.verdict,
        mode_b_margin=mode_b.margin,
        mode_b_time_seconds=mode_b.time_seconds,
        mode_b_lower=mode_b.lower,
        mode_b_upper=mode_b.upper,
        mode_b_error=mode_b.error,
        comparison_complete=complete,
        max_lower_diff=max_lower_diff,
        max_upper_diff=max_upper_diff,
        worst_lower_class=worst_lower_class,
        worst_upper_class=worst_upper_class,
        margin_diff=margin_diff,
        verdict_mismatch=verdict_mismatch,
        exceeds_tolerance=exceeds_tol,
        discrepancy_reason="; ".join(reasons) if reasons else None,
    )


def compare_sample_modes(
    network: MathematicalNetwork,
    sample: InputSample,
    epsilon: Real,
    *,
    tolerance: Real = DEFAULT_EQUIVALENCE_TOLERANCE,
    input_lower: Real = MNIST_PIXEL_LOWER,
    input_upper: Real = MNIST_PIXEL_UPPER,
) -> EquivalencePairRecord:
    """Evaluate one (sample, epsilon) pair under Mode A and Mode B on identical inputs."""
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(
            f"network must be a MathematicalNetwork, got {type(network).__name__}"
        )
    radius = _finite_real(epsilon, "epsilon")
    if radius < 0.0:
        raise ValueError(f"epsilon must be nonnegative, got {radius}")
    tol = _finite_real(tolerance, "tolerance")
    if tol < 0.0:
        raise ValueError(f"tolerance must be nonnegative, got {tol}")
    lower_bound, upper_bound = validate_input_bounds(input_lower, input_upper)

    region = InputRegion(
        sample.image, radius, input_lower=lower_bound, input_upper=upper_bound
    )
    target_class = sample.predicted

    mode_a = _run_single_mode(network, region, target_class, include_hidden_relu=True)
    mode_b = _run_single_mode(network, region, target_class, include_hidden_relu=False)
    return build_pair_record(sample, radius, mode_a, mode_b, tolerance=tol)


def _worker_compare_entry(
    payload: tuple[MathematicalNetwork, InputSample, float, float, float, float]
) -> EquivalencePairRecord:
    """Worker task helper for concurrent process execution."""
    net, sample, eps, tol, lower, upper = payload
    return compare_sample_modes(
        net,
        sample,
        eps,
        tolerance=tol,
        input_lower=lower,
        input_upper=upper,
    )


@dataclass(frozen=True)
class EquivalenceEpsilonSummary:
    """Aggregated equivalence statistics for one epsilon (or overall when epsilon=None)."""

    epsilon: Optional[float]
    total_pairs: int
    completed_pairs: int
    incomplete_pairs: int
    max_lower_diff: Optional[float]
    max_upper_diff: Optional[float]
    max_margin_diff: Optional[float]
    verdict_mismatches: int
    discrepancies_over_tolerance: int
    mode_a_total_lp_solves: int
    mode_b_total_lp_solves: int
    mode_a_rejected_targets_total: int
    mode_b_rejected_targets_total: int
    mode_a_mean_runtime: Optional[float]
    mode_a_median_runtime: Optional[float]
    mode_b_mean_runtime: Optional[float]
    mode_b_median_runtime: Optional[float]
    mean_runtime_speedup: Optional[float]
    median_runtime_speedup: Optional[float]
    worst_case: Optional[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to a JSON-safe dictionary."""
        return {
            "epsilon": self.epsilon,
            "total_pairs": self.total_pairs,
            "completed_pairs": self.completed_pairs,
            "incomplete_pairs": self.incomplete_pairs,
            "max_lower_diff": self.max_lower_diff,
            "max_upper_diff": self.max_upper_diff,
            "max_margin_diff": self.max_margin_diff,
            "verdict_mismatches": self.verdict_mismatches,
            "discrepancies_over_tolerance": self.discrepancies_over_tolerance,
            "mode_a_total_lp_solves": self.mode_a_total_lp_solves,
            "mode_b_total_lp_solves": self.mode_b_total_lp_solves,
            "mode_a_rejected_targets_total": self.mode_a_rejected_targets_total,
            "mode_b_rejected_targets_total": self.mode_b_rejected_targets_total,
            "mode_a_mean_runtime": self.mode_a_mean_runtime,
            "mode_a_median_runtime": self.mode_a_median_runtime,
            "mode_b_mean_runtime": self.mode_b_mean_runtime,
            "mode_b_median_runtime": self.mode_b_median_runtime,
            "mean_runtime_speedup": self.mean_runtime_speedup,
            "median_runtime_speedup": self.median_runtime_speedup,
            "worst_case": self.worst_case,
        }


def aggregate_equivalence_metrics(
    records: Sequence[EquivalencePairRecord],
    *,
    epsilon: Optional[float] = None,
) -> EquivalenceEpsilonSummary:
    """Aggregate comparison metrics over a sequence of pair records."""
    subset = (
        list(records)
        if epsilon is None
        else [rec for rec in records if math.isclose(rec.epsilon, epsilon, rel_tol=0.0, abs_tol=1e-15)]
    )
    completed = [rec for rec in subset if rec.comparison_complete]
    incomplete_count = len(subset) - len(completed)

    max_lower: Optional[float] = None
    max_upper: Optional[float] = None
    max_margin: Optional[float] = None
    worst_case: Optional[dict[str, Any]] = None

    if completed:
        lower_vals = [float(rec.max_lower_diff) for rec in completed if rec.max_lower_diff is not None]
        upper_vals = [float(rec.max_upper_diff) for rec in completed if rec.max_upper_diff is not None]
        margin_vals = [float(rec.margin_diff) for rec in completed if rec.margin_diff is not None]
        max_lower = max(lower_vals)
        max_upper = max(upper_vals)
        max_margin = max(margin_vals)

        worst_rec = max(
            completed,
            key=lambda rec: max(
                float(rec.max_lower_diff or 0.0),
                float(rec.max_upper_diff or 0.0),
                float(rec.margin_diff or 0.0),
            ),
        )
        worst_case = {
            "sample_index": worst_rec.sample_index,
            "epsilon": worst_rec.epsilon,
            "max_lower_diff": worst_rec.max_lower_diff,
            "worst_lower_class": worst_rec.worst_lower_class,
            "max_upper_diff": worst_rec.max_upper_diff,
            "worst_upper_class": worst_rec.worst_upper_class,
            "margin_diff": worst_rec.margin_diff,
        }

    times_a = [rec.mode_a_time_seconds for rec in subset if rec.mode_a_time_seconds is not None]
    times_b = [rec.mode_b_time_seconds for rec in subset if rec.mode_b_time_seconds is not None]

    mean_a = float(statistics.mean(times_a)) if times_a else None
    med_a = float(statistics.median(times_a)) if times_a else None
    mean_b = float(statistics.mean(times_b)) if times_b else None
    med_b = float(statistics.median(times_b)) if times_b else None

    mean_speedup = (
        float(mean_a / mean_b) if (mean_a is not None and mean_b is not None and mean_b > 0.0) else None
    )
    med_speedup = (
        float(med_a / med_b) if (med_a is not None and med_b is not None and med_b > 0.0) else None
    )

    return EquivalenceEpsilonSummary(
        epsilon=epsilon,
        total_pairs=len(subset),
        completed_pairs=len(completed),
        incomplete_pairs=incomplete_count,
        max_lower_diff=max_lower,
        max_upper_diff=max_upper,
        max_margin_diff=max_margin,
        verdict_mismatches=sum(1 for rec in subset if rec.verdict_mismatch),
        discrepancies_over_tolerance=sum(1 for rec in subset if rec.exceeds_tolerance),
        mode_a_total_lp_solves=sum(rec.mode_a_lp_solves or 0 for rec in subset),
        mode_b_total_lp_solves=sum(rec.mode_b_lp_solves or 0 for rec in subset),
        mode_a_rejected_targets_total=sum(rec.mode_a_rejected or 0 for rec in subset),
        mode_b_rejected_targets_total=sum(rec.mode_b_rejected or 0 for rec in subset),
        mode_a_mean_runtime=mean_a,
        mode_a_median_runtime=med_a,
        mode_b_mean_runtime=mean_b,
        mode_b_median_runtime=med_b,
        mean_runtime_speedup=mean_speedup,
        median_runtime_speedup=med_speedup,
        worst_case=worst_case,
    )


def write_equivalence_csv(
    records: Sequence[EquivalencePairRecord],
    path: Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Write detailed per-pair equivalence records to CSV atomically."""
    for rec in records:
        if not isinstance(rec, EquivalencePairRecord):
            raise TypeError(
                f"records must contain EquivalencePairRecord entries, got {type(rec).__name__}"
            )
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp_{target.name}_{uuid.uuid4().hex[:6]}"
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EquivalencePairRecord.CSV_FIELDNAMES))
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_row())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


@dataclass(frozen=True)
class EquivalenceValidationResult:
    """Complete result of one Mode A vs. Mode B equivalence experiment."""

    experiment_id: str
    timestamp: str
    config: EquivalenceConfig
    selection: SampleSelection
    epsilon_summaries: tuple[EquivalenceEpsilonSummary, ...]
    overall_summary: EquivalenceEpsilonSummary
    records: tuple[EquivalencePairRecord, ...]
    network_source: str
    dataset_size: int

    @property
    def equivalent(self) -> bool:
        """True iff all pairs completed with zero mismatches and zero over-tolerance diffs."""
        ov = self.overall_summary
        return (
            ov.total_pairs > 0
            and ov.incomplete_pairs == 0
            and ov.verdict_mismatches == 0
            and ov.discrepancies_over_tolerance == 0
        )

    def failing_records(self) -> tuple[EquivalencePairRecord, ...]:
        """Return any incomplete, mismatched, or over-tolerance records."""
        return tuple(
            rec
            for rec in self.records
            if (not rec.comparison_complete) or rec.verdict_mismatch or rec.exceeds_tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the equivalence experiment."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.config.experiment_name,
            "experiment_type": "equivalence_validation",
            "timestamp": self.timestamp,
            "status": "Equivalent" if self.equivalent else "DiscrepancyDetected",
            "equivalent": self.equivalent,
            "config": self.config.as_dict(),
            "model": {
                "checkpoint": str(self.config.checkpoint),
                "source_model": self.network_source,
            },
            "dataset": {
                "name": "MNIST",
                "split": "test",
                "total_samples": self.dataset_size,
            },
            "sample_selection": {
                "requested": self.selection.requested_count,
                "selected": self.selection.selected_count,
                "incorrect_excluded": self.selection.incorrect_excluded,
                "scanned": self.selection.scanned,
                "seed": self.selection.seed,
            },
            "overall": self.overall_summary.to_dict(),
            "by_epsilon": [entry.to_dict() for entry in self.epsilon_summaries],
            "failing_pairs": [
                {
                    "sample_index": rec.sample_index,
                    "epsilon": rec.epsilon,
                    "verdict_a": rec.mode_a_verdict,
                    "verdict_b": rec.mode_b_verdict,
                    "max_lower_diff": rec.max_lower_diff,
                    "worst_lower_class": rec.worst_lower_class,
                    "max_upper_diff": rec.max_upper_diff,
                    "worst_upper_class": rec.worst_upper_class,
                    "margin_diff": rec.margin_diff,
                    "reason": rec.discrepancy_reason,
                }
                for rec in self.failing_records()
            ],
        }

    def save(
        self,
        output_dir: Optional[Path] = None,
        *,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Export summary JSON and detailed CSV under ``output_dir``."""
        target_dir = Path(self.config.output_dir if output_dir is None else output_dir)
        csv_path = target_dir / f"{self.config.experiment_name}_samples.csv"
        json_path = target_dir / f"{self.config.experiment_name}_summary.json"
        if not overwrite and (csv_path.exists() or json_path.exists()):
            raise FileExistsError(
                f"Export already exists under {target_dir} for '{self.config.experiment_name}'"
            )
        write_equivalence_csv(self.records, csv_path, overwrite=overwrite)
        write_summary_json(self.to_dict(), json_path)
        return {"samples_csv": csv_path, "summary_json": json_path}


def run_equivalence_validation(
    config: Optional[EquivalenceConfig] = None,
    *,
    network: Optional[MathematicalNetwork] = None,
    dataset: Any = None,
    experiment_id: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, EquivalencePairRecord], None]] = None,
) -> EquivalenceValidationResult:
    """Run Mode A vs. Mode B equivalence validation over shared samples and epsilons."""
    cfg = config if config is not None else EquivalenceConfig()
    if not isinstance(cfg, EquivalenceConfig):
        raise TypeError(f"config must be an EquivalenceConfig, got {type(cfg).__name__}")

    net = load_trained_network(cfg.checkpoint) if network is None else network
    if not isinstance(net, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(net).__name__}")

    test_dataset = load_mnist_test_dataset(cfg.data_dir) if dataset is None else dataset
    selection = select_samples(
        net,
        test_dataset,
        sample_count=cfg.sample_count,
        seed=cfg.seed,
        batch_size=cfg.prediction_batch_size,
    )

    checkpoint_file = Path(cfg.output_dir) / f".{cfg.experiment_name}_checkpoint.jsonl"
    cached_records: dict[tuple[float, int], EquivalencePairRecord] = {}
    if checkpoint_file.is_file():
        try:
            with checkpoint_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        row = json.loads(line_str)
                        rec = EquivalencePairRecord.from_row(row)
                        cached_records[(round(rec.epsilon, 6), rec.sample_index)] = rec
        except Exception:
            cached_records.clear()

    total_pairs = len(cfg.epsilons) * len(selection.samples)
    pending_tasks: list[tuple[float, InputSample]] = []
    ordered_pairs: list[tuple[float, InputSample]] = []
    for eps in cfg.epsilons:
        for sample in selection.samples:
            ordered_pairs.append((eps, sample))
            if (round(eps, 6), sample.index) not in cached_records:
                pending_tasks.append((eps, sample))

    def _append_checkpoint(rec: EquivalencePairRecord) -> None:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_row()) + "\n")

    completed_count = len(cached_records)
    if progress_callback is not None and cached_records:
        for idx_so_far, (eps, sample) in enumerate(ordered_pairs, start=1):
            key = (round(eps, 6), sample.index)
            if key in cached_records:
                progress_callback(idx_so_far, total_pairs, cached_records[key])

    if pending_tasks:
        if cfg.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            task_payloads = [
                (
                    (net, sample, eps, cfg.tolerance, cfg.input_lower, cfg.input_upper),
                    (round(eps, 6), sample.index),
                )
                for eps, sample in pending_tasks
            ]
            with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
                future_to_key = {
                    executor.submit(_worker_compare_entry, payload): key
                    for payload, key in task_payloads
                }
                for future in as_completed(future_to_key):
                    rec = future.result()
                    key = (round(rec.epsilon, 6), rec.sample_index)
                    cached_records[key] = rec
                    _append_checkpoint(rec)
                    completed_count += 1
                    if progress_callback is not None:
                        progress_callback(completed_count, total_pairs, rec)
        else:
            for eps, sample in pending_tasks:
                rec = compare_sample_modes(
                    net,
                    sample,
                    eps,
                    tolerance=cfg.tolerance,
                    input_lower=cfg.input_lower,
                    input_upper=cfg.input_upper,
                )
                key = (round(rec.epsilon, 6), rec.sample_index)
                cached_records[key] = rec
                _append_checkpoint(rec)
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, total_pairs, rec)

    # Reconstruct in deterministic order: epsilon-major, then selection order
    records: list[EquivalencePairRecord] = []
    for eps, sample in ordered_pairs:
        records.append(cached_records[(round(eps, 6), sample.index)])

    # All pairs done, remove checkpoint file
    checkpoint_file.unlink(missing_ok=True)

    eps_summaries = tuple(
        aggregate_equivalence_metrics(records, epsilon=eps) for eps in cfg.epsilons
    )
    overall = aggregate_equivalence_metrics(records, epsilon=None)
    exp_id = experiment_id or generate_equivalence_experiment_id(cfg.experiment_name)
    ts = datetime.now(timezone.utc).isoformat()

    return EquivalenceValidationResult(
        experiment_id=exp_id,
        timestamp=ts,
        config=cfg,
        selection=selection,
        epsilon_summaries=eps_summaries,
        overall_summary=overall,
        records=tuple(records),
        network_source=net.source_model,
        dataset_size=len(test_dataset),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point to execute and save an equivalence validation study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument(
        "--epsilon",
        type=float,
        action="append",
        dest="epsilons",
        default=None,
    )
    parser.add_argument("--name", type=str, default="equivalence_seed39_n100")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_EQUIVALENCE_TOLERANCE)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of concurrent worker processes (default 1 for memory safety on Windows)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    epsilons = tuple(args.epsilons) if args.epsilons else (0.01, 0.03, 0.05, 0.1)
    cfg = EquivalenceConfig(
        sample_count=args.sample_count,
        epsilons=epsilons,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.name,
        tolerance=args.tolerance,
        workers=args.workers,
    )

    t_start = time.perf_counter()

    def _on_progress(done: int, total: int, rec: EquivalencePairRecord) -> None:
        if done % 20 == 0 or done == total or rec.verdict_mismatch or rec.exceeds_tolerance:
            elapsed = time.perf_counter() - t_start
            print(
                f"[{done}/{total}] eps={rec.epsilon} idx={rec.sample_index} "
                f"mismatch={rec.verdict_mismatch} tol_exceeded={rec.exceeds_tolerance} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    result = run_equivalence_validation(cfg, progress_callback=_on_progress)
    paths = result.save(overwrite=args.overwrite)
    ov = result.overall_summary
    print(f"Saved JSON: {paths['summary_json']}", flush=True)
    print(f"Saved CSV:  {paths['samples_csv']}", flush=True)
    print(
        f"Overall: pairs={ov.completed_pairs}/{ov.total_pairs} "
        f"mismatches={ov.verdict_mismatches} over_tol={ov.discrepancies_over_tolerance} "
        f"max_l={ov.max_lower_diff} max_u={ov.max_upper_diff} max_m={ov.max_margin_diff}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

