"""Per-sample execution of both verification methods.

This module drives the existing public verification APIs —
:func:`verification.pipeline.verify_robustness` for the baseline IBP verdict
and :func:`verification.integration.verify_robustness_refined` for the
LP-refined verdict — over identical :class:`~verification.regions.InputRegion`
queries and records both outcomes, their execution times, and their output
interval widths in one :class:`~evaluation.metrics.SampleRecord`.

No verification mathematics is reimplemented here: the module only
orchestrates existing public functions, times them, and measures the output
bounds they produce.
"""

from __future__ import annotations

import math
from numbers import Real
import time
from typing import Any, Optional

import numpy as np

from abstract_domain.propagation import propagate_network
from evaluation.dataset import (
    MNIST_PIXEL_LOWER,
    MNIST_PIXEL_UPPER,
    InputSample,
    validate_input_bounds,
)
from evaluation.metrics import SampleRecord, width_statistics
from models.model_loader import MathematicalNetwork
from verification.integration import verify_robustness_refined
from verification.pipeline import verify_robustness
from verification.regions import InputRegion

__all__ = ["evaluate_sample"]


def evaluate_sample(
    network: MathematicalNetwork,
    sample: InputSample,
    epsilon: Real,
    *,
    input_lower: Real = MNIST_PIXEL_LOWER,
    input_upper: Real = MNIST_PIXEL_UPPER,
) -> SampleRecord:
    """Verify one sample at one epsilon with both methods and record the outcome.

    The bounded input region is the L-infinity ball of radius ``epsilon``
    around the sample, clipped to the valid input pixel range. The nominal
    prediction is used as the target class for both methods, so both verdicts
    refer to identical samples, regions, and targets.

    The baseline verdict is produced by :func:`verify_robustness`; the refined
    verdict by :func:`verify_robustness_refined` (which preserves its own IBP
    verdict and never upgrades an inconclusive result without certified
    bounds). A method that raises is recorded with status ``"ERROR"``, its
    unavailable measurements stay ``None``, and evaluation continues.

    Output interval widths require both methods' output bounds: the baseline
    bounds come from an untimed :func:`propagate_network` call (a pure public
    IBP query, far cheaper than the LP stage), and the refined bounds from the
    refined verification's downstream trace. When either method failed, no
    width statistics are reported rather than zeros.

    Args:
        network: An extracted float64 MathematicalNetwork.
        sample: The selected evaluation sample.
        epsilon: Nonnegative finite L-infinity perturbation radius.
        input_lower: Valid input lower bound (MNIST pixels: 0.0).
        input_upper: Valid input upper bound (MNIST pixels: 1.0).

    Returns:
        The SampleRecord for this (sample, epsilon) evaluation.

    Raises:
        TypeError: If epsilon or an input bound is not a real scalar, or the
            network has the wrong type.
        ValueError: If epsilon is negative or non-finite, the input bounds are
            invalid, or the sample lies outside the valid input range.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(
            f"network must be a MathematicalNetwork, got {type(network).__name__}"
        )
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real scalar")
    radius = float(epsilon)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(f"epsilon must be finite and nonnegative, got {radius}")
    lower_bound, upper_bound = validate_input_bounds(input_lower, input_upper)

    region = InputRegion(
        sample.image, radius, input_lower=lower_bound, input_upper=upper_bound
    )
    target_class = sample.predicted

    baseline_status = "ERROR"
    baseline_margin: Optional[float] = None
    baseline_time: Optional[float] = None
    baseline_error: Optional[str] = None
    baseline_lower: Optional[np.ndarray] = None
    baseline_upper_bounds: Optional[np.ndarray] = None

    start = time.perf_counter()
    try:
        baseline = verify_robustness(network, region, target_class)
        baseline_time = time.perf_counter() - start
        baseline_status = baseline.status.value
        baseline_margin = baseline.verification_margin
        output_box = propagate_network(network, region)
        baseline_lower = np.asarray(output_box.lower, dtype=np.float64)
        baseline_upper_bounds = np.asarray(output_box.upper, dtype=np.float64)
    except Exception as exc:
        baseline_time = time.perf_counter() - start
        baseline_error = _format_error(exc)

    refined_status = "ERROR"
    refined_margin: Optional[float] = None
    refined_time: Optional[float] = None
    lp_time: Optional[float] = None
    refinement_status: Optional[str] = None
    refinement_targets: Optional[int] = None
    refinement_accepted: Optional[int] = None
    refinement_rejected: Optional[int] = None
    refined_lower: Optional[np.ndarray] = None
    refined_upper: Optional[np.ndarray] = None
    refined_error: Optional[str] = None

    start = time.perf_counter()
    try:
        refined = verify_robustness_refined(network, region, target_class)
        refined_time = time.perf_counter() - start
        refined_status = refined.refined_result.status.value
        refined_margin = refined.refined_result.verification_margin
        refinement = refined.refinement
        lp_time = refinement.execution_time_seconds
        refinement_status = refinement.status.value
        refinement_targets = refinement.num_targets
        refinement_accepted = refinement.num_accepted
        refinement_rejected = refinement.num_rejected
        refined_output = refined.refined_trace.output_bounds
        refined_lower = np.asarray(refined_output.lower, dtype=np.float64)
        refined_upper = np.asarray(refined_output.upper, dtype=np.float64)
    except Exception as exc:
        refined_time = time.perf_counter() - start
        refined_error = _format_error(exc)

    widths: Optional[dict[str, Any]] = None
    if baseline_lower is not None and refined_lower is not None:
        widths = width_statistics(
            baseline_lower, baseline_upper_bounds, refined_lower, refined_upper
        )

    error_parts = []
    if baseline_error is not None:
        error_parts.append(f"baseline verification failed ({baseline_error})")
    if refined_error is not None:
        error_parts.append(f"refined verification failed ({refined_error})")
    error_text = "; ".join(error_parts) if error_parts else None

    return SampleRecord(
        sample_index=sample.index,
        label=sample.label,
        predicted=sample.predicted,
        target_class=target_class,
        epsilon=radius,
        baseline_status=baseline_status,
        baseline_margin=baseline_margin,
        baseline_time_seconds=baseline_time,
        refined_status=refined_status,
        refined_margin=refined_margin,
        refined_time_seconds=refined_time,
        lp_refinement_time_seconds=lp_time,
        refinement_status=refinement_status,
        refinement_targets=refinement_targets,
        refinement_accepted=refinement_accepted,
        refinement_rejected=refinement_rejected,
        baseline_width_mean=None if widths is None else widths["baseline_width_mean"],
        refined_width_mean=None if widths is None else widths["refined_width_mean"],
        abs_width_reduction_mean=None
        if widths is None
        else widths["abs_width_reduction_mean"],
        rel_width_reduction_mean=None
        if widths is None
        else widths["rel_width_reduction_mean"],
        zero_width_baseline_intervals=None
        if widths is None
        else widths["zero_width_baseline_intervals"],
        error=error_text,
    )


def _format_error(exc: BaseException) -> str:
    """Format one exception as a stable ``Type: message`` string."""
    return f"{type(exc).__name__}: {exc}"
