"""Dashboard service layer: individual verification and experiment execution.

These helpers orchestrate the existing public APIs only; they contain no
verification mathematics and no evaluation logic of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
import time
from typing import Any, Optional

import numpy as np

from abstract_domain.propagation import propagate_network
from evaluation.dataset import InputSample
from evaluation.experiment import ExperimentConfig, ExperimentResult, run_experiment
from models.model_loader import MathematicalNetwork
from verification.integration import verify_robustness_refined
from verification.pipeline import verify_robustness
from verification.regions import InputRegion

MNIST_PIXEL_LOWER = 0.0
MNIST_PIXEL_UPPER = 1.0


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def validate_epsilon(epsilon: Real) -> float:
    """Validate a dashboard-supplied epsilon as a finite nonnegative radius."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real scalar")
    radius = float(epsilon)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(f"epsilon must be finite and nonnegative, got {radius}")
    return radius


@dataclass(frozen=True, eq=False)
class IndividualOutcome:
    """Outcome of one individual verification query on the dashboard.

    A status of ``"ERROR"`` means the corresponding stage raised; its
    measurements stay ``None`` and are displayed as missing, never zero.
    """

    sample_index: int
    epsilon: float
    label: int
    predicted: int
    target_class: int
    baseline_status: Optional[str]
    baseline_margin: Optional[float]
    baseline_time_seconds: Optional[float]
    baseline_lower: np.ndarray = field(default_factory=lambda: np.empty(0))
    baseline_upper: np.ndarray = field(default_factory=lambda: np.empty(0))
    refined_status: Optional[str] = None
    refined_margin: Optional[float] = None
    refined_time_seconds: Optional[float] = None
    lp_refinement_time_seconds: Optional[float] = None
    refinement_status: Optional[str] = None
    refinement_targets: Optional[int] = None
    refinement_accepted: Optional[int] = None
    refinement_rejected: Optional[int] = None
    refined_lower: np.ndarray = field(default_factory=lambda: np.empty(0))
    refined_upper: np.ndarray = field(default_factory=lambda: np.empty(0))
    error: Optional[str] = None


def run_baseline_verification(
    network: MathematicalNetwork,
    sample: InputSample,
    epsilon: Real,
) -> IndividualOutcome:
    """Run baseline IBP verification for one sample at one epsilon."""
    radius = validate_epsilon(epsilon)
    region = InputRegion(
        sample.image,
        radius,
        input_lower=MNIST_PIXEL_LOWER,
        input_upper=MNIST_PIXEL_UPPER,
    )
    start = time.perf_counter()
    try:
        result = verify_robustness(network, region, sample.predicted)
        output_box = propagate_network(network, region)
        return IndividualOutcome(
            sample_index=sample.index,
            epsilon=radius,
            label=sample.label,
            predicted=sample.predicted,
            target_class=sample.predicted,
            baseline_status=result.status.value,
            baseline_margin=result.verification_margin,
            baseline_time_seconds=time.perf_counter() - start,
            baseline_lower=np.asarray(output_box.lower, dtype=np.float64),
            baseline_upper=np.asarray(output_box.upper, dtype=np.float64),
        )
    except Exception as exc:
        return IndividualOutcome(
            sample_index=sample.index,
            epsilon=radius,
            label=sample.label,
            predicted=sample.predicted,
            target_class=sample.predicted,
            baseline_status="ERROR",
            baseline_margin=None,
            baseline_time_seconds=time.perf_counter() - start,
            error=_format_error(exc),
        )


def run_refined_verification(
    network: MathematicalNetwork,
    sample: InputSample,
    epsilon: Real,
) -> IndividualOutcome:
    """Run LP-refined verification for one sample at one epsilon."""
    radius = validate_epsilon(epsilon)
    region = InputRegion(
        sample.image,
        radius,
        input_lower=MNIST_PIXEL_LOWER,
        input_upper=MNIST_PIXEL_UPPER,
    )
    start = time.perf_counter()
    try:
        refined = verify_robustness_refined(network, region, sample.predicted)
        refined_result = refined.refined_result
        refinement = refined.refinement
        output_bounds = refined.refined_trace.output_bounds
        return IndividualOutcome(
            sample_index=sample.index,
            epsilon=radius,
            label=sample.label,
            predicted=sample.predicted,
            target_class=sample.predicted,
            baseline_status=refined.ibp_result.status.value,
            baseline_margin=refined.ibp_result.verification_margin,
            baseline_time_seconds=None,
            refined_status=refined_result.status.value,
            refined_margin=refined_result.verification_margin,
            refined_time_seconds=time.perf_counter() - start,
            lp_refinement_time_seconds=refinement.execution_time_seconds,
            refinement_status=refinement.status.value,
            refinement_targets=refinement.num_targets,
            refinement_accepted=refinement.num_accepted,
            refinement_rejected=refinement.num_rejected,
            refined_lower=np.asarray(output_bounds.lower, dtype=np.float64),
            refined_upper=np.asarray(output_bounds.upper, dtype=np.float64),
        )
    except Exception as exc:
        return IndividualOutcome(
            sample_index=sample.index,
            epsilon=radius,
            label=sample.label,
            predicted=sample.predicted,
            target_class=sample.predicted,
            baseline_status=None,
            baseline_margin=None,
            baseline_time_seconds=None,
            refined_status="ERROR",
            refined_time_seconds=time.perf_counter() - start,
            error=_format_error(exc),
        )


def predict_image(network: MathematicalNetwork, image: Any) -> tuple[np.ndarray, int]:
    """Return the network logits and predicted class for one flattened image."""
    logits = np.asarray(network.forward(image), dtype=np.float64)
    return logits, int(np.argmax(logits))


def execute_experiment(
    config: ExperimentConfig,
    network: MathematicalNetwork,
    dataset: Any,
) -> ExperimentResult:
    """Run one configured experiment with pre-loaded network and dataset."""
    return run_experiment(config, network=network, dataset=dataset)


def execute_and_save_experiment(
    config: ExperimentConfig,
    network: MathematicalNetwork,
    dataset: Any,
    experiment_type: str = "configured",
    experiment_id: Optional[str] = None,
    base_dir: Optional[Any] = None,
) -> tuple[ExperimentResult, Any]:
    """Run an experiment and atomically persist it to storage.

    Returns:
        Tuple of (ExperimentResult, ExperimentMetadata).
    """
    from dashboard.storage import save_experiment

    result = execute_experiment(config, network, dataset)
    meta = save_experiment(
        result,
        experiment_type=experiment_type,
        experiment_id=experiment_id,
        base_dir=base_dir,
    )
    return result, meta

