"""MiniCert experimental evaluation engine (Phase 4A).

This package measures whether LP-based bound refinement improves the
certification capability of Interval Bound Propagation (IBP) for neural
network robustness verification. It compares, on identical samples, target
classes, and perturbation regions:

1. Baseline IBP verification (:func:`verification.pipeline.verify_robustness`).
2. IBP verification with LP-based refinement
   (:func:`verification.integration.verify_robustness_refined`).

The package contains no verification mathematics of its own; it orchestrates
the existing public verification APIs, aggregates experimental metrics, and
exports reproducible CSV/JSON results.

Typical usage::

    from evaluation import ExperimentConfig, run_experiment

    result = run_experiment(ExperimentConfig(sample_count=100, seed=42))
    paths = result.save()
"""

from __future__ import annotations

from evaluation.dataset import (
    DEFAULT_PREDICTION_BATCH_SIZE,
    MNIST_PIXEL_LOWER,
    MNIST_PIXEL_UPPER,
    InputSample,
    SampleSelection,
    collect_nominal_predictions,
    flatten_image,
    load_mnist_test_dataset,
    select_samples,
    validate_input_bounds,
)
from evaluation.experiment import (
    ExperimentConfig,
    ExperimentResult,
    load_trained_network,
    run_experiment,
)
from evaluation.exports import write_samples_csv, write_summary_json
from evaluation.metrics import (
    BoundTighteningMetrics,
    CertificationMetrics,
    EpsilonMetrics,
    RuntimeMetrics,
    SampleRecord,
    aggregate_epsilon_metrics,
    bound_tightening_metrics,
    certification_metrics,
    interval_widths,
    runtime_metrics,
    width_statistics,
)
from evaluation.runner import evaluate_sample

__all__ = [
    "MNIST_PIXEL_LOWER",
    "MNIST_PIXEL_UPPER",
    "DEFAULT_PREDICTION_BATCH_SIZE",
    "InputSample",
    "SampleSelection",
    "flatten_image",
    "validate_input_bounds",
    "load_mnist_test_dataset",
    "collect_nominal_predictions",
    "select_samples",
    "ExperimentConfig",
    "ExperimentResult",
    "load_trained_network",
    "run_experiment",
    "write_samples_csv",
    "write_summary_json",
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
    "evaluate_sample",
]
