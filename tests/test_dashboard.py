"""Tests for the Streamlit dashboard helper modules.

The dashboard UI itself (dashboard/app.py) imports Streamlit and is exercised
manually; these tests cover the Streamlit-free layers instead:

- ``dashboard.components``: status display mapping, metric extraction, chart
  data preparation, epsilon parsing, per-sample table rows.
- ``dashboard.visualizations``: figure construction from real series data,
  empty-result fallbacks, missing values kept distinct from zero.
- ``dashboard.services``: individual verification orchestration (success and
  error paths), epsilon validation, prediction, experiment delegation.
- Export parity: files written by ``ExperimentResult.save`` match the
  evaluation engine's ``to_row`` / ``to_dict`` serialization contracts.

No test launches a browser, imports Streamlit, or executes a real LP
experiment. Verification engines are monkeypatched in the service tests.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dashboard import components
from dashboard import services
from dashboard import storage
from dashboard import comparison
from dashboard import visualizations
from evaluation.experiment import ExperimentConfig, ExperimentResult
from evaluation.metrics import (
    BoundTighteningMetrics,
    CertificationMetrics,
    EpsilonMetrics,
    RuntimeMetrics,
    SampleRecord,
)
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification import VerificationResult, VerificationStatus

MISSING = components.MISSING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _metrics(
    epsilon: float = 0.03,
    baseline_rate: float | None = None,
    refined_rate: float | None = None,
    improvement_pp: float | None = None,
    baseline_width: float | None = None,
    refined_width: float | None = None,
    baseline_time: float | None = None,
    lp_time: float | None = None,
    refined_time: float | None = None,
) -> EpsilonMetrics:
    """Build one EpsilonMetrics entry from the dashboard-facing aggregates."""
    certification = CertificationMetrics(
        epsilon=epsilon,
        total_test_samples=10000,
        evaluated_samples=0 if baseline_rate is None else 100,
        incorrect_excluded=12,
        baseline_certified=0,
        refined_certified=0,
        baseline_rate=baseline_rate,
        refined_rate=refined_rate,
        improvement_pp=improvement_pp,
        baseline_inconclusive=0,
        refined_inconclusive=0,
        baseline_errors=0,
        refined_errors=0,
    )
    bounds = BoundTighteningMetrics(
        mean_baseline_width=baseline_width,
        mean_refined_width=refined_width,
        mean_abs_width_reduction=None,
        mean_relative_width_reduction=None,
        zero_width_baseline_intervals=0,
        measured_samples=0 if baseline_width is None else 1,
    )
    runtime = RuntimeMetrics(
        mean_baseline_time=baseline_time,
        median_baseline_time=baseline_time,
        mean_lp_refinement_time=lp_time,
        median_lp_refinement_time=lp_time,
        mean_total_refined_time=refined_time,
        median_total_refined_time=refined_time,
        baseline_times_measured=0 if baseline_time is None else 1,
        lp_times_measured=0 if lp_time is None else 1,
        refined_times_measured=0 if refined_time is None else 1,
    )
    return EpsilonMetrics(epsilon=epsilon, certification=certification, bounds=bounds, runtime=runtime)


def _record(sample_index: int = 0, **overrides) -> SampleRecord:
    """Build one per-sample record; optional measurements default to None."""
    defaults: dict = dict(
        sample_index=sample_index,
        label=7,
        predicted=7,
        target_class=7,
        epsilon=0.03,
        baseline_status="CERTIFIED",
        baseline_margin=1.25,
        baseline_time_seconds=0.5,
        refined_status="INCONCLUSIVE",
        refined_margin=None,
        refined_time_seconds=None,
        lp_refinement_time_seconds=0.2,
        refinement_status="TIGHTENED",
        refinement_targets=3,
        refinement_accepted=2,
        refinement_rejected=1,
        baseline_width_mean=4.0,
        refined_width_mean=2.5,
        abs_width_reduction_mean=1.5,
        rel_width_reduction_mean=0.375,
        zero_width_baseline_intervals=0,
        error=None,
    )
    defaults.update(overrides)
    return SampleRecord(**defaults)


def _result(
    records,
    metrics=None,
    output_dir: Path = Path("unused_output"),
    epsilons: tuple[float, ...] = (0.03,),
    sample_count: int = 2,
    seed: int = 42,
) -> ExperimentResult:
    """Build an ExperimentResult around fabricated records/metrics."""
    config = ExperimentConfig(
        sample_count=sample_count,
        epsilons=epsilons,
        seed=seed,
        data_dir=Path("unused_data"),
        checkpoint=Path("unused.pth"),
        output_dir=output_dir,
        experiment_name="dashboard_test",
    )
    from evaluation.dataset import SampleSelection

    selection = SampleSelection(
        samples=(),
        requested_count=sample_count,
        incorrect_excluded=0,
        scanned=sample_count,
        seed=seed,
    )
    return ExperimentResult(
        config=config,
        selection=selection,
        metrics=tuple(metrics or ()),
        records=tuple(records),
        network_source="MiniCertNet",
        dataset_size=10000,
    )


def _network() -> MathematicalNetwork:
    """A tiny 4 -> Linear -> ReLU -> 2 network used by the service tests."""
    weight = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    bias = np.array([0.0, 0.0], dtype=np.float64)
    return MathematicalNetwork(
        layers=(
            LinearLayer(
                position=0,
                weight=weight,
                bias=bias,
                input_features=4,
                output_features=2,
            ),
            ReLULayer(position=1),
        ),
        input_features=4,
        output_features=2,
        source_model="TestNet",
    )


def _sample() -> object:
    from evaluation.dataset import InputSample

    return InputSample(index=0, image=np.full(4, 0.25), label=3, predicted=1)


# ---------------------------------------------------------------------------
# Status display mapping
# ---------------------------------------------------------------------------


def test_status_display_maps_known_statuses() -> None:
    assert components.status_display("CERTIFIED") == "CERTIFIED"
    assert components.status_display("INCONCLUSIVE") == "INCONCLUSIVE"
    assert components.status_display("ERROR") == "ERROR"


def test_status_display_maps_unknown_status_to_unknown() -> None:
    assert components.status_display("PENDING") == "UNKNOWN"
    assert components.status_display("") == "UNKNOWN"


# ---------------------------------------------------------------------------
# Missing-value formatting
# ---------------------------------------------------------------------------


def test_format_optional_keeps_missing_distinct_from_zero() -> None:
    assert components.format_optional(None) == MISSING
    assert components.format_optional(0.0) == "0"
    assert components.format_optional(1.5, digits=2) == "1.5"


# ---------------------------------------------------------------------------
# Epsilon parsing
# ---------------------------------------------------------------------------


def test_parse_epsilon_input_valid() -> None:
    assert components.parse_epsilon_input("0.01, 0.03, 0.05") == (0.01, 0.03, 0.05)
    assert components.parse_epsilon_input("0.03") == (0.03,)
    assert components.parse_epsilon_input("  0.1  ,0.2") == (0.1, 0.2)


def test_parse_epsilon_input_invalid() -> None:
    with pytest.raises(ValueError):
        components.parse_epsilon_input("")
    with pytest.raises(ValueError):
        components.parse_epsilon_input("   ,  ")
    with pytest.raises(ValueError):
        components.parse_epsilon_input("0.1, not-a-number")
    with pytest.raises(ValueError):
        components.parse_epsilon_input("-0.1")
    with pytest.raises(ValueError):
        components.parse_epsilon_input("inf")
    with pytest.raises(TypeError):
        components.parse_epsilon_input(0.1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chart data preparation (metric extraction)
# ---------------------------------------------------------------------------


def test_certification_rate_series_uses_percentages() -> None:
    metrics = [
        _metrics(epsilon=0.01, baseline_rate=0.80, refined_rate=0.85, improvement_pp=5.0),
        _metrics(epsilon=0.03, baseline_rate=0.60, refined_rate=0.72, improvement_pp=12.0),
    ]
    series = components.certification_rate_series(metrics)
    assert series == [
        {"epsilon": 0.01, "baseline_percent": 80.0, "refined_percent": 85.0},
        {"epsilon": 0.03, "baseline_percent": 60.0, "refined_percent": 72.0},
    ]


def test_certification_rate_series_keeps_missing_as_none() -> None:
    series = components.certification_rate_series(
        [_metrics(epsilon=0.05, baseline_rate=None, refined_rate=None)]
    )
    assert series == [
        {"epsilon": 0.05, "baseline_percent": None, "refined_percent": None}
    ]


def test_width_series_extracts_means() -> None:
    metrics = [
        _metrics(epsilon=0.01, baseline_width=4.0, refined_width=2.5),
        _metrics(epsilon=0.03, baseline_width=None, refined_width=None),
    ]
    assert components.width_series(metrics) == [
        {"epsilon": 0.01, "baseline_width": 4.0, "refined_width": 2.5},
        {"epsilon": 0.03, "baseline_width": None, "refined_width": None},
    ]


def test_runtime_series_extracts_means() -> None:
    metrics = [
        _metrics(
            epsilon=0.01,
            baseline_time=0.5,
            lp_time=0.2,
            refined_time=0.7,
        ),
        _metrics(epsilon=0.03),
    ]
    assert components.runtime_series(metrics) == [
        {
            "epsilon": 0.01,
            "baseline_time": 0.5,
            "lp_refinement_time": 0.2,
            "refined_total_time": 0.7,
        },
        {
            "epsilon": 0.03,
            "baseline_time": None,
            "lp_refinement_time": None,
            "refined_total_time": None,
        },
    ]


def test_improvement_series_extracts_percentage_points() -> None:
    metrics = [
        _metrics(epsilon=0.01, improvement_pp=5.0),
        _metrics(epsilon=0.03, improvement_pp=None),
    ]
    assert components.improvement_series(metrics) == [
        {"epsilon": 0.01, "improvement_pp": 5.0},
        {"epsilon": 0.03, "improvement_pp": None},
    ]


def test_series_empty_metrics_produces_empty_lists() -> None:
    assert components.certification_rate_series([]) == []
    assert components.width_series([]) == []
    assert components.runtime_series([]) == []
    assert components.improvement_series([]) == []


# ---------------------------------------------------------------------------
# Per-sample table rows
# ---------------------------------------------------------------------------


def test_records_table_rows_keep_missing_distinct_from_zero() -> None:
    missing = _record(sample_index=1, refined_margin=None, refinement_targets=None)
    result = _result([_record(sample_index=0), missing])
    rows = components.records_table_rows(result)
    assert len(rows) == 2
    first = rows[0]
    assert first["baseline_status"] == "CERTIFIED"
    assert first["baseline_margin"] == "1.25"
    assert first["refined_status"] == "INCONCLUSIVE"
    assert first["refined_margin"] == MISSING
    assert first["refinement_targets"] == "3"
    assert first["error"] == ""
    second = rows[1]
    assert second["refinement_targets"] == MISSING
    assert second["sample_index"] == "1"


def test_records_table_rows_zero_values_stay_zero() -> None:
    zeroed = _record(
        baseline_margin=0.0,
        baseline_time_seconds=0.0,
        refinement_accepted=0,
    )
    row = components.records_table_rows(_result([zeroed]))[0]
    assert row["baseline_margin"] == "0"
    assert row["baseline_time_seconds"] == "0"
    assert row["refinement_accepted"] == "0"


def test_records_table_rows_empty_result() -> None:
    assert components.records_table_rows(_result([])) == []


# ---------------------------------------------------------------------------
# Visualization figure construction
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass


def _chart_texts(axis) -> list[str]:
    return [text.get_text() for text in axis.texts]


def test_plot_functions_produce_figures_with_axes() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    series = [
        {
            "epsilon": 0.01,
            "baseline_percent": 80.0,
            "refined_percent": 85.0,
            "baseline_width": 4.0,
            "refined_width": 2.5,
            "baseline_time": 0.5,
            "lp_refinement_time": 0.2,
            "refined_total_time": 0.7,
            "improvement_pp": 5.0,
        },
        {
            "epsilon": 0.03,
            "baseline_percent": None,
            "refined_percent": 60.0,
            "baseline_width": None,
            "refined_width": None,
            "baseline_time": None,
            "lp_refinement_time": None,
            "refined_total_time": None,
            "improvement_pp": None,
        },
    ]
    for builder in (
        visualizations.plot_certification_rates,
        visualizations.plot_interval_widths,
        visualizations.plot_runtime,
        visualizations.plot_improvement,
    ):
        figure = builder(series)
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Epsilon"
        assert axis.get_title()
        assert tuple(figure.get_size_inches()) == visualizations.FIGURE_SIZE
        assert axis.title.get_color() == visualizations.TEXT_COLOR
        assert axis.xaxis.label.get_color() == visualizations.TEXT_COLOR
        assert axis.yaxis.label.get_color() == visualizations.TEXT_COLOR


def test_plot_functions_handle_empty_series() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    for builder in (
        visualizations.plot_certification_rates,
        visualizations.plot_interval_widths,
        visualizations.plot_runtime,
        visualizations.plot_improvement,
    ):
        figure = builder([])
        assert len(figure.axes) == 1
        assert any("no " in text.lower() for text in _chart_texts(figure.axes[0]))


def test_certification_rate_single_zero_epsilon_is_visible() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_certification_rates(
        [{"epsilon": 0.03, "baseline_percent": 0.0, "refined_percent": 0.0}]
    )
    axis = figure.axes[0]
    assert axis.get_ylim() == (0.0, 100.0)
    assert axis.get_ylabel() == "Certification rate (%)"
    assert len(axis.lines) == 2
    assert all(line.get_marker() in ("o", "s") for line in axis.lines)
    assert axis.lines[0].get_xdata()[0] != axis.lines[1].get_xdata()[0]
    assert _chart_texts(axis).count("0%") == 2
    assert [label.get_text() for label in axis.get_xticklabels()] == ["0.03"]


def test_certification_rate_multiple_epsilons_have_percentage_labels() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_certification_rates(
        [
            {"epsilon": 0.01, "baseline_percent": 80.0, "refined_percent": 85.0},
            {"epsilon": 0.03, "baseline_percent": 60.0, "refined_percent": 72.5},
        ]
    )
    axis = figure.axes[0]
    for expected in ("80%", "85%", "60%", "72.5%"):
        assert expected in _chart_texts(axis)
    assert [label.get_text() for label in axis.get_xticklabels()] == ["0.01", "0.03"]


def test_interval_width_zero_and_missing_values_are_distinct() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_interval_widths(
        [
            {"epsilon": 0.01, "baseline_width": 0.0, "refined_width": 0.0},
            {"epsilon": 0.03, "baseline_width": None, "refined_width": 2.5},
        ]
    )
    axis = figure.axes[0]
    assert axis.get_ylabel() == "Mean output interval width (logit units)"
    assert [patch.get_height() for patch in axis.patches] == [0.0, 0.0, 2.5]
    assert _chart_texts(axis).count("0") == 2
    assert _chart_texts(axis).count("n/a") == 1
    assert len(axis.collections) == 2


def test_interval_width_reduction_is_annotated() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_interval_widths(
        [{"epsilon": 0.01, "baseline_width": 4.0, "refined_width": 2.5}]
    )
    assert "37.5% narrower" in _chart_texts(figure.axes[0])


def test_runtime_wide_range_uses_log_scale_and_seconds() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_runtime(
        [
            {
                "epsilon": 0.03,
                "baseline_time": 0.0004,
                "lp_refinement_time": 6.3,
                "refined_total_time": 6.4,
            }
        ]
    )
    axis = figure.axes[0]
    assert axis.get_yscale() == "log"
    assert axis.get_ylabel() == "Mean runtime (s, log scale)"
    for expected in ("0.0004", "6.3", "6.4"):
        assert expected in _chart_texts(axis)


def test_runtime_missing_observations_are_not_zero_bars() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_runtime(
        [
            {
                "epsilon": 0.03,
                "baseline_time": None,
                "lp_refinement_time": 0.2,
                "refined_total_time": float("nan"),
            }
        ]
    )
    axis = figure.axes[0]
    assert [patch.get_height() for patch in axis.patches] == [0.2]
    assert _chart_texts(axis).count("n/a") == 2
    assert "0" not in _chart_texts(axis)


def test_improvement_zero_has_visible_marker_label_and_reference() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_improvement(
        [{"epsilon": 0.03, "improvement_pp": 0.0}]
    )
    axis = figure.axes[0]
    assert [patch.get_height() for patch in axis.patches] == [0.0]
    assert len(axis.collections) == 1
    assert "0 pp" in _chart_texts(axis)
    assert axis.get_ylim()[0] < 0.0 < axis.get_ylim()[1]
    assert any(
        len(line.get_ydata()) == 2 and all(value == 0.0 for value in line.get_ydata())
        for line in axis.lines
    )


def test_improvement_positive_negative_and_zero_are_labeled() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_improvement(
        [
            {"epsilon": 0.01, "improvement_pp": 5.0},
            {"epsilon": 0.03, "improvement_pp": -3.0},
            {"epsilon": 0.05, "improvement_pp": 0.0},
        ]
    )
    axis = figure.axes[0]
    assert [patch.get_height() for patch in axis.patches] == [5.0, -3.0, 0.0]
    for expected in ("+5 pp", "-3 pp", "0 pp"):
        assert expected in _chart_texts(axis)
    assert axis.get_ylim()[0] < -3.0
    assert axis.get_ylim()[1] > 5.0
    assert axis.get_ylabel() == "Certification improvement (percentage points)"


def test_improvement_missing_values_are_not_zero_observations() -> None:
    pytest.importorskip("matplotlib")
    from dashboard import visualizations

    figure = visualizations.plot_improvement(
        [
            {"epsilon": 0.01, "improvement_pp": None},
            {"epsilon": 0.03, "improvement_pp": float("nan")},
        ]
    )
    axis = figure.axes[0]
    assert len(axis.patches) == 0
    assert len(axis.collections) == 0
    assert _chart_texts(axis).count("n/a") == 2
    assert "0 pp" not in _chart_texts(axis)


# ---------------------------------------------------------------------------
# Service layer: epsilon validation
# ---------------------------------------------------------------------------


def test_validate_epsilon_accepts_finite_nonnegative() -> None:
    assert services.validate_epsilon(0.05) == 0.05
    assert services.validate_epsilon(0) == 0.0
    assert services.validate_epsilon(np.float64(0.1)) == 0.1


def test_validate_epsilon_rejects_invalid() -> None:
    with pytest.raises(TypeError):
        services.validate_epsilon(True)
    with pytest.raises(TypeError):
        services.validate_epsilon("0.1")
    with pytest.raises(ValueError):
        services.validate_epsilon(-0.1)
    with pytest.raises(ValueError):
        services.validate_epsilon(math.inf)
    with pytest.raises(ValueError):
        services.validate_epsilon(math.nan)


# ---------------------------------------------------------------------------
# Service layer: individual verification orchestration
# ---------------------------------------------------------------------------


def test_run_baseline_verification_error_path(monkeypatch) -> None:
    def boom(network, region, target_class):
        raise RuntimeError("propagation exploded")

    monkeypatch.setattr(services, "verify_robustness", boom)
    outcome = services.run_baseline_verification(_network(), _sample(), 0.05)
    assert outcome.baseline_status == "ERROR"
    assert outcome.baseline_margin is None
    assert outcome.refined_status is None
    assert outcome.error == "RuntimeError: propagation exploded"
    assert outcome.baseline_lower.size == 0
    assert outcome.epsilon == 0.05


def test_run_baseline_verification_success_path(monkeypatch) -> None:
    def fake_verify(network, region, target_class):
        return VerificationResult(
            status=VerificationStatus.CERTIFIED,
            target_class=target_class,
            target_lower_bound=2.0,
            max_competitor_upper_bound=1.0,
            explanation="fake certified verdict",
        )

    monkeypatch.setattr(services, "verify_robustness", fake_verify)
    outcome = services.run_baseline_verification(_network(), _sample(), 0.05)
    assert outcome.baseline_status == "CERTIFIED"
    assert outcome.baseline_margin == pytest.approx(1.0)
    assert outcome.error is None
    assert outcome.baseline_time_seconds is not None
    assert outcome.baseline_lower.shape == (2,)
    assert outcome.baseline_upper.shape == (2,)
    assert np.all(outcome.baseline_lower <= outcome.baseline_upper)


def test_run_refined_verification_error_path(monkeypatch) -> None:
    def boom(network, region, target_class, target_neurons=None):
        raise ValueError("LP solve failed")

    monkeypatch.setattr(services, "verify_robustness_refined", boom)
    outcome = services.run_refined_verification(_network(), _sample(), 0.05)
    assert outcome.refined_status == "ERROR"
    assert outcome.baseline_status is None
    assert outcome.baseline_margin is None
    assert outcome.error == "ValueError: LP solve failed"
    assert outcome.refined_lower.size == 0


def test_run_refined_verification_success_path(monkeypatch) -> None:
    lower = np.array([0.5, -0.5])
    upper = np.array([1.5, 0.5])

    def fake_refined(network, region, target_class, target_neurons=None):
        return SimpleNamespace(
            ibp_result=VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                target_class=target_class,
                target_lower_bound=0.0,
                max_competitor_upper_bound=1.0,
                explanation="fake ibp verdict",
            ),
            refined_result=VerificationResult(
                status=VerificationStatus.CERTIFIED,
                target_class=target_class,
                target_lower_bound=1.0,
                max_competitor_upper_bound=0.5,
                explanation="fake refined verdict",
            ),
            refinement=SimpleNamespace(
                status=SimpleNamespace(value="TIGHTENED"),
                execution_time_seconds=0.25,
                num_targets=2,
                num_accepted=2,
                num_rejected=0,
            ),
            refined_trace=SimpleNamespace(
                output_bounds=SimpleNamespace(lower=lower, upper=upper)
            ),
        )

    monkeypatch.setattr(services, "verify_robustness_refined", fake_refined)
    outcome = services.run_refined_verification(_network(), _sample(), 0.05)
    assert outcome.baseline_status == "INCONCLUSIVE"
    assert outcome.refined_status == "CERTIFIED"
    assert outcome.refined_margin == pytest.approx(0.5)
    assert outcome.lp_refinement_time_seconds == pytest.approx(0.25)
    assert outcome.refinement_status == "TIGHTENED"
    assert outcome.refinement_targets == 2
    assert outcome.refinement_accepted == 2
    assert outcome.refinement_rejected == 0
    assert np.array_equal(outcome.refined_lower, lower)
    assert np.array_equal(outcome.refined_upper, upper)
    assert outcome.error is None


def test_run_baseline_verification_rejects_bad_epsilon() -> None:
    with pytest.raises(ValueError):
        services.run_baseline_verification(_network(), _sample(), -1.0)


# ---------------------------------------------------------------------------
# Service layer: prediction and experiment delegation
# ---------------------------------------------------------------------------


def test_predict_image_returns_logits_and_class() -> None:
    logits, predicted = services.predict_image(_network(), np.array([1.0, -2.0, 0.5, 0.0]))
    assert logits.shape == (2,)
    assert logits.dtype == np.float64
    assert predicted == 0
    assert logits[0] == pytest.approx(1.0)
    assert logits[1] == pytest.approx(0.0)


def test_execute_experiment_delegates_with_preloaded_inputs(monkeypatch) -> None:
    sentinel = object()
    captured = {}

    def fake_run_experiment(config, *, network=None, dataset=None):
        captured["config"] = config
        captured["network"] = network
        captured["dataset"] = dataset
        return sentinel

    monkeypatch.setattr(services, "run_experiment", fake_run_experiment)
    config = ExperimentConfig(sample_count=1, epsilons=(0.03,), seed=42)
    network = _network()
    dataset = object()
    assert services.execute_experiment(config, network, dataset) is sentinel
    assert captured["config"] is config
    assert captured["network"] is network
    assert captured["dataset"] is dataset


# ---------------------------------------------------------------------------
# Export parity: saved files match the evaluation engine's serialization
# ---------------------------------------------------------------------------


def test_saved_csv_rows_match_sample_record_rows(tmp_path) -> None:
    records = [_record(sample_index=0), _record(sample_index=1, refined_margin=None, error="boom")]
    result = _result(records, output_dir=tmp_path)
    paths = result.save(tmp_path)
    with open(paths["samples_csv"], newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(SampleRecord.CSV_FIELDNAMES)
        written = list(reader)
    assert written == [record.to_row() for record in records]


def test_saved_summary_json_matches_result_dict(tmp_path) -> None:
    metrics = [
        _metrics(
            epsilon=0.03,
            baseline_rate=0.80,
            refined_rate=0.85,
            improvement_pp=5.0,
            baseline_width=4.0,
            refined_width=2.5,
            baseline_time=0.5,
            lp_time=0.2,
            refined_time=0.7,
        )
    ]
    result = _result([_record()], metrics=metrics, output_dir=tmp_path)
    paths = result.save(tmp_path)
    with open(paths["summary_json"], encoding="utf-8") as handle:
        written = json.load(handle)
    assert written == result.to_dict()


def test_save_with_empty_records_still_writes_valid_files(tmp_path) -> None:
    result = _result([], metrics=[], output_dir=tmp_path)
    paths = result.save(tmp_path)
    with open(paths["samples_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []
    with open(paths["summary_json"], encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["results"] == []


# ---------------------------------------------------------------------------
# Persistent experiment storage & comparison tests
# ---------------------------------------------------------------------------


def test_generate_experiment_id() -> None:
    id1 = storage.generate_experiment_id()
    id2 = storage.generate_experiment_id()
    assert id1 != id2
    assert len(id1) >= 15
    with_prefix = storage.generate_experiment_id("dev")
    assert with_prefix.startswith("dev_")


def test_save_and_load_experiment_creates_files_and_reconstructs(tmp_path: Path) -> None:
    records = [
        _record(sample_index=0, baseline_status="CERTIFIED", refined_status="CERTIFIED"),
        _record(sample_index=1, baseline_status="INCONCLUSIVE", refined_status="CERTIFIED", refined_margin=None),
    ]
    metrics = [
        _metrics(
            epsilon=0.03,
            baseline_rate=0.5,
            refined_rate=1.0,
            improvement_pp=50.0,
            baseline_width=4.0,
            refined_width=2.5,
            baseline_time=0.01,
            lp_time=0.05,
            refined_time=0.06,
        )
    ]
    orig_result = _result(records, metrics=metrics, output_dir=tmp_path)

    meta = storage.save_experiment(
        orig_result,
        experiment_type="development",
        experiment_id="test_exp_01",
        base_dir=tmp_path,
        timestamp="2026-11-12T10:00:00Z",
    )

    assert meta.experiment_id == "test_exp_01"
    assert meta.experiment_type == "development"
    assert meta.status == "Completed"
    assert meta.sample_count == 2
    assert meta.epsilons == (0.03,)
    assert meta.seed == 42
    assert meta.folder_name == "experiment_test_exp_01"
    exp_dir = tmp_path / "experiment_test_exp_01"
    assert exp_dir.is_dir()
    assert (exp_dir / "config.json").is_file()
    assert (exp_dir / "summary.json").is_file()
    assert (exp_dir / "samples.csv").is_file()

    # Load data directly
    config_dict, summary_dict, loaded_records = storage.load_experiment_data(exp_dir)
    assert config_dict["experiment_id"] == "test_exp_01"
    assert config_dict["experiment_type"] == "development"
    assert len(loaded_records) == 2
    assert loaded_records[1].refined_margin is None

    # Load reconstructed ExperimentResult
    reconstructed = storage.load_experiment_result(exp_dir)
    assert reconstructed.config.sample_count == 2
    assert reconstructed.config.epsilons == (0.03,)
    assert reconstructed.config.seed == 42
    assert reconstructed.selection.selected_count == 2
    assert len(reconstructed.metrics) == 1
    loaded_metric = reconstructed.metrics[0]
    assert loaded_metric.epsilon == 0.03
    assert loaded_metric.certification.baseline_rate == 0.5
    assert loaded_metric.certification.refined_rate == 1.0
    assert loaded_metric.certification.improvement_pp == 50.0
    assert loaded_metric.bounds.mean_baseline_width == 4.0
    assert loaded_metric.bounds.mean_refined_width == 2.5
    assert len(reconstructed.records) == 2


def test_save_experiment_prevents_overwrite(tmp_path: Path) -> None:
    orig_result = _result([_record()], output_dir=tmp_path)
    storage.save_experiment(orig_result, experiment_id="exp_unique_123", base_dir=tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        storage.save_experiment(orig_result, experiment_id="exp_unique_123", base_dir=tmp_path)


def test_save_experiment_type_and_argument_validation(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be an ExperimentResult"):
        storage.save_experiment("not an experiment result", base_dir=tmp_path)  # type: ignore[arg-type]

    orig_result = _result([_record()], output_dir=tmp_path)
    meta = storage.save_experiment(orig_result, experiment_type="dev_run", experiment_id="dev_run_1", base_dir=tmp_path)
    assert meta.experiment_type == "development"


def test_persistence_across_simulated_dashboard_reruns(tmp_path: Path) -> None:
    # Run 1: execute and save experiment
    records = [_record(sample_index=0)]
    metrics = [_metrics(epsilon=0.03, baseline_rate=1.0, refined_rate=1.0)]
    result1 = _result(records, metrics=metrics, output_dir=tmp_path)
    storage.save_experiment(result1, experiment_id="run_1", base_dir=tmp_path)

    # Run 2: simulated fresh session state (memory cleared)
    # Reading from disk without prior session state
    experiments = storage.list_experiments(base_dir=tmp_path)
    assert len(experiments) == 1
    assert experiments[0].experiment_id == "run_1"
    assert experiments[0].status == "Completed"

    loaded_result = storage.load_experiment_result(experiments[0].path)
    assert loaded_result.config.experiment_name == result1.config.experiment_name
    assert len(loaded_result.records) == 1
    assert loaded_result.records[0].sample_index == 0


def test_list_experiments_multiple_and_sorted(tmp_path: Path) -> None:
    res = _result([_record()], output_dir=tmp_path)
    storage.save_experiment(res, experiment_id="exp_older", base_dir=tmp_path, timestamp="2026-11-10T12:00:00Z")
    storage.save_experiment(res, experiment_id="exp_newer", base_dir=tmp_path, timestamp="2026-11-12T12:00:00Z")
    storage.save_experiment(res, experiment_id="exp_middle", base_dir=tmp_path, timestamp="2026-11-11T12:00:00Z")

    experiments = storage.list_experiments(base_dir=tmp_path)
    assert len(experiments) == 3
    # Newest first
    assert [e.experiment_id for e in experiments] == ["exp_newer", "exp_middle", "exp_older"]


def test_list_and_load_corrupted_or_incomplete_experiment(tmp_path: Path) -> None:
    # 1. Incomplete folder: missing samples.csv
    incomplete_dir = tmp_path / "experiment_incomplete_01"
    incomplete_dir.mkdir()
    (incomplete_dir / "config.json").write_text(json.dumps({"experiment_id": "incomplete_01", "sample_count": 5, "epsilons": [0.03], "seed": 42}), encoding="utf-8")
    (incomplete_dir / "summary.json").write_text(json.dumps({"results": []}), encoding="utf-8")

    # 2. Corrupted folder: invalid JSON in config.json
    corrupted_dir = tmp_path / "experiment_corrupted_01"
    corrupted_dir.mkdir()
    (corrupted_dir / "config.json").write_text("{invalid json", encoding="utf-8")
    (corrupted_dir / "summary.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (corrupted_dir / "samples.csv").write_text("sample_index\n0\n", encoding="utf-8")

    experiments = storage.list_experiments(base_dir=tmp_path)
    status_by_id = {e.folder_name: e.status for e in experiments}
    assert status_by_id["experiment_incomplete_01"] == "Incomplete"
    assert status_by_id["experiment_corrupted_01"] == "Corrupted"

    # Loading incomplete/corrupted should raise ValueError with informative message
    with pytest.raises(ValueError, match="missing required files"):
        storage.load_experiment_result(incomplete_dir)

    with pytest.raises(ValueError, match="cannot be loaded"):
        storage.load_experiment_result(corrupted_dir)


def test_import_legacy_experiments(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "legacy"
    experiments_dir = tmp_path / "experiments"
    legacy_dir.mkdir()
    experiments_dir.mkdir()

    # Create dummy legacy files
    dev_summary = {
        "config": {"sample_count": 1, "epsilons": [0.03], "seed": 42, "experiment_name": "dev_1"},
        "results": [{"epsilon": 0.03, "certification": {"evaluated_samples": 1, "baseline_certified": 0, "refined_certified": 0, "baseline_rate": 0.0, "refined_rate": 0.0, "improvement_pp": 0.0}}],
    }
    (legacy_dir / "evaluation_seed42_n5_summary.json").write_text(json.dumps(dev_summary), encoding="utf-8")
    sample_csv_text = "sample_index,label,predicted,target_class,epsilon,baseline_status,baseline_margin,baseline_time_seconds,refined_status,refined_margin,refined_time_seconds,lp_refinement_time_seconds,refinement_status,refinement_targets,refinement_accepted,refinement_rejected,baseline_width_mean,refined_width_mean,abs_width_reduction_mean,rel_width_reduction_mean,zero_width_baseline_intervals,error\n0,7,7,7,0.03,INCONCLUSIVE,-0.5,0.001,INCONCLUSIVE,-0.2,0.5,0.49,TIGHTENED,16,14,2,1.2,0.9,0.3,0.25,0,\n"
    (legacy_dir / "evaluation_seed42_n5_samples.csv").write_text(sample_csv_text, encoding="utf-8")

    imported = storage.import_legacy_experiments(source_dir=legacy_dir, target_base_dir=experiments_dir)
    assert len(imported) == 1
    assert imported[0].experiment_type == "development"
    assert imported[0].sample_count == 1
    assert imported[0].epsilons == (0.03,)

    # Running import a second time should not duplicate or error
    imported_second = storage.import_legacy_experiments(source_dir=legacy_dir, target_base_dir=experiments_dir)
    assert len(imported_second) == 0


def test_import_dynamic_discovery_and_incomplete_handling(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()

    # 1. Complete newly finished experiment: evaluation_seed39_n100 (mock 6 samples * 2 epsilons = 12 records)
    conf_summary = {
        "config": {"sample_count": 6, "epsilons": [0.01, 0.03], "seed": 39, "experiment_name": "evaluation_seed39_n100"},
        "results": [
            {"epsilon": 0.01, "certification": {"evaluated_samples": 6, "baseline_certified": 1, "refined_certified": 2, "baseline_rate": 0.5, "refined_rate": 1.0, "improvement_pp": 50.0}},
            {"epsilon": 0.03, "certification": {"evaluated_samples": 6, "baseline_certified": 0, "refined_certified": 1, "baseline_rate": 0.0, "refined_rate": 0.5, "improvement_pp": 50.0}},
        ],
    }
    (src_dir / "evaluation_seed39_n100_summary.json").write_text(json.dumps(conf_summary), encoding="utf-8")
    header = "sample_index,label,predicted,target_class,epsilon,baseline_status,baseline_margin,baseline_time_seconds,refined_status,refined_margin,refined_time_seconds,lp_refinement_time_seconds,refinement_status,refinement_targets,refinement_accepted,refinement_rejected,baseline_width_mean,refined_width_mean,abs_width_reduction_mean,rel_width_reduction_mean,zero_width_baseline_intervals,error"
    conf_csv_lines = [header]
    for eps in (0.01, 0.03):
        for s_idx in range(6):
            conf_csv_lines.append(f"{s_idx},7,7,7,{eps},CERTIFIED,0.5,0.01,CERTIFIED,0.5,0.02,0.01,UNCHANGED,0,0,0,1.0,1.0,0.0,0.0,0,")
    (src_dir / "evaluation_seed39_n100_samples.csv").write_text("\n".join(conf_csv_lines) + "\n", encoding="utf-8")

    # 2. Incomplete experiment: missing CSV file
    (src_dir / "evaluation_incomplete_no_csv_summary.json").write_text(json.dumps(conf_summary), encoding="utf-8")

    # 3. Incomplete experiment: record count mismatch (only 2 records instead of 12)
    (src_dir / "evaluation_incomplete_partial_summary.json").write_text(json.dumps(conf_summary), encoding="utf-8")
    (src_dir / "evaluation_incomplete_partial_samples.csv").write_text(conf_csv_lines[0] + "\n" + conf_csv_lines[1] + "\n", encoding="utf-8")

    imported = storage.import_legacy_experiments(source_dir=src_dir, target_base_dir=dst_dir)
    assert len(imported) == 1
    assert imported[0].experiment_id == "evaluation_seed39_n100"
    assert imported[0].experiment_type == "configured"
    assert imported[0].sample_count == 6
    assert imported[0].seed == 39
    assert imported[0].status == "Completed"

    # Both incomplete experiments must NOT have been imported
    all_exps = storage.list_experiments(base_dir=dst_dir)
    assert len(all_exps) == 1
    assert all_exps[0].experiment_id == "evaluation_seed39_n100"


def test_status_display_experiment_statuses() -> None:
    assert components.status_display("Completed") == "Completed"
    assert components.status_display("COMPLETED") == "Completed"
    assert components.status_display("Failed") == "Failed"
    assert components.status_display("Incomplete") == "Incomplete"
    assert components.status_display("Corrupted") == "Corrupted"


def test_comparison_analysis_matching_epsilons(tmp_path: Path) -> None:
    # Exp 1: epsilons (0.01, 0.03, 0.05)
    rec1 = [_record(sample_index=0, epsilon=0.01), _record(sample_index=0, epsilon=0.03), _record(sample_index=0, epsilon=0.05)]
    met1 = [_metrics(0.01), _metrics(0.03), _metrics(0.05)]
    res1 = _result(rec1, met1, output_dir=tmp_path, epsilons=(0.01, 0.03, 0.05))
    meta1 = storage.save_experiment(res1, experiment_id="exp_1", base_dir=tmp_path)

    # Exp 2: epsilons (0.03, 0.05, 0.10)
    rec2 = [_record(sample_index=0, epsilon=0.03), _record(sample_index=0, epsilon=0.05), _record(sample_index=0, epsilon=0.10)]
    met2 = [_metrics(0.03), _metrics(0.05), _metrics(0.10)]
    res2 = _result(rec2, met2, output_dir=tmp_path, epsilons=(0.03, 0.05, 0.10))
    meta2 = storage.save_experiment(res2, experiment_id="exp_2", base_dir=tmp_path)

    item1 = comparison.ComparisonItem(metadata=meta1, result=res1)
    item2 = comparison.ComparisonItem(metadata=meta2, result=res2)

    analysis = comparison.build_comparison_analysis([item1, item2])
    assert analysis.matching_epsilons == (0.03, 0.05)
    assert analysis.all_epsilons == (0.01, 0.03, 0.05, 0.10)
    assert analysis.differing_epsilons["exp_1"] == (0.01,)
    assert analysis.differing_epsilons["exp_2"] == (0.10,)
    assert any("Epsilon values differ" in notice for notice in analysis.notices)


def test_comparison_differing_sample_counts_and_seeds(tmp_path: Path) -> None:
    # Exp A: n=2, seed=42
    res_a = _result([_record(sample_index=0), _record(sample_index=1)], output_dir=tmp_path, sample_count=2, seed=42)
    meta_a = storage.save_experiment(res_a, experiment_id="exp_a", base_dir=tmp_path)

    # Exp B: n=1, seed=42
    res_b = _result([_record(sample_index=0)], output_dir=tmp_path, sample_count=1, seed=42)
    meta_b = storage.save_experiment(res_b, experiment_id="exp_b", base_dir=tmp_path)

    # Exp C: n=2, seed=99
    res_c = _result([_record(sample_index=0), _record(sample_index=1)], output_dir=tmp_path, sample_count=2, seed=99)
    meta_c = storage.save_experiment(res_c, experiment_id="exp_c", base_dir=tmp_path)

    item_a = comparison.ComparisonItem(metadata=meta_a, result=res_a)
    item_b = comparison.ComparisonItem(metadata=meta_b, result=res_b)
    item_c = comparison.ComparisonItem(metadata=meta_c, result=res_c)

    # A vs B: differing sample count
    analysis_ab = comparison.build_comparison_analysis([item_a, item_b])
    assert analysis_ab.sample_counts_differ is True
    assert analysis_ab.seeds_differ is False
    assert analysis_ab.identical_samples is False
    assert any("Sample counts differ" in notice for notice in analysis_ab.notices)

    # A vs C: differing seeds
    analysis_ac = comparison.build_comparison_analysis([item_a, item_c])
    assert analysis_ac.sample_counts_differ is False
    assert analysis_ac.seeds_differ is True
    assert analysis_ac.identical_samples is False
    assert any("Random seeds differ" in notice for notice in analysis_ac.notices)


def test_comparison_metric_table_missing_metrics_distinct_from_zero(tmp_path: Path) -> None:
    # Exp 1 with values
    met1 = [_metrics(0.03, baseline_rate=0.8, refined_rate=0.9, improvement_pp=10.0, baseline_width=2.0, refined_width=1.5, baseline_time=0.1, lp_time=0.2, refined_time=0.3)]
    res1 = _result([_record(sample_index=0)], metrics=met1, output_dir=tmp_path)
    meta1 = storage.save_experiment(res1, experiment_id="exp_1", base_dir=tmp_path)

    # Exp 2 with missing metrics (None)
    met2 = [_metrics(0.03, baseline_rate=None, refined_rate=None, improvement_pp=None, baseline_width=None, refined_width=None, baseline_time=None, lp_time=None, refined_time=None)]
    res2 = _result([_record(sample_index=0)], metrics=met2, output_dir=tmp_path)
    meta2 = storage.save_experiment(res2, experiment_id="exp_2", base_dir=tmp_path)

    item1 = comparison.ComparisonItem(metadata=meta1, result=res1)
    item2 = comparison.ComparisonItem(metadata=meta2, result=res2)

    analysis = comparison.build_comparison_analysis([item1, item2])
    table = comparison.build_comparison_metric_table(analysis, epsilon=0.03)

    assert len(table) == 10  # 10 core metrics
    col1 = item1.display_label
    col2 = item2.display_label

    # Rows 1 through 9 are quantitative metrics (rates, widths, runtimes)
    for row in table[:9]:
        assert col1 in row
        assert col2 in row
        # In exp 2, all metric values are missing and must be "—", never "0" or "0.0"
        assert row[col2] == "—"

    # Also verify zero values remain "0" or "0.0%", not "—"
    met3 = [_metrics(0.03, baseline_rate=0.0, refined_rate=0.0, improvement_pp=0.0, baseline_width=0.0, refined_width=0.0, baseline_time=0.0, lp_time=0.0, refined_time=0.0)]
    res3 = _result([_record(sample_index=0)], metrics=met3, output_dir=tmp_path)
    meta3 = storage.save_experiment(res3, experiment_id="exp_3", base_dir=tmp_path)
    item3 = comparison.ComparisonItem(metadata=meta3, result=res3)

    analysis_zero = comparison.build_comparison_analysis([item1, item3])
    table_zero = comparison.build_comparison_metric_table(analysis_zero, epsilon=0.03)
    col3 = item3.display_label
    row_rate = next(r for r in table_zero if "Baseline certification rate" in r["Metric"])
    assert row_rate[col3] == "0.00%"


def test_historical_chart_generation_and_comparison_plots(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    met = [
        _metrics(0.01, baseline_rate=0.8, refined_rate=0.85, improvement_pp=5.0, baseline_width=3.0, refined_width=2.0, baseline_time=0.1, lp_time=0.2, refined_time=0.3),
        _metrics(0.03, baseline_rate=0.6, refined_rate=0.7, improvement_pp=10.0, baseline_width=4.0, refined_width=2.5, baseline_time=0.15, lp_time=0.4, refined_time=0.55),
    ]
    res1 = _result([_record(sample_index=0, epsilon=0.01), _record(sample_index=0, epsilon=0.03)], metrics=met, output_dir=tmp_path, epsilons=(0.01, 0.03))
    meta1 = storage.save_experiment(res1, experiment_id="exp_hist_1", base_dir=tmp_path)

    # 1. Historical inspection charts without rerunning verification
    loaded = storage.load_experiment_result(meta1.path)
    cert_series = components.certification_rate_series(loaded.metrics)
    fig1 = visualizations.plot_certification_rates(cert_series)
    assert len(fig1.axes) == 1

    width_series = components.width_series(loaded.metrics)
    fig2 = visualizations.plot_interval_widths(width_series)
    assert len(fig2.axes) == 1

    rt_series = components.runtime_series(loaded.metrics)
    fig3 = visualizations.plot_runtime(rt_series)
    assert len(fig3.axes) == 1

    imp_series = components.improvement_series(loaded.metrics)
    fig4 = visualizations.plot_improvement(imp_series)
    assert len(fig4.axes) == 1

    # 2. Multi-experiment comparison charts
    res2 = _result([_record(sample_index=0, epsilon=0.01), _record(sample_index=0, epsilon=0.03)], metrics=met, output_dir=tmp_path, epsilons=(0.01, 0.03))
    meta2 = storage.save_experiment(res2, experiment_id="exp_hist_2", base_dir=tmp_path)

    item1 = comparison.ComparisonItem(metadata=meta1, result=loaded)
    item2 = comparison.ComparisonItem(metadata=meta2, result=res2)
    analysis = comparison.build_comparison_analysis([item1, item2])
    comp_series = comparison.get_comparison_series_for_plotting(analysis)

    fig_cmp1 = visualizations.plot_comparison_certification(comp_series)
    assert len(fig_cmp1.axes) == 1

    fig_cmp2 = visualizations.plot_comparison_widths(comp_series)
    assert len(fig_cmp2.axes) == 1

    fig_cmp3 = visualizations.plot_comparison_runtime(comp_series)
    assert len(fig_cmp3.axes) == 1

    fig_cmp4 = visualizations.plot_comparison_improvement(comp_series)
    assert len(fig_cmp4.axes) == 1


def test_execute_and_save_experiment(monkeypatch, tmp_path: Path) -> None:
    res = _result([_record()], metrics=[_metrics(0.03)], output_dir=tmp_path)
    monkeypatch.setattr(services, "execute_experiment", lambda cfg, net, ds: res)
    monkeypatch.setattr(storage, "DEFAULT_EXPERIMENTS_DIR", tmp_path)

    config = ExperimentConfig(sample_count=1, epsilons=(0.03,), seed=42)
    result_out, meta = services.execute_and_save_experiment(
        config=config,
        network=_network(),
        dataset=object(),
        experiment_type="development",
        base_dir=tmp_path,
    )
    assert result_out is res
    assert meta.experiment_type == "development"
    assert (tmp_path / meta.folder_name).is_dir()


def test_history_table_rows() -> None:
    meta = storage.ExperimentMetadata(
        experiment_id="test_exp",
        experiment_type="development",
        timestamp="2026-11-12T10:00:00Z",
        sample_count=5,
        epsilons=(0.03,),
        seed=42,
        status="Completed",
        experiment_name="dev_run",
        folder_name="experiment_test_exp",
        path=Path("/fake/path"),
    )
    rows = components.history_table_rows([meta])
    assert len(rows) == 1
    assert rows[0]["Experiment ID"] == "test_exp"
    assert rows[0]["Type"] == "Development"
    assert rows[0]["Samples"] == 5
    assert rows[0]["Epsilon values"] == "0.03"
    assert rows[0]["Seed"] == 42
    assert rows[0]["Status"] == "Completed"

