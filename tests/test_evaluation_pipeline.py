"""Deterministic tests for the evaluation pipeline orchestration.

These tests use a small in-memory stub dataset and a hand-built float64
network so the sample-selection, configuration, export, and failure-handling
behavior is verified without MNIST data, model training, or large LP
experiments. Failure and inconclusive handling is exercised by monkeypatching
the verification entry points inside :mod:`evaluation.runner`.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from evaluation.dataset import (
    InputSample,
    flatten_image,
    select_samples,
    validate_input_bounds,
)
from evaluation.experiment import ExperimentConfig, ExperimentResult, run_experiment
from evaluation.metrics import SampleRecord
from evaluation.runner import evaluate_sample
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification import VerificationResult, VerificationStatus
from verification.regions import InputRegion

CERTIFIED = VerificationStatus.CERTIFIED.value
INCONCLUSIVE = VerificationStatus.INCONCLUSIVE.value


# =====================================================================
# Stub dataset and a deterministic 4 -> 3 -> 2 evaluation network
# =====================================================================


class _StubDataset:
    """Minimal sized (image, label) collection mimicking a dataset."""

    def __init__(self, items: list[tuple[np.ndarray, int]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self._items[index]


def _stub_dataset() -> _StubDataset:
    """Ten 4-feature items; even indices predict class 1, odd class 0.

    Labels match the predictions except for indices 1, 5, and 9, which are
    mislabeled as the opposite class (3 incorrectly classified samples).
    """
    items = []
    for index in range(10):
        image = np.zeros(4, dtype=np.float64)
        image[0] = float(index % 2)
        image[1] = 0.25
        label = 1 - index % 2
        if index in (1, 5, 9):
            label = index % 2
        items.append((image, label))
    return _StubDataset(items)


def _evaluation_network() -> MathematicalNetwork:
    """4 -> 3 -> 2 ReLU network predicting argmax(x0 - x1, x1 - x0)."""
    first = LinearLayer(
        position=0,
        weight=np.array(
            [[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]],
            dtype=np.float64,
        ),
        bias=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        input_features=4,
        output_features=3,
    )
    second = LinearLayer(
        position=2,
        weight=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        ),
        bias=np.zeros(2, dtype=np.float64),
        input_features=3,
        output_features=2,
    )
    return MathematicalNetwork(
        layers=(first, ReLULayer(position=1), second),
        input_features=4,
        output_features=2,
        source_model="EvaluationStubNet",
    )


def _config(**overrides: object) -> ExperimentConfig:
    """Build an ExperimentConfig against the stub fixtures."""
    defaults: dict[str, object] = {
        "sample_count": 4,
        "epsilons": (0.02, 0.05),
        "seed": 42,
        "prediction_batch_size": 4,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _record(**overrides: object) -> SampleRecord:
    """Build one fabricated per-sample record for serialization tests."""
    fields: dict[str, object] = {
        "sample_index": 3,
        "label": 0,
        "predicted": 0,
        "target_class": 0,
        "epsilon": 0.05,
        "baseline_status": CERTIFIED,
        "baseline_margin": 0.5,
        "baseline_time_seconds": 0.01,
        "refined_status": CERTIFIED,
        "refined_margin": 0.75,
        "refined_time_seconds": 0.05,
        "lp_refinement_time_seconds": 0.03,
        "refinement_status": "TIGHTENED",
        "refinement_targets": 6,
        "refinement_accepted": 5,
        "refinement_rejected": 1,
        "baseline_width_mean": 1.0,
        "refined_width_mean": 0.5,
        "abs_width_reduction_mean": 0.5,
        "rel_width_reduction_mean": 0.5,
        "zero_width_baseline_intervals": 0,
        "error": None,
    }
    fields.update(overrides)
    return SampleRecord(**fields)


# =====================================================================
# Sample selection
# =====================================================================


class TestSampleSelection:
    def test_selection_is_reproducible_for_a_fixed_seed(self) -> None:
        network = _evaluation_network()
        dataset = _stub_dataset()

        first = select_samples(network, dataset, 4, 42)
        second = select_samples(network, dataset, 4, 42)

        assert [sample.index for sample in first.samples] == [
            sample.index for sample in second.samples
        ]
        assert [sample.image.tolist() for sample in first.samples] == [
            sample.image.tolist() for sample in second.samples
        ]
        assert first.incorrect_excluded == second.incorrect_excluded
        assert first.scanned == second.scanned

    def test_selected_samples_are_all_correctly_classified(self) -> None:
        selection = select_samples(_evaluation_network(), _stub_dataset(), 4, 42)

        assert selection.selected_count == 4
        assert all(sample.correctly_classified for sample in selection.samples)
        assert all(sample.predicted == sample.label for sample in selection.samples)

    def test_incorrect_samples_are_excluded_and_counted(self) -> None:
        network = _evaluation_network()
        dataset = _stub_dataset()
        selection = select_samples(network, dataset, 10, 0)

        incorrect_indices = {1, 5, 9}
        assert all(sample.index not in incorrect_indices for sample in selection.samples)
        assert selection.selected_count == 7
        assert selection.incorrect_excluded <= len(incorrect_indices)
        assert selection.scanned == selection.selected_count + selection.incorrect_excluded
        assert selection.scanned <= len(dataset)

    def test_selection_images_are_flattened_float64(self) -> None:
        selection = select_samples(_evaluation_network(), _stub_dataset(), 1, 42)
        sample = selection.samples[0]

        assert sample.image.dtype == np.float64
        assert sample.image.shape == (4,)
        assert np.all(np.isfinite(sample.image))

    def test_sample_count_validation(self) -> None:
        network = _evaluation_network()

        with pytest.raises(ValueError):
            select_samples(network, _stub_dataset(), 0, 42)
        with pytest.raises(TypeError):
            select_samples(network, _stub_dataset(), True, 42)  # type: ignore[arg-type]


# =====================================================================
# Configuration
# =====================================================================


class TestExperimentConfig:
    def test_default_configuration_matches_suggested_setup(self) -> None:
        config = ExperimentConfig()

        assert config.sample_count == 100
        assert config.epsilons == (0.01, 0.03, 0.05, 0.1)
        assert config.seed == 42

    def test_epsilon_configuration_is_respected_in_order(self) -> None:
        config = _config(epsilons=(0.02, 0.07, 0.11))

        assert config.epsilons == (0.02, 0.07, 0.11)

    def test_default_experiment_name_is_deterministic(self) -> None:
        config = _config(sample_count=25, seed=7)

        assert config.experiment_name == "evaluation_seed7_n25"

    def test_invalid_configurations_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            _config(sample_count=0)
        with pytest.raises(ValueError):
            _config(epsilons=())
        with pytest.raises(ValueError):
            _config(epsilons=(-0.01,))
        with pytest.raises(TypeError):
            _config(seed=1.5)  # type: ignore[arg-type]

    def test_input_bounds_validation(self) -> None:
        assert validate_input_bounds(0.0, 1.0) == (0.0, 1.0)
        with pytest.raises(ValueError):
            validate_input_bounds(1.0, 0.0)
        with pytest.raises(ValueError):
            validate_input_bounds(float("nan"), 1.0)


# =====================================================================
# End-to-end experiment over the stub fixtures (real verification)
# =====================================================================


class TestRunExperiment:
    def test_identical_populations_for_both_methods_and_all_epsilons(self) -> None:
        network = _evaluation_network()
        dataset = _stub_dataset()
        result = run_experiment(_config(), network=network, dataset=dataset)

        selection = select_samples(network, dataset, 4, 42)
        expected_indices = [sample.index for sample in selection.samples]
        by_epsilon: dict[float, list[SampleRecord]] = {}
        for record in result.records:
            by_epsilon.setdefault(record.epsilon, []).append(record)

        assert sorted(by_epsilon) == [0.02, 0.05]
        for epsilon, records in by_epsilon.items():
            assert [record.sample_index for record in records] == expected_indices
            assert all(record.target_class == record.predicted for record in records)
            assert all(record.label == record.predicted for record in records)

    def test_metrics_cover_every_configured_epsilon_in_order(self) -> None:
        result = run_experiment(
            _config(epsilons=(0.02, 0.05)),
            network=_evaluation_network(),
            dataset=_stub_dataset(),
        )

        assert [metrics.epsilon for metrics in result.metrics] == [0.02, 0.05]
        for metrics in result.metrics:
            assert metrics.certification.evaluated_samples == 4
            assert metrics.certification.total_test_samples == 10
            assert metrics.runtime.baseline_times_measured == 4
            assert metrics.runtime.refined_times_measured == 4

    def test_records_carry_epsilon_association(self) -> None:
        result = run_experiment(
            _config(),
            network=_evaluation_network(),
            dataset=_stub_dataset(),
        )

        assert len(result.records) == 8
        assert {record.epsilon for record in result.records} == {0.02, 0.05}

    def test_selection_is_performed_once_and_shared(self) -> None:
        network = _evaluation_network()
        dataset = _stub_dataset()
        result = run_experiment(_config(), network=network, dataset=dataset)

        assert result.selection.selected_count == 4
        assert result.selection.requested_count == 4
        assert result.dataset_size == 10
        assert result.network_source == "EvaluationStubNet"

    def test_refined_and_baseline_share_target_classes(self) -> None:
        result = run_experiment(
            _config(),
            network=_evaluation_network(),
            dataset=_stub_dataset(),
        )

        for record in result.records:
            assert record.baseline_status in (CERTIFIED, INCONCLUSIVE)
            assert record.refined_status in (CERTIFIED, INCONCLUSIVE)


# =====================================================================
# Failure and inconclusive handling (mocked verification entry points)
# =====================================================================


class TestFailureHandling:
    def test_failed_refined_verification_is_recorded_not_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(network: object, region: object, target: object) -> object:
            raise RuntimeError("LP solver exploded")

        monkeypatch.setattr("evaluation.runner.verify_robustness_refined", _raise)
        network = _evaluation_network()
        sample = InputSample(
            index=0,
            image=flatten_image(np.array([0.75, 0.25, 0.0, 0.0])),
            label=0,
            predicted=0,
        )

        record = evaluate_sample(network, sample, 0.05)

        assert record.baseline_status in (CERTIFIED, INCONCLUSIVE)
        assert record.refined_status == "ERROR"
        assert record.error is not None
        assert "RuntimeError" in record.error
        assert record.lp_refinement_time_seconds is None
        assert record.refined_margin is None
        assert record.refinement_targets is None
        assert record.baseline_width_mean is None
        assert record.refined_width_mean is None

    def test_failed_baseline_verification_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(network: object, region: object, target: object) -> object:
            raise ValueError("bad region")

        monkeypatch.setattr("evaluation.runner.verify_robustness", _raise)
        network = _evaluation_network()
        sample = InputSample(
            index=0,
            image=flatten_image(np.array([0.75, 0.25, 0.0, 0.0])),
            label=0,
            predicted=0,
        )

        record = evaluate_sample(network, sample, 0.05)

        assert record.baseline_status == "ERROR"
        assert record.baseline_margin is None
        assert record.error is not None
        assert "ValueError" in record.error

    def test_inconclusive_results_are_counted_not_treated_as_certified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _inconclusive(network: object, region: object, target: object):
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                target_class=target,
                target_lower_bound=0.0,
                max_competitor_upper_bound=1.0,
                explanation="mocked inconclusive verdict",
            )

        monkeypatch.setattr("evaluation.runner.verify_robustness", _inconclusive)
        network = _evaluation_network()
        sample = InputSample(
            index=0,
            image=flatten_image(np.array([0.75, 0.25, 0.0, 0.0])),
            label=0,
            predicted=0,
        )

        record = evaluate_sample(network, sample, 0.05)

        assert record.baseline_status == INCONCLUSIVE
        assert record.baseline_certified is False
        assert record.baseline_inconclusive is True
        assert record.baseline_margin == pytest.approx(-1.0)

    def test_inconclusive_region_is_valid_but_uncertified(self) -> None:
        network = _evaluation_network()
        sample = InputSample(
            index=0,
            image=flatten_image(np.array([0.75, 0.25, 0.0, 0.0])),
            label=0,
            predicted=0,
        )

        record = evaluate_sample(network, sample, 0.9)

        assert record.baseline_status == INCONCLUSIVE


# =====================================================================
# Exports
# =====================================================================


class TestExports:
    def _small_result(self) -> ExperimentResult:
        return run_experiment(
            _config(sample_count=2, epsilons=(0.02,)),
            network=_evaluation_network(),
            dataset=_stub_dataset(),
        )

    def test_csv_export_contains_all_records_with_epsilon(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        result = self._small_result()
        csv_path = tmp_path / "out" / "evaluation_seed42_n2_samples.csv"
        written = result.save(output_dir=tmp_path / "out")["samples_csv"]

        assert written == csv_path
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == len(result.records)
        assert {row["epsilon"] for row in rows} == {"0.02"}
        assert all(row["refined_status"] in (CERTIFIED, INCONCLUSIVE) for row in rows)
        rebuilt = [SampleRecord.from_row(row) for row in rows]
        assert rebuilt == list(result.records)

    def test_json_summary_contains_config_seed_and_metrics(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        result = self._small_result()
        json_path = tmp_path / "out" / "evaluation_seed42_n2_summary.json"
        written = result.save(output_dir=tmp_path / "out")["summary_json"]

        assert written == json_path
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        assert payload["config"]["seed"] == 42
        assert payload["config"]["epsilons"] == [0.02]
        assert payload["config"]["sample_count"] == 2
        assert payload["sample_selection"]["selected"] == 2
        assert payload["sample_selection"]["seed"] == 42
        assert len(payload["results"]) == 1
        entry = payload["results"][0]
        assert entry["epsilon"] == pytest.approx(0.02)
        assert entry["certification"]["evaluated_samples"] == 2
        assert entry["runtime"]["baseline_times_measured"] == 2

    def test_save_uses_deterministic_config_filenames(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        result = self._small_result()
        paths = result.save(output_dir=tmp_path)

        assert paths["samples_csv"].name == "evaluation_seed42_n2_samples.csv"
        assert paths["summary_json"].name == "evaluation_seed42_n2_summary.json"

    def test_input_region_respects_valid_pixel_range(self) -> None:
        network = _evaluation_network()
        sample = InputSample(
            index=0,
            image=flatten_image(np.array([0.01, 0.99, 0.5, 0.5])),
            label=0,
            predicted=0,
        )

        record = evaluate_sample(
            network, sample, 0.5, input_lower=0.0, input_upper=1.0
        )

        region = InputRegion(
            sample.image, 0.5, input_lower=0.0, input_upper=1.0
        )
        assert region.lower[0] == pytest.approx(0.0)
        assert region.upper[1] == pytest.approx(1.0)
        assert record.baseline_status in (CERTIFIED, INCONCLUSIVE)
