"""Focused tests for 100-sample Mode A vs. Mode B equivalence validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dashboard.storage import import_legacy_experiments, list_experiments
from evaluation.dataset import InputSample
from evaluation.equivalence import (
    DEFAULT_EQUIVALENCE_TOLERANCE,
    EquivalenceConfig,
    EquivalencePairRecord,
    _ModeOutcome,
    build_pair_record,
    compare_sample_modes,
    generate_equivalence_experiment_id,
    run_equivalence_validation,
)
from evaluation.exports import write_summary_json
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification import VerificationStatus


class _StubDataset:
    def __init__(self, items: list[tuple[np.ndarray, int]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self._items[index]


def _two_layer_net() -> MathematicalNetwork:
    """4 -> 3 -> 2 ReLU network with an unstable hidden neuron for eps >= 0.3."""
    l0 = LinearLayer(
        position=0,
        weight=np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        bias=np.array([0.0, 0.0, -0.2], dtype=np.float64),
        input_features=4,
        output_features=3,
    )
    l1 = ReLULayer(position=1)
    l2 = LinearLayer(
        position=2,
        weight=np.array(
            [
                [1.0, 0.0, 0.25],
                [0.0, 1.0, -0.25],
            ],
            dtype=np.float64,
        ),
        bias=np.zeros(2, dtype=np.float64),
        input_features=3,
        output_features=2,
    )
    return MathematicalNetwork(
        layers=(l0, l1, l2),
        input_features=4,
        output_features=2,
        source_model="EquivalenceStubNet",
    )


def _stub_dataset() -> _StubDataset:
    items = []
    for i in range(6):
        img = np.array([0.8 if i % 2 == 0 else 0.2, 0.2 if i % 2 == 0 else 0.8, 0.3, 0.3], dtype=np.float64)
        label = 0 if i % 2 == 0 else 1
        items.append((img, label))
    return _StubDataset(items)


def _outcome(
    *,
    targets: int = 2,
    verdict: str = VerificationStatus.CERTIFIED.value,
    margin: float = 0.5,
    lower: tuple[float, ...] = (0.5, -0.2),
    upper: tuple[float, ...] = (1.2, 0.0),
    error: str | None = None,
) -> _ModeOutcome:
    return _ModeOutcome(
        targets=None if error else targets,
        lp_solves=None if error else 2 * targets,
        accepted=None if error else targets,
        rejected=None if error else 0,
        refinement_status="ERROR" if error else "TIGHTENED",
        verdict="ERROR" if error else verdict,
        margin=None if error else margin,
        time_seconds=0.01,
        lower=None if error else lower,
        upper=None if error else upper,
        error=error,
    )


def test_both_target_selection_modes_and_equivalence(tmp_path: Path) -> None:
    """Mode A (hidden+output) and Mode B (output-only) match on identical pairs."""
    net = _two_layer_net()
    ds = _stub_dataset()
    cfg = EquivalenceConfig(
        sample_count=3,
        epsilons=(0.05, 0.4),
        seed=39,
        output_dir=tmp_path,
        experiment_name="eq_stub_test",
        prediction_batch_size=3,
    )
    result = run_equivalence_validation(cfg, network=net, dataset=ds)

    assert len(result.records) == 6
    assert result.overall_summary.total_pairs == 6
    assert result.overall_summary.completed_pairs == 6
    assert result.overall_summary.incomplete_pairs == 0
    assert result.overall_summary.verdict_mismatches == 0
    assert result.overall_summary.discrepancies_over_tolerance == 0
    assert result.equivalent is True

    # At eps=0.4 hidden neuron 2 spans zero so Mode A must include > 2 targets,
    # whereas Mode B stays at exactly 2 output targets.
    eps04 = [r for r in result.records if r.epsilon == 0.4]
    assert all(r.mode_b_targets == 2 and r.mode_b_lp_solves == 4 for r in eps04)
    assert any((r.mode_a_targets or 0) > 2 and (r.mode_a_lp_solves or 0) > 4 for r in eps04)


def test_identical_sample_and_epsilon_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    """compare_sample_modes passes exact identical region & target to both modes."""
    net = _two_layer_net()
    sample = InputSample(
        index=17,
        image=np.array([0.8, 0.2, 0.3, 0.3], dtype=np.float64),
        label=0,
        predicted=0,
    )
    seen_calls: list[tuple[np.ndarray, np.ndarray, float, int, bool]] = []
    import evaluation.equivalence as eq_mod

    orig_fn = eq_mod.verify_robustness_refined

    def spy_verify(network, region, target_class, target_neurons=None, *, include_hidden_relu=False):
        seen_calls.append(
            (
                region.lower.copy(),
                region.upper.copy(),
                float(region.radius),
                int(target_class),
                bool(include_hidden_relu),
            )
        )
        return orig_fn(
            network,
            region,
            target_class,
            target_neurons,
            include_hidden_relu=include_hidden_relu,
        )

    monkeypatch.setattr(eq_mod, "verify_robustness_refined", spy_verify)
    rec = compare_sample_modes(net, sample, 0.05)

    assert len(seen_calls) == 2
    assert seen_calls[0][4] is True
    assert seen_calls[1][4] is False
    np.testing.assert_array_equal(seen_calls[0][0], seen_calls[1][0])
    np.testing.assert_array_equal(seen_calls[0][1], seen_calls[1][1])
    assert seen_calls[0][2] == seen_calls[1][2] == 0.05
    assert seen_calls[0][3] == seen_calls[1][3] == 0
    assert rec.sample_index == 17


def test_numerical_tolerance_handling_flags_exact_discrepancies() -> None:
    """Differences strictly above tolerance flag exact bound/margin class details."""
    sample = InputSample(index=4, image=np.zeros(4, dtype=np.float64), label=0, predicted=0)
    tol = DEFAULT_EQUIVALENCE_TOLERANCE  # 1e-12

    within = build_pair_record(
        sample,
        0.03,
        _outcome(lower=(0.5, -0.2), upper=(1.2, 0.0), margin=0.5),
        _outcome(lower=(0.5 + 0.5 * tol, -0.2), upper=(1.2, 0.0), margin=0.5),
        tolerance=tol,
    )
    assert within.comparison_complete is True
    assert within.exceeds_tolerance is False
    assert within.verdict_mismatch is False

    over_upper = build_pair_record(
        sample,
        0.03,
        _outcome(lower=(0.5, -0.2), upper=(1.2, 0.0), margin=0.5),
        _outcome(lower=(0.5, -0.2), upper=(1.2, 10.0 * tol), margin=0.5),
        tolerance=tol,
    )
    assert over_upper.comparison_complete is True
    assert over_upper.exceeds_tolerance is True
    assert over_upper.worst_upper_class == 1
    assert over_upper.max_upper_diff == pytest.approx(10.0 * tol)
    assert "class 1" in (over_upper.discrepancy_reason or "")


def test_verdict_mismatch_detection() -> None:
    """Mode A vs Mode B verdict divergence must be explicitly flagged."""
    sample = InputSample(index=9, image=np.zeros(4, dtype=np.float64), label=0, predicted=0)
    rec = build_pair_record(
        sample,
        0.01,
        _outcome(verdict=VerificationStatus.CERTIFIED.value),
        _outcome(verdict=VerificationStatus.INCONCLUSIVE.value),
        tolerance=DEFAULT_EQUIVALENCE_TOLERANCE,
    )
    assert rec.verdict_mismatch is True
    assert "verdict mismatch" in (rec.discrepancy_reason or "")


def test_failed_or_incomplete_comparison_handling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Solver/pipeline errors mark pairs incomplete without crashing or skipping."""
    net = _two_layer_net()
    ds = _stub_dataset()
    import evaluation.equivalence as eq_mod

    orig_fn = eq_mod.verify_robustness_refined

    def flaky_verify(network, region, target_class, target_neurons=None, *, include_hidden_relu=False):
        if not include_hidden_relu:
            raise RuntimeError("simulated solver failure")
        return orig_fn(
            network,
            region,
            target_class,
            target_neurons,
            include_hidden_relu=include_hidden_relu,
        )

    monkeypatch.setattr(eq_mod, "verify_robustness_refined", flaky_verify)
    cfg = EquivalenceConfig(
        sample_count=2,
        epsilons=(0.01,),
        seed=39,
        output_dir=tmp_path,
        experiment_name="eq_failure_test",
        prediction_batch_size=2,
    )
    res = run_equivalence_validation(cfg, network=net, dataset=ds)

    assert res.equivalent is False
    assert res.overall_summary.total_pairs == 2
    assert res.overall_summary.completed_pairs == 0
    assert res.overall_summary.incomplete_pairs == 2
    assert len(res.failing_records()) == 2
    assert all(r.mode_b_verdict == "ERROR" and "RuntimeError" in (r.mode_b_error or "") for r in res.records)


def test_export_integrity_and_unique_experiment_ids(tmp_path: Path) -> None:
    """CSV/JSON round-trip integrity, unique IDs, and non-overwriting guard."""
    ids = {generate_equivalence_experiment_id("equivalence_seed39_n100") for _ in range(10)}
    assert len(ids) == 10

    net = _two_layer_net()
    ds = _stub_dataset()
    cfg = EquivalenceConfig(
        sample_count=2,
        epsilons=(0.01, 0.05),
        seed=39,
        output_dir=tmp_path,
        experiment_name="equivalence_test_export",
        prediction_batch_size=2,
    )
    res1 = run_equivalence_validation(cfg, network=net, dataset=ds)
    res2 = run_equivalence_validation(cfg, network=net, dataset=ds)
    assert res1.experiment_id != res2.experiment_id

    paths = res1.save(overwrite=False)
    assert paths["samples_csv"].is_file()
    assert paths["summary_json"].is_file()

    with paths["summary_json"].open(encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["experiment_id"] == res1.experiment_id
    assert summary["experiment_name"] == "equivalence_test_export"
    assert summary["overall"]["total_pairs"] == 4

    with paths["samples_csv"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert rows[0]["comparison_complete"] == "True"
    assert len(json.loads(rows[0]["mode_a_lower"])) == 2

    with pytest.raises(FileExistsError):
        res2.save(overwrite=False)


def test_historical_experiments_preserved_and_storage_unaffected(tmp_path: Path) -> None:
    """Validation export does not touch historical folders or break storage scanning."""
    results_dir = tmp_path / "results"
    exp_dir = results_dir / "experiments"
    legacy_folder = exp_dir / "experiment_dev_seed42_n5"
    legacy_folder.mkdir(parents=True)
    sentinel_payload = {"experiment_id": "dev_seed42_n5", "status": "Completed"}
    write_summary_json(sentinel_payload, legacy_folder / "summary.json")

    net = _two_layer_net()
    ds = _stub_dataset()
    cfg = EquivalenceConfig(
        sample_count=2,
        epsilons=(0.01,),
        seed=39,
        output_dir=results_dir,
        experiment_name="equivalence_seed39_n100",
        prediction_batch_size=2,
    )
    res = run_equivalence_validation(cfg, network=net, dataset=ds)
    paths = res.save(overwrite=False)

    assert paths["summary_json"].parent == results_dir
    assert json.loads((legacy_folder / "summary.json").read_text(encoding="utf-8")) == sentinel_payload
    assert import_legacy_experiments(source_dir=results_dir, target_base_dir=exp_dir) == []
    listed = list_experiments(exp_dir)
    assert [m.experiment_id for m in listed] == ["dev_seed42_n5"]


def test_checkpoint_resumption_and_workers(tmp_path: Path) -> None:
    """Pre-existing checkpoint records are restored and remaining run in parallel."""
    net = _two_layer_net()
    ds = _stub_dataset()
    cfg = EquivalenceConfig(
        sample_count=4,
        epsilons=(0.01, 0.05),
        seed=39,
        output_dir=tmp_path,
        experiment_name="eq_resumption_test",
        prediction_batch_size=4,
        workers=2,
    )
    checkpoint_file = tmp_path / ".eq_resumption_test_checkpoint.jsonl"

    # Pre-populate checkpoint with 1 completed record
    sample0 = ds[0]
    input_sample0 = InputSample(index=0, image=sample0[0], label=sample0[1], predicted=0)
    rec0 = compare_sample_modes(net, input_sample0, 0.01)
    row0 = rec0.to_row()
    recovered_rec0 = EquivalencePairRecord.from_row(row0)
    assert recovered_rec0.sample_index == rec0.sample_index
    assert recovered_rec0.comparison_complete == rec0.comparison_complete
    assert recovered_rec0.max_lower_diff == rec0.max_lower_diff

    with checkpoint_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row0) + "\n")

    # Run validation with workers=2; must read the checkpoint and complete all 8
    res = run_equivalence_validation(cfg, network=net, dataset=ds)
    assert len(res.records) == 8
    assert res.overall_summary.completed_pairs == 8
    assert not checkpoint_file.exists()

