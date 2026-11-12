"""Persistent storage service for MiniCert evaluation experiments.

This module manages the storage and retrieval of completed evaluation experiments
in independent directories under ``results/experiments/experiment_<unique_id>/``:

    results/
    └── experiments/
        ├── experiment_<unique_id>/
        │   ├── config.json
        │   ├── summary.json
        │   └── samples.csv
        └── experiment_<unique_id>/
            ├── config.json
            ├── summary.json
            └── samples.csv

Key guarantees:
- Every run receives a unique experiment ID.
- Existing experiment folders are never overwritten.
- All writes are atomic (written to temporary folders and renamed).
- Missing values and corrupted/incomplete folders are handled gracefully.
- Legacy experiment files in ``results/`` (e.g. the 5-sample dev experiment)
  are imported faithfully without modifying any metric values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence
import uuid

import numpy as np

from evaluation.dataset import InputSample, SampleSelection
from evaluation.experiment import ExperimentConfig, ExperimentResult
from evaluation.exports import write_samples_csv, write_summary_json
from evaluation.metrics import (
    BoundTighteningMetrics,
    CertificationMetrics,
    EpsilonMetrics,
    RuntimeMetrics,
    SampleRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

__all__ = [
    "DEFAULT_EXPERIMENTS_DIR",
    "ExperimentMetadata",
    "generate_experiment_id",
    "save_experiment",
    "list_experiments",
    "load_experiment_data",
    "load_experiment_result",
    "import_legacy_experiments",
]


@dataclass(frozen=True)
class ExperimentMetadata:
    """High-level metadata of an experiment folder on disk.

    Attributes:
        experiment_id: Unique identifier for the experiment run.
        experiment_type: Classification: ``"development"`` or ``"configured"``.
        timestamp: ISO-8601 execution timestamp string.
        sample_count: Number of evaluated correctly classified samples.
        epsilons: Perturbation configurations (radii).
        seed: Random seed used for deterministic selection.
        status: Run status: ``"Completed"``, ``"Failed"``, ``"Incomplete"``, or ``"Corrupted"``.
        experiment_name: Configured or generated experiment name.
        folder_name: Subdirectory name under the experiments root.
        path: Absolute or relative Path to the experiment directory.
        error_message: Optional detail string when status is not ``"Completed"``.
    """

    experiment_id: str
    experiment_type: str
    timestamp: str
    sample_count: int
    epsilons: tuple[float, ...]
    seed: int
    status: str
    experiment_name: str
    folder_name: str
    path: Path
    error_message: Optional[str] = None


def generate_experiment_id(prefix: str = "") -> str:
    """Generate a collision-resistant unique experiment ID with timestamp and hex salt."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    salt = uuid.uuid4().hex[:6]
    base = f"{now_str}_{salt}"
    return f"{prefix}_{base}" if prefix else base


def _sanitize_folder_name(experiment_id: str) -> str:
    """Derive the directory name according to the experiment_<unique_id> convention."""
    if experiment_id.startswith("experiment_"):
        return experiment_id
    return f"experiment_{experiment_id}"


def save_experiment(
    result: ExperimentResult,
    experiment_type: str = "configured",
    experiment_id: Optional[str] = None,
    base_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> ExperimentMetadata:
    """Save an experiment atomically into its own directory under base_dir.

    Args:
        result: The complete ExperimentResult to persist.
        experiment_type: Either ``"development"`` or ``"configured"``.
        experiment_id: Optional explicit unique ID; auto-generated if omitted.
        base_dir: Root directory for experiments (defaults to results/experiments).
        timestamp: Optional explicit ISO timestamp; auto-generated if omitted.

    Returns:
        ExperimentMetadata describing the saved experiment.

    Raises:
        FileExistsError: If a folder with the generated/given ID already exists.
        TypeError: If result is not an ExperimentResult.
    """
    if not isinstance(result, ExperimentResult):
        raise TypeError(f"result must be an ExperimentResult, got {type(result).__name__}")
    if experiment_type not in ("development", "configured"):
        experiment_type = "development" if "dev" in str(experiment_type).lower() else "configured"

    root = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    root.mkdir(parents=True, exist_ok=True)

    exp_id = experiment_id or generate_experiment_id()
    folder_name = _sanitize_folder_name(exp_id)
    target_dir = root / folder_name

    if target_dir.exists():
        raise FileExistsError(
            f"Experiment folder '{target_dir}' already exists. Overwriting is forbidden."
        )

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    # Determine status from result
    all_errors = all(
        (rec.baseline_status == "ERROR" and rec.refined_status == "ERROR")
        for rec in result.records
    ) if result.records else False
    status = "Failed" if all_errors else "Completed"

    # Atomic write pattern: write to a temporary folder, then rename
    temp_dir = root / f".tmp_{folder_name}_{uuid.uuid4().hex[:6]}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        config_dict = {
            "experiment_id": exp_id,
            "experiment_type": experiment_type,
            "timestamp": ts,
            "status": status,
            "sample_count": result.config.sample_count,
            "epsilons": list(result.config.epsilons),
            "seed": result.config.seed,
            "data_dir": str(result.config.data_dir),
            "checkpoint": str(result.config.checkpoint),
            "output_dir": str(result.config.output_dir),
            "experiment_name": result.config.experiment_name,
            "prediction_batch_size": result.config.prediction_batch_size,
            "input_lower": result.config.input_lower,
            "input_upper": result.config.input_upper,
        }
        config_path = temp_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
            f.write("\n")

        summary_dict = result.to_dict()
        summary_dict["experiment_id"] = exp_id
        summary_dict["experiment_type"] = experiment_type
        summary_dict["timestamp"] = ts
        summary_dict["status"] = status
        summary_path = temp_dir / "summary.json"
        write_summary_json(summary_dict, summary_path)

        samples_path = temp_dir / "samples.csv"
        write_samples_csv(result.records, samples_path)

        # Atomic rename
        os.replace(temp_dir, target_dir)
    except Exception:
        # Clean up temp_dir on failure
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return ExperimentMetadata(
        experiment_id=exp_id,
        experiment_type=experiment_type,
        timestamp=ts,
        sample_count=result.config.sample_count,
        epsilons=tuple(result.config.epsilons),
        seed=result.config.seed,
        status=status,
        experiment_name=result.config.experiment_name,
        folder_name=folder_name,
        path=target_dir,
    )


def list_experiments(base_dir: Optional[Path] = None) -> list[ExperimentMetadata]:
    """Scan base_dir and return metadata for all stored experiments.

    Handles incomplete or corrupted experiment folders gracefully without
    raising unhandled exceptions. Results are sorted by timestamp descending.
    """
    root = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
    if not root.is_dir():
        return []

    experiments: list[ExperimentMetadata] = []
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith(".tmp"):
            continue
        if not entry.name.startswith("experiment_"):
            continue

        raw_id = entry.name.removeprefix("experiment_")
        config_path = entry / "config.json"
        summary_path = entry / "summary.json"
        samples_path = entry / "samples.csv"

        if not config_path.is_file():
            # Incomplete or corrupted folder
            experiments.append(
                ExperimentMetadata(
                    experiment_id=raw_id,
                    experiment_type="unknown",
                    timestamp="",
                    sample_count=0,
                    epsilons=(),
                    seed=0,
                    status="Incomplete",
                    experiment_name=raw_id,
                    folder_name=entry.name,
                    path=entry,
                    error_message="config.json is missing",
                )
            )
            continue

        try:
            with config_path.open(encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            experiments.append(
                ExperimentMetadata(
                    experiment_id=raw_id,
                    experiment_type="unknown",
                    timestamp="",
                    sample_count=0,
                    epsilons=(),
                    seed=0,
                    status="Corrupted",
                    experiment_name=raw_id,
                    folder_name=entry.name,
                    path=entry,
                    error_message=f"Failed to parse config.json: {exc}",
                )
            )
            continue

        exp_id = cfg.get("experiment_id", raw_id)
        exp_type = cfg.get("experiment_type", "configured")
        ts = cfg.get("timestamp", "")
        sample_count = int(cfg.get("sample_count", 0))
        epsilons = tuple(float(e) for e in cfg.get("epsilons", ()))
        seed = int(cfg.get("seed", 0))
        exp_name = cfg.get("experiment_name", exp_id)
        status = cfg.get("status", "Completed")

        # Check completeness of summary and samples
        error_msg = None
        if not summary_path.is_file() or not samples_path.is_file():
            status = "Incomplete"
            missing = []
            if not summary_path.is_file():
                missing.append("summary.json")
            if not samples_path.is_file():
                missing.append("samples.csv")
            error_msg = f"Missing required file(s): {', '.join(missing)}"
        else:
            # Validate summary.json readability
            try:
                with summary_path.open(encoding="utf-8") as f:
                    json.load(f)
            except Exception as exc:
                status = "Corrupted"
                error_msg = f"Corrupted summary.json: {exc}"

        experiments.append(
            ExperimentMetadata(
                experiment_id=exp_id,
                experiment_type=exp_type,
                timestamp=ts,
                sample_count=sample_count,
                epsilons=epsilons,
                seed=seed,
                status=status,
                experiment_name=exp_name,
                folder_name=entry.name,
                path=entry,
                error_message=error_msg,
            )
        )

    # Sort descending by timestamp (newest first)
    experiments.sort(key=lambda m: m.timestamp or m.folder_name, reverse=True)
    return experiments


def load_experiment_data(
    experiment_id_or_path: str | Path, base_dir: Optional[Path] = None
) -> tuple[dict[str, Any], dict[str, Any], tuple[SampleRecord, ...]]:
    """Load the raw config, summary, and sample records of an experiment.

    Args:
        experiment_id_or_path: Unique experiment ID string or directory Path.
        base_dir: Optional base experiments directory.

    Raises:
        FileNotFoundError: If the experiment folder is missing.
        ValueError: If any experiment file is missing or corrupted.
    """
    if isinstance(experiment_id_or_path, Path):
        target_dir = experiment_id_or_path
        experiment_id = target_dir.name
    else:
        experiment_id = str(experiment_id_or_path)
        folder_name = _sanitize_folder_name(experiment_id)
        root = Path(base_dir or DEFAULT_EXPERIMENTS_DIR)
        target_dir = root / folder_name

    if not target_dir.is_dir():
        raise FileNotFoundError(f"Experiment folder not found at '{target_dir}'")

    config_path = target_dir / "config.json"
    summary_path = target_dir / "summary.json"
    samples_path = target_dir / "samples.csv"

    if not config_path.is_file():
        raise ValueError(f"Experiment '{experiment_id}' is missing required files: config.json")
    if not summary_path.is_file():
        raise ValueError(f"Experiment '{experiment_id}' is missing required files: summary.json")
    if not samples_path.is_file():
        raise ValueError(f"Experiment '{experiment_id}' is missing required files: samples.csv")

    try:
        with config_path.open(encoding="utf-8") as f:
            config_dict = json.load(f)
    except Exception as exc:
        raise ValueError(f"Corrupted config.json in '{experiment_id}': cannot be loaded ({exc})") from exc

    try:
        with summary_path.open(encoding="utf-8") as f:
            summary_dict = json.load(f)
    except Exception as exc:
        raise ValueError(f"Corrupted summary.json in '{experiment_id}': cannot be loaded ({exc})") from exc

    try:
        with samples_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = tuple(SampleRecord.from_row(row) for row in reader)
    except Exception as exc:
        raise ValueError(f"Corrupted samples.csv in '{experiment_id}': cannot be loaded ({exc})") from exc

    return config_dict, summary_dict, records


def load_experiment_result(
    experiment_id_or_path: str | Path, base_dir: Optional[Path] = None
) -> ExperimentResult:
    """Reconstruct an immutable ExperimentResult from persistent storage.

    All metrics, records, and parameters are reconstituted with exact parity
    without recomputing any verification queries.
    """
    config_dict, summary_dict, records = load_experiment_data(experiment_id_or_path, base_dir)

    cfg_data = summary_dict.get("config", config_dict)
    config = ExperimentConfig(
        sample_count=int(cfg_data["sample_count"]),
        epsilons=tuple(float(e) for e in cfg_data["epsilons"]),
        seed=int(cfg_data["seed"]),
        data_dir=Path(cfg_data.get("data_dir", PROJECT_ROOT / "data")),
        checkpoint=Path(cfg_data.get("checkpoint", PROJECT_ROOT / "saved_model.pth")),
        output_dir=Path(cfg_data.get("output_dir", PROJECT_ROOT / "results")),
        experiment_name=cfg_data.get("experiment_name", config_dict.get("experiment_name", "")),
        prediction_batch_size=int(cfg_data.get("prediction_batch_size", 1024)),
        input_lower=float(cfg_data.get("input_lower", 0.0)),
        input_upper=float(cfg_data.get("input_upper", 1.0)),
    )

    # Reconstruct InputSamples for SampleSelection
    seen: set[int] = set()
    samples: list[InputSample] = []
    for r in records:
        if r.sample_index not in seen:
            seen.add(r.sample_index)
            samples.append(
                InputSample(
                    index=r.sample_index,
                    image=np.empty(0, dtype=np.float64),
                    label=r.label,
                    predicted=r.predicted,
                )
            )

    sel_data = summary_dict.get("sample_selection", {})
    selection = SampleSelection(
        samples=tuple(samples),
        requested_count=int(sel_data.get("requested", config.sample_count)),
        incorrect_excluded=int(sel_data.get("incorrect_excluded", 0)),
        scanned=int(sel_data.get("scanned", len(samples))),
        seed=int(sel_data.get("seed", config.seed)),
    )

    metrics_list: list[EpsilonMetrics] = []
    for entry in summary_dict.get("results", []):
        cert = CertificationMetrics(**entry["certification"])
        bounds = BoundTighteningMetrics(**entry["bounds"])
        runtime = RuntimeMetrics(**entry["runtime"])
        metrics_list.append(
            EpsilonMetrics(
                epsilon=float(entry["epsilon"]),
                certification=cert,
                bounds=bounds,
                runtime=runtime,
            )
        )

    model_data = summary_dict.get("model", {})
    network_source = model_data.get("source_model", "MiniCertNet")
    dataset_data = summary_dict.get("dataset", {})
    dataset_size = int(dataset_data.get("total_samples", 10000))

    return ExperimentResult(
        config=config,
        selection=selection,
        metrics=tuple(metrics_list),
        records=records,
        network_source=network_source,
        dataset_size=dataset_size,
    )


def import_legacy_experiments(
    source_dir: Optional[Path] = None,
    target_base_dir: Optional[Path] = None,
) -> list[ExperimentMetadata]:
    """Import existing legacy and completed experiments from results/ into results/experiments/.

    Discovers:
    - Known legacy baseline experiments:
      - ``evaluation_seed42_n5`` -> ``dev_seed42_n5`` (development).
      - ``evaluation_seed42_n20`` -> ``conf_seed42_n20`` (configured).
    - Any newly completed evaluation experiments (``evaluation_*_summary.json``
      and matching ``evaluation_*_samples.csv``).

    Guarantees:
    - All original metrics, sample records, seed, epsilons, and identities are preserved.
    - Idempotent: repeated imports never create duplicate records or directories.
    - Existing records are never overwritten.
    - Incomplete or corrupted runs are handled safely and never imported as completed.
    """
    src = Path(source_dir or DEFAULT_RESULTS_DIR)
    dst = Path(target_base_dir or DEFAULT_EXPERIMENTS_DIR)
    if not src.is_dir():
        return []

    dst.mkdir(parents=True, exist_ok=True)
    imported: list[ExperimentMetadata] = []

    # 1. Fixed legacy mappings for backwards-compatible IDs
    legacy_candidates = [
        ("evaluation_seed42_n5", "dev_seed42_n5", "development"),
        ("evaluation_seed42_n20", "conf_seed42_n20", "configured"),
    ]
    candidates: list[tuple[str, str, Optional[str]]] = [
        (base, tid, exp_type) for base, tid, exp_type in legacy_candidates
    ]
    legacy_bases = {c[0] for c in legacy_candidates}

    # 2. Dynamic discovery of other completed evaluation experiments
    for json_path in sorted(src.glob("evaluation_*_summary.json")):
        base_name = json_path.name[: -len("_summary.json")]
        if base_name in legacy_bases:
            continue
        candidates.append((base_name, base_name, None))

    for base_name, target_id, exp_type in candidates:
        target_folder = dst / f"experiment_{target_id}"
        if target_folder.exists():
            # Already imported - do not overwrite
            continue

        json_file = src / f"{base_name}_summary.json"
        csv_file = src / f"{base_name}_samples.csv"

        # Incomplete experiment: missing summary or CSV artifact
        if not json_file.is_file() or not csv_file.is_file():
            continue

        try:
            with json_file.open(encoding="utf-8") as f:
                summary = json.load(f)

            cfg_data = summary.get("config", {})
            sample_count = int(cfg_data.get("sample_count", 0))
            epsilons = tuple(float(e) for e in cfg_data.get("epsilons", []))
            results = summary.get("results", [])

            with csv_file.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = tuple(SampleRecord.from_row(row) for row in reader)

            # Completeness verification: must have non-empty samples and results
            if sample_count <= 0 or not epsilons or not records:
                continue

            expected_records = sample_count * len(epsilons)
            if len(records) != expected_records or len(results) != len(epsilons):
                # Partial or interrupted experiment; never import as completed
                continue

            # Determine experiment type
            if exp_type is None:
                if sample_count <= 5 or "dev" in base_name.lower():
                    exp_type = "development"
                else:
                    exp_type = "configured"

            # Determine execution status
            all_errors = all(
                (rec.baseline_status == "ERROR" and rec.refined_status == "ERROR")
                for rec in records
            )
            status = "Failed" if all_errors else "Completed"

            # Get timestamp from file modification time
            mtime = json_file.stat().st_mtime
            ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

            config_dict = {
                "experiment_id": target_id,
                "experiment_type": exp_type,
                "timestamp": ts,
                "status": status,
                "sample_count": sample_count,
                "epsilons": list(epsilons),
                "seed": int(cfg_data.get("seed", 42)),
                "data_dir": cfg_data.get("data_dir", str(PROJECT_ROOT / "data")),
                "checkpoint": cfg_data.get("checkpoint", str(PROJECT_ROOT / "saved_model.pth")),
                "output_dir": str(dst),
                "experiment_name": base_name,
                "prediction_batch_size": int(cfg_data.get("prediction_batch_size", 1024)),
                "input_lower": float(cfg_data.get("input_lower", 0.0)),
                "input_upper": float(cfg_data.get("input_upper", 1.0)),
            }

            temp_dir = dst / f".tmp_import_{target_id}_{uuid.uuid4().hex[:4]}"
            temp_dir.mkdir(parents=True, exist_ok=False)

            with (temp_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)
                f.write("\n")

            summary["experiment_id"] = target_id
            summary["experiment_type"] = exp_type
            summary["timestamp"] = ts
            summary["status"] = status
            with (temp_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
                f.write("\n")

            write_samples_csv(records, temp_dir / "samples.csv")

            os.replace(temp_dir, target_folder)

            imported.append(
                ExperimentMetadata(
                    experiment_id=target_id,
                    experiment_type=exp_type,
                    timestamp=ts,
                    sample_count=config_dict["sample_count"],
                    epsilons=tuple(config_dict["epsilons"]),
                    seed=config_dict["seed"],
                    status=status,
                    experiment_name=base_name,
                    folder_name=target_folder.name,
                    path=target_folder,
                )
            )
        except Exception:
            # Skip failed or unreadable imports without crashing
            continue

    return imported
