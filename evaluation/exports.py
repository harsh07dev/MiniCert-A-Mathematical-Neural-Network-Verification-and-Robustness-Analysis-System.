"""Deterministic CSV and JSON exports for evaluation experiments.

The writers produce one CSV file with all per-sample records (each row
carrying its epsilon value) and one JSON file with the experiment
configuration, the sample-selection statistics, and the aggregated metrics
per epsilon. Filenames are derived from the configured experiment name, so
repeated runs with identical configuration overwrite deterministic paths.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.metrics import SampleRecord

__all__ = ["write_samples_csv", "write_summary_json"]


def write_samples_csv(records: Sequence[SampleRecord], path: Path) -> Path:
    """Write all per-sample records to a CSV file.

    Args:
        records: The per-sample records of the experiment.
        path: Target CSV path; parent directories are created.

    Returns:
        The written path.

    Raises:
        TypeError: If a record is not a SampleRecord.
    """
    for record in records:
        if not isinstance(record, SampleRecord):
            raise TypeError(
                f"records must contain SampleRecord entries, got {type(record).__name__}"
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SampleRecord.CSV_FIELDNAMES))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return target


def write_summary_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Write the JSON experiment summary.

    The payload is written with fixed key order and ``allow_nan=False``, so a
    summary containing non-finite floats fails loudly instead of emitting
    invalid JSON.

    Args:
        payload: The JSON-serializable summary (see
            :meth:`ExperimentResult.to_dict`).
        path: Target JSON path; parent directories are created.

    Returns:
        The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return target
