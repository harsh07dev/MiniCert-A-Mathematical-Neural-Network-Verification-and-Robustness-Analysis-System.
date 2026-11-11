"""Command-line entry point for the Phase 4A evaluation experiment.

Example::

    python -m evaluation --sample-count 100 --epsilon 0.01 --epsilon 0.03 \\
        --epsilon 0.05 --epsilon 0.1 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from evaluation.experiment import ExperimentConfig, PROJECT_ROOT, run_experiment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments into an argparse namespace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="number of correctly classified test samples to evaluate",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        action="append",
        dest="epsilons",
        default=None,
        help="L-infinity perturbation radius; repeatable",
    )
    parser.add_argument("--seed", type=int, default=42, help="sample-selection seed")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="MNIST data directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "saved_model.pth",
        help="trained MiniCertNet checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="directory for the CSV and JSON exports",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="experiment name base for the export filenames",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="forward-pass batch size for nominal predictions",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured experiment, export results, and print a summary."""
    args = parse_args(argv)
    epsilons = tuple(args.epsilons) if args.epsilons else (0.01, 0.03, 0.05, 0.1)
    config = ExperimentConfig(
        sample_count=args.sample_count,
        epsilons=epsilons,
        seed=args.seed,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        experiment_name=args.name,
        prediction_batch_size=args.batch_size,
    )
    result = run_experiment(config)
    paths = result.save()

    selection = result.selection
    print(
        f"Experiment '{config.experiment_name}': "
        f"selected {selection.selected_count} correctly classified samples "
        f"(seed {selection.seed}, {selection.incorrect_excluded} incorrect excluded)"
    )
    print(f"Samples CSV:    {paths['samples_csv']}")
    print(f"Summary JSON:   {paths['summary_json']}")
    for metrics in result.metrics:
        cert = metrics.certification
        bounds = metrics.bounds
        runtime = metrics.runtime
        baseline_rate = (
            "n/a" if cert.baseline_rate is None else f"{cert.baseline_rate * 100:.2f}%"
        )
        refined_rate = (
            "n/a" if cert.refined_rate is None else f"{cert.refined_rate * 100:.2f}%"
        )
        improvement = (
            "n/a" if cert.improvement_pp is None else f"{cert.improvement_pp:+.2f}pp"
        )
        mean_baseline = (
            "n/a"
            if bounds.mean_baseline_width is None
            else f"{bounds.mean_baseline_width:.6g}"
        )
        mean_refined = (
            "n/a"
            if bounds.mean_refined_width is None
            else f"{bounds.mean_refined_width:.6g}"
        )
        mean_refined_time = (
            "n/a"
            if runtime.mean_total_refined_time is None
            else f"{runtime.mean_total_refined_time:.4f}s"
        )
        print(
            f"epsilon={metrics.epsilon:<6g} "
            f"evaluated={cert.evaluated_samples:<4d} "
            f"baseline_certified={cert.baseline_certified:<4d} ({baseline_rate})  "
            f"refined_certified={cert.refined_certified:<4d} ({refined_rate})  "
            f"improvement={improvement}  "
            f"mean_widths {mean_baseline} -> {mean_refined}  "
            f"mean_refined_time={mean_refined_time}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
