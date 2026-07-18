"""Generate full or fault-specific MF_IA fault-injection datasets without figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.config import (
    BATCH_SEVERITIES,
    DEFAULT_HEALTHY_NOISE,
    DEFAULT_N_SAMPLES,
    DEFAULT_RESULTS_DIR,
    FAULT_TYPES,
    fault_dataset_matrix,
)
from src.experiment.dataset import write_dataset_summary
from src.experiment.runner import run_experiment


def parse_args() -> argparse.Namespace:
    """Parse a single-fault or complete seven-fault batch request."""
    parser = argparse.ArgumentParser(description="Generate MF_IA fault-injection dataset batches")
    parser.add_argument(
        "--fault",
        choices=("all", *FAULT_TYPES),
        default="all",
        help="Fault family to generate; default runs all seven",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Base seed for independent cases")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Time-series length")
    parser.add_argument(
        "--healthy-noise",
        type=float,
        default=DEFAULT_HEALTHY_NOISE,
        help="Relative healthy measurement noise on the selected sensor",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "batch_runs",
        help="Root directory containing one subdirectory per fault type",
    )
    return parser.parse_args()


def run_fault_batch(args: argparse.Namespace, fault_type: str) -> list[dict[str, object]]:
    """Run one 80-case fault dataset with CSV-only outputs."""
    output_root = args.results_dir / fault_type
    configs = fault_dataset_matrix(
        fault_types=(fault_type,),
        random_seed=args.random_seed,
        n_samples=args.n_samples,
        healthy_noise=args.healthy_noise,
        repetitions=1,
        results_dir=output_root,
    )
    expected_cases = 80
    if len(configs) != expected_cases:
        raise RuntimeError(f"Expected {expected_cases} {fault_type} cases, received {len(configs)}")
    if any(config.generate_figures for config in configs):
        raise RuntimeError("Batch configurations must not generate figures")

    rows: list[dict[str, object]] = []
    for config in configs:
        result = run_experiment(config)
        rows.append(result["metrics"])
        print(
            f"Completed {fault_type}: {config.representative_id} / {config.sensor} "
            f"/ severity {config.severity:g}"
        )

    summary_path = write_dataset_summary(rows, output_root)
    print(f"{fault_type} dataset saved to: {summary_path}")
    return rows


def print_run_summary(args: argparse.Namespace, total_cases: int, selected_faults: tuple[str, ...]) -> None:
    """Print resolved configuration and repeatable command/output references."""
    print("\nBatch configuration")
    print(f"Total experiments: {total_cases}")
    print(f"Severity schedule: {BATCH_SEVERITIES}")
    print(f"Registered faults: {list(FAULT_TYPES)}")
    print(f"Generated fault batches: {list(selected_faults)}")
    print("\nCommands")
    for fault_type in FAULT_TYPES:
        print(f"python experiments/batch_run.py --fault {fault_type}")
    print("python experiments/batch_run.py --fault all")
    print("\nOutput directories")
    for fault_type in selected_faults:
        print(args.results_dir / fault_type)
    if args.fault == "all":
        print(f"Aggregate summary: {args.results_dir / 'fault_dataset.csv'}")


def main() -> None:
    args = parse_args()
    selected_faults = FAULT_TYPES if args.fault == "all" else (args.fault,)
    all_rows: list[dict[str, object]] = []
    for fault_type in selected_faults:
        all_rows.extend(run_fault_batch(args, fault_type))

    expected_total = 560 if args.fault == "all" else 80
    if len(all_rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} total cases, received {len(all_rows)}")
    if args.fault == "all":
        summary_path = write_dataset_summary(all_rows, args.results_dir)
        print(f"Full 560-case dataset summary saved to: {summary_path}")
    print_run_summary(args, len(all_rows), selected_faults)


if __name__ == "__main__":
    main()
