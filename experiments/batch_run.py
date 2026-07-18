"""Generate full or fault-specific MF_IA fault-injection datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.config import (
    DEFAULT_HEALTHY_NOISE,
    DEFAULT_N_SAMPLES,
    DEFAULT_REPETITIONS,
    DEFAULT_RESULTS_DIR,
    FAULT_TYPES,
    fault_dataset_matrix,
)
from src.experiment.dataset import write_dataset_summary
from src.experiment.runner import run_experiment


def parse_args() -> argparse.Namespace:
    """Parse the optional fault filter for a full or segregated batch run."""
    parser = argparse.ArgumentParser(description="Generate MF_IA fault-injection dataset batches")
    parser.add_argument(
        "--fault",
        choices=FAULT_TYPES,
        default=None,
        help="Run one fault type; omit to generate all eight fault datasets",
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
    """Run one isolated 160-case fault dataset and return its summary rows."""
    output_root = args.results_dir / fault_type
    configs = fault_dataset_matrix(
        fault_types=(fault_type,),
        random_seed=args.random_seed,
        n_samples=args.n_samples,
        healthy_noise=args.healthy_noise,
        repetitions=DEFAULT_REPETITIONS,
        results_dir=output_root,
    )
    expected_cases = 160
    if len(configs) != expected_cases:
        raise RuntimeError(f"Expected {expected_cases} {fault_type} cases, received {len(configs)}")

    rows: list[dict[str, object]] = []
    for config in configs:
        result = run_experiment(config)
        rows.append(result["metrics"])
        print(
            f"Completed {fault_type}: {config.representative_id} / {config.sensor} "
            f"/ severity {config.severity:g} / repetition {config.repetition}"
        )

    summary_path = write_dataset_summary(rows, output_root)
    print(f"{fault_type} dataset saved to: {summary_path}")
    return rows


def main() -> None:
    args = parse_args()
    selected_faults = (args.fault,) if args.fault else FAULT_TYPES
    all_rows: list[dict[str, object]] = []
    for fault_type in selected_faults:
        all_rows.extend(run_fault_batch(args, fault_type))

    if args.fault is None:
        if len(all_rows) != 1280:
            raise RuntimeError(f"Expected 1280 total cases, received {len(all_rows)}")
        summary_path = write_dataset_summary(all_rows, args.results_dir)
        print(f"Full 1280-case dataset summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
