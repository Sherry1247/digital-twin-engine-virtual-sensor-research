"""Run the 12 configured virtual-sensor fault experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.config import DEFAULT_N_SAMPLES, SEVERITY_LEVELS, default_experiment_matrix
from src.experiment.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all configured ANN virtual-sensor fault experiments")
    parser.add_argument("--fault", default="gaussian", help="Fault type to inject")
    parser.add_argument("--severity", default="0.05", help="Fault severity fraction or preset name")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for stochastic faults")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Synthetic time-series length")
    parser.add_argument("--results-dir", type=Path, default=None, help="Output directory for experiment results")
    return parser.parse_args()


def parse_severity(value: str) -> float:
    if value in SEVERITY_LEVELS:
        return SEVERITY_LEVELS[value]
    return float(value)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir if args.results_dir is not None else None
    configs = default_experiment_matrix(
        fault_type=args.fault,
        severity=parse_severity(args.severity),
        random_seed=args.random_seed,
        n_samples=args.n_samples,
        **({"results_dir": results_dir} if results_dir is not None else {}),
    )

    rows = []
    for config in configs:
        result = run_experiment(config)
        rows.append(result["metrics"])
        print(f"Completed {config.target} / {config.representative_id} / {config.sensor}")

    summary = pd.DataFrame(rows)
    summary_path = configs[0].results_dir / "batch_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Batch summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
