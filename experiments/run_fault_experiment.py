"""Command-line entry point for one virtual-sensor fault experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.config import DEFAULT_N_SAMPLES, SEVERITY_LEVELS, ExperimentConfig
from src.experiment.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ANN virtual-sensor fault experiment")
    parser.add_argument("--target", required=True, help="Target output, e.g. MF_IA")
    parser.add_argument("--representative", required=True, help="Representative ID, e.g. MF_IA_High")
    parser.add_argument("--sensor", required=True, help="Physical sensor to corrupt")
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
    config_kwargs = {
        "target": args.target,
        "representative_id": args.representative,
        "sensor": args.sensor,
        "fault_type": args.fault,
        "severity": parse_severity(args.severity),
        "random_seed": args.random_seed,
        "n_samples": args.n_samples,
    }
    if args.results_dir is not None:
        config_kwargs["results_dir"] = args.results_dir

    result = run_experiment(ExperimentConfig(**config_kwargs))
    metrics = result["metrics"]

    print("Virtual Sensor Fault Experiment")
    print(f"Target: {metrics['target']}")
    print(f"Representative: {metrics['representative_id']}")
    print(f"Sensor: {metrics['sensor']}")
    print(f"Fault: {metrics['fault_type']}")
    print(f"Severity: {metrics['severity']}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"Results: {result['output_dir']}")


if __name__ == "__main__":
    main()
