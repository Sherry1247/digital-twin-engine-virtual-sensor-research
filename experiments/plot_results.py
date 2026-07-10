"""Recreate the standard figures from a saved experiment time-series CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.figures import save_experiment_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figures A-C from a saved time_series.csv")
    parser.add_argument("--time-series", type=Path, required=True, help="Path to an experiment time_series.csv")
    parser.add_argument("--target", required=True, help="Target output name")
    parser.add_argument("--sensor", required=True, help="Injected sensor name")
    parser.add_argument("--fault", default="gaussian", help="Injected fault type")
    parser.add_argument("--severity", type=float, default=0.05, help="Fault severity fraction")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for regenerated figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.time_series.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    time_series = pd.read_csv(args.time_series)
    figure_paths = save_experiment_figures(
        time_series=time_series,
        target=args.target,
        sensor=args.sensor,
        fault_type=args.fault,
        severity=args.severity,
        output_dir=output_dir,
    )
    for name, path in figure_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
