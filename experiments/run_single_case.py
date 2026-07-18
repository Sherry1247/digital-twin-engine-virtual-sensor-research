"""Run one MF_IA fault case for debugging and figure validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.config import (
    ANN_ARTIFACT_PATH,
    DEFAULT_HEALTHY_NOISE,
    DEFAULT_N_SAMPLES,
    DEFAULT_RESULTS_DIR,
    FAULT_TYPES,
    TARGET,
)
from src.experiment.evaluator import (
    extract_residual_features,
    load_saved_ann_model,
    predict_with_ann,
    residual_statistics,
)
from src.experiment.healthy_signal import generate_healthy_signal
from src.experiment.representative import load_representative_point
from src.experiment.runner import FAULT_REGISTRY


REPRESENTATIVES = ("MF_IA_High", "MF_IA_Low")
SENSORS = ("P_IM", "EGR_Rate")


def parse_args() -> argparse.Namespace:
    """Parse one independently runnable debugging experiment."""
    parser = argparse.ArgumentParser(description="Run one MF_IA fault-injection debugging case")
    parser.add_argument("--representative", choices=REPRESENTATIVES, required=True)
    parser.add_argument("--sensor", choices=SENSORS, required=True)
    parser.add_argument("--fault", choices=FAULT_TYPES, required=True)
    parser.add_argument("--severity", type=float, required=True, help="Relative fault severity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the fault case")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--healthy-noise", type=float, default=DEFAULT_HEALTHY_NOISE)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "single_cases",
        help="Root directory for this debugging case",
    )
    return parser.parse_args()


def case_output_dir(args: argparse.Namespace) -> Path:
    """Create a reproducible, non-colliding output directory for one case."""
    severity = f"{args.severity:g}".replace(".", "p")
    return args.results_dir / args.representative / f"{args.fault}_{args.sensor}_severity_{severity}_seed_{args.seed}"


def run_single_case(args: argparse.Namespace) -> dict[str, Path]:
    """Execute the healthy-signal, fault, ANN, residual, and plotting stages."""
    if args.severity < 0:
        raise ValueError("severity must be non-negative")
    if args.healthy_noise < 0:
        raise ValueError("healthy-noise must be non-negative")

    representative = load_representative_point(
        csv_path=REPO_ROOT / "results" / "representative_points" / "representative_operating_points.csv",
        representative_id=args.representative,
        target=TARGET,
    )
    model, scaler_x, scaler_y, feature_columns = load_saved_ann_model(ANN_ARTIFACT_PATH, TARGET)
    sensor_index = feature_columns.index(args.sensor)

    # Stage 1: vary only the selected physical sensor under normal operation.
    base_inputs = representative.loc[feature_columns].to_numpy(dtype=float)
    healthy_sensor = generate_healthy_signal(
        representative_value=base_inputs[sensor_index],
        n_samples=args.n_samples,
        relative_noise=args.healthy_noise,
        random_state=args.seed + 1_000_000,
    )
    healthy_inputs = np.repeat(base_inputs.reshape(1, -1), args.n_samples, axis=0)
    healthy_inputs[:, sensor_index] = healthy_sensor

    # Stage 2: derive one fault from the clean baseline, never from healthy noise.
    pure_sensor = np.full(args.n_samples, base_inputs[sensor_index], dtype=float)
    faulty_sensor = FAULT_REGISTRY[args.fault](pure_sensor, args.severity, random_state=args.seed)
    faulty_inputs = healthy_inputs.copy()
    faulty_inputs[:, sensor_index] = faulty_sensor

    # Stage 3: infer with the existing saved ANN; no model training occurs here.
    healthy_prediction = predict_with_ann(model, scaler_x, scaler_y, healthy_inputs)
    faulty_prediction = predict_with_ann(model, scaler_x, scaler_y, faulty_inputs)
    ground_truth = np.full(args.n_samples, float(representative[TARGET]), dtype=float)
    residual = faulty_prediction - healthy_prediction
    prediction_error = ground_truth - faulty_prediction

    time_series = pd.DataFrame(
        {
            "sample_index": np.arange(args.n_samples),
            "healthy_sensor": healthy_sensor,
            "faulty_sensor": faulty_sensor,
            "healthy_ground_truth": ground_truth,
            "healthy_prediction": healthy_prediction,
            "faulty_prediction": faulty_prediction,
            "residual": residual,
            "prediction_error": prediction_error,
            "fault_type": args.fault,
            "severity": args.severity,
        }
    )
    summary = {
        "target": TARGET,
        "representative_id": args.representative,
        "sensor": args.sensor,
        "fault_type": args.fault,
        "severity": args.severity,
        "random_seed": args.seed,
        "n_samples": args.n_samples,
        "healthy_noise": args.healthy_noise,
        **residual_statistics(residual),
        "prediction_error_mae": float(np.mean(np.abs(prediction_error))),
        "prediction_error_rmse": float(np.sqrt(np.mean(prediction_error**2))),
    }

    output_dir = case_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.csv",
        "time_series": output_dir / "time_series.csv",
        "residual_features": output_dir / "residual_features.csv",
    }
    pd.DataFrame([summary]).to_csv(paths["summary"], index=False)
    time_series.to_csv(paths["time_series"], index=False)
    extract_residual_features(residual, window_size=30, step_size=10).to_csv(
        paths["residual_features"], index=False
    )
    # Plotting is loaded only after CSV exports so command-line validation does
    # not require a plotting backend.
    from src.visualization.figures import save_experiment_figures

    paths.update(
        save_experiment_figures(
            time_series=time_series,
            target=TARGET,
            sensor=args.sensor,
            fault_type=args.fault,
            severity=args.severity,
            output_dir=output_dir,
        )
    )
    return paths


def main() -> None:
    args = parse_args()
    paths = run_single_case(args)
    print("Single-case experiment completed")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
