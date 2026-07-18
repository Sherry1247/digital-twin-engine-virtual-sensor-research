"""Numerical dry-run for the production linear-drift signal path.

This file is intentionally isolated from the production experiment runners.
It documents the input matrices and can optionally query the saved ANN when
the local Python environment has the training dependencies installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FEATURES = ("Torque", "p_0", "T_IM", "P_IM", "EGR_Rate", "ECU_VTG_Pos")
BASELINE = np.array((100.0, 1.0, 300.0, 2.8, 15.0, 50.0), dtype=float)
P_IM_INDEX = FEATURES.index("P_IM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace clean-baseline linear drift without changing production code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--healthy-noise", type=float, default=0.02)
    parser.add_argument("--severity", type=float, default=0.05)
    parser.add_argument("--evaluate-ann", action="store_true", help="Load the saved ANN and print actual predictions")
    return parser.parse_args()


def build_matrices(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the relevant input-construction lines in run_single_case.py."""
    rng = np.random.default_rng(args.seed + 1_000_000)
    healthy_noise = rng.normal(0.0, args.healthy_noise * BASELINE[P_IM_INDEX], size=args.n_samples)
    healthy_sensor = BASELINE[P_IM_INDEX] + healthy_noise

    healthy_inputs = np.repeat(BASELINE.reshape(1, -1), args.n_samples, axis=0)
    healthy_inputs[:, P_IM_INDEX] = healthy_sensor

    # The fault starts from a separate pure P_IM array, not healthy_sensor.
    pure_sensor = np.full(args.n_samples, BASELINE[P_IM_INDEX], dtype=float)
    drift = np.linspace(0.0, args.severity, num=args.n_samples)
    faulty_sensor = pure_sensor + pure_sensor * drift
    faulty_inputs = healthy_inputs.copy()
    faulty_inputs[:, P_IM_INDEX] = faulty_sensor
    return healthy_inputs, faulty_inputs


def print_trace(healthy_inputs: np.ndarray, faulty_inputs: np.ndarray) -> None:
    """Print t=0, midpoint, and final input rows in the production feature order."""
    indices = (0, healthy_inputs.shape[0] // 2, healthy_inputs.shape[0] - 1)
    print("Feature order:", ", ".join(FEATURES))
    print("\nInput matrix trace")
    print("sample | healthy input row                                      | faulty input row")
    for index in indices:
        healthy = ", ".join(f"{value:.6f}" for value in healthy_inputs[index])
        faulty = ", ".join(f"{value:.6f}" for value in faulty_inputs[index])
        print(f"{index:6d} | [{healthy}] | [{faulty}]")


def print_ann_predictions(healthy_inputs: np.ndarray, faulty_inputs: np.ndarray) -> None:
    """Optionally run the real saved model; no synthetic ANN is substituted."""
    try:
        from src.experiment.config import ANN_ARTIFACT_PATH
        from src.experiment.evaluator import load_saved_ann_model, predict_with_ann

        model, scaler_x, scaler_y, features = load_saved_ann_model(ANN_ARTIFACT_PATH, "MF_IA")
        if tuple(features) != FEATURES:
            raise ValueError(f"Unexpected saved-model feature order: {features}")
        healthy_prediction = predict_with_ann(model, scaler_x, scaler_y, healthy_inputs)
        faulty_prediction = predict_with_ann(model, scaler_x, scaler_y, faulty_inputs)
        residual = faulty_prediction - healthy_prediction
        print("\nActual saved-ANN trace")
        for index in (0, healthy_inputs.shape[0] // 2, healthy_inputs.shape[0] - 1):
            print(
                f"sample={index}: healthy={healthy_prediction[index]:.6f}, "
                f"faulty={faulty_prediction[index]:.6f}, residual={residual[index]:.6f}"
            )
    except ModuleNotFoundError as error:
        print(f"\nActual ANN trace unavailable: missing dependency '{error.name}'.")
    except Exception as error:
        print(f"\nActual ANN trace unavailable: {error}")


def main() -> None:
    args = parse_args()
    healthy_inputs, faulty_inputs = build_matrices(args)
    print_trace(healthy_inputs, faulty_inputs)
    if args.evaluate_ann:
        print_ann_predictions(healthy_inputs, faulty_inputs)


if __name__ == "__main__":
    main()
