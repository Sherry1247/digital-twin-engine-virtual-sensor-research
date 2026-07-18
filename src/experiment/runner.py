"""Orchestrate one labelled fault-injection dataset case."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.fault_injection.bias import inject_bias
from src.fault_injection.dropout import inject_dropout
from src.fault_injection.exponential_drift import inject_exponential_drift
from src.fault_injection.gaussian_noise import inject_relative_gaussian_noise
from src.fault_injection.linear_drift import inject_linear_drift
from src.fault_injection.oscillation import inject_oscillation
from src.fault_injection.saturation import inject_saturation
from src.fault_injection.spike import inject_spike
from .config import ExperimentConfig
from .dataset import write_case_outputs
from .evaluator import (
    extract_residual_features,
    load_saved_ann_model,
    predict_with_ann,
    regression_metrics,
    residual_statistics,
)
from .healthy_signal import generate_healthy_signal
from .representative import load_representative_point


FAULT_REGISTRY = {
    "gaussian": inject_relative_gaussian_noise,
    "bias": inject_bias,
    "linear_drift": inject_linear_drift,
    "exp_drift": inject_exponential_drift,
    "spike": inject_spike,
    "dropout": inject_dropout,
    "saturation": inject_saturation,
    "oscillation": inject_oscillation,
}


def _experiment_output_dir(config: ExperimentConfig) -> Path:
    """Return a unique directory, including a label for duplicate severities."""
    severity_label = f"{config.severity:g}".replace(".", "p")
    return (
        config.results_dir
        / config.target
        / config.representative_id
        / f"{config.fault_type}_{config.sensor}_severity_{severity_label}_rep_{config.repetition:02d}"
    )


def _healthy_seed(fault_seed: int | None) -> int | None:
    """Keep normal measurement noise independent from the injected fault noise."""
    return None if fault_seed is None else fault_seed + 1_000_000


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Generate, evaluate, and export one independently labelled fault case."""
    if config.fault_type not in FAULT_REGISTRY:
        supported = ", ".join(sorted(FAULT_REGISTRY))
        raise ValueError(f"Unsupported fault type '{config.fault_type}'. Supported fault types: {supported}")

    representative = load_representative_point(
        csv_path=config.representative_points_path,
        representative_id=config.representative_id,
        target=config.target,
    )
    model, scaler_x, scaler_y, feature_columns = load_saved_ann_model(config.ann_artifact_path, config.target)
    if config.sensor not in feature_columns:
        raise ValueError(f"Sensor '{config.sensor}' is not one of the ANN inputs: {feature_columns}")

    # Stage 1: only the selected physical sensor varies under healthy operation.
    base_inputs = representative.loc[feature_columns].to_numpy(dtype=float)
    sensor_index = feature_columns.index(config.sensor)
    healthy_sensor = generate_healthy_signal(
        representative_value=base_inputs[sensor_index],
        n_samples=config.n_samples,
        relative_noise=config.healthy_noise,
        random_state=_healthy_seed(config.random_seed),
    )
    healthy_inputs = np.repeat(base_inputs.reshape(1, -1), config.n_samples, axis=0)
    healthy_inputs[:, sensor_index] = healthy_sensor
    healthy_ground_truth = np.full(config.n_samples, float(representative[config.target]), dtype=float)

    # Stage 2: generate the fault profile directly from the clean constant baseline.
    fault_function = FAULT_REGISTRY[config.fault_type]
    pure_sensor = np.full(config.n_samples, base_inputs[sensor_index], dtype=float)
    faulty_sensor = fault_function(pure_sensor, config.severity, random_state=config.random_seed)
    faulty_inputs = healthy_inputs.copy()
    faulty_inputs[:, sensor_index] = faulty_sensor

    # Stage 3: infer with the saved ANN only; models are never retrained here.
    healthy_prediction = predict_with_ann(model, scaler_x, scaler_y, healthy_inputs)
    faulty_prediction = predict_with_ann(model, scaler_x, scaler_y, faulty_inputs)
    # Diagnostic residual measures virtual-sensor behavior divergence.
    residual = faulty_prediction - healthy_prediction
    # Prediction error measures faulty virtual-sensor accuracy against physics.
    prediction_error = healthy_ground_truth - faulty_prediction
    metrics = {
        **regression_metrics(healthy_ground_truth, faulty_prediction),
        **residual_statistics(residual),
    }

    # Stage 4: retain the complete signal record and classifier labels for this case.
    time_series = pd.DataFrame(
        {
            "sample_index": np.arange(config.n_samples),
            "healthy_sensor": healthy_sensor,
            "faulty_sensor": faulty_sensor,
            "healthy_ground_truth": healthy_ground_truth,
            "healthy_prediction": healthy_prediction,
            "faulty_prediction": faulty_prediction,
            "prediction_error": prediction_error,
            "residual": residual,
            "fault_type": config.fault_type,
            "severity": config.severity,
            "repetition": config.repetition,
        }
    )
    metrics_row = {
        "target": config.target,
        "representative_id": config.representative_id,
        "sensor": config.sensor,
        "fault_type": config.fault_type,
        "severity": config.severity,
        "random_seed": config.random_seed,
        "repetition": config.repetition,
        "healthy_noise": config.healthy_noise,
        "n_samples": config.n_samples,
        **metrics,
    }

    output_dir = _experiment_output_dir(config)
    output_paths = write_case_outputs(
        output_dir=output_dir,
        time_series=time_series,
        metrics_row=metrics_row,
        residual_features=extract_residual_features(residual, window_size=30, step_size=10),
    )

    figure_paths: dict[str, Path] = {}
    if config.generate_figures:
        # Import plotting only for the four representative publication cases.
        from src.visualization.figures import save_experiment_figures

        figure_paths = save_experiment_figures(
            time_series=time_series,
            target=config.target,
            sensor=config.sensor,
            fault_type=config.fault_type,
            severity=config.severity,
            output_dir=output_dir,
        )

    return {
        "config": config,
        "output_dir": output_dir,
        "metrics": metrics_row,
        "figure_paths": figure_paths,
        **output_paths,
    }
