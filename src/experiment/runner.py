"""Execution path for one virtual-sensor fault experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.fault_injection.gaussian_noise import inject_relative_gaussian_noise
from src.visualization.figures import save_experiment_figures

from .config import ExperimentConfig
from .evaluator import extract_residual_features, load_saved_ann_model, predict_with_ann, regression_metrics
from .representative import load_representative_point


FAULT_REGISTRY = {
    "gaussian": inject_relative_gaussian_noise,
}


def build_healthy_timeseries(
    representative_row: pd.Series,
    n_samples: int,
    target: str,
    feature_columns: list[str],
    sensor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeat a representative operating point into a constant time series."""
    if sensor not in feature_columns:
        raise ValueError(f"Sensor '{sensor}' is not one of the ANN inputs: {feature_columns}")

    base_inputs = representative_row.loc[feature_columns].to_numpy(dtype=float)
    healthy_ground_truth = np.full(n_samples, float(representative_row[target]), dtype=float)
    healthy_sensor = np.full(n_samples, base_inputs[feature_columns.index(sensor)], dtype=float)
    healthy_inputs = np.repeat(base_inputs.reshape(1, -1), n_samples, axis=0)
    return healthy_inputs, healthy_ground_truth, healthy_sensor


def _experiment_output_dir(config: ExperimentConfig) -> Path:
    severity_label = f"{config.severity:g}".replace(".", "p")
    return (
        config.results_dir
        / config.target
        / config.representative_id
        / f"{config.fault_type}_{config.sensor}_severity_{severity_label}"
    )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Execute one configured fault experiment and save metrics and figures."""
    if config.fault_type not in FAULT_REGISTRY:
        supported = ", ".join(sorted(FAULT_REGISTRY))
        raise ValueError(f"Unsupported fault type '{config.fault_type}'. Supported fault types: {supported}")

    representative = load_representative_point(
        csv_path=config.representative_points_path,
        representative_id=config.representative_id,
        target=config.target,
    )
    model, scaler_x, scaler_y, feature_columns = load_saved_ann_model(config.ann_artifact_path, config.target)

    healthy_inputs, healthy_ground_truth, healthy_sensor = build_healthy_timeseries(
        representative_row=representative,
        n_samples=config.n_samples,
        target=config.target,
        feature_columns=feature_columns,
        sensor=config.sensor,
    )

    fault_function = FAULT_REGISTRY[config.fault_type]
    faulty_sensor = fault_function(healthy_sensor.copy(), config.severity, random_state=config.random_seed)
    faulty_inputs = healthy_inputs.copy()
    faulty_inputs[:, feature_columns.index(config.sensor)] = faulty_sensor

    healthy_prediction = predict_with_ann(model, scaler_x, scaler_y, healthy_inputs)
    faulty_prediction = predict_with_ann(model, scaler_x, scaler_y, faulty_inputs)
    residual = faulty_prediction - healthy_ground_truth
    prediction_error = faulty_prediction - healthy_prediction
    metrics = regression_metrics(healthy_ground_truth, faulty_prediction)

    sample_index = np.arange(config.n_samples)
    time_series = pd.DataFrame(
        {
            "sample_index": sample_index,
            "healthy_sensor": healthy_sensor,
            "faulty_sensor": faulty_sensor,
            "healthy_ground_truth": healthy_ground_truth,
            "healthy_prediction": healthy_prediction,
            "faulty_prediction": faulty_prediction,
            "prediction_error": prediction_error,
            "residual": residual,
            "severity": np.full(config.n_samples, config.severity, dtype=float),
        }
    )

    output_dir = _experiment_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_series_path = output_dir / "time_series.csv"
    metrics_path = output_dir / "metrics.csv"
    residual_features_path = output_dir / "residual_features.csv"
    time_series.to_csv(time_series_path, index=False)

    metrics_row = {
        "target": config.target,
        "representative_id": config.representative_id,
        "sensor": config.sensor,
        "fault_type": config.fault_type,
        "severity": config.severity,
        "random_seed": config.random_seed,
        "n_samples": config.n_samples,
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    residual_features = extract_residual_features(residual, window_size=30, step_size=10)
    residual_features.to_csv(residual_features_path, index=False)

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
        "metrics_path": metrics_path,
        "residual_features_path": residual_features_path,
        "time_series_path": time_series_path,
        "figure_paths": figure_paths,
    }
