"""Prediction and metric helpers for fault experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_saved_ann_model(artifact_path: Path, target: str) -> tuple[object, object, object, list[str]]:
    """Load a saved ANN model entry and its scalers without retraining."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Saved ANN artifact not found: {artifact_path}")

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict) or target not in payload:
        raise KeyError(f"ANN artifact does not contain a saved model for target '{target}'")

    entry = payload[target]
    if not isinstance(entry, dict):
        raise TypeError(f"ANN artifact entry for '{target}' must be a dictionary")

    model = entry.get("model")
    scaler_x = entry.get("scaler_X")
    scaler_y = entry.get("scaler_y")
    features = entry.get("features")
    if model is None or scaler_x is None or scaler_y is None:
        raise KeyError(f"ANN artifact entry for '{target}' is missing model/scaler fields")
    if not features:
        raise KeyError(f"ANN artifact entry for '{target}' is missing feature names")

    return model, scaler_x, scaler_y, list(features)


def predict_with_ann(model: object, scaler_x: object, scaler_y: object, inputs: np.ndarray) -> np.ndarray:
    """Run a saved ANN and return predictions in the original target scale."""
    transformed_inputs = scaler_x.transform(inputs)
    predictions = model.predict(transformed_inputs)
    return scaler_y.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Compute prediction-error metrics for one experiment."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    error = prediction - truth
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
        "residual_mean": float(np.mean(error)),
        "residual_std": float(np.std(error, ddof=0)),
        "residual_min": float(np.min(error)),
        "residual_max": float(np.max(error)),
    }


def extract_residual_features(
    residual: Sequence[float],
    window_size: int = 30,
    step_size: int = 10,
) -> pd.DataFrame:
    """Extract sliding-window diagnostic features from a residual signal."""
    values = np.asarray(residual, dtype=float)
    if values.ndim != 1:
        raise ValueError("residual must be a 1D sequence")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if values.size == 0:
        return pd.DataFrame()

    rows = []
    starts = list(range(0, max(values.size - window_size + 1, 1), step_size))
    if starts[-1] != max(values.size - window_size, 0):
        starts.append(max(values.size - window_size, 0))

    for start in sorted(set(starts)):
        end = min(start + window_size, values.size)
        window = values[start:end]
        rows.append(_window_residual_features(window, start, end))

    return pd.DataFrame(rows)


def _window_residual_features(window: np.ndarray, start: int, end: int) -> dict[str, float | int]:
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    centered = window - mean
    rms = float(np.sqrt(np.mean(window**2)))
    max_error = float(np.max(np.abs(window)))
    slope = _linear_slope(window)
    skewness = _skewness(centered, std)
    kurtosis = _kurtosis(centered, std)
    autocorrelation = _lag1_autocorrelation(window)
    dominant_frequency_energy = _dominant_frequency_energy(window)
    wavelet_energy = _haar_detail_energy(window)
    dropout_ratio = _dropout_ratio(window)
    spike_count = _spike_count(window, mean, std)

    return {
        "window_start": int(start),
        "window_end": int(end),
        "mean": mean,
        "standard_deviation": std,
        "rms": rms,
        "maximum_error": max_error,
        "slope": slope,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "autocorrelation_lag1": autocorrelation,
        "dominant_frequency_energy": dominant_frequency_energy,
        "wavelet_detail_energy": wavelet_energy,
        "dropout_ratio": dropout_ratio,
        "spike_count": spike_count,
    }


def _linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x_axis = np.arange(values.size, dtype=float)
    slope, _ = np.polyfit(x_axis, values, deg=1)
    return float(slope)


def _skewness(centered: np.ndarray, std: float) -> float:
    if std == 0.0:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _kurtosis(centered: np.ndarray, std: float) -> float:
    if std == 0.0:
        return 0.0
    return float(np.mean((centered / std) ** 4))


def _lag1_autocorrelation(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    previous = values[:-1]
    current = values[1:]
    if np.std(previous) == 0.0 or np.std(current) == 0.0:
        return 0.0
    return float(np.corrcoef(previous, current)[0, 1])


def _dominant_frequency_energy(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    spectrum = np.fft.rfft(values - np.mean(values))
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return 0.0
    return float(np.max(power[1:]))


def _haar_detail_energy(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    even_length = values.size - (values.size % 2)
    paired = values[:even_length].reshape(-1, 2)
    details = (paired[:, 0] - paired[:, 1]) / np.sqrt(2.0)
    return float(np.sum(details**2))


def _dropout_ratio(values: np.ndarray, tolerance: float = 1e-9) -> float:
    if values.size < 2:
        return 0.0
    flat_steps = np.isclose(np.diff(values), 0.0, atol=tolerance, rtol=0.0)
    return float(np.mean(flat_steps))


def _spike_count(values: np.ndarray, mean: float, std: float) -> int:
    if values.size < 3 or std == 0.0:
        return 0
    threshold = 2.0 * std
    local_peak = (np.abs(values[1:-1] - mean) > threshold) & (
        np.abs(values[1:-1]) >= np.abs(values[:-2])
    ) & (np.abs(values[1:-1]) >= np.abs(values[2:]))
    return int(np.sum(local_peak))
