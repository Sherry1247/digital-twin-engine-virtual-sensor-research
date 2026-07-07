"""
Linear drift fault injection.

Provides `inject_linear_drift` to add a smoothly increasing drift across
samples, optionally combined with random measurement noise:

    faulty(t) = x(t) * (1 + drift(t)) + noise(t)    [relative mode]
    faulty(t) = x(t) + drift(t) + noise(t)          [absolute mode]

where drift(t) ramps linearly from 0 to `drift_level` over the signal, and
noise(t) ~ N(0, noise_std) is i.i.d. Gaussian noise representing ordinary
sensor measurement noise on top of the deterministic calibration drift.

This module also owns the final fault-robustness deliverable for the
linear-drift fault type: sweeping over DRIFT_LEVELS and producing

    results/figures_chart/fault_robustness_summary.csv
    results/figures_chart/Summary_MAE_vs_Drift.png

via `generate_fault_robustness_summary()`. Per-level diagnostic figures
(Figure A/B/C) live in plot_fault_robustness.py; this module is only
responsible for the fault-injection logic and the final summary table/plot
for this fault type.
"""
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Supported drift levels for experiments (fractions)
DRIFT_LEVELS = [
    0.01,  # 1%
    0.02,  # 2%
    0.05,  # 5%
    0.10,  # 10%
    0.20,  # 20%
]


def inject_linear_drift(
    sensor_values: Sequence[float],
    drift_level: float,
    mode: str = "relative",
    noise_std: float = 0.0,
    noise_mode: Optional[str] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Inject a linear drift (optionally plus random noise) into a 1D sensor signal.

    The deterministic drift component starts at zero and increases linearly
    to `drift_level` over the length of the array. Two drift modes are
    supported:

    - ``mode='relative'`` (default): multiplicative drift
      ``faulty = original * (1 + drift(i)) + noise(i)``
      where drift(i) ranges [0, drift_level]
    - ``mode='absolute'``: additive drift
      ``faulty = original + drift(i) + noise(i)``
      where drift(i) ranges [0, drift_level]

    On top of the drift, an independent Gaussian noise term is added at each
    sample to emulate ordinary sensor measurement noise:

    - ``noise_mode='absolute'`` (default when noise_mode is None): noise(i) ~ N(0, noise_std)
      -- noise_std is in the same units as the sensor signal.
    - ``noise_mode='relative'``: noise(i) ~ N(0, noise_std * |original[i]|)
      -- noise_std is a fraction of each sample's own magnitude.

    Set ``noise_std=0.0`` (the default) to get the original pure-drift
    behavior with no random component.

    Args:
        sensor_values: 1D sequence or numpy array of sensor samples.
        drift_level: maximum drift at the last sample (non-negative).
        mode: drift application mode, either "relative" or "absolute".
        noise_std: standard deviation of the Gaussian noise term (non-negative).
            Use 0.0 (default) to disable noise entirely.
        noise_mode: how noise_std is interpreted, either "absolute" (fixed
            units) or "relative" (fraction of each sample's magnitude).
            Defaults to "absolute" if not specified.
        random_state: optional seed (or None) for reproducible noise draws.
            If None, noise is drawn from the default global RNG state
            (non-reproducible across calls unless you pass a seed).

    Returns:
        np.ndarray: New array with drift (+ noise) applied. The original
        input is not modified.

    Raises:
        ValueError: if inputs are invalid.
    """
    arr = np.asarray(sensor_values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("sensor_values must be a 1D sequence of shape (N,)")

    if drift_level < 0:
        raise ValueError("drift_level must be non-negative")

    if mode not in ("relative", "absolute"):
        raise ValueError("mode must be 'relative' or 'absolute'")

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if noise_mode is None:
        noise_mode = "absolute"
    if noise_mode not in ("relative", "absolute"):
        raise ValueError("noise_mode must be 'relative' or 'absolute'")

    N = arr.shape[0]
    if N == 0:
        return arr.copy()

    # drift(i) = drift_level * (i / (N - 1))
    if N == 1:
        drift = np.array([drift_level])
    else:
        drift = drift_level * np.linspace(0.0, 1.0, N)

    # --- random noise term: noise(i) ~ N(0, noise_std) ---------------------
    rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()
    if noise_std > 0:
        if noise_mode == "absolute":
            noise = rng.normal(loc=0.0, scale=noise_std, size=N)
        else:  # relative: scale noise by each sample's own magnitude
            noise = rng.normal(loc=0.0, scale=1.0, size=N) * (noise_std * np.abs(arr))
    else:
        noise = np.zeros(N)

    if mode == "relative":
        factors = 1.0 + drift
        return arr.copy() * factors + noise
    else:
        return arr.copy() + drift + noise


# -------------------------------------------------------------------------
# Fault-robustness summary deliverable for this fault type
# -------------------------------------------------------------------------

def _predict_with_scalers(model, X: np.ndarray, scaler_X, scaler_y) -> np.ndarray:
    X_scaled = X
    if scaler_X is not None:
        X_scaled = scaler_X.transform(X_scaled)
    y_pred = model.predict(X_scaled)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
    else:
        y_pred = np.asarray(y_pred).ravel()
    return y_pred


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def generate_fault_robustness_summary(
    model,
    scaler_X,
    scaler_y,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    sensor_idx: int,
    drift_levels: Optional[List[float]] = None,
    mode: str = "relative",
    noise_std: float = 0.0,
    noise_mode: str = "relative",
    random_state: Optional[int] = 42,
    result_dir: Path = Path("results/figures_chart"),
    target_name: str = "MF_IA",
    model_name: str = "ANN",
    sensor_name: str = "P_IM",
) -> pd.DataFrame:
    """
    Sweep `inject_linear_drift` over `drift_levels` (with a 0.0 / no-fault
    baseline prepended) and produce the final linear-drift fault-robustness
    deliverable:

        <result_dir>/fault_robustness_summary.csv  -- drift_pct, mae, rmse, r2
        <result_dir>/Summary_MAE_vs_Drift.png       -- MAE vs Drift Magnitude (%)

    Args:
        model: trained model exposing .predict().
        scaler_X, scaler_y: optional fitted scalers (or None) used the same
            way as in the rest of the fault-robustness pipeline.
        X_eval: (N, n_features) array of evaluation inputs, in original
            (un-shuffled, log-point) order.
        y_eval: (N,) array of ground-truth targets, same order as X_eval.
        sensor_idx: column index of the sensor to inject drift into.
        drift_levels: drift magnitudes to sweep (fractions). Defaults to
            module-level DRIFT_LEVELS. 0.0 (clean baseline) is always
            prepended automatically if not already present.
        mode: passed through to inject_linear_drift ("relative"/"absolute").
        noise_std, noise_mode, random_state: passed through to
            inject_linear_drift.
        result_dir: directory the CSV/PNG are written to (created if needed).
        target_name, model_name, sensor_name: only used for labeling.

    Returns:
        pd.DataFrame with columns [drift_pct, mae, rmse, r2], one row per
        drift level, sorted by drift_pct ascending.
    """
    if drift_levels is None:
        drift_levels = list(DRIFT_LEVELS)
    levels = sorted(set([0.0] + list(drift_levels)))

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for level in levels:
        X_fault = X_eval.copy()
        sensor_faulted = inject_linear_drift(
            X_eval[:, sensor_idx],
            level,
            mode=mode,
            noise_std=noise_std,
            noise_mode=noise_mode,
            random_state=random_state,
        )
        X_fault[:, sensor_idx] = sensor_faulted

        y_pred_fault = _predict_with_scalers(model, X_fault, scaler_X, scaler_y)
        mae, rmse, r2 = _regression_metrics(y_eval, y_pred_fault)
        rows.append({"drift_pct": level * 100.0, "mae": mae, "rmse": rmse, "r2": r2})

    summary_df = pd.DataFrame(rows).sort_values("drift_pct").reset_index(drop=True)

    csv_path = result_dir / "fault_robustness_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(
        summary_df["drift_pct"],
        summary_df["mae"],
        marker="o",
        markersize=8,
        linewidth=2,
    )
    plt.xlabel("Drift Magnitude (%)", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title(f"{model_name} MAE vs {sensor_name} Linear Drift ({target_name})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = result_dir / "Summary_MAE_vs_Drift.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Fault-robustness summary saved to '{csv_path}'")
    print(summary_df.to_string(index=False))
    print(f"Summary plot saved to '{plot_path}'")

    return summary_df
