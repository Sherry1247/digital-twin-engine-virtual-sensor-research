import os
import pickle
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fault_injection.linear_drift import inject_linear_drift, DRIFT_LEVELS

# -------------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------------
# Everything that changes between experiments (model, target, sensor, fault
# type) lives here. Adding a new fault type (noise / spike / stuck-at /
# exponential drift / ...) only requires registering it in FAULT_REGISTRY
# and switching FAULT_TYPE below; nothing else in the script needs to change.

MODEL_NAME = "ANN"
TARGET_OUTPUT = "MF_IA"
SENSOR_NAME = "P_IM"

FAULT_TYPE = "linear_drift"

# Sweep every severity level defined in DRIFT_LEVELS (1%, 2%, 5%, 10%, 20%)
# instead of hardcoding a single level. Each level gets its own Figure A/B/C
# plus its own data-driven thresholds (the thresholds only depend on the
# CLEAN baseline, so they are identical across levels -- only the faulted
# curves differ).
SWEEP_DRIFT_LEVELS: List[float] = list(DRIFT_LEVELS)  # [0.01, 0.02, 0.05, 0.10, 0.20]
FAULT_MODE = "relative"

# Noise added on top of the deterministic drift, to emulate ordinary sensor
# measurement noise (see linear_drift.py). Set NOISE_STD=0.0 to disable.
NOISE_STD = 0.01          # in NOISE_MODE units (relative -> fraction of |signal|)
NOISE_MODE = "relative"   # "relative" (fraction of |signal|) or "absolute" (fixed units)
NOISE_RANDOM_STATE = 42   # fixed seed so figures are reproducible across runs

# Registry: fault_type -> callable(sensor_array, drift_level, **extra) -> faulted_array
FAULT_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "linear_drift": lambda x, drift_level: inject_linear_drift(
        x,
        drift_level,
        mode=FAULT_MODE,
        noise_std=NOISE_STD,
        noise_mode=NOISE_MODE,
        random_state=NOISE_RANDOM_STATE,
    ),
    # "noise": inject_noise,
    # "spike": inject_spike,
    # "stuck_at": inject_stuck_at,
    # "exponential_drift": inject_exponential_drift,
}

SENSOR_TOLERANCE_PCT = 0.05  # +-5% sensor spec band (this one IS a real spec, keep it)

# Data-driven error thresholds (replace the old hardcoded 100% / 200% bands).
# Percentiles are computed from the CLEAN/baseline prediction error distribution.
GREEN_PERCENTILE = 95.0   # below this -> small degradation
YELLOW_PERCENTILE = 99.0  # below this -> moderate degradation; above -> large degradation

RESULT_DIR = Path("results/figures_chart")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ARTIFACT = ROOT / "experiments" / "ann" / "test_results" / "visualizations" / "ann_all_inputs_results.pkl"
DATA_FILE = ROOT / "data" / "processed" / "df_processed.csv"
DEFAULT_FEATURES = ["Torque", "p_0", "T_IM", "P_IM", "EGR_Rate", "ECU_VTG_Pos"]

# -------------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------------

def load_saved_ann_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Saved {MODEL_NAME} artifact not found: {path}\n"
            f"Please generate the {MODEL_NAME} results artifact before running this plot script."
        )
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Unable to load the saved artifact because a required package is missing. "
                "This artifact depends on scikit-learn classes, so please run this script "
                "in the same environment used to train the model or install scikit-learn."
            ) from exc


def load_processed_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {path}\n"
            "Ensure df_processed.csv exists in data/processed."
        )
    return pd.read_csv(path)


def get_features(entry: dict) -> list[str]:
    features = entry.get("features")
    return list(features) if features is not None else DEFAULT_FEATURES.copy()


def predict_with_scalers(model, X: np.ndarray, scaler_X, scaler_y) -> np.ndarray:
    X_scaled = X
    if scaler_X is not None:
        X_scaled = scaler_X.transform(X_scaled)

    y_pred = model.predict(X_scaled)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
    else:
        y_pred = np.asarray(y_pred).ravel()
    return y_pred


def apply_fault(sensor_values: np.ndarray, fault_type: str, drift_level: float) -> np.ndarray:
    if fault_type not in FAULT_REGISTRY:
        raise KeyError(f"Unknown fault type '{fault_type}'. Registered: {list(FAULT_REGISTRY)}")
    return FAULT_REGISTRY[fault_type](sensor_values, drift_level)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    error = y_pred - y_true
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return mae, rmse, r2


def plot_tolerance_region(ax, x, lower, upper, color, label):
    ax.fill_between(x, lower, upper, color=color, alpha=0.2, label=label)


def level_tag(drift_level: float) -> str:
    """e.g. 0.2 -> '20pct', 0.01 -> '1pct' -- used in filenames/titles."""
    pct = drift_level * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}pct"
    return f"{pct:g}pct".replace(".", "_")


# -------------------------------------------------------------------------
# 3. Load Model & Data (done once, shared across all drift levels)
# -------------------------------------------------------------------------
model_results = load_saved_ann_artifact(MODEL_ARTIFACT)
if TARGET_OUTPUT not in model_results:
    raise KeyError(
        f"Target '{TARGET_OUTPUT}' is not available in the saved {MODEL_NAME} artifact."
    )

entry = model_results[TARGET_OUTPUT]
features = get_features(entry)
if SENSOR_NAME not in features:
    raise KeyError(
        f"Sensor '{SENSOR_NAME}' is not available in model features: {features}"
    )

model = entry["model"]
scaler_X = entry.get("scaler_X")
scaler_y = entry.get("scaler_y")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found at {DATA_FILE}."
    )

df = load_processed_data(DATA_FILE)
missing_features = [feat for feat in features if feat not in df.columns]
if missing_features:
    raise KeyError(
        f"Missing required features in processed data: {missing_features}"
    )

# --- Use the full log in its original order, NOT a random shuffled split. ---
# A random test split destroys the time/log-point ordering of the data, so a
# "trajectory" plotted over a random split has no physical meaning (sample i
# and sample i+1 are unrelated log points). For fault-robustness figures we
# need to walk the engine log in order, so we evaluate on the entire
# (ordered) dataset.
X_eval = df[features].values
y_eval = df[TARGET_OUTPUT].values

sensor_idx = features.index(SENSOR_NAME)
sensor_clean = X_eval[:, sensor_idx].copy()

sample_count = len(y_eval)
log_point = np.arange(sample_count)  # x-axis = actual log point index in the CSV

# --- Clean-input baseline (computed once; identical for every drift level) ---
y_pred_clean = predict_with_scalers(model, X_eval, scaler_X, scaler_y)
baseline_mae, baseline_rmse, baseline_r2 = regression_metrics(y_eval, y_pred_clean)
baseline_mae = float(np.asarray(baseline_mae).item())

# Data-driven thresholds, derived from the baseline (clean-input) error
# distribution, instead of hardcoded 100% / 200% bands. This makes the
# zones meaningful regardless of the output's units/scale (MF_IA vs NOx vs
# SOC, etc.), and is defensible to a reviewer because the threshold is
# derived from the model's own observed performance, not chosen by hand.
# These thresholds only depend on clean data, so they are computed ONCE and
# reused across every drift level in the sweep.
baseline_error = np.abs(y_pred_clean - y_eval)
green_limit = float(np.percentile(baseline_error, GREEN_PERCENTILE))
yellow_limit = float(np.percentile(baseline_error, YELLOW_PERCENTILE))

print(f"Baseline MAE={baseline_mae:.4f}  P{GREEN_PERCENTILE:.0f} error={green_limit:.4f}  P{YELLOW_PERCENTILE:.0f} error={yellow_limit:.4f}")
print(f"Sweeping {FAULT_TYPE} over drift levels: {SWEEP_DRIFT_LEVELS}")

# -------------------------------------------------------------------------
# 4. Sweep: generate Figure A/B/C for every drift level
# -------------------------------------------------------------------------
for drift_level in SWEEP_DRIFT_LEVELS:
    tag = level_tag(drift_level)

    sensor_faulted = apply_fault(sensor_clean, FAULT_TYPE, drift_level)

    X_eval_fault = X_eval.copy()
    X_eval_fault[:, sensor_idx] = sensor_faulted

    y_pred_fault = predict_with_scalers(model, X_eval_fault, scaler_X, scaler_y)

    # ---------------------------------------------------------------------
    # FIGURE A: Sensor Level Fault & Specification Band
    # ---------------------------------------------------------------------
    plt.figure(figsize=(11, 5))

    sensor_upper = sensor_clean * (1 + SENSOR_TOLERANCE_PCT)
    sensor_lower = sensor_clean * (1 - SENSOR_TOLERANCE_PCT)

    plot_tolerance_region(
        plt, log_point, sensor_lower, sensor_upper,
        color="gray", label=f"Sensor Spec Band (±{SENSOR_TOLERANCE_PCT * 100:.0f}%)"
    )
    plt.plot(log_point, sensor_clean, color="black", linewidth=1.5, label=f"Original {SENSOR_NAME}")
    plt.plot(
        log_point,
        sensor_faulted,
        color="crimson",
        linewidth=2,
        linestyle="--",
        label=f"Faulted {SENSOR_NAME} (drift {drift_level*100:.0f}% + noise std={NOISE_STD} [{NOISE_MODE}])"
    )

    sensor_violation = (sensor_faulted > sensor_upper) | (sensor_faulted < sensor_lower)
    if np.any(sensor_violation):
        first_breach = log_point[sensor_violation][0]
        plt.axvline(x=first_breach, color="red", linestyle=":", alpha=0.8)
        plt.text(
            first_breach + sample_count * 0.02,
            np.min(sensor_clean),
            f"Spec Breach\n(Log Point {first_breach})",
            color="red",
            fontsize=9,
        )

    plt.xlabel("Log Point", fontsize=11)
    plt.ylabel(f"{SENSOR_NAME} Value", fontsize=11)
    plt.title(
        f"Figure A: {SENSOR_NAME} Sensor Linear Drift ({drift_level*100:.0f}%) + Noise vs Tolerance Region",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"Figure_A_Sensor_Drift_{tag}.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # FIGURE B: Model Performance Relative to Data-Driven Error Thresholds
    # ---------------------------------------------------------------------
    plt.figure(figsize=(11, 5))

    abs_error_fault = np.abs(y_pred_fault - y_eval)

    plt.axhspan(
        0,
        green_limit,
        color="limegreen",
        alpha=0.15,
        label=f"Small degradation (<= P{GREEN_PERCENTILE:.0f} baseline error)"
    )
    plt.axhspan(
        green_limit,
        yellow_limit,
        color="gold",
        alpha=0.2,
        label=f"Moderate degradation (P{GREEN_PERCENTILE:.0f}-P{YELLOW_PERCENTILE:.0f} baseline error)"
    )
    plt.axhspan(
        yellow_limit,
        max(abs_error_fault.max() * 1.05, yellow_limit * 1.1),
        color="crimson",
        alpha=0.12,
        label=f"Large degradation (> P{YELLOW_PERCENTILE:.0f} baseline error)"
    )

    plt.plot(
        log_point,
        abs_error_fault,
        color="navy",
        linewidth=2,
        label=f"|Prediction error| under fault ({TARGET_OUTPUT})"
    )
    plt.axhline(
        green_limit,
        color="darkgreen",
        linestyle="--",
        linewidth=1.2,
        label=f"P{GREEN_PERCENTILE:.0f} baseline error = {green_limit:.3g}"
    )
    plt.axhline(
        yellow_limit,
        color="darkorange",
        linestyle="--",
        linewidth=1.2,
        label=f"P{YELLOW_PERCENTILE:.0f} baseline error = {yellow_limit:.3g}"
    )

    plt.xlabel("Log Point", fontsize=11)
    plt.ylabel(f"Absolute Prediction Error ({TARGET_OUTPUT} units)", fontsize=11)
    plt.title(
        f"Figure B: {MODEL_NAME} Prediction Degradation vs Data-Driven Error Thresholds (drift {drift_level*100:.0f}%)",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sample_count - 1)
    plt.ylim(0, max(abs_error_fault.max() * 1.05, yellow_limit * 1.1))
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"Figure_B_Error_Zones_{tag}.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # FIGURE C: Prediction Comparison under Input Fault Injection
    # ---------------------------------------------------------------------
    plt.figure(figsize=(11, 5.5))

    # Acceptable-prediction band derived from the baseline error distribution
    # (P95) rather than an arbitrary +-5%, so it is unit-aware per target.
    prediction_upper = y_eval + green_limit
    prediction_lower = y_eval - green_limit
    plot_tolerance_region(
        plt,
        log_point,
        prediction_lower,
        prediction_upper,
        color="limegreen",
        label=f"Acceptable prediction region (±P{GREEN_PERCENTILE:.0f} baseline error)"
    )

    plt.plot(log_point, y_eval, color="black", linewidth=1.5, label=f"Ground Truth ({TARGET_OUTPUT})")
    plt.plot(log_point, y_pred_clean, color="blue", linewidth=1.5, alpha=0.8, label=f"{MODEL_NAME} Prediction (clean input)")
    plt.plot(log_point, y_pred_fault, color="red", linewidth=1.8, label=f"{MODEL_NAME} Prediction ({SENSOR_NAME} faulted)")

    out_of_bounds = (y_pred_fault > prediction_upper) | (y_pred_fault < prediction_lower)
    plt.scatter(
        log_point[out_of_bounds],
        y_pred_fault[out_of_bounds],
        color="darkred",
        s=20,
        zorder=5,
        label="Out-of-spec prediction points"
    )

    plt.xlabel("Log Point", fontsize=11)
    plt.ylabel(f"{TARGET_OUTPUT} Prediction", fontsize=11)
    plt.title(
        f"Figure C: {MODEL_NAME} Output Comparison Before/After {SENSOR_NAME} Drift Injection (drift {drift_level*100:.0f}%)",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"Figure_C_Prediction_Tolerance_{tag}.png", dpi=300)
    plt.close()

    fault_mae, fault_rmse, fault_r2 = regression_metrics(y_eval, y_pred_fault)
    n_out_of_spec = int(np.sum(out_of_bounds))
    print(
        f"[drift={drift_level*100:>5.1f}%] MAE={fault_mae:.4f} RMSE={fault_rmse:.4f} "
        f"R2={fault_r2:.4f}  out-of-spec points={n_out_of_spec}/{sample_count} "
        f"-> Figure_A/B/C_..._{tag}.png"
    )

print(f"\nDone. Generated Figure A/B/C for {len(SWEEP_DRIFT_LEVELS)} drift levels in '{RESULT_DIR.absolute()}'")
