"""
Evaluate robustness of a saved regression model to sensor linear drift.

This command-line tool loads a saved model result artifact, injects drift
into a single sensor column, and computes MAE, RMSE, and R² across multiple
failure magnitudes.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.utils import get_data_path, regression_metrics
from fault_injection.linear_drift import inject_linear_drift, DRIFT_LEVELS

MODEL_RESULT_FILES = {
    'ann': ROOT / 'experiments' / 'ann' / 'test_results' / 'visualizations' / 'ann_all_inputs_results.pkl',
    'linear_regression': SRC_ROOT / 'models' / 'lr_baseline_results.pkl',
    'random_forest': SRC_ROOT / 'models' / 'rf_results.pkl',
    'svr': SRC_ROOT / 'models' / 'svr_results.pkl',
}

DEFAULT_FEATURES = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate model robustness to linear sensor drift.'
    )
    parser.add_argument(
        '--model',
        choices=['ann', 'random_forest', 'svr', 'linear_regression'],
        default='ann',
        help='Saved model type to evaluate.',
    )
    parser.add_argument(
        '--target',
        choices=['MF_IA', 'NOx_EO', 'SOC'],
        default='MF_IA',
        help='Target output variable to evaluate.',
    )
    parser.add_argument(
        '--sensor',
        default='P_IM',
        help='Input sensor to inject with drift.',
    )
    parser.add_argument(
        '--mode',
        choices=['relative', 'absolute'],
        default='relative',
        help='Drift mode to apply to the selected sensor.',
    )
    parser.add_argument(
        '--levels',
        nargs='*',
        type=float,
        help='Drift levels to evaluate, e.g. --levels 0.01 0.02 0.05',
    )
    return parser.parse_args()


def get_model_path(model_name: str) -> Path:
    if model_name not in MODEL_RESULT_FILES:
        raise ValueError(f'Unsupported model: {model_name}')
    return MODEL_RESULT_FILES[model_name]


def load_model_results(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f'Saved model result file not found: {path}\n'
            'Generate or place the expected artifact before running this experiment.'
        )
    with path.open('rb') as f:
        return pickle.load(f)


def get_feature_list(entry: dict) -> List[str]:
    features = entry.get('features')
    if features is None:
        return DEFAULT_FEATURES.copy()
    return list(features)


def build_output_paths(
    out_dir: Path,
    model_name: str,
    target: str,
    sensor: str,
    mode: str,
) -> Dict[str, Path]:
    suffix = f'{model_name}_{target}_{sensor}_{mode}'
    return {
        'csv': out_dir / f'linear_drift_metrics_{suffix}.csv',
        'figure': out_dir / f'mae_vs_linear_drift_{suffix}.png',
    }


def validate_sensor(sensor: str, features: List[str]) -> None:
    if sensor not in features:
        raise ValueError(
            f"Sensor '{sensor}' is not available in the model's feature list: {features}"
        )


def compute_predictions(
    model,
    X_mod: np.ndarray,
    scaler_X,
    scaler_y,
) -> np.ndarray:
    if scaler_X is not None:
        X_mod = scaler_X.transform(X_mod)
    y_pred = model.predict(X_mod)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).ravel()
    else:
        y_pred = np.asarray(y_pred).ravel()
    return y_pred


def run_experiment(
    model_name: str,
    target: str,
    sensor: str,
    mode: str,
    levels: Optional[Iterable[float]],
) -> None:
    levels = list(levels) if levels else list(DRIFT_LEVELS)
    if len(levels) == 0:
        levels = list(DRIFT_LEVELS)
    if any(level < 0 for level in levels):
        raise ValueError('Drift levels must be non-negative.')

    model_path = get_model_path(model_name)
    model_results = load_model_results(model_path)

    if target not in model_results:
        raise KeyError(f"Target '{target}' is not available in saved model results.")

    entry = model_results[target]
    features = get_feature_list(entry)
    validate_sensor(sensor, features)

    model = entry['model']
    scaler_X = entry.get('scaler_X')
    scaler_y = entry.get('scaler_y')

    data_path = ROOT / 'data' / 'processed' / 'df_processed.csv'
    if not data_path.exists():
        data_path = Path(get_data_path('df_processed.csv'))

    if not data_path.exists():
        raise FileNotFoundError(
            f'Could not find df_processed.csv in {ROOT / "data" / "processed"} or via utils.get_data_path.'
        )

    df = pd.read_csv(data_path)

    missing = [feat for feat in features if feat not in df.columns]
    if missing:
        raise KeyError(f'Missing features in processed data: {missing}')

    X = df[features].values
    y = df[target].values

    idx = np.arange(X.shape[0])
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    X_test = X[test_idx].copy()
    y_test = y[test_idx]

    sensor_idx = features.index(sensor)
    records = []

    for level in [0.0] + levels:
        X_mod = X_test.copy()
        X_mod[:, sensor_idx] = inject_linear_drift(
            X_mod[:, sensor_idx],
            level,
            mode=mode,
        )
        y_pred = compute_predictions(model, X_mod, scaler_X, scaler_y)
        mae, rmse, r2 = regression_metrics(y_test, y_pred)
        mae_value = float(np.asarray(mae).item())
        rmse_value = float(np.asarray(rmse).item())
        r2_array = np.asarray(r2).ravel()
        records.append({
            'drift_pct': float(level * 100.0),
            'mae': mae_value,
            'rmse': rmse_value,
            'r2': float(r2_array[0]) if r2_array.size > 0 else float('nan'),
        })

    out_dir = ROOT / 'results' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = build_output_paths(out_dir, model_name, target, sensor, mode)

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(paths['csv'], index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics['drift_pct'], df_metrics['mae'], marker='o', linestyle='-')
    plt.xlabel('Drift Magnitude (%)')
    plt.ylabel('MAE')
    plt.title(f'{model_name.upper()} MAE vs {sensor} Linear Drift ({target})')
    plt.grid(True, alpha=0.3)
    plt.savefig(paths['figure'], dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Saved metrics CSV: {paths["csv"]}')
    print(f'✓ Saved MAE plot: {paths["figure"]}')


def main() -> None:
    args = parse_args()
    print(f'Model: {args.model}')
    print(f'Target: {args.target}')
    print(f'Faulty Sensor: {args.sensor}')
    print(f'Fault Mode: {args.mode}')
    levels = args.levels if args.levels else DRIFT_LEVELS
    if len(levels) == 0:
        levels = DRIFT_LEVELS
    print('Drift Levels:', ', '.join(f'{float(l) * 100:.2f}%' for l in levels))
    print()
    run_experiment(
        model_name=args.model,
        target=args.target,
        sensor=args.sensor,
        mode=args.mode,
        levels=levels,
    )


if __name__ == '__main__':
    main()
