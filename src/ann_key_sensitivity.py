# ============================================================================
# ANN KEY INPUTS – PERMUTATION-BASED SENSITIVITY ANALYSIS
# ============================================================================

import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Paths and basic configuration
# --------------------------------------------------
processed_filepath = (
    "/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-"
    "/virtual_sensor/df_processed.csv"
)
save_directory = os.path.dirname(processed_filepath)
os.makedirs(save_directory, exist_ok=True)

key_inputs = ["Torque", "p_0", "T_IM", "P_IM", "EGR_Rate", "ECU_VTG_Pos"]
outputs = ["MF_IA", "NOx_EO", "SOC"]

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
try:
    df_processed = pd.read_csv(processed_filepath)
    print(f"✓ Loaded data: {processed_filepath}")
except FileNotFoundError:
    print("✗ Error: processed CSV not found.")
    raise SystemExit

X_all = df_processed[key_inputs].values

# global X scaler (shared across outputs, as in your ANN script)
scaler_X = StandardScaler()
X_all_norm = scaler_X.fit_transform(X_all)

print("\n" + "=" * 80)
print("ANN MODEL – KEY INPUTS (RETRAIN FOR SENSITIVITY)")
print("=" * 80)

# --------------------------------------------------
# 3. Train one ANN per output (same architecture as original code)
# --------------------------------------------------
ann_results_key = {}

for output in outputs:
    print(f"\nTraining ANN for output: {output}")
    print("-" * 60)

    y = df_processed[output].values
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all_norm, y_norm, test_size=0.2, random_state=42
    )

    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
    )

    ann.fit(X_train, y_train)

    # predictions in normalized space
    y_pred_train_norm = ann.predict(X_train)
    y_pred_test_norm = ann.predict(X_test)

    # back-transform to physical units
    y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_train_orig = scaler_y.inverse_transform(
        y_pred_train_norm.reshape(-1, 1)
    ).flatten()
    y_pred_test_orig = scaler_y.inverse_transform(
        y_pred_test_norm.reshape(-1, 1)
    ).flatten()

    mae_train = mean_absolute_error(y_train_orig, y_pred_train_orig)
    rmse_train = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
    r2_train = r2_score(y_train_orig, y_pred_train_orig)

    mae_test = mean_absolute_error(y_test_orig, y_pred_test_orig)
    rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    r2_test = r2_score(y_test_orig, y_pred_test_orig)

    ann_results_key[output] = {
        "model": ann,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "X_test_norm": X_test,
        "y_test_norm": y_test,
        "y_test_orig": y_test_orig,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test,
    }

    print(f"  Train: MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R²={r2_train:.4f}")
    print(f"  Test:  MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R²={r2_test:.4f}")

print("\n" + "=" * 80)
print("✓ ANN TRAINING COMPLETE – STARTING SENSITIVITY ANALYSIS")
print("=" * 80)

# --------------------------------------------------
# 4. Permutation-based sensitivity analysis
# --------------------------------------------------
def permutation_importance_ann_results(
    ann_results_dict, key_inputs, outputs, n_repeats=20, random_state=42
):
    """
    Compute permutation-based sensitivity for each ANN (one per output).
    For each output:
      - use its own X_test_norm and y_test_orig,
      - permute each input in normalized space,
      - compute ΔMAE in original units.
    Returns DataFrame with Input, Target, Delta_MAE, Rel_Importance.
    """
    rng = np.random.RandomState(random_state)
    rows = []

    for out in outputs:
        res = ann_results_dict[out]
        model = res["model"]
        X_test_norm = res["X_test_norm"]
        y_test_orig = res["y_test_orig"]

        # baseline MAE
        y_pred_base_norm = model.predict(X_test_norm)
        y_pred_base_orig = res["scaler_y"].inverse_transform(
            y_pred_base_norm.reshape(-1, 1)
        ).flatten()
        mae_base = mean_absolute_error(y_test_orig, y_pred_base_orig)

        print(f"\nOutput {out}: baseline MAE (test) = {mae_base:.4f}")

        for j, fname in enumerate(key_inputs):
            delta_mae_reps = []

            for _ in range(n_repeats):
                X_perm = deepcopy(X_test_norm)
                perm_idx = rng.permutation(X_perm.shape[0])
                X_perm[:, j] = X_perm[perm_idx, j]

                y_pred_perm_norm = model.predict(X_perm)
                y_pred_perm_orig = res["scaler_y"].inverse_transform(
                    y_pred_perm_norm.reshape(-1, 1)
                ).flatten()

                mae_perm = mean_absolute_error(y_test_orig, y_pred_perm_orig)
                delta_mae_reps.append(mae_perm - mae_base)

            delta_mae_mean = np.mean(delta_mae_reps)
            rows.append(
                {
                    "Input": fname,
                    "Target": out,
                    "Delta_MAE": delta_mae_mean,
                }
            )
            print(f"  Input {fname:>10s}: ΔMAE = {delta_mae_mean:.4f}")

    df_imp = pd.DataFrame(rows)

    # relative importance per target (0–1)
    df_imp["Rel_Importance"] = df_imp.groupby("Target")["Delta_MAE"].transform(
        lambda v: v / (v.max() + 1e-12)
    )

    return df_imp


df_sens = permutation_importance_ann_results(
    ann_results_dict=ann_results_key,
    key_inputs=key_inputs,
    outputs=outputs,
    n_repeats=20,
    random_state=42,
)

sens_csv_path = os.path.join(
    save_directory, "ann_key_permutation_sensitivity_standalone.csv"
)
df_sens.to_csv(sens_csv_path, index=False)
print(f"\n✓ Sensitivity table saved to: {sens_csv_path}")
print("\nSensitivity table (head):")
print(df_sens.head())

# --------------------------------------------------
# 5. Bar plots per output
# --------------------------------------------------
def plot_sensitivity_bars(df_imp, target_name, save_dir):
    df_t = df_imp[df_imp["Target"] == target_name].copy()
    df_t = df_t.sort_values("Rel_Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        df_t["Input"], df_t["Rel_Importance"],
        color="#3498db", edgecolor="black", alpha=0.85
    )

    ax.set_ylabel("Relative importance (ΔMAE normalized)",
                  fontsize=11, fontweight="bold")
    ax.set_xlabel("Input variable", fontsize=11, fontweight="bold")
    ax.set_title(f"ANN (Key Inputs) Sensitivity – {target_name}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                h + 0.02,
                f"{h:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    fname = f"ann_key_sensitivity_{target_name}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {fname}")

print("\nGenerating sensitivity bar plots...")

for out in outputs:
    plot_sensitivity_bars(df_sens, out, save_directory)

print("\n" + "=" * 80)
print("✓ SENSITIVITY ANALYSIS (STANDALONE) COMPLETE")
print("=" * 80)
