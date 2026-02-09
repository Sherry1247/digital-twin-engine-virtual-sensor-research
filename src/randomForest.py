# ================================
# MULTI-OUTPUT RANDOM FOREST REGRESSION
# ================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    r2   = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return mae, rmse, np.array(r2)


def plot_pred_vs_actual_clear_circles(
    y_test_dict, outputs, model_name, save_directory, file_name_suffix
):
    print("\n1. Generating Predicted vs Actual (Clear Circles for Overlap Detection)...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name}: Predicted vs Actual - Clear Circle Overlap View (Test Set)',
                 fontsize=16, fontweight='bold', y=0.99)

    for idx, output in enumerate(outputs):
        ax = axes[idx]

        y_actual = y_test_dict[output]['y_actual']
        y_pred   = y_test_dict[output]['y_pred']
        mae      = y_test_dict[output]['mae']
        r2       = y_test_dict[output]['r2']

        ax.scatter(
            y_actual, y_actual,
            alpha=0.5, s=120,
            edgecolors='darkgreen', linewidth=2,
            color='lightgreen',
            label='Actual (Reference)',
            marker='o', zorder=3
        )

        ax.scatter(
            y_actual, y_pred,
            alpha=0.8, s=80,
            edgecolors='darkblue', linewidth=2,
            color='lightblue',
            label='Predicted',
            marker='o', zorder=4
        )

        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            'r--', linewidth=2.5,
            label='Perfect Fit', zorder=2
        )

        ax.set_xlabel('Actual Values', fontweight='bold', fontsize=12)
        ax.set_ylabel('Predicted Values', fontweight='bold', fontsize=12)
        ax.set_title(
            f'{output}\nMAE={mae:.4f}, R²={r2:.4f}\n(Green=Actual, Blue=Predicted)',
            fontweight='bold', fontsize=13
        )
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        if idx == 0:
            ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    fname = f'viz_predicted_vs_actual_clear_circles_{file_name_suffix}.png'
    plt.savefig(os.path.join(save_directory, fname), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {fname}")


def plot_mae_comparison_percent(
    mae_train_abs, mae_test_abs, y_test, outputs,
    model_name, save_directory, file_name_suffix
):
    """
    Plot MAE as percentage of mean |actual| on the test set.
    """
    print("\n2. Generating Normalized MAE Comparison (Percent of |mean actual|)...")

    fig, ax = plt.subplots(figsize=(9, 5))

    x_pos = np.arange(len(outputs))
    width = 0.28

    # Mean absolute actual value per output (test data only)
    mean_abs_test = np.array([
        np.mean(np.abs(y_test[:, i])) for i in range(len(outputs))
    ])
    mean_abs_test = np.where(mean_abs_test == 0, 1e-8, mean_abs_test)

    # Convert absolute MAE to percentage
    mae_train_pct = 100.0 * mae_train_abs / mean_abs_test
    mae_test_pct  = 100.0 * mae_test_abs  / mean_abs_test

    bars1 = ax.bar(
        x_pos - width/2, mae_train_pct, width,
        label='Training MAE (%)', alpha=0.8,
        edgecolor='black', color='#3498db'
    )
    bars2 = ax.bar(
        x_pos + width/2, mae_test_pct, width,
        label='Test MAE (%)', alpha=0.8,
        edgecolor='black', color='#e74c3c'
    )

    ax.set_xlabel('Output Variable', fontweight='bold', fontsize=12)
    ax.set_ylabel('MAE (% of mean |actual|)', fontweight='bold', fontsize=12)
    ax.set_title(f'{model_name}: Normalized MAE Comparison',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(outputs, fontsize=11)

    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3, axis='y')

    # value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.0,
                h + 0.3,
                f'{h:.2f}%',
                ha='center', va='bottom',
                fontsize=8.5, fontweight='bold'
            )

    plt.tight_layout()
    fname = f'viz_mae_comparison_pct_{file_name_suffix}.png'
    plt.savefig(os.path.join(save_directory, fname), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {fname}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    save_directory = "/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/figures_rf"
    os.makedirs(save_directory, exist_ok=True)

    df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/df_processed.csv')

    feature_cols = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
    target_cols  = ['MF_IA', 'NOx_EO', 'SOC']

    X = df[feature_cols].values
    y = df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred  = rf.predict(X_test)

    mae_train, rmse_train, r2_train = regression_metrics(y_train, y_train_pred)
    mae_test,  rmse_test,  r2_test  = regression_metrics(y_test,  y_test_pred)

    print("Random Forest – Test metrics")
    for i, t in enumerate(target_cols):
        print(f"{t}: MAE={mae_test[i]:.4f}, RMSE={rmse_test[i]:.4f}, R²={r2_test[i]:.4f}")

    # dict for scatter plots
    y_test_dict = {}
    for i, out in enumerate(target_cols):
        y_test_dict[out] = {
            'y_actual': y_test[:, i],
            'y_pred'  : y_test_pred[:, i],
            'mae'     : mae_test[i],
            'r2'      : r2_test[i]
        }

    plot_pred_vs_actual_clear_circles(
        y_test_dict, target_cols,
        model_name='Random Forest Model',
        save_directory=save_directory,
        file_name_suffix='rf'
    )

    # MAE percentage plot
    plot_mae_comparison_percent(
        mae_train_abs=mae_train,
        mae_test_abs=mae_test,
        y_test=y_test,
        outputs=target_cols,
        model_name='Random Forest Model',
        save_directory=save_directory,
        file_name_suffix='rf'
    )
