# ============================================================================
# REVISED: ANN KEY INPUTS MODEL WITH IMPROVED VISUALIZATIONS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# --- File Path Configuration ---
processed_filepath = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/df_processed.csv'
save_directory = os.path.dirname(processed_filepath)

# --- Load data ---
try:
    df_processed = pd.read_csv(processed_filepath)
    print(f"✓ Successfully loaded data from: {processed_filepath}")
except FileNotFoundError:
    print(f"✗ Error: File not found.")
    raise SystemExit

print("\n" + "="*80)
print("ANN MODEL - KEY INPUTS ONLY (REVISED WITH CONFUSION MATRIX)")
print("="*80)

# Define features
key_inputs = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
outputs = ['MF_IA', 'NOx_EO', 'SOC']

# Prepare and normalize data
X = df_processed[key_inputs].values
scaler_X = StandardScaler()
X_norm = scaler_X.fit_transform(X)

# Train-test split
X_train, X_test = train_test_split(X_norm, test_size=0.2, random_state=42)

print(f"\nDataset: {X_norm.shape[0]} samples, {len(key_inputs)} key inputs")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

# Store ANN results
ann_results_key = {}

# Train ANN for each output
for output in outputs:
    print(f"\n{output}:")
    print("-" * 60)
    
    y = df_processed[output].values
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    _, _, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
    
    # Build ANN: 6 → 64 → 32 → 16 → 1 (Information Funnel)
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20
    )
    
    ann.fit(X_train, y_train)
    
    # Predictions (normalized)
    y_pred_train_norm = ann.predict(X_train)
    y_pred_test_norm = ann.predict(X_test)
    
    # Inverse transform to original scale
    y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_train_orig = scaler_y.inverse_transform(y_pred_train_norm.reshape(-1, 1)).flatten()
    y_pred_test_orig = scaler_y.inverse_transform(y_pred_test_norm.reshape(-1, 1)).flatten()
    
    # Metrics
    mae_train = mean_absolute_error(y_train_orig, y_pred_train_orig)
    rmse_train = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
    r2_train = r2_score(y_train_orig, y_pred_train_orig)
    
    mae_test = mean_absolute_error(y_test_orig, y_pred_test_orig)
    rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    r2_test = r2_score(y_test_orig, y_pred_test_orig)
    
    # Store
    ann_results_key[output] = {
        'model': ann,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'y_train': y_train_orig,
        'y_test': y_test_orig,
        'y_pred_train': y_pred_train_orig,
        'y_pred_test': y_pred_test_orig,
        'features': key_inputs
    }
    
    print(f"  Train: MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R²={r2_train:.4f}")
    print(f"  Test:  MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R²={r2_test:.4f}")

print("\n" + "="*80)
print("✓ ANN MODEL TRAINING COMPLETE")
print("="*80)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

print("\n" + "="*80)
print("GENERATING REVISED VISUALIZATIONS")
print("="*80)

# ============================================================================
# FIGURE 1: PREDICTED vs ACTUAL (CLEAR CIRCLES - OVERLAPPING VISUALIZATION)
# ============================================================================
print("\n1. Generating Predicted vs Actual (Clear Circles for Overlap Detection)...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ANN Model: Predicted vs Actual - Clear Circle Overlap View (Test Set)', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, output in enumerate(outputs):
    ax = axes[idx]
    
    y_actual = ann_results_key[output]['y_test']
    y_pred = ann_results_key[output]['y_pred_test']
    mae = ann_results_key[output]['mae_test']
    r2 = ann_results_key[output]['r2_test']
    
    # GREEN circles for Actual values (larger, more transparent)
    ax.scatter(y_actual, y_actual, alpha=0.5, s=120, edgecolors='darkgreen', 
               linewidth=2, color='lightgreen', label='Actual (Reference)', marker='o', zorder=3)
    
    # BLUE circles for Predicted values (smaller, less transparent to see overlap)
    ax.scatter(y_actual, y_pred, alpha=0.8, s=80, edgecolors='darkblue', 
               linewidth=2, color='lightblue', label='Predicted', marker='o', zorder=4)
    
    # Perfect prediction line (red dashed)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, 
            label='Perfect Fit', zorder=2)
    
    ax.set_xlabel('Actual Values', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Values', fontweight='bold', fontsize=12)
    ax.set_title(f'{output}\nMAE={mae:.4f}, R²={r2:.4f}\n(Green=Actual, Blue=Predicted)', 
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Equal aspect ratio for better overlap visualization
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_1_predicted_vs_actual_clear_circles.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: viz_1_predicted_vs_actual_clear_circles.png")

# ============================================================================
# FIGURE 2: NORMALIZED MAE COMPARISON (PERCENT OF MEAN ABSOLUTE VALUE)
# ============================================================================
print("\n3. Generating Normalized MAE Comparison (Percent of |mean actual|)...")

fig, ax = plt.subplots(figsize=(12, 6)) 

x_pos = np.arange(len(outputs))
width = 0.28 

mae_train_abs = np.array([ann_results_key[out]['mae_train'] for out in outputs])
mae_test_abs  = np.array([ann_results_key[out]['mae_test']  for out in outputs])

mean_abs_test = np.array([
    np.mean(np.abs(ann_results_key[out]['y_test'])) for out in outputs
])
mean_abs_test = np.where(mean_abs_test == 0, 1e-8, mean_abs_test)

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
ax.set_title('ANN Model (Key Inputs): Normalized MAE Comparison (Train vs Test)',
             fontweight='bold', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(outputs, fontsize=11)

# Legend on right
ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.grid(True, alpha=0.3, axis='y')

# Value labels closer to bars, slightly smaller
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,                 
            f'{height:.2f}%',
            ha='center', va='bottom',
            fontsize=8.5, fontweight='bold'
        )

plt.tight_layout()
plt.savefig(os.path.join(save_directory,
                         'viz_3_mae_comparison_normalized.png'),
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: viz_3_mae_comparison_normalized.png")


# ============================================================================
# FIGURE 3: R² SCORE COMPARISON
# ============================================================================
print("\n4. Generating R² Score Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

r2_train = [ann_results_key[out]['r2_train'] for out in outputs]
r2_test = [ann_results_key[out]['r2_test'] for out in outputs]

bars1 = ax.bar(x_pos - width/2, r2_train, width, label='Training R²', alpha=0.8, 
               edgecolor='black', color='#2ecc71')
bars2 = ax.bar(x_pos + width/2, r2_test, width, label='Test R²', alpha=0.8, 
               edgecolor='black', color='#f39c12')

ax.set_xlabel('Output Variable', fontweight='bold', fontsize=13)
ax.set_ylabel('R² Score', fontweight='bold', fontsize=13)
ax.set_title('ANN Model: R² Score Comparison (Training vs Test)', fontweight='bold', fontsize=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(outputs, fontsize=12)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.05])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_4_r2_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: viz_4_r2_comparison.png")

# ============================================================================
# FIGURE 4: RESIDUALS DISTRIBUTION
# ============================================================================
print("\n5. Generating Residuals Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ANN Model: Prediction Error Distribution', fontsize=16, fontweight='bold', y=1.02)

for idx, output in enumerate(outputs):
    ax = axes[idx]
    
    y_actual = ann_results_key[output]['y_test']
    y_pred = ann_results_key[output]['y_pred_test']
    residuals = y_actual - y_pred
    mae = ann_results_key[output]['mae_test']
    
    colors = ['red' if r > 0 else 'green' for r in residuals]
    ax.bar(range(len(residuals)), residuals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=2.5, label='Perfect Prediction')
    ax.axhline(y=mae, color='orange', linestyle=':', linewidth=2, label=f'+MAE ({mae:.4f})', alpha=0.7)
    ax.axhline(y=-mae, color='orange', linestyle=':', linewidth=2, label=f'-MAE ({mae:.4f})', alpha=0.7)
    
    ax.set_xlabel('Test Sample Index', fontweight='bold', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontweight='bold', fontsize=12)
    ax.set_title(f'{output}', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_5_residuals_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: viz_5_residuals_distribution.png")

# ============================================================================
# FIGURE 5: PERFORMANCE METRICS HEATMAP
# ============================================================================
print("\n6. Generating Performance Metrics Heatmap...")

fig, ax = plt.subplots(figsize=(10, 6))

metrics_data = []
for output in outputs:
    metrics_data.append([
        ann_results_key[output]['mae_train'],
        ann_results_key[output]['mae_test'],
        ann_results_key[output]['rmse_train'],
        ann_results_key[output]['rmse_test'],
        ann_results_key[output]['r2_train'],
        ann_results_key[output]['r2_test']
    ])

metrics_df = pd.DataFrame(metrics_data, 
                          index=outputs,
                          columns=['MAE Train', 'MAE Test', 'RMSE Train', 'RMSE Test', 'R² Train', 'R² Test'])

sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='RdYlGn', center=0.5, 
            cbar_kws={'label': 'Score'}, ax=ax, linewidths=2, linecolor='black')

ax.set_title('ANN Model: Performance Metrics Heatmap', fontweight='bold', fontsize=15, pad=20)
ax.set_ylabel('Output Variables', fontweight='bold', fontsize=12)
ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_6_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: viz_6_metrics_heatmap.png")

# ============================================================================
# COMPREHENSIVE ANALYSIS & STATISTICS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MODEL ANALYSIS & CONFUSION MATRIX INTERPRETATION")
print("="*80)

print("\n" + "-"*80)
print("CONFUSION MATRIX ANALYSIS")
print("-"*80)

for output in outputs:
    print(f"\n{output}:")
    print("=" * 60)
    
    y_actual = ann_results_key[output]['y_test']
    y_pred = ann_results_key[output]['y_pred_test']
    mae = ann_results_key[output]['mae_test']
    r2 = ann_results_key[output]['r2_test']
    
    errors = np.abs(y_actual - y_pred)
    median_actual = np.median(y_actual)
    
    actual_high = y_actual >= median_actual
    actual_low = y_actual < median_actual
    pred_correct = errors < mae
    pred_incorrect = errors >= mae
    
    # Calculate confusion matrix
    TP = np.sum(actual_high & pred_correct)
    TN = np.sum(actual_low & pred_correct)
    FP = np.sum(actual_low & pred_incorrect)
    FN = np.sum(actual_high & pred_incorrect)
    
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total * 100
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Confusion Matrix Breakdown:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │ True Positive (TP):    {TP:3d} (Actual High, Pred Correct)")
    print(f"  │ True Negative (TN):    {TN:3d} (Actual Low, Pred Correct)")
    print(f"  │ False Positive (FP):   {FP:3d} (Actual Low, Pred Incorrect)")
    print(f"  │ False Negative (FN):   {FN:3d} (Actual High, Pred Incorrect)")
    print(f"  └─────────────────────────────────────────┘")
    print(f"\n  Performance Metrics (based on MAE threshold={mae:.4f}):")
    print(f"  ├─ Accuracy:           {accuracy:.2f}% ({TP + TN}/{total} correct)")
    print(f"  ├─ Precision:          {precision:.2f}% (of predicted correct, {TP} were truly correct)")
    print(f"  ├─ Recall/Sensitivity: {recall:.2f}% (detected {TP} of {TP + FN} actual high values)")
    print(f"  └─ F1 Score:           {f1:.2f}% (harmonic mean of precision & recall)")
    
    print(f"\n  Prediction Error Analysis:")
    print(f"  ├─ MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"  ├─ RMSE (Root Mean Squared Error): {ann_results_key[output]['rmse_test']:.4f}")
    print(f"  ├─ R² Score:                       {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"  └─ Sample size:                    {total} test samples")
    
    print(f"\n  Error Distribution:")
    print(f"  ├─ Min Error:  {errors.min():.4f}")
    print(f"  ├─ Max Error:  {errors.max():.4f}")
    print(f"  ├─ Mean Error: {errors.mean():.4f}")
    print(f"  └─ Std Error:  {errors.std():.4f}")
    
    # Percentage within threshold
    within_threshold = np.sum(errors < mae) / len(errors) * 100
    print(f"\n  Accuracy within MAE threshold: {within_threshold:.2f}%")
    print(f"  (i.e., {np.sum(errors < mae)} out of {len(errors)} predictions within ±{mae:.4f})")

print("\n" + "-"*80)
print("PERFORMANCE SUMMARY TABLE")
print("-"*80)

summary_table = []
for output in outputs:
    y_actual = ann_results_key[output]['y_test']
    y_pred = ann_results_key[output]['y_pred_test']
    errors = np.abs(y_actual - y_pred)
    mae = ann_results_key[output]['mae_test']
    within_mae = np.sum(errors < mae) / len(errors) * 100
    
    summary_table.append({
        'Output': output,
        'MAE Train': f"{ann_results_key[output]['mae_train']:.4f}",
        'MAE Test': f"{ann_results_key[output]['mae_test']:.4f}",
        'R² Test': f"{ann_results_key[output]['r2_test']:.4f}",
        'Within MAE %': f"{within_mae:.2f}%"
    })

summary_df = pd.DataFrame(summary_table)
print("\n" + summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(save_directory, 'ann_key_summary_metrics.csv'), index=False)
print(f"\n✓ Summary saved to: ann_key_summary_metrics.csv")

# ============================================================================
# DETAILED STATISTICS
# ============================================================================
print("\n" + "-"*80)
print("DETAILED STATISTICS BY OUTPUT")
print("-"*80)

for output in outputs:
    print(f"\n{output}:")
    y_actual = ann_results_key[output]['y_test']
    y_pred = ann_results_key[output]['y_pred_test']
    residuals = y_actual - y_pred
    abs_errors = np.abs(residuals)
    
    print(f"\n  Actual Values Statistics:")
    print(f"    Min: {y_actual.min():.4f}, Max: {y_actual.max():.4f}")
    print(f"    Mean: {y_actual.mean():.4f}, Median: {np.median(y_actual):.4f}, Std: {y_actual.std():.4f}")
    
    print(f"\n  Predicted Values Statistics:")
    print(f"    Min: {y_pred.min():.4f}, Max: {y_pred.max():.4f}")
    print(f"    Mean: {y_pred.mean():.4f}, Median: {np.median(y_pred):.4f}, Std: {y_pred.std():.4f}")
    
    print(f"\n  Residuals (Actual - Predicted):")
    print(f"    Min: {residuals.min():.4f}, Max: {residuals.max():.4f}")
    print(f"    Mean: {residuals.mean():.4f}, Median: {np.median(residuals):.4f}, Std: {residuals.std():.4f}")
    
    print(f"\n  Absolute Errors:")
    print(f"    Min: {abs_errors.min():.4f}, Max: {abs_errors.max():.4f}")
    print(f"    Mean: {abs_errors.mean():.4f} (MAE), Median: {np.median(abs_errors):.4f}")
    
    # Percentage error
    pct_errors = (abs_errors / np.abs(y_actual)) * 100
    print(f"\n  Percentage Errors:")
    print(f"    Mean: {pct_errors.mean():.2f}%, Median: {np.median(pct_errors):.2f}%")
    print(f"    Min: {pct_errors.min():.2f}%, Max: {pct_errors.max():.2f}%")

print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS & ANALYSIS COMPLETE")
print("="*80)
