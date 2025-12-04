# ============================================================================
# COMPLETE REVISED CODE: ANN MODEL WITH KEY + POTENTIAL INPUTS
# FULL VISUALIZATION PIPELINE (WITH CONFUSION MATRIX & ANALYSIS)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, accuracy_score
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA LOADING & MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("ANN MODEL - KEY + POTENTIAL INPUTS (WITH COMPREHENSIVE VISUALIZATIONS)")
print("="*80)

# --- File Path Configuration (UPDATE THIS PATH AS NEEDED) ---
processed_filepath = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/df_processed.csv'  
save_directory = os.path.dirname(processed_filepath) 

# --- Load data ---
try:
    df_processed = pd.read_csv(processed_filepath)
    print(f"✓ Successfully loaded data from: {processed_filepath}")
except FileNotFoundError:
    print(f"⚠ Warning: File not found. Using alternative path...")
    # Try common locations
    for alt_path in ['/tmp/df_processed.csv', './df_processed.csv', '../df_processed.csv']:
        if os.path.exists(alt_path):
            df_processed = pd.read_csv(alt_path)
            processed_filepath = alt_path
            save_directory = os.path.dirname(processed_filepath) or '/tmp/'
            print(f"✓ Loaded from: {alt_path}")
            break
    else:
        print(f"✗ Error: File not found in any location.")
        raise SystemExit

# Define features
key_inputs = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
potential_inputs = ['MF_FUEL', 'p_21', 'p_31', 'T_21', 'T_31', 'q_MI', 'Rail Pressure']
outputs = ['MF_IA', 'NOx_EO', 'SOC']

# Verify all potential inputs exist
available_potential = [col for col in potential_inputs if col in df_processed.columns]
print(f"\nAvailable potential inputs: {available_potential}")

# Use available potential inputs
all_inputs = key_inputs + available_potential

# Prepare and normalize data
X = df_processed[all_inputs].values
scaler_X = StandardScaler()
X_norm = scaler_X.fit_transform(X)

# Train-test split
X_train, X_test = train_test_split(X_norm, test_size=0.2, random_state=42)

print(f"\nDataset Configuration:")
print(f"  Total samples: {X_norm.shape[0]}")
print(f"  Key inputs: {len(key_inputs)}")
print(f"  Potential inputs: {len(available_potential)}")
print(f"  TOTAL inputs: {len(all_inputs)}")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}\n")

# Store ANN results
ann_results_all = {}

# Train ANN for each output
print("Training ANN models for each output...\n")
for output in outputs:
    print(f"{output}:")
    print("-" * 60)
    
    y = df_processed[output].values
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    _, _, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
    
    # Build ANN with larger hidden layers (more inputs)
    # 13 → 128 → 64 → 32 → 1 (Information Funnel, larger for more features)
    ann = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        verbose=0
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
    
    # Store all results
    ann_results_all[output] = {
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
        'features': all_inputs
    }
    
    print(f"  Train: MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R²={r2_train:.4f}")
    print(f"  Test:  MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R²={r2_test:.4f}")

print("\n" + "="*80)
print("✓ ANN MODEL TRAINING COMPLETE")
print("="*80)

# Save model results
with open(os.path.join(save_directory, 'ann_all_inputs_results.pkl'), 'wb') as f:
    pickle.dump(ann_results_all, f)
print(f"✓ Results saved to: {os.path.join(save_directory, 'ann_all_inputs_results.pkl')}")

# ============================================================================
# PART 2: VISUALIZATION PIPELINE
# ============================================================================

print("\n" + "="*80)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# ============================================================================
# VIZ 1: PREDICTED vs ACTUAL (CLEAR CIRCLES - OVERLAPPING VISUALIZATION)
# ============================================================================
print("\n[1/6] Generating Predicted vs Actual (Clear Circles for Overlap Detection)...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ANN Model (All Inputs): Predicted vs Actual - Clear Circle Overlap View (Test Set)', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, output in enumerate(outputs):
    ax = axes[idx]
    
    y_actual = ann_results_all[output]['y_test']
    y_pred = ann_results_all[output]['y_pred_test']
    mae = ann_results_all[output]['mae_test']
    r2 = ann_results_all[output]['r2_test']
    
    # GREEN circles for Actual values (larger, more transparent)
    ax.scatter(y_actual, y_actual, alpha=0.5, s=120, edgecolors='darkgreen', 
               linewidth=2, color='lightgreen', label='Actual (Reference)', marker='o', zorder=3)
    
    # BLUE circles for Predicted values (smaller, less transparent to see overlap)
    ax.scatter(y_actual, y_pred, alpha=0.8, s=80, edgecolors='darkblue', 
               linewidth=2, color='lightblue', label='Predicted', marker='o', zorder=4)
    
    # Perfect fit line (diagonal)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Fit', zorder=2)
    
    ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_title(f'{output}\nMAE={mae:.4f}, R²={r2:.4f}\n(Green=Actual, Blue=Predicted)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_all_1_predicted_vs_actual_circles.jpg'), dpi=300, bbox_inches='tight')
print("✓ Saved: viz_all_1_predicted_vs_actual_circles.jpg")
plt.close()

# ============================================================================
# VIZ 2: CONFUSION MATRIX - PREDICTION ACCURACY CLASSIFICATION
# ============================================================================
print("[2/6] Generating Confusion Matrix (Binary Classification)...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('ANN Model (All Inputs): Confusion Matrix - Prediction Accuracy Classification', 
             fontsize=16, fontweight='bold', y=1.02)

for idx, output in enumerate(outputs):
    ax = axes[idx]
    
    y_actual = ann_results_all[output]['y_test']
    y_pred = ann_results_all[output]['y_pred_test']
    mae = ann_results_all[output]['mae_test']
    
    # Binary classification: split at median
    median_actual = np.median(y_actual)
    
    # Create binary labels: 1 if above median (High), 0 if below (Low)
    actual_binary = (y_actual > median_actual).astype(int)
    
    # For predictions: "correct" if within MAE, "incorrect" otherwise
    pred_error = np.abs(y_actual - y_pred)
    pred_correct = (pred_error <= mae).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(actual_binary, pred_correct, labels=[0, 1])
    
    # Compute accuracy
    accuracy = accuracy_score(actual_binary, pred_correct)
    
    # Extract TP, TN, FP, FN
    TN = cm[0, 0]  # Actual Low, Predicted Correct
    FP = cm[0, 1]  # Actual Low, Predicted Incorrect
    FN = cm[1, 0]  # Actual High, Predicted Incorrect
    TP = cm[1, 1]  # Actual High, Predicted Correct
    
    # Create confusion matrix for display
    cm_display = np.array([[TP, FP], [FN, TN]])
    
    # Plot heatmap
    im = ax.imshow(cm_display, cmap='Blues', aspect='auto', vmin=0, vmax=max(cm_display.flatten()))
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, str(cm_display[i, j]), ha="center", va="center",
                          color="white" if cm_display[i, j] > cm_display.max()/2 else "black",
                          fontsize=16, fontweight='bold')
    
    # Set labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Correct Pred', 'Incorrect Pred'], fontsize=10)
    ax.set_yticklabels(['Actual High', 'Actual Low'], fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    
    ax.set_title(f'{output}\nMAE={mae:.4f}, Accuracy={accuracy*100:.1f}%', 
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_all_2_confusion_matrix.jpg'), dpi=300, bbox_inches='tight')
print("✓ Saved: viz_all_2_confusion_matrix.jpg")
plt.close()

# ============================================================================
# VIZ 3: MAE COMPARISON (TRAINING vs TEST)
# ============================================================================
print("[3/6] Generating MAE Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

outputs_list = list(ann_results_all.keys())
mae_train = [ann_results_all[out]['mae_train'] for out in outputs_list]
mae_test = [ann_results_all[out]['mae_test'] for out in outputs_list]

x = np.arange(len(outputs_list))
width = 0.35

bars1 = ax.bar(x - width/2, mae_train, width, label='Training MAE', color='skyblue', edgecolor='navy', linewidth=1.5)
bars2 = ax.bar(x + width/2, mae_test, width, label='Test MAE', color='salmon', edgecolor='darkred', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Output Variable', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax.set_title('ANN Model (All Inputs): MAE Comparison (Training vs Test)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(outputs_list, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_all_3_mae_comparison.jpg'), dpi=300, bbox_inches='tight')
print("✓ Saved: viz_all_3_mae_comparison.jpg")
plt.close()

# ============================================================================
# VIZ 4: R² SCORE COMPARISON (TRAINING vs TEST)
# ============================================================================
print("[4/6] Generating R² Score Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

r2_train = [ann_results_all[out]['r2_train'] for out in outputs_list]
r2_test = [ann_results_all[out]['r2_test'] for out in outputs_list]

x = np.arange(len(outputs_list))
width = 0.35

bars1 = ax.bar(x - width/2, r2_train, width, label='Training R²', color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
bars2 = ax.bar(x + width/2, r2_test, width, label='Test R²', color='gold', edgecolor='orange', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Output Variable', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('ANN Model (All Inputs): R² Score Comparison (Training vs Test)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(outputs_list, fontsize=11)
ax.set_ylim([0.97, 1.01])
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_all_4_r2_comparison.jpg'), dpi=300, bbox_inches='tight')
print("✓ Saved: viz_all_4_r2_comparison.jpg")
plt.close()

# ============================================================================
# VIZ 5: RESIDUALS DISTRIBUTION (ERROR ANALYSIS)
# ============================================================================
print("[5/6] Generating Residuals Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ANN Model (All Inputs): Prediction Error Distribution', fontsize=16, fontweight='bold', y=1.02)

for idx, output in enumerate(outputs):
    ax = axes[idx]
    
    y_actual = ann_results_all[output]['y_test']
    y_pred = ann_results_all[output]['y_pred_test']
    mae = ann_results_all[output]['mae_test']
    
    residuals = y_actual - y_pred
    
    # Create bar plot
    x_indices = np.arange(len(residuals))
    colors = ['red' if r >= 0 else 'green' for r in residuals]
    
    ax.bar(x_indices, residuals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line
    ax.axhline(y=0, color='blue', linestyle='-', linewidth=2.5, label='Perfect Prediction', zorder=5)
    
    # ±MAE threshold
    ax.axhline(y=mae, color='orange', linestyle='--', linewidth=2, label=f'+MAE ({mae:.4f})', alpha=0.8)
    ax.axhline(y=-mae, color='orange', linestyle='--', linewidth=2, label=f'-MAE (-{mae:.4f})', alpha=0.8)
    
    ax.set_xlabel('Test Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax.set_title(f'{output}\nMAE={mae:.4f}, R²={ann_results_all[output]["r2_test"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_all_5_residuals_distribution.jpg'), dpi=300, bbox_inches='tight')
print("✓ Saved: viz_all_5_residuals_distribution.jpg")
plt.close()

# ============================================================================
# VIZ 6: PERFORMANCE METRICS HEATMAP
# ============================================================================
print("[6/6] Generating Performance Metrics Heatmap...")

# Prepare data for heatmap
metrics_data = []
for output in outputs_list:
    metrics_data.append([
        ann_results_all[output]['mae_train'],
        ann_results_all[output]['mae_test'],
        ann_results_all[output]['rmse_train'],
        ann_results_all[output]['rmse_test'],
        ann_results_all[output]['r2_train'],
        ann_results_all[output]['r2_test']
    ])

metrics_df = pd.DataFrame(
    metrics_data,
    columns=['MAE Train', 'MAE Test', 'RMSE Train', 'RMSE Test', 'R² Train', 'R² Test'],
    index=outputs_list
)

fig, ax = plt.subplots(figsize=(12, 5))

# Create heatmap
im = ax.imshow(metrics_df.values, cmap='RdYlGn_r', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(metrics_df.columns)))
ax.set_yticks(np.arange(len(metrics_df.index)))
ax.set_xticklabels(metrics_df.columns, fontsize=11)
ax.set_yticklabels(metrics_df.index, fontsize=11)

# Add text annotations
for i in range(len(metrics_df.index)):
    for j in range(len(metrics_df.columns)):
        text = ax.text(j, i, f'{metrics_df.values[i, j]:.4f}',
                      ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('ANN Model (All Inputs): Performance Metrics Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Output Variables', fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Metric Value', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'viz_all_6_metrics_heatmap.jpg'), dpi=300, bbox_inches='tight')
print("✓ Saved: viz_all_6_metrics_heatmap.jpg")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*80)

print(f"\n📊 Generated Visualization Files:")
print(f"  1. viz_all_1_predicted_vs_actual_circles.jpg    [Clear circle overlap view]")
print(f"  2. viz_all_2_confusion_matrix.jpg               [Binary classification accuracy]")
print(f"  3. viz_all_3_mae_comparison.jpg                 [Training vs Test MAE]")
print(f"  4. viz_all_4_r2_comparison.jpg                  [Training vs Test R²]")
print(f"  5. viz_all_5_residuals_distribution.jpg         [Error distribution analysis]")
print(f"  6. viz_all_6_metrics_heatmap.jpg                [Overall performance metrics]")

print(f"\n📁 All files saved to: {save_directory}")

print(f"\n📈 Model Performance Summary (All Inputs):")
print(f"\n{'Output':<12} {'MAE Train':<15} {'MAE Test':<15} {'R² Test':<12}")
print("-" * 60)
for output in outputs_list:
    mae_tr = ann_results_all[output]['mae_train']
    mae_te = ann_results_all[output]['mae_test']
    r2_te = ann_results_all[output]['r2_test']
    print(f"{output:<12} {mae_tr:<15.4f} {mae_te:<15.4f} {r2_te:<12.4f}")

print("\n✓ Model training and visualization pipeline complete!")
print("="*80)
