# ============================================================================
# SIMPLE LINEAR REGRESSION BASELINE
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    print(f"✗ Error: File not found at {processed_filepath}.")
    raise SystemExit

print("\n" + "="*80)
print("LINEAR REGRESSION BASELINE")
print("="*80)

# Define feature groups
key_inputs = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
outputs = ['MF_IA', 'NOx_EO', 'SOC']

# Prepare data
X = df_processed[key_inputs].values
y_dict = {output: df_processed[output].values for output in outputs}

# Train-test split (80-20)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Store results
lr_results = {}

print(f"\nDataset: {X.shape[0]} samples, {len(key_inputs)} key inputs")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

# Train models for each output
for output in outputs:
    print(f"\n{output}:")
    print("-" * 60)
    
    y = y_dict[output]
    _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    
    # Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    # Store
    lr_results[output] = {
        'model': lr,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }
    
    print(f"  Train: MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R²={r2_train:.4f}")
    print(f"  Test:  MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R²={r2_test:.4f}")

print("\n" + "="*80)
print("✓ LINEAR REGRESSION BASELINE COMPLETE")
print("="*80)

# Save results
import pickle
with open(os.path.join(save_directory, 'lr_baseline_results.pkl'), 'wb') as f:
    pickle.dump(lr_results, f)
print(f"✓ Results saved to lr_baseline_results.pkl")
