# ============================================================================
# Step 1: DATA PROCESSING & SAVING
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/Data_vaibhav_colored.csv')

print("="*80)
print("DATA PROCESSING PIPELINE")
print("="*80)

# Remove the first row (units/descriptions)
df = df[1:].reset_index(drop=True)
print(f"\n1. Removed units row")
print(f"   Current shape: {df.shape}")

# Convert all columns to numeric (except Log Point)
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\n2. Converted all columns to numeric")

# Define feature groups based on your color coding
dark_green_inputs = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
light_green_inputs = ['MF_FUEL', 'p_21', 'p_31', 'T_21', 'T_31', 'q_MI', 'Rail Pressure']
outputs = ['MF_IA', 'NOx_EO', 'SOC']

# Check which columns exist in dataset
available_cols = df.columns.tolist()

# Corrected column names based on actual data
dark_green_corrected = [col for col in dark_green_inputs if col in available_cols]
light_green_corrected = [col for col in light_green_inputs if col in available_cols]
outputs_corrected = [col for col in outputs if col in available_cols]

print(f"\n3. Feature groups identified:")
print(f"   Dark Green Inputs (6): {dark_green_corrected}")
print(f"   Light Green Inputs (7): {light_green_corrected}")
print(f"   Outputs (3): {outputs_corrected}")

# Remove rows with NaN values
df_clean = df.dropna()
print(f"\n4. Removed NaN rows")
print(f"   Final shape: {df_clean.shape}")
print(f"   Samples: {df_clean.shape[0]}, Features: {df_clean.shape[1]}")

# Statistical summary of key variables
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print("\nDark Green Inputs Statistics:")
print(df_clean[dark_green_corrected].describe())
print("\nOutputs Statistics:")
print(df_clean[outputs_corrected].describe())

# Store for later use
df_processed = df_clean.copy()

# Save processed data
output_path = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/df_processed.csv'
df_processed.to_csv(output_path, index=False)

print("\n" + "="*80)
print(f"✓ Processed data saved to:")
print(f"  {output_path}")
print("="*80)
