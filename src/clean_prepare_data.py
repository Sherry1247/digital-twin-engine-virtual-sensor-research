# ============================================================================
# Step 1: DATA PROCESSING & SAVING
# ============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from utils import get_data_path, get_save_directory


def validate_data_quality(df, columns):
    """
    Validate data quality and report issues.
    
    Args:
        df (DataFrame): Data to validate
        columns (list): Columns to validate
        
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'null_counts': df[columns].isnull().sum().to_dict(),
        'dtypes': df[columns].dtypes.to_dict(),
        'outliers': {}
    }
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            validation_results['outliers'][col] = {
                'count': outliers,
                'percentage': (outliers / len(df)) * 100
            }
    
    return validation_results


# Load data
try:
    input_path = get_data_path('Data_vaibhav_colored.csv')
    df = pd.read_csv(input_path)
    print(f"✓ Data loaded: {input_path}")
except FileNotFoundError:
    print(f"✗ Error: Data file not found at {input_path}")
    raise SystemExit(1)

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
print(f"   Dark Green Inputs ({len(dark_green_corrected)}): {dark_green_corrected}")
print(f"   Light Green Inputs ({len(light_green_corrected)}): {light_green_corrected}")
print(f"   Outputs ({len(outputs_corrected)}): {outputs_corrected}")

# Validate before cleaning
all_cols = dark_green_corrected + light_green_corrected + outputs_corrected
print(f"\n4. Data quality validation (before cleaning):")
validation_before = validate_data_quality(df, all_cols)
print(f"   Total rows: {validation_before['total_rows']}")
for col, nulls in validation_before['null_counts'].items():
    if nulls > 0:
        print(f"   {col}: {nulls} null values")

# Remove rows with NaN values
initial_rows = len(df)
df_clean = df.dropna(subset=all_cols)
removed_rows = initial_rows - len(df_clean)

print(f"\n5. Removed rows with NaN values")
print(f"   Initial rows: {initial_rows}")
print(f"   Removed rows: {removed_rows}")
print(f"   Final shape: {df_clean.shape}")
print(f"   Samples: {df_clean.shape[0]}, Features: {df_clean.shape[1]}")

# Validate after cleaning
print(f"\n6. Data quality validation (after cleaning):")
validation_after = validate_data_quality(df_clean, all_cols)
print(f"   Total rows: {validation_after['total_rows']}")
print(f"   Outliers detected:")
for col, outlier_info in validation_after['outliers'].items():
    if outlier_info['count'] > 0:
        print(f"     {col}: {outlier_info['count']} ({outlier_info['percentage']:.2f}%)")

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
save_dir = get_save_directory()
output_path = os.path.join(save_dir, 'df_processed.csv')
df_processed.to_csv(output_path, index=False)

print("\n" + "="*80)
print(f"✓ Processed data saved:")
print(f"  {output_path}")
print(f"  Rows: {len(df_processed)}, Columns: {len(df_processed.columns)}")
print("="*80)
