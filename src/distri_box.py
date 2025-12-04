# ============================================================================
# DISTRIBUTION & BOX PLOTS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# --- File Path Configuration ---
# Use the correct file path as saved in the data processing step
processed_filepath = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/df_processed.csv'

# Define the directory for saving the plots
save_directory = os.path.dirname(processed_filepath)

try:
    # Load processed data
    df_processed = pd.read_csv(processed_filepath)
    print(f"Successfully loaded data from: {processed_filepath}")
except FileNotFoundError:
    print(f"Error: Processed data file not found at {processed_filepath}. Please ensure data_processing_save.py was run correctly.")
    exit()

# Define feature groups
dark_green_inputs = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
outputs = ['MF_IA', 'NOx_EO', 'SOC']

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14

print("="*80)
print("DISTRIBUTION & BOX PLOT ANALYSIS")
print("="*80)

# ===== FIGURE 1: Distribution of Dark Green Inputs =====
print("\nGenerating input distributions...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# FIX: Removed the 'pad' argument
fig.suptitle('Distribution of Dark Green Inputs (Primary Sensors)', 
             fontsize=18, fontweight='bold')

for idx, feature in enumerate(dark_green_inputs):
    ax = axes[idx // 3, idx % 3]
    
    # Check if feature exists
    if feature not in df_processed.columns:
        print(f"Skipping plot for {feature}: Data missing.")
        ax.set_title(f"Data Missing: {feature}")
        continue

    ax.hist(df_processed[feature], bins=30, color='darkgreen', 
            alpha=0.7, edgecolor='black', linewidth=1.2)
    
    mean_val = df_processed[feature].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_val:.2f}')
    
    ax.set_xlabel(feature, fontweight='bold', fontsize=13)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=13)
    ax.set_title(feature, fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

# FIX: Added rect to tight_layout to handle the title spacing
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
save_path_1 = os.path.join(save_directory, 'distribution_inputs.png')
plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: distribution_inputs.png to {save_path_1}")

# ===== FIGURE 2: Distribution of Outputs =====
print("\nGenerating output distributions...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# FIX: Removed the 'pad' argument
fig.suptitle('Distribution of Outputs (Virtual Sensor Targets)', 
             fontsize=18, fontweight='bold')

colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

for idx, output in enumerate(outputs):
    ax = axes[idx]

    # Check if output exists
    if output not in df_processed.columns:
        print(f"Skipping plot for {output}: Data missing.")
        ax.set_title(f"Data Missing: {output}")
        continue
    
    ax.hist(df_processed[output], bins=30, color=colors[idx], 
            alpha=0.7, edgecolor='black', linewidth=1.2)
    
    mean_val = df_processed[output].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_val:.2f}')
    
    ax.set_xlabel(output, fontweight='bold', fontsize=13)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=13)
    ax.set_title(output, fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

# FIX: Added rect to tight_layout to handle the title spacing
plt.tight_layout(rect=[0, 0, 1, 0.9])
save_path_2 = os.path.join(save_directory, 'distribution_outputs.png')
plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: distribution_outputs.png to {save_path_2}")

# ===== FIGURE 3: Box Plots of Inputs =====
print("\nGenerating input box plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# FIX: Removed the 'pad' argument
fig.suptitle('Box Plots: Dark Green Inputs (Outlier Detection)', 
             fontsize=18, fontweight='bold')

for idx, feature in enumerate(dark_green_inputs):
    ax = axes[idx // 3, idx % 3]

    # Check if feature exists
    if feature not in df_processed.columns:
        print(f"Skipping plot for {feature}: Data missing.")
        ax.set_title(f"Data Missing: {feature}")
        continue
    
    bp = ax.boxplot(df_processed[feature].dropna(), vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Box plots are often better displayed without X ticks/labels for single variable
    ax.set_xticks([]) 
    ax.set_ylabel(feature, fontweight='bold', fontsize=13)
    ax.set_title(feature, fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

# FIX: Added rect to tight_layout to handle the title spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path_3 = os.path.join(save_directory, 'boxplot_inputs.png')
plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: boxplot_inputs.png to {save_path_3}")

# ===== FIGURE 4: Box Plots of Outputs =====
print("\nGenerating output box plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# FIX: Removed the 'pad' argument
fig.suptitle('Box Plots: Outputs (Outlier Detection)', 
             fontsize=18, fontweight='bold')

for idx, output in enumerate(outputs):
    ax = axes[idx]

    # Check if output exists
    if output not in df_processed.columns:
        print(f"Skipping plot for {output}: Data missing.")
        ax.set_title(f"Data Missing: {output}")
        continue
    
    bp = ax.boxplot(df_processed[output].dropna(), vert=True, patch_artist=True,
                     boxprops=dict(facecolor=colors[idx], alpha=0.7),
                     medianprops=dict(color='darkred', linewidth=2.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    ax.set_xticks([])
    ax.set_ylabel(output, fontweight='bold', fontsize=13)
    ax.set_title(output, fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

# FIX: Added rect to tight_layout to handle the title spacing
plt.tight_layout(rect=[0, 0, 1, 0.9])
save_path_4 = os.path.join(save_directory, 'boxplot_outputs.png')
plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: boxplot_outputs.png to {save_path_4}")

print("\n" + "="*80)
print("✓ ALL DISTRIBUTION & BOX PLOTS COMPLETED")
print("="*80)
