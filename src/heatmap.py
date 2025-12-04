# ============================================================================
# CORRELATION HEATMAP - INPUT-OUTPUT RELATIONSHIPS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load processed data
df_processed = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/df_processed.csv')

# Define feature groups
dark_green_inputs = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
outputs = ['MF_IA', 'NOx_EO', 'SOC']

# Set font sizes
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

print("="*80)
print("CORRELATION HEATMAP ANALYSIS")
print("="*80)

# ===== FIGURE 1: Dark Green Inputs vs Outputs Correlation =====
print("\nGenerating correlation heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

# Calculate correlation matrix
relevant_cols = dark_green_inputs + outputs
corr_matrix = df_processed[relevant_cols].corr()

# Create heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    center=0,
    square=True,
    linewidths=2,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    ax=ax,
    vmin=-1,
    vmax=1,
    annot_kws={'size': 11, 'weight': 'bold'}
)

ax.set_title('Correlation Matrix: Dark Green Inputs → Outputs\n(Virtual Sensor Feature Analysis)', 
             fontsize=17, fontweight='bold', pad=20)

# Rotate labels for readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/correlation_heatmap.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: correlation_heatmap.png")

# ===== FIGURE 2: Focused Correlation - Inputs to Outputs Only =====
print("\nGenerating focused correlation plot...")

fig, ax = plt.subplots(figsize=(8, 10))

# Extract correlation of inputs to outputs only
corr_focused = corr_matrix.loc[dark_green_inputs, outputs]

sns.heatmap(
    corr_focused,
    annot=True,
    fmt='.3f',
    cmap='coolwarm',
    center=0,
    linewidths=2,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    ax=ax,
    vmin=-1,
    vmax=1,
    annot_kws={'size': 12, 'weight': 'bold'}
)

ax.set_title('Input-Output Correlation Strength\n(Feature Importance for Virtual Sensor)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Outputs (Targets)', fontweight='bold', fontsize=14)
ax.set_ylabel('Inputs (Dark Green Features)', fontweight='bold', fontsize=14)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/correlation_input_output.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: correlation_input_output.png")

# Print top correlations
print("\n" + "="*80)
print("TOP CORRELATIONS")
print("="*80)

for output in outputs:
    print(f"\n{output}:")
    corr_values = corr_focused[output].sort_values(ascending=False, key=abs)
    for feature, corr_val in corr_values.items():
        print(f"  {feature:15s}: {corr_val:7.4f}")

print("\n" + "="*80)
print("✓ CORRELATION ANALYSIS COMPLETED")
print("="*80)
