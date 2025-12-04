# ============================================================================
# Step 2: PAIRPLOT - INPUT-OUTPUT RELATIONSHIPS
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

# Set style for better readability
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

print("="*80)
print("GENERATING PAIRPLOTS")
print("="*80)

# ===== PAIRPLOT 1: Dark Green Inputs vs MF_IA =====
print("\nCreating Pairplot 1: Inputs vs MF_IA...")
data_mf_ia = df_processed[dark_green_inputs + ['MF_IA']].copy()

fig = sns.pairplot(
    data_mf_ia,
    diag_kind='hist',
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
    diag_kws={'bins': 25, 'edgecolor': 'black', 'alpha': 0.7},
    corner=False,
    height=2.5
)

fig.fig.suptitle('Pairplot: Dark Green Inputs → MF_IA (Intake Air Mass Flow)', 
                 fontsize=18, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/pairplot_MF_IA.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: pairplot_MF_IA.png")

# ===== PAIRPLOT 2: Dark Green Inputs vs NOx_EO =====
print("\nCreating Pairplot 2: Inputs vs NOx_EO...")
data_nox = df_processed[dark_green_inputs + ['NOx_EO']].copy()

fig = sns.pairplot(
    data_nox,
    diag_kind='hist',
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
    diag_kws={'bins': 25, 'edgecolor': 'black', 'alpha': 0.7},
    corner=False,
    height=2.5
)

fig.fig.suptitle('Pairplot: Dark Green Inputs → NOx_EO (NOx Emissions)', 
                 fontsize=18, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/pairplot_NOx_EO.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: pairplot_NOx_EO.png")

# ===== PAIRPLOT 3: Dark Green Inputs vs SOC =====
print("\nCreating Pairplot 3: Inputs vs SOC...")
data_soc = df_processed[dark_green_inputs + ['SOC']].copy()

fig = sns.pairplot(
    data_soc,
    diag_kind='hist',
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'black', 'linewidth': 0.5},
    diag_kws={'bins': 25, 'edgecolor': 'black', 'alpha': 0.7},
    corner=False,
    height=2.5
)

fig.fig.suptitle('Pairplot: Dark Green Inputs → SOC (Start of Combustion)', 
                 fontsize=18, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/pairplot_SOC.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: pairplot_SOC.png")

print("\n" + "="*80)
print("✓ ALL PAIRPLOTS COMPLETED")
print("="*80)
