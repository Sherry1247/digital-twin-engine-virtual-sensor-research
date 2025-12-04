# ============================================================================
# NONLINEAR REGRESSION FOR SELECTED PAIRS (WITH R²)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# --- File Path Configuration ---
processed_filepath = '/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/virtual_sensor/df_processed.csv'
save_directory = os.path.dirname(processed_filepath)

# --- Load data ---
try:
    df_processed = pd.read_csv(processed_filepath)
    print(f"Successfully loaded data from: {processed_filepath}")
except FileNotFoundError:
    print(f"Error: Processed data file not found at {processed_filepath}.")
    raise SystemExit

# Helper to compute R² for non‑linear transforms
def fit_and_plot(ax, x, y, x_label, y_label, title, mode='linear', color='C0'):
    """
    mode:
      'linear'  : y = a * x + b
      'exp'     : y = a * exp(b*x)     -> fit ln(y) on x
      'logx'    : y = a * log(x) + b  -> fit y on ln(x)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # mask out non‑positive values when logs are used
    if mode == 'exp':
        mask = y > 0
        x, y = x[mask], y[mask]
        y_trans = np.log(y)
        X_fit = x.reshape(-1, 1)
    elif mode == 'logx':
        mask = x > 0
        x, y = x[mask], y[mask]
        x_trans = np.log(x)
        y_trans = y
        X_fit = x_trans.reshape(-1, 1)
    else:  # linear
        y_trans = y
        X_fit = x.reshape(-1, 1)

    # fit linear model in transformed space
    lr = LinearRegression()
    lr.fit(X_fit, y_trans)
    y_pred_trans = lr.predict(X_fit)

    # back‑transform prediction for plotting and R² in original space
    if mode == 'exp':
        y_pred = np.exp(y_pred_trans)
    else:
        y_pred = y_pred_trans

    r2 = r2_score(y, y_pred)

    # scatter
    ax.scatter(x, y, alpha=0.6, s=45, edgecolors='black', linewidth=0.5, color=color)

    # sort for smooth line
    sort_idx = np.argsort(x)
    ax.plot(x[sort_idx], y_pred[sort_idx], 'r-', linewidth=3,
            label=f'Best fit ({mode}, R²={r2:.3f})')

    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)

    return r2


plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

print("="*80)
print("NONLINEAR REGRESSION FOR SELECTED VARIABLE PAIRS")
print("="*80)

# ----------------------------------------------------------------------------
# FIGURE 1: MF_IA vs Torque (exponential) & P_IM (linear)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MF_IA Relationships', fontsize=18, fontweight='bold')

# Torque vs MF_IA  (exponential)
r2_torque_mf = fit_and_plot(
    ax=axes[0],
    x=df_processed['Torque'],
    y=df_processed['MF_IA'],
    x_label='Torque (N·m)',
    y_label='MF_IA (kg/h)',
    title='Torque vs MF_IA (exponential fit)',
    mode='exp'
)

# P_IM vs MF_IA (linear)
r2_pim_mf = fit_and_plot(
    ax=axes[1],
    x=df_processed['P_IM'],
    y=df_processed['MF_IA'],
    x_label='P_IM (bar)',
    y_label='MF_IA (kg/h)',
    title='P_IM vs MF_IA (linear fit)',
    mode='linear'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path_1 = os.path.join(save_directory, 'reg_MF_IA_torque_exp_PIM_lin.png')
plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {save_path_1}")
print(f"  R² Torque→MF_IA (exp): {r2_torque_mf:.4f}")
print(f"  R² P_IM→MF_IA (lin):   {r2_pim_mf:.4f}")

# ----------------------------------------------------------------------------
# FIGURE 2: NOx_EO vs T_IM, EGR_Rate, ECU_VTG_Pos (log / exponential‑type)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('NOx_EO Relationships', fontsize=18, fontweight='bold')

# T_IM vs NOx_EO  (use log(x): NOx = a*log(T_IM)+b)
r2_tim_nox = fit_and_plot(
    ax=axes[0],
    x=df_processed['T_IM'],
    y=df_processed['NOx_EO'],
    x_label='T_IM (°C)',
    y_label='NOx_EO (ppm)',
    title='T_IM vs NOx_EO (log‑x fit)',
    mode='logx'
)

# EGR_Rate vs NOx_EO (exponential decay: fit exp on EGR_Rate)
r2_egr_nox = fit_and_plot(
    ax=axes[1],
    x=df_processed['EGR_Rate'],
    y=df_processed['NOx_EO'],
    x_label='EGR_Rate (%)',
    y_label='NOx_EO (ppm)',
    title='EGR_Rate vs NOx_EO (exp fit)',
    mode='exp'
)

# ECU_VTG_Pos vs NOx_EO (log‑x or exp; here choose log‑x)
r2_vtg_nox = fit_and_plot(
    ax=axes[2],
    x=df_processed['ECU_VTG_Pos'],
    y=df_processed['NOx_EO'],
    x_label='ECU_VTG_Pos (%)',
    y_label='NOx_EO (ppm)',
    title='ECU_VTG_Pos vs NOx_EO (log‑x fit)',
    mode='logx'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path_2 = os.path.join(save_directory, 'reg_NOx_special_pairs.png')
plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {save_path_2}")
print(f"  R² T_IM→NOx_EO (log‑x):     {r2_tim_nox:.4f}")
print(f"  R² EGR_Rate→NOx_EO (exp):   {r2_egr_nox:.4f}")
print(f"  R² ECU_VTG_Pos→NOx_EO(log): {r2_vtg_nox:.4f}")

# ----------------------------------------------------------------------------
# FIGURE 3: SOC vs Torque and P_IM (exponential)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('SOC Relationships', fontsize=18, fontweight='bold')

# Torque vs SOC (exp)
r2_torque_soc = fit_and_plot(
    ax=axes[0],
    x=df_processed['Torque'],
    y=df_processed['SOC'],
    x_label='Torque (N·m)',
    y_label='SOC (deg)',
    title='Torque vs SOC (exp fit)',
    mode='exp'
)

# P_IM vs SOC (exp)
r2_pim_soc = fit_and_plot(
    ax=axes[1],
    x=df_processed['P_IM'],
    y=df_processed['SOC'],
    x_label='P_IM (bar)',
    y_label='SOC (deg)',
    title='P_IM vs SOC (exp fit)',
    mode='exp'
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path_3 = os.path.join(save_directory, 'reg_SOC_torque_PIM_exp.png')
plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Saved: {save_path_3}")
print(f"  R² Torque→SOC (exp): {r2_torque_soc:.4f}")
print(f"  R² P_IM→SOC (exp):   {r2_pim_soc:.4f}")

print("\n" + "="*80)
print("✓ ALL SELECTED NONLINEAR REGRESSIONS COMPLETED")
print("="*80)
