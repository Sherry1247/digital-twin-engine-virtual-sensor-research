# ============================================================================
# NONLINEAR REGRESSION FOR SELECTED VARIABLE PAIRS
# ============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

from utils import get_data_path


def fit_and_plot(ax, x, y, x_label, y_label, title, mode='linear', color='C0', poly_degree=2):
    """
    Fit and plot regression models (linear, exponential, polynomial).
    
    Args:
        ax: Matplotlib axis
        x: Independent variable
        y: Dependent variable
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        mode: 'linear', 'exp', or 'poly'
        color: Plot color
        poly_degree: Degree of polynomial (for poly mode)
        
    Returns:
        float: R² score
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    X_fit = x.reshape(-1, 1)
    
    if mode == 'exp':
        # For exponential: shift y to positive if needed
        y_min = y.min()
        if y_min <= 0:
            y_shifted = y - y_min + 1
        else:
            y_shifted = y
        
        y_trans = np.log(y_shifted)
        lr = LinearRegression()
        lr.fit(X_fit, y_trans)
        y_pred_trans = lr.predict(X_fit)
        y_pred = np.exp(y_pred_trans)
        
        if y_min <= 0:
            y_pred = y_pred + y_min - 1
            
    elif mode == 'poly':
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(X_fit)
        lr = LinearRegression()
        lr.fit(X_poly, y)
        y_pred = lr.predict(X_poly)
        
    else:  # linear
        lr = LinearRegression()
        lr.fit(X_fit, y)
        y_pred = lr.predict(X_fit)
    
    r2 = r2_score(y, y_pred)
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=30, edgecolors='black', linewidth=0.3, color=color)
    
    # Fitted line
    if mode == 'poly':
        X_smooth = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
        X_smooth_poly = poly.transform(X_smooth)
        y_smooth = lr.predict(X_smooth_poly)
        ax.plot(X_smooth, y_smooth, 'r-', linewidth=2.5,
                label=f'Best fit (poly-{poly_degree}, R²={r2:.3f})')
    else:
        sort_idx = np.argsort(x)
        ax.plot(x[sort_idx], y_pred[sort_idx], 'r-', linewidth=2.5,
                label=f'Best fit ({mode}, R²={r2:.3f})')
    
    ax.set_xlabel(x_label, fontweight='bold', fontsize=11)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    return r2


def main():
    """Main execution for nonlinear regression analysis."""
    
    # Load data
    try:
        data_path = get_data_path('df_processed.csv')
        df = pd.read_csv(data_path)
        print(f"✓ Data loaded: {data_path}")
    except FileNotFoundError:
        print(f"✗ Error: Processed data not found at {data_path}")
        print("   Please run clean_prepare_data.py first.")
        raise SystemExit(1)
    
    # Helper function for fitting and plotting
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    
    print("="*80)
    print("NONLINEAR REGRESSION FOR SELECTED VARIABLE PAIRS")
    print("="*80)
    
    # ============================================================================
    # FIGURE 1: MF_IA vs Torque (exponential) & P_IM (linear)
    # ============================================================================
    print("\n[1/3] Analyzing MF_IA relationships...")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('MF_IA Relationships', fontsize=16, fontweight='bold')
    
    r2_torque_mf = fit_and_plot(
        ax=axes[0],
        x=df['Torque'],
        y=df['MF_IA'],
        x_label='Torque (N·m)',
        y_label='MF_IA (kg/h)',
        title='Torque vs MF_IA',
        mode='exp'
    )
    
    r2_pim_mf = fit_and_plot(
        ax=axes[1],
        x=df['P_IM'],
        y=df['MF_IA'],
        x_label='P_IM (bar)',
        y_label='MF_IA (kg/h)',
        title='P_IM vs MF_IA',
        mode='linear'
    )
    
    plt.tight_layout()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(save_dir)
    plt.savefig(os.path.join(parent_dir, 'regression_MF_IA.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: regression_MF_IA.png")
    print(f"  R² Torque→MF_IA (exp):    {r2_torque_mf:.4f}")
    print(f"  R² P_IM→MF_IA (linear):   {r2_pim_mf:.4f}")
    
    # ============================================================================
    # FIGURE 2: NOx_EO vs EGR_Rate, T_IM, ECU_VTG_Pos
    # ============================================================================
    print("\n[2/3] Analyzing NOx_EO relationships...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('NOx_EO Relationships', fontsize=16, fontweight='bold')
    
    r2_egr_nox = fit_and_plot(
        ax=axes[0],
        x=df['EGR_Rate'],
        y=df['NOx_EO'],
        x_label='EGR_Rate (%)',
        y_label='NOx_EO (ppm)',
        title='EGR_Rate vs NOx_EO',
        mode='exp'
    )
    
    r2_tim_nox = fit_and_plot(
        ax=axes[1],
        x=df['T_IM'],
        y=df['NOx_EO'],
        x_label='T_IM (°C)',
        y_label='NOx_EO (ppm)',
        title='T_IM vs NOx_EO',
        mode='poly',
        poly_degree=2
    )
    
    r2_vtg_nox = fit_and_plot(
        ax=axes[2],
        x=df['ECU_VTG_Pos'],
        y=df['NOx_EO'],
        x_label='ECU_VTG_Pos (%)',
        y_label='NOx_EO (ppm)',
        title='ECU_VTG_Pos vs NOx_EO',
        mode='poly',
        poly_degree=2
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'regression_NOx_EO.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: regression_NOx_EO.png")
    print(f"  R² EGR_Rate→NOx_EO (exp):         {r2_egr_nox:.4f}")
    print(f"  R² T_IM→NOx_EO (poly-2):         {r2_tim_nox:.4f}")
    print(f"  R² ECU_VTG_Pos→NOx_EO (poly-2):  {r2_vtg_nox:.4f}")
    
    # ============================================================================
    # FIGURE 3: SOC vs Torque & P_IM (POLYNOMIAL DEGREE 2 - U-shaped)
    # ============================================================================
    print("\n[3/3] Analyzing SOC relationships...")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('SOC Relationships', fontsize=16, fontweight='bold')
    
    r2_torque_soc = fit_and_plot(
        ax=axes[0],
        x=df['Torque'],
        y=df['SOC'],
        x_label='Torque (N·m)',
        y_label='SOC (deg)',
        title='Torque vs SOC',
        mode='poly',
        poly_degree=2
    )
    
    r2_pim_soc = fit_and_plot(
        ax=axes[1],
        x=df['P_IM'],
        y=df['SOC'],
        x_label='P_IM (bar)',
        y_label='SOC (deg)',
        title='P_IM vs SOC',
        mode='poly',
        poly_degree=2
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'regression_SOC.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: regression_SOC.png")
    print(f"  R² Torque→SOC (poly-2):   {r2_torque_soc:.4f}")
    print(f"  R² P_IM→SOC (poly-2):     {r2_pim_soc:.4f}")
    
    print("\n" + "="*80)
    print("✓ ALL REGRESSIONS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
