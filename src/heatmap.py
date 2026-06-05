# ============================================================================
# CORRELATION HEATMAP - INPUT-OUTPUT RELATIONSHIPS
# ============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from utils import get_data_path, get_save_directory


def main():
    """Main execution for correlation heatmap analysis."""
    
    # Load processed data
    try:
        data_path = get_data_path('df_processed.csv')
        df_processed = pd.read_csv(data_path)
        print(f"✓ Loaded data: {data_path}")
    except FileNotFoundError:
        print(f"✗ Error: Processed data not found at {data_path}")
        print("   Please run clean_prepare_data.py first.")
        raise SystemExit(1)
    
    save_directory = get_save_directory()
    
    # Define feature groups
    dark_green_inputs = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
    outputs = ['MF_IA', 'NOx_EO', 'SOC']
    
    # Verify columns exist
    missing_cols = [col for col in dark_green_inputs + outputs if col not in df_processed.columns]
    if missing_cols:
        print(f"✗ Error: Missing columns: {missing_cols}")
        raise SystemExit(1)
    
    # Set font sizes
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    print("="*80)
    print("CORRELATION HEATMAP ANALYSIS")
    print("="*80)
    
    # ===== FIGURE 1: Full Correlation Matrix =====
    print("\n[1/2] Generating full correlation heatmap...")
    
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
    
    ax.set_title('Correlation Matrix: Inputs → Outputs\n(Virtual Sensor Feature Analysis)', 
                 fontsize=17, fontweight='bold', pad=20)
    
    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    plt.tight_layout()
    save_path_1 = os.path.join(save_directory, 'correlation_heatmap.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: correlation_heatmap.png")
    
    # ===== FIGURE 2: Focused Correlation - Inputs to Outputs Only =====
    print("[2/2] Generating focused correlation plot...")
    
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
    save_path_2 = os.path.join(save_directory, 'correlation_input_output.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: correlation_input_output.png")
    
    # Print top correlations
    print("\n" + "="*80)
    print("TOP CORRELATIONS BY OUTPUT")
    print("="*80)
    
    for output in outputs:
        print(f"\n{output}:")
        corr_values = corr_focused[output].sort_values(ascending=False, key=abs)
        for feature, corr_val in corr_values.items():
            print(f"  {feature:15s}: {corr_val:7.4f}")
    
    print("\n" + "="*80)
    print("✓ CORRELATION ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
