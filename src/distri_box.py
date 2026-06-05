# ============================================================================
# DISTRIBUTION & BOX PLOTS
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
    """Main execution for distribution and box plot analysis."""
    
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
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    
    print("="*80)
    print("DISTRIBUTION & BOX PLOT ANALYSIS")
    print("="*80)
    
    # ===== FIGURE 1: Distribution of Dark Green Inputs =====
    print("\n[1/4] Generating input distributions...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution of Dark Green Inputs (Primary Sensors)', 
                 fontsize=18, fontweight='bold')
    
    for idx, feature in enumerate(dark_green_inputs):
        ax = axes[idx // 3, idx % 3]
        
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
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path_1 = os.path.join(save_directory, 'distribution_inputs.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: distribution_inputs.png")
    
    # ===== FIGURE 2: Distribution of Outputs =====
    print("[2/4] Generating output distributions...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribution of Outputs (Virtual Sensor Targets)', 
                 fontsize=18, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        
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
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_path_2 = os.path.join(save_directory, 'distribution_outputs.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: distribution_outputs.png")
    
    # ===== FIGURE 3: Box Plots of Inputs =====
    print("[3/4] Generating input box plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Box Plots: Dark Green Inputs (Outlier Detection)', 
                 fontsize=18, fontweight='bold')
    
    for idx, feature in enumerate(dark_green_inputs):
        ax = axes[idx // 3, idx % 3]
        
        bp = ax.boxplot(df_processed[feature].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        ax.set_xticks([])
        ax.set_ylabel(feature, fontweight='bold', fontsize=13)
        ax.set_title(feature, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path_3 = os.path.join(save_directory, 'boxplot_inputs.png')
    plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: boxplot_inputs.png")
    
    # ===== FIGURE 4: Box Plots of Outputs =====
    print("[4/4] Generating output box plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Box Plots: Outputs (Outlier Detection)', 
                 fontsize=18, fontweight='bold')
    
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        
        bp = ax.boxplot(df_processed[output].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor=colors[idx], alpha=0.7),
                        medianprops=dict(color='darkred', linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        ax.set_xticks([])
        ax.set_ylabel(output, fontweight='bold', fontsize=13)
        ax.set_title(output, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_path_4 = os.path.join(save_directory, 'boxplot_outputs.png')
    plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: boxplot_outputs.png")
    
    print("\n" + "="*80)
    print("✓ DISTRIBUTION & BOX PLOT ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
