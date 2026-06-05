"""
Shared utilities for virtual sensor analysis.

This module provides common functions used across multiple analysis scripts
to reduce code duplication and ensure consistency.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_data_path(filename):
    """
    Get path to data file relative to this script.
    
    Args:
        filename (str): Name of the file in the parent directory
        
    Returns:
        str: Full path to the file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, filename)


def get_save_directory(subfolder=None):
    """
    Get save directory for outputs relative to script location.
    
    Args:
        subfolder (str, optional): Subdirectory name. If None, uses parent directory.
        
    Returns:
        str: Full path to save directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    if subfolder:
        save_dir = os.path.join(parent_dir, subfolder)
    else:
        save_dir = parent_dir
    
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def regression_metrics(y_true, y_pred, multioutput='raw_values'):
    """
    Compute regression metrics (MAE, RMSE, R²).
    
    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values
        multioutput (str): How to handle multi-output. Default 'raw_values' for array output
        
    Returns:
        tuple: (mae, rmse, r2) arrays
    """
    mae = mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput=multioutput))
    
    if y_true.ndim == 1:
        r2 = np.array([r2_score(y_true, y_pred)])
    else:
        r2 = np.array([r2_score(y_true[:, i], y_pred[:, i]) 
                       for i in range(y_true.shape[1])])
    
    return mae, rmse, r2


def plot_pred_vs_actual_clear_circles(y_test_dict, outputs, model_name, 
                                       save_directory, file_name_suffix):
    """
    Create scatter plot comparing predicted vs actual values.
    
    Args:
        y_test_dict (dict): Dictionary with keys as output names and values containing
                           'y_actual', 'y_pred', 'mae', 'r2'
        outputs (list): List of output variable names
        model_name (str): Name of the model for title
        save_directory (str): Directory to save the figure
        file_name_suffix (str): Suffix for the saved filename
    """
    print("\n[VIZ] Generating Predicted vs Actual (Clear Circles for Overlap Detection)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'{model_name}: Predicted vs Actual - Clear Circle Overlap View (Test Set)',
        fontsize=16, fontweight='bold', y=0.99
    )
    
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        
        y_actual = y_test_dict[output]['y_actual']
        y_pred = y_test_dict[output]['y_pred']
        mae = y_test_dict[output]['mae']
        r2 = y_test_dict[output]['r2']
        
        # GREEN circles for Actual values
        ax.scatter(
            y_actual, y_actual,
            alpha=0.5, s=120,
            edgecolors='darkgreen', linewidth=2,
            color='lightgreen',
            label='Actual (Reference)',
            marker='o', zorder=3
        )
        
        # BLUE circles for Predicted values
        ax.scatter(
            y_actual, y_pred,
            alpha=0.8, s=80,
            edgecolors='darkblue', linewidth=2,
            color='lightblue',
            label='Predicted',
            marker='o', zorder=4
        )
        
        # Perfect fit line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            'r--', linewidth=2.5,
            label='Perfect Fit', zorder=2
        )
        
        ax.set_xlabel('Actual Values', fontweight='bold', fontsize=12)
        ax.set_ylabel('Predicted Values', fontweight='bold', fontsize=12)
        ax.set_title(
            f'{output}\nMAE={mae:.4f}, R²={r2:.4f}\n(Green=Actual, Blue=Predicted)',
            fontweight='bold', fontsize=13
        )
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        if idx == 0:
            ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    fname = f'viz_predicted_vs_actual_clear_circles_{file_name_suffix}.png'
    plt.savefig(os.path.join(save_directory, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {fname}")


def plot_mae_comparison_percent(mae_train_abs, mae_test_abs, y_test, outputs,
                                  model_name, save_directory, file_name_suffix):
    """
    Plot MAE as percentage of mean |actual| on the test set.
    
    Args:
        mae_train_abs (array): Training MAE values
        mae_test_abs (array): Test MAE values
        y_test (array): Test target values
        outputs (list): List of output variable names
        model_name (str): Name of the model for title
        save_directory (str): Directory to save the figure
        file_name_suffix (str): Suffix for the saved filename
    """
    print("[VIZ] Generating Normalized MAE Comparison (Percent of |mean actual|)...")
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x_pos = np.arange(len(outputs))
    width = 0.28
    
    # Mean absolute actual (test) for each output
    mean_abs_test = np.array([
        np.mean(np.abs(y_test[:, i])) for i in range(len(outputs))
    ])
    mean_abs_test = np.where(mean_abs_test == 0, 1e-8, mean_abs_test)
    
    # Convert absolute MAE to %
    mae_train_pct = 100.0 * mae_train_abs / mean_abs_test
    mae_test_pct = 100.0 * mae_test_abs / mean_abs_test
    
    bars1 = ax.bar(
        x_pos - width/2, mae_train_pct, width,
        label='Training MAE (%)', alpha=0.8,
        edgecolor='black', color='#3498db'
    )
    bars2 = ax.bar(
        x_pos + width/2, mae_test_pct, width,
        label='Test MAE (%)', alpha=0.8,
        edgecolor='black', color='#e74c3c'
    )
    
    ax.set_xlabel('Output Variable', fontweight='bold', fontsize=12)
    ax.set_ylabel('MAE (% of mean |actual|)', fontweight='bold', fontsize=12)
    ax.set_title(f'{model_name}: Normalized MAE Comparison',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(outputs, fontsize=11)
    
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.3,
                f'{h:.2f}%',
                ha='center', va='bottom',
                fontsize=8.5, fontweight='bold'
            )
    
    plt.tight_layout()
    fname = f'viz_mae_comparison_pct_{file_name_suffix}.png'
    plt.savefig(os.path.join(save_directory, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {fname}")
