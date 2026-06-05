# ================================
# MULTI-OUTPUT RANDOM FOREST REGRESSION
# ================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    get_data_path, get_save_directory, regression_metrics,
    plot_pred_vs_actual_clear_circles, plot_mae_comparison_percent
)


def main():
    """Main execution function for Random Forest regression analysis."""
    
    # Load processed data
    try:
        data_path = get_data_path('df_processed.csv')
        df = pd.read_csv(data_path)
        print(f"✓ Loaded data: {data_path}")
    except FileNotFoundError:
        print(f"✗ Error: Processed data not found at {data_path}")
        print("   Please run clean_prepare_data.py first.")
        raise SystemExit(1)
    
    save_directory = get_save_directory('figures_rf')
    
    print("="*80)
    print("MULTI-OUTPUT RANDOM FOREST REGRESSION")
    print("="*80)
    
    feature_cols = ['Torque', 'p_0', 'T_IM', 'P_IM', 'EGR_Rate', 'ECU_VTG_Pos']
    target_cols = ['MF_IA', 'NOx_EO', 'SOC']
    
    # Verify all columns exist
    missing_features = [col for col in feature_cols + target_cols if col not in df.columns]
    if missing_features:
        print(f"✗ Error: Missing columns: {missing_features}")
        raise SystemExit(1)
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print(f"\nDataset: {len(df)} samples, {len(feature_cols)} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    rf.fit(X_train, y_train)
    print("✓ Training complete")
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    mae_train, rmse_train, r2_train = regression_metrics(y_train, y_train_pred)
    mae_test, rmse_test, r2_test = regression_metrics(y_test, y_test_pred)
    
    print("\nRandom Forest – Test Metrics")
    print("-" * 80)
    for i, t in enumerate(target_cols):
        print(f"{t}: MAE={mae_test[i]:.4f}, RMSE={rmse_test[i]:.4f}, R²={r2_test[i]:.4f}")
    
    # Prepare results dictionary for plotting
    y_test_dict = {}
    for i, out in enumerate(target_cols):
        y_test_dict[out] = {
            'y_actual': y_test[:, i],
            'y_pred': y_test_pred[:, i],
            'mae': mae_test[i],
            'r2': r2_test[i]
        }
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_pred_vs_actual_clear_circles(
        y_test_dict, target_cols,
        model_name='Random Forest Model',
        save_directory=save_directory,
        file_name_suffix='rf'
    )
    
    plot_mae_comparison_percent(
        mae_train_abs=mae_train,
        mae_test_abs=mae_test,
        y_test=y_test,
        outputs=target_cols,
        model_name='Random Forest Model',
        save_directory=save_directory,
        file_name_suffix='rf'
    )
    
    print("\n✓ Random Forest analysis complete!")


if __name__ == "__main__":
    main()
