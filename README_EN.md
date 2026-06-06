# Virtual Sensor - Mass Air Flow (MAF) Prediction

A comprehensive machine learning framework for predicting engine Mass Air Flow (MAF) using multiple regression models and neural networks. This project includes data preprocessing, model development, hyperparameter tuning, and extensive visualization analysis.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Project Structure](#project-structure)
5. [Scripts Overview](#scripts-overview)
6. [Quick Start Guide](#quick-start-guide)
7. [Model Performance](#model-performance)
8. [Output Files](#output-files)
9. [Troubleshooting](#troubleshooting)
10. [Future Improvements](#future-improvements)

---

## Project Overview

This virtual sensor project develops predictive models for three critical engine outputs:
- **MF_IA** (Mass Air Flow - Intake)
- **NOx_EO** (Nitrogen Oxides - Engine Out)
- **SOC** (State of Charge)

### Key Features

✅ **Multi-Model Comparison**
- Linear Regression (OLS)
- Support Vector Regression (SVR)
- Random Forest Regression
- Artificial Neural Networks (ANN)

✅ **Advanced Analysis**
- Nonlinear regression fitting (exponential, polynomial)
- Feature sensitivity analysis (permutation importance)
- Multi-dimensional data visualization

✅ **Data Quality Control**
- Outlier detection (IQR method)
- Missing value reporting
- Data validation and normalization

✅ **Scalable Architecture**
- Centralized utilities module (utils.py)
- Relative path management for cross-platform compatibility
- Modular design for easy extension

---

## Environment Setup

### Prerequisites
- Python 3.7 or later
- pip or conda package manager

### Installation

```bash
# Navigate to project directory
cd virtual_sensor

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.19 | Numerical computation |
| pandas | ≥1.1 | Data manipulation |
| scikit-learn | ≥0.24 | Machine learning models |
| tensorflow/keras | ≥2.4 | Neural networks |
| matplotlib | ≥3.3 | Data visualization |
| seaborn | ≥0.11 | Statistical visualization |

---

## Data Preparation

### Input Data Requirements

**File Location**: Virtual sensor root directory (same level as `src/` folder)

**Filename**: `Data_vaibhav_colored.csv`

**Data Format**:
- CSV format with comma separators
- Columns: 44 total (6 key inputs + 7 potential inputs + 3 outputs + metadata)
- Target variable: `MF_IA`, `NOx_EO`, `SOC`
- Feature variables: All input columns

### Data Processing Workflow

```
Raw Data (Data_vaibhav_colored.csv)
           ↓
    [clean_prepare_data.py]  ← RUN FIRST
           ↓
  Processed Data (df_processed.csv)
           ↓
    [All analysis scripts]
```

### Feature Groups

**Key Inputs (6 features)**:
- Torque
- p_0 (Ambient Pressure)
- T_IM (Intake Manifold Temperature)
- P_IM (Intake Manifold Pressure)
- EGR_Rate (Exhaust Gas Recirculation Rate)
- ECU_VTG_Pos (Variable Geometry Turbocharger Position)

**Potential Inputs (7 features)**:
- MF_FUEL (Mass Flow of Fuel)
- p_21, p_31 (Additional pressure measurements)
- T_21, T_31 (Additional temperature measurements)
- q_MI (Manifold Charge)
- Rail Pressure

**Output Variables (3 targets)**:
- MF_IA
- NOx_EO
- SOC

---

## Project Structure

```
virtual_sensor/
├── README_EN.md                    # English documentation
├── README.md                       # Chinese documentation
├── Data_vaibhav_colored.csv        # Input data (original)
├── df_processed.csv                # Processed data (generated)
├── src/
│   ├── utils.py                    # Shared utilities module (220 lines)
│   ├── clean_prepare_data.py       # Data cleaning & preprocessing
│   ├── OLS_linear_reg.py           # OLS linear regression baseline
│   ├── simple_reg.py               # Simple regression baseline
│   ├── SVR.py                      # Support Vector Regression
│   ├── randomForest.py             # Random Forest model
│   ├── linear_reg.py               # Nonlinear regression analysis
│   ├── distri_box.py               # Distribution & box plots
│   ├── heatmap.py                  # Correlation heatmap
│   ├── pairplot_visual.py          # Pair plot visualization
│   ├── ann_only_key+visual.py      # ANN with key inputs only
│   ├── ann_key_potential+visual.py # ANN with all inputs
│   └── ann_key_sensitivity.py      # Feature sensitivity analysis
└── test_results/                   # Test outputs & visualizations
    ├── figures_rf/
    ├── figures_svr/
    └── visualizations/
```

---

## Scripts Overview

### 1. Data Preparation

#### `clean_prepare_data.py` - Essential First Step

**Purpose**: Data cleaning, feature engineering, and quality validation

**Input**: `Data_vaibhav_colored.csv`

**Output**: `df_processed.csv` (saved in root directory)

**Processing Steps**:
```python
1. Remove metadata row (units)
2. Convert all columns to numeric
3. Identify feature groups (dark/light green inputs, outputs)
4. Remove rows with NaN values
5. Detect outliers using IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
6. Generate data quality report
```

**Quality Metrics Computed**:
- Number of samples per group
- Data type consistency
- Null value counts
- Outlier statistics (count and percentage)

**Run Command**:
```bash
cd src
python3 clean_prepare_data.py
```

**Expected Output**:
```
✓ Data loaded successfully
================================================================================
DATA PROCESSING PIPELINE
================================================================================
1. Removed units row
2. Converted all columns to numeric
3. Feature groups identified...
4. Data quality validation (before cleaning)...
5. Removed rows with NaN values...
6. Data quality validation (after cleaning)...

================================================================================
STATISTICAL SUMMARY
================================================================================
[Detailed statistics for each feature group]

✓ Data processing complete!
```

---

### 2. Baseline Models

#### `OLS_linear_reg.py` - Ordinary Least Squares Linear Regression

**Model**: Ordinary Least Squares (OLS)

**Characteristics**:
- Linear relationship assumption
- Fast training
- Interpretable coefficients
- Complete statistical output

**Features Used**: 6 key inputs only

**Training Configuration**:
- Train/Test Split: 80/20
- Standardization: Yes (StandardScaler)

**Output Files**:
- `viz_predicted_vs_actual_clear_circles_ols.png`
- `viz_mae_comparison_pct_ols.png`

**Run Command**:
```bash
python3 OLS_linear_reg.py
```

**Performance Metrics**:
```
MF_IA: MAE=49.90, RMSE=63.12, R²=0.9729
NOx_EO: MAE=89.91, RMSE=130.60, R²=0.8780
SOC: MAE=1.30, RMSE=1.63, R²=0.6795
```

---

#### `simple_reg.py` - Simple Regression Baseline

**Purpose**: Minimal baseline for rapid prototyping

**Characteristics**:
- Simplified implementation
- Reference model for comparison

**Run Command**:
```bash
python3 simple_reg.py
```

---

#### `SVR.py` - Support Vector Regression with RBF Kernel

**Model**: Support Vector Regression (RBF kernel)

**Characteristics**:
- Non-linear regression
- Multi-output support
- Robust to outliers
- Slower training than linear models

**Hyperparameters**:
- Kernel: RBF (Radial Basis Function)
- C, gamma: Default scikit-learn values

**Output Files**:
- `viz_predicted_vs_actual_clear_circles_svr.png`
- `viz_mae_comparison_pct_svr.png`

**Run Command**:
```bash
python3 SVR.py
```

**Performance Metrics**:
```
MF_IA: MAE=162.04, RMSE=227.66, R²=0.6469
NOx_EO: MAE=156.67, RMSE=325.19, R²=0.2438
SOC: MAE=0.22, RMSE=0.37, R²=0.9835
```

---

#### `randomForest.py` - Random Forest Regression

**Model**: Random Forest (300 estimators)

**Characteristics**:
- Ensemble learning method
- Multi-output support
- Feature importance ranking
- Excellent generalization

**Hyperparameters**:
- n_estimators: 300
- random_state: 42
- max_features: auto

**Output Files**:
- Feature importance rankings
- `viz_predicted_vs_actual_clear_circles_rf.png`
- `viz_mae_comparison_pct_rf.png`

**Run Command**:
```bash
python3 randomForest.py
```

**Performance Metrics** (Best among traditional ML models):
```
MF_IA: MAE=40.43, RMSE=53.52, R²=0.9805
NOx_EO: MAE=37.49, RMSE=59.95, R²=0.9743
SOC: MAE=0.40, RMSE=0.59, R²=0.9584
```

---

### 3. Advanced Models

#### `linear_reg.py` - Nonlinear Regression Analysis

**Purpose**: Analyze specific feature relationships with multiple fitting strategies

**Supported Modes**:
- **Exponential Fitting**: `y = a * exp(b*x)`
- **Polynomial Fitting**: `y = a*x² + b*x + c` (degree 2)
- **Linear Fitting**: `y = a*x + b`

**Feature Pairs Analyzed**:
1. MF_IA: Torque→MF_IA (exponential), P_IM→MF_IA (linear)
2. NOx_EO: EGR_Rate→NOx_EO (exponential), T_IM→NOx_EO (polynomial)
3. SOC: Torque→SOC (polynomial), P_IM→SOC (polynomial)

**Output Files**:
- `regression_MF_IA.png` - Multi-panel visualization
- `regression_NOx_EO.png` - Multi-panel visualization
- `regression_SOC.png` - Multi-panel visualization

**Run Command**:
```bash
python3 linear_reg.py
```

**Key Findings**:
- EGR_Rate → NOx_EO shows strongest exponential relationship (R²=0.9269)
- P_IM → MF_IA shows strong linear relationship (R²=0.7788)

---

### 4. Neural Network Models

#### `ann_only_key+visual.py` - ANN with Key Inputs

**Model Architecture**:
- Input Layer: 6 key features (standardized)
- Hidden Layers: 2-3 fully connected layers
- Output Layer: 3 targets (MF_IA, NOx_EO, SOC)
- Activation: ReLU
- Optimizer: Adam

**Training Configuration**:
- Epochs: 1000
- Batch Size: 32
- Validation Split: 20%
- Early Stopping: Yes (patience=50)

**Features Used**:
- Torque, p_0, T_IM, P_IM, EGR_Rate, ECU_VTG_Pos

**Output Visualization (6 components)**:
1. Predicted vs Actual (scatter with clear circles)
2. Confusion Matrix Analysis (text-based performance breakdown)
3. MAE Comparison (normalized percentage)
4. R² Score Comparison (all outputs)
5. Residuals Distribution (error analysis)
6. Performance Metrics Heatmap

**Run Command**:
```bash
python3 ann_only_key+visual.py
```

**Performance Metrics**:
```
MF_IA: MAE=22.57, RMSE=28.54, R²=0.9945
NOx_EO: MAE=29.77, RMSE=39.00, R²=0.9891
SOC: MAE=0.27, RMSE=0.36, R²=0.9841
Average R²: 0.9892 ⭐ Excellent
```

---

#### `ann_key_potential+visual.py` - ANN with All Inputs

**Model Architecture**: Same as above, but with additional input features

**Features Used (13 total)**:
- All 6 key inputs (Torque, p_0, T_IM, P_IM, EGR_Rate, ECU_VTG_Pos)
- Plus 7 potential inputs (MF_FUEL, p_21, p_31, T_21, T_31, q_MI, Rail Pressure)

**Why More Inputs?**
- Capture additional physical relationships
- Improve prediction accuracy
- Enable more comprehensive sensitivity analysis

**Output Visualization**: Same 6 components as key inputs model

**Run Command**:
```bash
python3 ann_key_potential+visual.py
```

**Performance Metrics**:
```
MF_IA: MAE=16.12, RMSE=20.99, R²=0.9970 (+0.0025)
NOx_EO: MAE=25.14, RMSE=35.08, R²=0.9912 (+0.0021)
SOC: MAE=0.29, RMSE=0.40, R²=0.9802 (-0.0039)
Average R²: 0.9895 ⭐⭐ Outstanding
```

**Conclusion**: Using more features further improves prediction accuracy, especially for MF_IA.

---

#### `ann_key_sensitivity.py` - Feature Importance Analysis

**Method**: Permutation Importance

**Process**:
1. Train ANN model on all 6 key features
2. For each feature:
   - Randomly shuffle values
   - Measure increase in MAE
   - ΔMAE indicates feature importance

**Output Files**:
- `ann_key_sensitivity_MF_IA.png` - Bar chart of feature importance
- `ann_key_sensitivity_NOx_EO.png` - Bar chart of feature importance
- `ann_key_sensitivity_SOC.png` - Bar chart of feature importance
- `ann_key_permutation_sensitivity_standalone.csv` - Numerical results

**Run Command**:
```bash
python3 ann_key_sensitivity.py
```

**Key Findings** (Top 3 Most Important Features):

MF_IA Prediction:
1. P_IM (Intake Manifold Pressure) - ΔMAE=181.87
2. Torque - ΔMAE=116.64
3. p_0 (Ambient Pressure) - ΔMAE=108.14

NOx_EO Prediction:
1. EGR_Rate (Exhaust Gas Recirculation) - ΔMAE=203.33 ⭐ Critical
2. T_IM (Intake Manifold Temperature) - ΔMAE=111.53
3. Torque - ΔMAE=58.92

SOC Prediction:
1. Torque - ΔMAE=1.88
2. P_IM - ΔMAE=0.76
3. T_IM - ΔMAE=0.74

---

### 5. Data Visualization Scripts

#### `distri_box.py` - Distribution and Box Plots

**Visualizations**:
1. Input feature distributions (histogram)
2. Output feature distributions (histogram)
3. Input feature box plots
4. Output feature box plots

**Use Cases**:
- Identify data distribution shapes
- Detect outliers visually
- Compare feature scales
- Assess normality assumptions

**Run Command**:
```bash
python3 distri_box.py
```

**Output Files**:
- `distribution_inputs.png`
- `distribution_outputs.png`
- `boxplot_inputs.png`
- `boxplot_outputs.png`

---

#### `heatmap.py` - Correlation Analysis

**Visualizations**:
1. Full correlation matrix heatmap (all 13 features)
2. Input-output correlation heatmap (focused view)

**Information Provided**:
- Pearson correlation coefficients
- Color intensity indicates strength
- Identifies collinearity issues

**Run Command**:
```bash
python3 heatmap.py
```

**Output Files**:
- `correlation_heatmap.png`
- `correlation_input_output.png`

**Key Correlations Found**:
- MF_IA most correlated with: P_IM (r=0.8825)
- NOx_EO most correlated with: EGR_Rate (r=-0.8856)
- SOC most correlated with: Torque (r=-0.7072)

---

#### `pairplot_visual.py` - Multivariate Relationships

**Purpose**: Visualize all pairwise relationships between features and outputs

**Visualizations**:
- Scatter plots for each feature vs output combination
- Diagonal shows univariate distributions
- Separate pairplot for each target variable

**Run Command**:
```bash
python3 pairplot_visual.py
```

**Output Files**:
- `pairplot_MF_IA.png` - MF_IA vs all inputs (matrix)
- `pairplot_NOx_EO.png` - NOx_EO vs all inputs (matrix)
- `pairplot_SOC.png` - SOC vs all inputs (matrix)

**Note**: Memory intensive for large datasets. May require optimization for >10K samples.

---

## Quick Start Guide

### Minimal Example (2 minutes)

```bash
# Navigate to project
cd virtual_sensor

# Run data preparation
cd src
python3 clean_prepare_data.py

# Run fastest model
python3 OLS_linear_reg.py

# Run best model
python3 ann_only_key+visual.py
```

### Complete Analysis (15 minutes)

```bash
# All 12 scripts in sequence
cd virtual_sensor/src

# Data preparation (required first)
python3 clean_prepare_data.py

# Baseline models
python3 OLS_linear_reg.py
python3 simple_reg.py
python3 SVR.py
python3 randomForest.py

# Advanced analysis
python3 linear_reg.py

# Neural networks
python3 ann_only_key+visual.py
python3 ann_key_potential+visual.py
python3 ann_key_sensitivity.py

# Visualizations
python3 distri_box.py
python3 heatmap.py
python3 pairplot_visual.py
```

---

## Model Performance

### Ranking by Average R² Score

| Rank | Model | MF_IA | NOx_EO | SOC | Average |
|------|-------|-------|--------|-----|---------|
| 🥇 1 | ANN (All Inputs) | 0.9970 | 0.9912 | 0.9802 | **0.9895** |
| 🥈 2 | ANN (Key Inputs) | 0.9945 | 0.9891 | 0.9841 | **0.9892** |
| 🥉 3 | Random Forest | 0.9805 | 0.9743 | 0.9584 | **0.9711** |
| 4 | OLS Linear | 0.9729 | 0.8780 | 0.6795 | **0.8435** |
| 5 | SVR (RBF) | 0.6469 | 0.2438 | 0.9835 | **0.6247** |

### Model Selection Guide

**For Maximum Accuracy**: Use ANN with all inputs
```python
# Expected R² > 0.98 for all outputs
python3 ann_key_potential+visual.py
```

**For Interpretability**: Use Random Forest or OLS
```python
# Feature importance and coefficients available
python3 randomForest.py  # Better accuracy
python3 OLS_linear_reg.py  # More interpretable
```

**For Speed**: Use OLS Linear Regression
```python
# Trains in seconds, good enough for many applications
python3 OLS_linear_reg.py
```

**For Feature Analysis**: Use ANN + Sensitivity
```python
# Understand which inputs matter most
python3 ann_key_sensitivity.py
```

---

## Output Files

### Generated Outputs Directory: `test_results/`

All test outputs, visualizations, and intermediate results are stored here:

```
test_results/
├── figures_rf/                          # Random Forest outputs
├── figures_svr/                         # SVR outputs
├── visualizations/                      # ANN and other visualizations
│   ├── ann_all_inputs_results.pkl
│   ├── viz_all_1_predicted_vs_actual_circles.jpg
│   ├── viz_all_2_confusion_matrix.jpg
│   ├── viz_all_3_mae_comparison.jpg
│   ├── viz_all_4_r2_comparison.jpg
│   ├── viz_all_5_residuals_distribution.jpg
│   └── viz_all_6_metrics_heatmap.jpg
├── ann_key_sensitivity_*.png            # Sensitivity analysis plots
├── boxplot_*.png                        # Box plot visualizations
├── correlation_*.png                    # Correlation heatmaps
├── distribution_*.png                   # Distribution histograms
├── pairplot_*.png                       # Pairwise relationships
└── regression_*.png                     # Nonlinear regression fits
```

### Key Metrics Output

**Console Output** from each script contains:
- Training and test metrics (MAE, RMSE, R²)
- Feature importance rankings (if applicable)
- Data quality reports
- File save confirmations

### Saved Models and Data

| File | Purpose | Generated By |
|------|---------|--------------|
| `df_processed.csv` | Cleaned & normalized data | clean_prepare_data.py |
| `ann_key_permutation_sensitivity_standalone.csv` | Feature importance scores | ann_key_sensitivity.py |

---

## Utilities Module (utils.py)

The `utils.py` module provides shared functions used by all analysis scripts:

### Path Management Functions

#### `get_data_path(filename)`
Returns absolute path to data file in parent directory.

```python
from utils import get_data_path
df = pd.read_csv(get_data_path('df_processed.csv'))
```

#### `get_save_directory(subfolder=None)`
Returns absolute path to save directory, creates if needed.

```python
from utils import get_save_directory
save_dir = get_save_directory('visualizations')
# Returns: ../visualizations/ (with auto-creation)
```

### Analysis Functions

#### `regression_metrics(y_true, y_pred)`
Computes R², MAE, RMSE for predictions.

```python
from utils import regression_metrics
metrics = regression_metrics(y_test, y_pred)
# Returns: {'R2': 0.95, 'MAE': 10.5, 'RMSE': 15.2}
```

#### `plot_pred_vs_actual_clear_circles(y_true, y_pred, title, save_path)`
Creates prediction vs actual scatter plot.

#### `plot_mae_comparison_percent(models_dict, save_path)`
Creates MAE comparison bar chart across multiple models.

---

## Troubleshooting

### ❌ Error: FileNotFoundError

**Problem**: Script cannot find `df_processed.csv`

**Solution**:
1. Run `clean_prepare_data.py` first
2. Ensure `Data_vaibhav_colored.csv` is in virtual_sensor root directory
3. Run scripts from `src/` directory:
   ```bash
   cd virtual_sensor/src
   python3 OLS_linear_reg.py
   ```

### ❌ Error: ModuleNotFoundError

**Problem**: Missing package (numpy, pandas, etc.)

**Solution**:
```bash
pip install -r requirements.txt
# or individually:
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

### ⚠️ Warning: Memory Usage High

**Problem**: Running `pairplot_visual.py` uses excessive memory

**Solution**:
1. Reduce sample size in script before running
2. Run on machine with more RAM
3. Process subsets of data sequentially

### ❌ Error: Path Issues

**Problem**: Scripts fail with relative path errors

**Solution**:
- Always run scripts from `src/` directory
- Ensure data file is in parent directory (same level as `src/`)
- utils.py handles path resolution automatically

---

## Future Improvements

### Short Term (High Priority)
- [ ] Rename ANN files to remove `+` character (Unix convention)
- [ ] Implement missing confusion matrix visualization in ANN script
- [ ] Add comprehensive unit tests (pytest)
- [ ] Create requirements.txt with locked versions

### Medium Term
- [ ] Consolidate duplicate ANN code
- [ ] Add CLI interface for batch processing
- [ ] Implement cross-validation for robust evaluation
- [ ] Add hyperparameter tuning (Bayesian optimization)

### Long Term
- [ ] Refactor as installable Python package
- [ ] Add CI/CD pipeline (GitHub Actions)
- [ ] Develop web dashboard for results visualization
- [ ] Create production inference pipeline

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{virtual_sensor_maf,
  title={Virtual Sensor for Mass Air Flow Prediction},
  author={Dai, Siqi},
  year={2024},
  howpublished={\url{https://github.com/Sherry1247/URS-Applied-ML-Research}}
}
```

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact

For questions or feedback:
- 📧 Email: sdai66@wisc.edu
- 🐙 GitHub: [Sherry1247/URS-Applied-ML-Research](https://github.com/Sherry1247/URS-Applied-ML-Research)

---

**Last Updated**: June 2024
**Maintainer**: Siqi Dai
**Status**: Active Development
