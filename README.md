# Digital Twin Engine: Virtual Sensor Research

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Active_Research-success)

This repository contains an advanced research framework for evaluating the robustness, sensitivity, and diagnostic capabilities of Artificial Neural Network (ANN)-based **Virtual Sensors** within a Digital Twin engine environment. 

The project focuses on injecting precise physical faults (e.g., Gaussian noise, bias) into hardware sensors (like `P_IM`, `EGR_Rate`) and mapping the degradation path of target virtual sensors (like `MF_IA`). It features an automated pipeline for generating publication-grade diagnostic visualizations and extracting high-dimensional residual features for predictive maintenance.

---

## 🚀 Key Features

*   **Configurable Fault Injection**: Inject single or multi-sensor faults (e.g., Gaussian noise) with adjustable severity levels (e.g., `0.05` for 5% deviation) mapped directly to physical baselines.
*   **Academic-Grade Visualization**: Automated generation of self-explanatory, high-contrast plots including:
    *   **Figure A**: Physical sensor timeseries (Healthy vs. Corrupted).
    *   **Figure B**: Residual diagnostics featuring multi-level semantic severity bands (Safe $\pm 1\sigma$, Warning $1\sigma \sim 2\sigma$, Failure $>2\sigma$) and red anomaly markers.
    *   **Figure C**: Virtual sensor performance degradation path (3-line comparison: Ground Truth vs. Healthy Prediction vs. Faulty Prediction).
*   **Advanced Feature Engineering Pipeline**: Automatically transforms continuous prediction residuals into a 12-dimensional sliding-window feature matrix (including Mean, Std, RMS, Skewness, Kurtosis, FFT/Wavelet energy, Dropout Ratio, etc.) saved as `residual_features.csv` for downstream ML classification.
*   **Batch Experiment Automation**: Execute predefined experiment matrices evaluating multiple sensors, targets, and operating points seamlessly.

---

## 📂 Project Structure

\`\`\`text
digital-twin-engine-virtual-sensor-research/
├── data/                       # Datasets, ANN model pickles (e.g., ann_all_inputs_results.pkl)
├── experiments/                # Executable experiment scripts
│   ├── batch_run.py            # Executes the full matrix of 12 standard experiments
│   ├── plot_results.py         # Standalone script to regenerate visualizations
│   └── run_fault_experiment.py # Runs a highly configurable single fault experiment
├── src/                        # Core engine modules
│   ├── experiment/             # Experiment runners, configuration, and feature evaluation
│   ├── visualization/          # First-principles plotting geometry and style configurations
│   └── ...
├── results/                    # Generated output directory
│   └── fault_experiments/      # Contains CSVs and plots organized by target/sensor/fault
└── README.md
\`\`\`

---

## 💻 Quick Start & Usage

### 1. Run a Single Fault Experiment
To evaluate the sensitivity of the virtual sensor to a specific physical sensor failure, use the `run_fault_experiment.py` script. 

**Example**: Injecting 5% Gaussian noise into the intake manifold pressure sensor (`P_IM`) to observe the degradation of the intake air mass flow (`MF_IA`) prediction:

\`\`\`bash
python experiments/run_fault_experiment.py --target MF_IA --representative MF_IA_High --sensor P_IM --fault gaussian --severity 0.05
\`\`\`

### 2. Execute the Full Experiment Matrix
To run the pre-configured suite of experiments (evaluating multiple sensors like `P_IM` and `EGR_Rate` across different operating points) and generate all datasets and plots automatically:

\`\`\`bash
python experiments/batch_run.py
\`\`\`

### 3. Regenerate Visualizations Only
If you have updated the visualization logic in `src/visualization/figures.py` and want to redraw the A/B/C figures for existing experimental data without rerunning the ANN inference:

\`\`\`bash
python experiments/plot_results.py
\`\`\`

---

## 📊 Output Artifacts

Running an experiment will automatically generate a dedicated folder in `results/fault_experiments/...` containing the following artifacts:

1.  **`metrics.csv`**: Macroscopic evaluation metrics (MAE, RMSE, R²).
2.  **`time_series.csv`**: The raw, tick-by-tick predictions, ground truth, and residual arrays.
3.  **`residual_features.csv`**: A sliding-window feature matrix containing 12 extracted statistical, frequency, and time-frequency domain diagnostic features.
4.  **`figure_a_sensor_fault.png`**: The injected physical disturbance.
5.  **`figure_b_error_residual_severity.png`**: The core diagnostic plot with adaptive severity shading.
6.  **`figure_c_predictions.png`**: The virtual sensor degradation mapping.

---

## 🔬 Research Context & Physical Principles

This engine demonstrates the critical discrepancy in physical sensitivity among different sensors. For example, neural networks exhibit high structural dependence on **`P_IM`** (directly dictating volumetric efficiency and mass flow), meaning slight perturbations cause immediate catastrophic degradation in virtual sensor confidence limits. Conversely, sensors like **`EGR_Rate`** act as secondary correction factors, yielding high system robustness even under fault conditions. The extracted `residual_features` act as unique topological "fingerprints" to classify and isolate these faults in production.
