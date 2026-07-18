"""Paper-ready figures for virtual-sensor fault experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "figure.dpi": 300,
        }
    )


def _context_title(
    figure_label: str,
    description: str,
    target: str,
    sensor: str,
    fault_type: str,
    severity: float,
) -> str:
    return (
        f"Figure {figure_label}: {description} | Input Sensor: {sensor} "
        f"({fault_type} @ Severity {severity:g}) -> Target Virtual Sensor: {target}"
    )


def _add_top_padding(axis: plt.Axes, values: list[np.ndarray], padding_fraction: float = 0.25) -> None:
    finite_values = np.concatenate([value[np.isfinite(value)] for value in values if value.size > 0])
    if finite_values.size == 0:
        return
    ymin = float(np.min(finite_values))
    ymax = float(np.max(finite_values))
    value_range = ymax - ymin
    if value_range == 0.0:
        value_range = max(abs(ymax), 1.0) * 0.1
    axis.set_ylim(ymin - 0.05 * value_range, ymax + padding_fraction * value_range)


def save_figure_a(
    time_series: pd.DataFrame,
    target: str,
    sensor: str,
    fault_type: str,
    severity: float,
    output_path: Path,
) -> Path:
    """Figure A: healthy sensor versus faulty sensor."""
    _apply_style()
    fig, axis = plt.subplots(figsize=(16, 5.5))
    sample_index = time_series["sample_index"].to_numpy()
    healthy_sensor = time_series["healthy_sensor"].to_numpy()
    faulty_sensor = time_series["faulty_sensor"].to_numpy()

    axis.plot(sample_index, healthy_sensor, label="Healthy Sensor", linewidth=2.4, color="#1f77b4")
    axis.plot(sample_index, faulty_sensor, label="Faulty Sensor", linewidth=2.0, color="#d62728", alpha=0.9)
    axis.set_xlabel("Sample Index")
    axis.set_ylabel(sensor)
    axis.set_title(
        _context_title("A", "Healthy Sensor vs. Faulty Sensor", target, sensor, fault_type, severity),
        fontweight="bold",
        pad=16,
    )
    _add_top_padding(axis, [healthy_sensor, faulty_sensor])
    axis.legend(loc="upper right", frameon=True, framealpha=0.92)
    axis.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_figure_b(
    time_series: pd.DataFrame,
    target: str,
    sensor: str,
    fault_type: str,
    severity: float,
    output_path: Path,
) -> Path:
    """Figure B: residual diagnostics with sigma-based severity bands."""
    _apply_style()
    sample_index = time_series["sample_index"].to_numpy()
    residual = time_series["residual"].to_numpy()

    mean_residual = float(np.mean(residual))
    std_residual = float(np.std(residual, ddof=0))
    if std_residual == 0.0:
        std_residual = max(abs(mean_residual), 1.0) * 0.05

    lower_1 = mean_residual - std_residual
    upper_1 = mean_residual + std_residual
    lower_2 = mean_residual - 2.0 * std_residual
    upper_2 = mean_residual + 2.0 * std_residual
    outside_mask = (residual < lower_2) | (residual > upper_2)

    residual_min = float(np.min(residual))
    residual_max = float(np.max(residual))
    plot_min = min(residual_min, lower_2)
    plot_max = max(residual_max, upper_2)
    value_range = plot_max - plot_min
    if value_range == 0.0:
        value_range = max(abs(plot_max), 1.0) * 0.1
    y_min = plot_min - 0.08 * value_range
    y_max = plot_max + 0.20 * value_range

    fig, axis = plt.subplots(figsize=(16, 5.5))
    axis.set_ylim(y_min, y_max)
    _add_sigma_bands(axis, y_min, y_max, lower_1, upper_1, lower_2, upper_2)
    axis.plot(sample_index, residual, label="Residual", linewidth=2.0, color="#2ca02c", zorder=4)
    axis.scatter(
        sample_index[outside_mask],
        residual[outside_mask],
        color="#d62728",
        edgecolor="white",
        linewidth=0.6,
        s=42,
        label="Outside +/-2 sigma",
        zorder=5,
    )

    axis.axhline(mean_residual, color="black", linewidth=1.2, linestyle="--", label="Mean")
    axis.axhline(upper_1, color="#2ca02c", linewidth=1.25, linestyle=":", label="+/-1 sigma")
    axis.axhline(lower_1, color="#2ca02c", linewidth=1.25, linestyle=":")
    axis.axhline(upper_2, color="#ffbf00", linewidth=1.25, linestyle=":", label="+/-2 sigma")
    axis.axhline(lower_2, color="#ffbf00", linewidth=1.25, linestyle=":")

    axis.set_xlabel("Sample Index")
    axis.set_ylabel(f"Residual ({target})")
    axis.set_title(
        _context_title("B", "Residual Diagnostics with Semantic Severity Bands", target, sensor, fault_type, severity),
        fontweight="bold",
        pad=16,
    )
    axis.legend(loc="upper right", frameon=True, framealpha=0.92, ncol=2)
    axis.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _add_sigma_bands(
    axis: plt.Axes,
    y_min: float,
    y_max: float,
    lower_1: float,
    upper_1: float,
    lower_2: float,
    upper_2: float,
) -> None:
    axis.axhspan(lower_1, upper_1, color="#2ca02c", alpha=0.18, label="Safe/Low Band (+/-1 sigma)", zorder=0)
    axis.axhspan(upper_1, upper_2, color="#ffbf00", alpha=0.18, label="Warning/Medium Band (1-2 sigma)", zorder=0)
    axis.axhspan(lower_2, lower_1, color="#ffbf00", alpha=0.18, zorder=0)
    axis.axhspan(upper_2, y_max, color="#ff7f0e", alpha=0.16, label="Failure/High Band (>2 sigma)", zorder=0)
    axis.axhspan(y_min, lower_2, color="#ff7f0e", alpha=0.16, zorder=0)


def save_figure_c(
    time_series: pd.DataFrame,
    target: str,
    sensor: str,
    fault_type: str,
    severity: float,
    output_path: Path,
) -> Path:
    """Figure C: healthy prediction, faulty prediction, and healthy ground truth."""
    _apply_style()
    fig, axis = plt.subplots(figsize=(16, 5.5))
    sample_index = time_series["sample_index"].to_numpy()
    healthy_ground_truth = time_series["healthy_ground_truth"].to_numpy()
    healthy_prediction = time_series["healthy_prediction"].to_numpy()
    faulty_prediction = time_series["faulty_prediction"].to_numpy()

    axis.plot(sample_index, healthy_ground_truth, label="Healthy Ground Truth", linewidth=2.4, color="#1f77b4")
    axis.plot(sample_index, healthy_prediction, label="Healthy Prediction", linewidth=2.0, color="#2ca02c")
    axis.plot(sample_index, faulty_prediction, label="Faulty Prediction", linewidth=2.0, color="#ff7f0e", alpha=0.9)
    axis.set_xlabel("Sample Index")
    axis.set_ylabel(target)
    axis.set_title(
        _context_title("C", "Prediction Performance Degradation Path", target, sensor, fault_type, severity),
        fontweight="bold",
        pad=16,
    )
    _add_top_padding(axis, [healthy_ground_truth, healthy_prediction, faulty_prediction])
    axis.legend(loc="upper right", frameon=True, framealpha=0.92)
    axis.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_experiment_figures(
    time_series: pd.DataFrame,
    target: str,
    sensor: str,
    fault_type: str,
    severity: float,
    output_dir: Path,
) -> dict[str, Path]:
    """Save all three standard figures for an experiment."""
    return {
        "figure_a": save_figure_a(
            time_series,
            target,
            sensor,
            fault_type,
            severity,
            output_dir / "figure_a_sensor_fault.png",
        ),
        "figure_b": save_figure_b(
            time_series,
            target,
            sensor,
            fault_type,
            severity,
            output_dir / "figure_b_error_residual_severity.png",
        ),
        "figure_c": save_figure_c(
            time_series,
            target,
            sensor,
            fault_type,
            severity,
            output_dir / "figure_c_predictions.png",
        ),
    }
