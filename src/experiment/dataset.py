"""CSV export helpers for fault-classification datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def write_case_outputs(
    output_dir: Path,
    time_series: pd.DataFrame,
    metrics_row: Mapping[str, object],
    residual_features: pd.DataFrame,
) -> dict[str, Path]:
    """Persist one case's signals, labels, residual statistics, and features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "time_series_path": output_dir / "time_series.csv",
        "metrics_path": output_dir / "metrics.csv",
        "residual_features_path": output_dir / "residual_features.csv",
    }
    time_series.to_csv(paths["time_series_path"], index=False)
    pd.DataFrame([metrics_row]).to_csv(paths["metrics_path"], index=False)
    residual_features.to_csv(paths["residual_features_path"], index=False)
    return paths


def write_dataset_summary(rows: list[Mapping[str, object]], output_dir: Path) -> Path:
    """Write one classifier-ready row per generated fault case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "fault_dataset.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path
