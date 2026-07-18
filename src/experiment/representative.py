"""Representative operating-point loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import INPUT_COLUMNS


def load_representative_point(csv_path: Path, representative_id: str, target: str) -> pd.Series:
    """Load one representative point and validate required ANN fields."""
    representatives = pd.read_csv(csv_path)
    row = representatives.loc[representatives["Representative_ID"] == representative_id]
    if row.empty:
        available = ", ".join(representatives["Representative_ID"].astype(str).tolist())
        raise ValueError(f"Representative ID '{representative_id}' not found. Available IDs: {available}")

    representative = row.iloc[0].copy()
    expected_output = representative.get("Output")
    if pd.notna(expected_output) and str(expected_output) != target:
        raise ValueError(f"Representative '{representative_id}' is for '{expected_output}', not '{target}'")

    missing = [column for column in [*INPUT_COLUMNS, target] if column not in representative.index]
    if missing:
        raise ValueError(f"Representative CSV is missing required columns: {missing}")

    if pd.isna(representative[target]):
        raise ValueError(f"Representative '{representative_id}' does not contain target value '{target}'")

    return representative
