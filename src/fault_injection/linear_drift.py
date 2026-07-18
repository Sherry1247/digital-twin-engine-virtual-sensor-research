"""Relative linear-drift fault injection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_linear_drift(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Increase from clean ``y`` to ``y + severity * y`` over the series."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    values = np.asarray(sensor_values, dtype=float)
    if values.size == 0:
        return values.copy()
    drift = np.linspace(0.0, severity, num=values.size)
    return values + values * drift
