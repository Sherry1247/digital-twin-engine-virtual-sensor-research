"""Exponential-drift fault injection from a clean signal baseline."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_exponential_drift(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Add a saturating drift that reaches 95% of ``severity * y`` at the end."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    values = np.asarray(sensor_values, dtype=float)
    if values.size == 0:
        return values.copy()
    tau = -np.log(0.05) / max(values.size - 1, 1)
    time = np.arange(values.size, dtype=float)
    return values + severity * values * (1.0 - np.exp(-tau * time))
