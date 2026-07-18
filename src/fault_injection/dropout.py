"""Contiguous sensor-dropout fault injection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_dropout(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Set the sensor signal to the strict missing-data value of zero."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    values = np.asarray(sensor_values, dtype=float)
    return np.zeros_like(values)
