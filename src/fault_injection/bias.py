"""Persistent relative bias fault injection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_bias(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Apply the constant additive bias ``severity * y`` to clean signal ``y``."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    values = np.asarray(sensor_values, dtype=float)
    return values + severity * values
