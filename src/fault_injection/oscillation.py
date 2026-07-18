"""Oscillation fault injection from a clean signal baseline."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_oscillation(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Add a five-cycle sinusoid with amplitude ``severity * y``."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    values = np.asarray(sensor_values, dtype=float)
    if values.size == 0:
        return values.copy()
    phase = np.linspace(0.0, 10.0 * np.pi, num=values.size)
    return values + severity * values * np.sin(phase)
