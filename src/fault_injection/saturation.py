"""Upper-saturation fault injection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_saturation(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Clip clean values at the upper saturation limit ``y * (1 - severity)``."""
    if not 0 <= severity <= 1:
        raise ValueError("saturation severity must be between 0 and 1")
    values = np.asarray(sensor_values, dtype=float)
    nominal_peak = float(np.mean(values)) if values.size else 0.0
    upper_limit = nominal_peak * (1.0 - severity)
    return np.minimum(values, upper_limit)
