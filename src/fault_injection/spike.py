"""Intermittent relative spike fault injection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def inject_spike(
    sensor_values: Sequence[float], severity: float, random_state: int | None = None
) -> np.ndarray:
    """Inject +/- ``3 * severity * y`` spikes into 2% of clean samples."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    values = np.asarray(sensor_values, dtype=float)
    faulty = values.copy()
    if faulty.size == 0 or severity == 0:
        return faulty
    rng = np.random.default_rng(random_state)
    count = min(faulty.size, max(1, int(np.ceil(0.02 * faulty.size))))
    positions = rng.choice(faulty.size, size=count, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    faulty[positions] += signs * 3.0 * severity * values[positions]
    return faulty
