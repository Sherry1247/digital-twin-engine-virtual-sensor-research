"""Bias fault injection placeholder."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def inject_bias(sensor_values: Sequence[float], *args: Any, **kwargs: Any) -> np.ndarray:
    """Reserve the bias fault interface for future experiments."""
    raise NotImplementedError("inject_bias is not implemented yet.")
