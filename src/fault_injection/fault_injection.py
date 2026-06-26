"""
Fault injection module for sensor robustness evaluation.

Provides functions to inject various fault types into sensor data arrays.

Author: TODO (add your name)
"""

from typing import Any
import numpy as np

# Supported drift levels for experiments
DRIFT_LEVELS = [
    0.01,  # 1%
    0.02,  # 2%
    0.05,  # 5%
    0.10,  # 10%
    0.20,  # 20%
]

def inject_linear_drift(sensor_values: np.ndarray, drift_level: float) -> np.ndarray:
    """
    Injects a linear drift into the sensor values.

    The drift starts at 0 and increases linearly to drift_level over the dataset.
    Each value is scaled by (1 + drift(i)), where drift(i) = drift_level * (i / (N-1)).

    Args:
        sensor_values (np.ndarray): Original sensor data (1D array).
        drift_level (float): Maximum drift percentage (e.g., 0.05 for 5%).

    Returns:
        np.ndarray: Faulty sensor values with linear drift applied.
    """
    sensor_values = np.asarray(sensor_values)
    N = sensor_values.shape[0]
    if N < 2:
        raise ValueError("sensor_values must contain at least 2 samples for drift injection.")
    drift = drift_level * np.linspace(0, 1, N)
    return sensor_values * (1 + drift)


def inject_exponential_drift(sensor_values: np.ndarray, drift_level: float) -> np.ndarray:
    """
    TODO: Implement exponential drift injection.
    """
    raise NotImplementedError("inject_exponential_drift is not implemented yet.")


def inject_spike(sensor_values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    TODO: Implement spike fault injection.
    """
    raise NotImplementedError("inject_spike is not implemented yet.")


def inject_noise(sensor_values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    TODO: Implement noise injection.
    """
    raise NotImplementedError("inject_noise is not implemented yet.")


def inject_dropout(sensor_values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    TODO: Implement dropout fault injection.
    """
    raise NotImplementedError("inject_dropout is not implemented yet.")


def inject_stuck_at(sensor_values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    TODO: Implement stuck-at fault injection.
    """
    raise NotImplementedError("inject_stuck_at is not implemented yet.")


def inject_saturation(sensor_values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    TODO: Implement saturation fault injection.
    """
    raise NotImplementedError("inject_saturation is not implemented yet.")


def inject_dead_zone(sensor_values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    TODO: Implement dead-zone fault injection.
    """
    raise NotImplementedError("inject_dead_zone is not implemented yet.")


def inject_oscillation(sensor_values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    TODO: Implement oscillation fault injection.
    """
    raise NotImplementedError("inject_oscillation is not implemented yet.")
