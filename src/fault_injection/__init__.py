"""
Fault injection package exports.

Expose individual fault injection functions for easy import.
"""
from .linear_drift import inject_linear_drift, DRIFT_LEVELS
from .exponential_drift import inject_exponential_drift
from .spike import inject_spike
from .noise import inject_noise
from .dropout import inject_dropout
from .stuck_at import inject_stuck_at
from .saturation import inject_saturation
from .dead_zone import inject_dead_zone
from .oscillation import inject_oscillation

__all__ = [
    "inject_linear_drift",
    "DRIFT_LEVELS",
    "inject_exponential_drift",
    "inject_spike",
    "inject_noise",
    "inject_dropout",
    "inject_stuck_at",
    "inject_saturation",
    "inject_dead_zone",
    "inject_oscillation",
]
