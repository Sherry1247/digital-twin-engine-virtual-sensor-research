"""Central configuration for the MF_IA fault-dataset generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_ARTIFACT_PATH = REPO_ROOT / "experiments" / "ann" / "test_results" / "visualizations" / "ann_all_inputs_results.pkl"
REPRESENTATIVE_POINTS_PATH = REPO_ROOT / "results" / "representative_points" / "representative_operating_points.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "fault_dataset"

INPUT_COLUMNS = ["Torque", "p_0", "T_IM", "P_IM", "EGR_Rate", "ECU_VTG_Pos"]
TARGET = "MF_IA"
SENSITIVE_SENSORS = ("P_IM", "EGR_Rate")
REPRESENTATIVE_IDS = ("MF_IA_High", "MF_IA_Low")

DEFAULT_N_SAMPLES = 300
DEFAULT_HEALTHY_NOISE = 0.02
DEFAULT_REPETITIONS = 2
FAULT_TYPES = (
    "gaussian",
    "bias",
    "linear_drift",
    "exp_drift",
    "spike",
    "dropout",
    "saturation",
    "oscillation",
)
BATCH_SEVERITIES = tuple(round(level / 100.0, 2) for level in range(1, 21))


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime parameters for one independently generated fault case."""

    target: str
    representative_id: str
    sensor: str
    fault_type: str = "gaussian"
    severity: float = 0.05
    random_seed: int | None = 42
    repetition: int = 1
    n_samples: int = DEFAULT_N_SAMPLES
    healthy_noise: float = DEFAULT_HEALTHY_NOISE
    generate_figures: bool = False
    results_dir: Path = DEFAULT_RESULTS_DIR
    ann_artifact_path: Path = ANN_ARTIFACT_PATH
    representative_points_path: Path = REPRESENTATIVE_POINTS_PATH


def fault_dataset_matrix(
    fault_types: tuple[str, ...] = FAULT_TYPES,
    random_seed: int | None = 42,
    n_samples: int = DEFAULT_N_SAMPLES,
    healthy_noise: float = DEFAULT_HEALTHY_NOISE,
    repetitions: int = DEFAULT_REPETITIONS,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[ExperimentConfig]:
    """Build the 1,280-case batch matrix for the eight supported failures."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    unsupported = sorted(set(fault_types) - set(FAULT_TYPES))
    if unsupported:
        raise ValueError(f"Unsupported fault types: {', '.join(unsupported)}")

    configs: list[ExperimentConfig] = []
    case_offset = 0
    for representative_id in REPRESENTATIVE_IDS:
        for sensor in SENSITIVE_SENSORS:
            for fault_type in fault_types:
                for severity in BATCH_SEVERITIES:
                    for repetition in range(1, repetitions + 1):
                        case_seed = None if random_seed is None else random_seed + case_offset
                        configs.append(
                            ExperimentConfig(
                                target=TARGET,
                                representative_id=representative_id,
                                sensor=sensor,
                                fault_type=fault_type,
                                severity=severity,
                                random_seed=case_seed,
                                repetition=repetition,
                                n_samples=n_samples,
                                healthy_noise=healthy_noise,
                                results_dir=results_dir,
                            )
                        )
                        case_offset += 1
    return configs
