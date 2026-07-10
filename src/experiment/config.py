"""Central configuration for virtual-sensor fault experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_ARTIFACT_PATH = REPO_ROOT / "experiments" / "ann" / "test_results" / "visualizations" / "ann_all_inputs_results.pkl"
REPRESENTATIVE_POINTS_PATH = REPO_ROOT / "results" / "representative_points" / "representative_operating_points.csv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "fault_experiments"

INPUT_COLUMNS = ["Torque", "p_0", "T_IM", "P_IM", "EGR_Rate", "ECU_VTG_Pos"]
TARGETS = ["MF_IA", "NOx_EO", "SOC"]
DEFAULT_N_SAMPLES = 300

SEVERITY_LEVELS = {
    "low": 0.02,
    "medium": 0.05,
    "high": 0.10,
    "failure": 0.20,
}

SENSITIVE_SENSORS = {
    "MF_IA": {"most": "P_IM", "least": "EGR_Rate"},
    "NOx_EO": {"most": "EGR_Rate", "least": "ECU_VTG_Pos"},
    "SOC": {"most": "Torque", "least": "ECU_VTG_Pos"},
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime parameters for one target/representative/sensor/fault experiment."""

    target: str
    representative_id: str
    sensor: str
    fault_type: str = "gaussian"
    severity: float = 0.05
    random_seed: int | None = 42
    n_samples: int = DEFAULT_N_SAMPLES
    results_dir: Path = DEFAULT_RESULTS_DIR
    ann_artifact_path: Path = ANN_ARTIFACT_PATH
    representative_points_path: Path = REPRESENTATIVE_POINTS_PATH


def default_experiment_matrix(
    fault_type: str = "gaussian",
    severity: float = 0.05,
    random_seed: int | None = 42,
    n_samples: int = DEFAULT_N_SAMPLES,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> list[ExperimentConfig]:
    """Return the 12 configured experiments in the study design."""
    configs: list[ExperimentConfig] = []
    for target in TARGETS:
        for level in ("High", "Low"):
            representative_id = f"{target}_{level}"
            for sensor in SENSITIVE_SENSORS[target].values():
                configs.append(
                    ExperimentConfig(
                        target=target,
                        representative_id=representative_id,
                        sensor=sensor,
                        fault_type=fault_type,
                        severity=severity,
                        random_seed=random_seed,
                        n_samples=n_samples,
                        results_dir=results_dir,
                    )
                )
    return configs
