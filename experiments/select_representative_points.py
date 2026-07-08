"""Select representative operating points for ANN output variables.

This standalone script inspects the processed dataset, generates histograms for
three ANN outputs, and selects one high-value and one low-value representative
sample for each output based on percentile groups and median proximity.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "df_processed.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "representative_points"
OUTPUT_CSV = OUTPUT_DIR / "representative_operating_points.csv"
OUTPUTS = ["MF_IA", "NOx_EO", "SOC"]
INPUT_COLUMNS = ["Torque", "p_0", "T_IM", "P_IM", "EGR_Rate", "ECU_VTG_Pos"]
LOW_PERCENTILE = 0.10
HIGH_PERCENTILE = 0.90
BIN_COUNT = 30


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load the processed dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    return pd.read_csv(path)


def ensure_output_directory(path: Path) -> None:
    """Create the output directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def plot_histogram(df: pd.DataFrame, output_name: str, output_dir: Path) -> None:
    """Create and save a histogram for the requested output variable."""
    values = df[output_name].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(values, bins=BIN_COUNT, color="steelblue", edgecolor="black")

    median_value = np.median(values)
    low_percentile = np.percentile(values, LOW_PERCENTILE * 100)
    high_percentile = np.percentile(values, HIGH_PERCENTILE * 100)

    ax.axvline(median_value, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_value:.4f}")
    ax.axvline(low_percentile, color="green", linestyle=":", linewidth=1.5, label=f"10th percentile: {low_percentile:.4f}")
    ax.axvline(high_percentile, color="orange", linestyle=":", linewidth=1.5, label=f"90th percentile: {high_percentile:.4f}")

    ax.set_title(f"Histogram of {output_name}")
    ax.set_xlabel(output_name)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()

    save_path = output_dir / f"histogram_{output_name}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def select_representative_point(df: pd.DataFrame, output_name: str) -> pd.Series:
    """Select one low and one high representative sample for an output."""
    values = df[output_name].to_numpy()
    lower_threshold = np.percentile(values, LOW_PERCENTILE * 100)
    upper_threshold = np.percentile(values, HIGH_PERCENTILE * 100)

    low_mask = values <= lower_threshold
    high_mask = values >= upper_threshold

    low_group = df.loc[low_mask]
    high_group = df.loc[high_mask]

    if low_group.empty or high_group.empty:
        raise ValueError(f"Not enough samples to form percentile groups for {output_name}")

    low_group_median = low_group[output_name].median()
    high_group_median = high_group[output_name].median()

    low_candidate = low_group.iloc[(np.abs(low_group[output_name] - low_group_median)).argmin()]
    high_candidate = high_group.iloc[(np.abs(high_group[output_name] - high_group_median)).argmin()]

    return pd.Series(
        {
            "Output": output_name,
            "Low_Group_Median": low_group_median,
            "High_Group_Median": high_group_median,
            "Low_Representative": low_candidate,
            "High_Representative": high_candidate,
        }
    )


def build_representative_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Build a row-wise summary of representative operating points."""
    rows = []

    for output_name in OUTPUTS:
        result = select_representative_point(df, output_name)

        for group_name, candidate in (("Low", result["Low_Representative"]), ("High", result["High_Representative"])):
            sample_index = candidate.get("Log Point", candidate.name)
            group_median = result["Low_Group_Median"] if group_name == "Low" else result["High_Group_Median"]
            output_value = candidate[output_name]
            abs_diff = abs(output_value - group_median)

            row = {
                "Representative_ID": f"{output_name}_{group_name}",
                "Sample_Index": sample_index,
                "Output": output_name,
                "Group": group_name,
                "Output_Value": output_value,
                "Group_Median": group_median,
                "Abs_Diff_To_Group_Median": abs_diff,
            }
            row.update({column: candidate[column] for column in INPUT_COLUMNS})
            row[output_name] = output_value
            rows.append(row)

    return pd.DataFrame(rows)


def print_summary(representatives: pd.DataFrame) -> None:
    """Print a compact summary report for all selected representative points."""
    print("=" * 90)
    print("Representative Operating Points")
    print("=" * 90)
    for _, row in representatives.iterrows():
        print(
            f"{row['Output']} {row['Group']}: "
            f"Sample Index: {row['Sample_Index']}, "
            f"Output Value: {row['Output_Value']:.6f}, "
            f"Median of {row['Group']} Group: {row['Group_Median']:.6f}, "
            f"Abs Diff: {row['Abs_Diff_To_Group_Median']:.6f}"
        )
    print("=" * 90)


def main() -> None:
    """Run the representative-point selection workflow."""
    ensure_output_directory(OUTPUT_DIR)
    df = load_processed_data(DATA_PATH)

    for output_name in OUTPUTS:
        plot_histogram(df, output_name, OUTPUT_DIR)

    representatives = build_representative_rows(df)
    representatives.to_csv(OUTPUT_CSV, index=False)
    print_summary(representatives)
    print(f"Saved histograms to: {OUTPUT_DIR}")
    print(f"Saved representative points to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
