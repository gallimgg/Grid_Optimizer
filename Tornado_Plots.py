from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILES = [
    Path("TornadoRawSameOut.csv"),
]

OUTPUT_DIR = Path("tornado_plots")

SCENARIO_COL = "scenario_name"
STATUS_COL = "status"
BASELINE_NAME = "baseline"

METRIC_COLS = [
    "Combined_LCOE",
    "Carbon_Adjusted_LCOE_SCC_51",
    "Carbon_Adjusted_LCOE_SCC_120",
    "Carbon_Adjusted_LCOE_SCC_190",
    "Carbon_Adjusted_LCOE_SCC_340",
]


def load_all_successful_runs(input_files: list[Path], metric_col: str) -> pd.DataFrame:
    frames = []

    for path in input_files:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        df = pd.read_csv(path)

        required = {SCENARIO_COL, STATUS_COL, metric_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        df = df.copy()
        df["source_file"] = path.name
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

        df = df[
            (df[STATUS_COL].astype(str).str.lower() == "success")
            & df[metric_col].notna()
        ]

        frames.append(df)

    if not frames:
        raise ValueError("No usable successful run data found.")

    return pd.concat(frames, ignore_index=True)

def add_sigma_bands(ax, baseline_mean: float, baseline_std: float) -> None:
    bands = [
        (3, 0.08),
        (2, 0.12),
        (1, 0.18),
    ]

    for sigma, alpha in bands:
        ax.axvspan(
            baseline_mean - sigma * baseline_std,
            baseline_mean + sigma * baseline_std,
            alpha=alpha,
            zorder=0,
            label=f"±{sigma}σ baseline",
        )
def clean_label(name: str) -> str:
    label = name

    replacements = {
        "_capex_": " CAPEX ",
        "_connect_cost_": " connect cost ",
        "_battery_cost_": " EV battery cost ",
        "_ev_battery_cost_": " EV battery cost ",
        "_": " ",
        "minus": "-",
        "plus": "+",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    return label


def scenario_sort_key(name: str) -> tuple:
    lower = name.lower()

    if lower == BASELINE_NAME:
        return (0, lower)

    order = {
        "wind": 1,
        "solar": 2,
        "nuclear": 3,
        "grid": 4,
        "battery": 5,
        "v2g": 6,
        "ev": 7,
    }

    for key, value in order.items():
        if key in lower:
            return (value, lower)

    return (99, lower)


def make_boxplot(metric_col: str) -> None:
    df = load_all_successful_runs(INPUT_FILES, metric_col)

    baseline_df = df[df[SCENARIO_COL].str.lower() == BASELINE_NAME]

    if baseline_df.empty:
        raise ValueError("No successful baseline rows found.")

    baseline_mean = baseline_df[metric_col].mean()
    baseline_std = baseline_df[metric_col].std()

    plot_df = df[df[SCENARIO_COL].str.lower() != BASELINE_NAME]

    scenarios = sorted(plot_df[SCENARIO_COL].unique(), key=scenario_sort_key)

    values = [
        plot_df.loc[plot_df[SCENARIO_COL] == scenario, metric_col].values
        for scenario in scenarios
    ]

    labels = [clean_label(scenario) for scenario in scenarios]

    fig_height = max(7, 0.45 * len(scenarios))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    add_sigma_bands(ax, baseline_mean, baseline_std)

    ax.boxplot(
        values,
        vert=False,
        labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=False,
    )

    ax.axvline(
        baseline_mean,
        linestyle=":",
        linewidth=2.5,
        label=f"Baseline mean = {baseline_mean:.3f}",
        zorder=3,
    )

    ax.set_title(f"Tornado Sensitivity: {metric_col}", fontweight="bold")
    ax.set_xlabel("LCOE ($/MWh)")
    ax.set_ylabel("Scenario")
    ax.grid(axis="x", alpha=0.35)
    ax.legend()

    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"tornado_boxplot_{metric_col}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main() -> None:
    for metric_col in METRIC_COLS:
        make_boxplot(metric_col)


if __name__ == "__main__":
    main()