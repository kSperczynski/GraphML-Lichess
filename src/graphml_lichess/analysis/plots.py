"""Plot summaries without pooling incompatible protocols or model variants."""

from pathlib import Path

import numpy as np
import pandas as pd


def plot_summary(
    summary: pd.DataFrame, output: Path, metric: str = "rmse", phase: str = "test", interval: str = "sd"
) -> list[Path]:
    if metric not in {"mae", "rmse", "mse", "nmse", "r2"} or phase not in {"train", "validation", "test"}:
        raise ValueError("Unsupported metric or evaluation phase.")
    if interval not in {"sd", "ci95"}:
        raise ValueError("Interval must be sd or ci95.")
    if interval == "ci95" and (summary["unit"] != "split_mean").any():
        raise ValueError("Confidence intervals require --unit split, not training seeds.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = summary[summary.phase == phase].copy()
    selected = selected[np.isfinite(selected[f"{metric}_mean"])]
    if selected.empty:
        raise ValueError(f"No finite {phase} {metric} values available.")
    if interval == "ci95" and ((selected["n_units"] < 3).any() or not np.allclose(selected["confidence"], 0.95)):
        raise ValueError("CI95 plots require at least three splits and 95% summary intervals.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    paths = []

    def errors(frame):
        if interval == "sd":
            return frame[f"{metric}_std"].fillna(0).to_numpy()
        center = frame[f"{metric}_mean"].to_numpy()
        return np.vstack([center - frame[f"{metric}_ci_low"].to_numpy(), frame[f"{metric}_ci_high"].to_numpy() - center])

    def finish(fig, ax, stem, quality):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        label = metric.upper()
        units = (
            "rating points" if metric in {"mae", "rmse"} else "rating points squared" if metric == "mse" else "dimensionless"
        )
        ax.set_ylabel(f"{phase.title()} {label} ({units})")
        uncertainty = (
            "SD across training seeds"
            if selected["unit"].iloc[0] == "training_seed"
            else ("descriptive 95% t interval across split means" if interval == "ci95" else "SD across split means")
        )
        footer = f"Error bars: {uncertainty}.\nMissing uncertainty is not zero uncertainty."
        if quality == "legacy_unverified":
            footer += "\nLegacy input identity unverified."
        fig.text(0.02, 0.01, footer, fontsize=8)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        for suffix in ("png", "svg"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=170)
            paths.append(path)
        plt.close(fig)

    cohort_keys = ["data_id", "protocol_id", "split_set_id"]

    def token(value):
        return str(value).removeprefix("legacy-")[:12]

    for (data, protocol, split_set), group in selected.groupby(cohort_keys):
        group = group.sort_values(["model_type", "config_id"])
        fig, ax = plt.subplots(figsize=(max(7, len(group) * 0.8), 4.8))
        x = np.arange(len(group))
        labels = [f"{row.model_type}\n{row.config_id[:6]} (n={row.n_units})" for row in group.itertuples()]
        ax.bar(x, group[f"{metric}_mean"], width=0.55, yerr=errors(group), capsize=4, color="#2166AC")
        ax.set_xlim(-0.8, max(len(group) - 0.2, 0.8))
        ax.set_xticks(x, labels, rotation=0 if len(group) <= 4 else 20, ha="center" if len(group) <= 4 else "right")
        ax.set_title(f"Model variants | {phase} | split set {split_set[:6]}")
        finish(
            fig,
            ax,
            f"models_{metric}_{phase}_{token(data)}_{token(protocol)}_{token(split_set)}",
            group.provenance_quality.iloc[0],
        )

    sweeps = selected[selected.alpha.notna()]
    for key, group in sweeps.groupby(cohort_keys + ["sweep_id"]):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for zero, series in group.groupby("zero_edges"):
            series = series.sort_values("alpha")
            if series["alpha"].duplicated().any():
                plt.close(fig)
                raise ValueError("Ambiguous alpha sweep: multiple model variants share an alpha and graph setting.")
            ax.errorbar(
                series.alpha,
                series[f"{metric}_mean"],
                yerr=errors(series),
                fmt="o-",
                capsize=4,
                label="Zero edges retained" if zero else "Zero edges absent/removed",
            )
        ax.set_xlabel("DirGNN alpha")
        ax.set_title("Alpha and zero-edge experiment")
        ax.legend(frameon=False)
        stem = f"alpha_{metric}_{phase}_" + "_".join(token(v) for v in key)
        finish(fig, ax, stem, group.provenance_quality.iloc[0])
    return paths
