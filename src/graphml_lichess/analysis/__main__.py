"""CLI for read-only experiment analysis; no model training is triggered."""

import argparse
from pathlib import Path

import numpy as np

from graphml_lichess.analysis.experiments import ExperimentCollection


def add_arguments(parser):
    parser.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="One or more run directories, searched recursively"
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--unit", choices=["training", "split"], default="training")
    parser.add_argument("--metric", choices=["mae", "rmse", "mse", "nmse", "r2"], default="rmse")
    parser.add_argument("--phase", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--interval", choices=["sd", "ci95"], default="sd")
    parser.add_argument("--no-plots", action="store_true")


def run_analysis(args):
    if args.interval == "ci95" and args.unit != "split":
        raise ValueError("--interval ci95 requires --unit split")
    collection = ExperimentCollection.from_roots(args.runs)
    summary, _ = collection.summarize(args.unit)
    if not args.no_plots:
        from graphml_lichess.analysis.plots import plot_summary

        eligible = summary[(summary.phase == args.phase) & np.isfinite(summary[f"{args.metric}_mean"])]
        if eligible.empty:
            raise ValueError(f"No finite {args.phase} {args.metric} values; old runs may lack the NMSE baseline.")
        if args.interval == "ci95" and (eligible.n_units < 3).any():
            raise ValueError("At least three distinct data splits per variant are required for CI plots.")
    collection.export(args.output, args.unit)
    if not args.no_plots:
        plot_summary(summary, args.output, args.metric, args.phase, args.interval)
    print(
        summary[
            ["model_type", "config_id", "phase", "split_id", "n_units", "unit", f"{args.metric}_mean", f"{args.metric}_std"]
        ].to_string(index=False)
    )
    print(f"Saved analysis to {args.output}; inspect diagnostics.json for skipped or unverified runs.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        run_analysis(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
