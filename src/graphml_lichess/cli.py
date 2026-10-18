import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from graphml_lichess.config import load_config
from graphml_lichess.data_processing.data_splits import make_split
from graphml_lichess.training.pipeline import TrainingPipeline, ExplainabilityPipeline, load_graph
from graphml_lichess.tuning.tuning import tune
from graphml_lichess.utils.logging import configure_logging


def cmd_make_split(args) -> None:
    """Method for creating a split in the data"""
    config = load_config(args.config, args.set)
    cfg = config["split"]

    path = Path(cfg["path"])
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists. Use --force to overwrite (this invalidates old comparisons).")

    graph = load_graph(config)
    split = make_split(
        graph.usernames,
        graph.data.y.squeeze(-1).numpy(),  # type: ignore
        cfg["train_ratio"],
        cfg["validation_ratio"],
        cfg["test_ratio"],
        seed=cfg["seed"],
        stratify=cfg.get("stratify", True),
    )

    split.save(path)
    print(
        f"Saved split {split.fingerprint} to {path}: "
        f"{len(split.train)} train / {len(split.validation)} val / {len(split.test)} test"
    )


def cmd_train(args) -> None:
    """Method for training a model. Trains the model for every provided seed, on the same data split."""
    config = load_config(args.config, args.set)
    for seed in args.seeds or [config.get("seed", 42)]:
        run_config = copy.deepcopy(config)
        run_config["seed"] = seed
        run_dir = TrainingPipeline(run_config).run()
        print(f"[seed {seed}] -> {run_dir}")


def cmd_train_multiple_alphas(args) -> None:
    """Method for training models with multiple different alphas."""
    config = load_config(args.config, args.set)

    assert "dir_alpha" in config["model"], "ERR: trying to train a model that does not take dir_alpha with multiple alphas"

    # use different alpha params
    DIR_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    for dir_alpha in DIR_ALPHAS:
        # run each alpha on multiple seeds
        for seed in args.seeds or [config.get("seed", 42)]:
            run_config = copy.deepcopy(config)

            # set seed
            run_config["seed"] = seed

            # set alpha
            run_config["model"]["dir_alpha"] = dir_alpha

            run_dir = TrainingPipeline(run_config).run()
            print(f"[seed {seed}] -> {run_dir}")


def cmd_tune(args) -> None:
    """Method for tuning the models."""
    config = load_config(args.config, args.set)
    search_path = args.search or Path("config/search") / f"{args.config.stem}.yaml"
    search = yaml.safe_load(search_path.read_text())
    logger = configure_logging(Path("runs/tuning", f"{config['name']}.log"), f"tune-{config['name']}")
    tuned = tune(config, search, args.config, logger, n_trials=args.trials)
    print(f"Wrote {tuned}\nNext: train --config {tuned} --seeds 0 1 2 3 4")


def cmd_report(args) -> None:
    """Aggregate finished runs into a comparison table."""
    rows = []
    for run_dir in sorted(Path(args.runs_dir).glob("*/")):
        info_path, metrics_path = run_dir / "run_info.json", run_dir / "metrics.json"
        if not (info_path.exists() and metrics_path.exists()):
            continue  # unfinished or failed run, or a non-run folder such as runs/tuning

        info = json.loads(info_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "model": info.get("name"),
                "split": info.get("data_split_fingerprint"),
                "seed": info.get("seed"),
                **{f"val_{k}": v for k, v in metrics["validation"].items()},
                **{f"test_{k}": v for k, v in metrics["test"].items()},
            }
        )

    if not rows:
        raise SystemExit(f"No finished runs found in {args.runs_dir}")

    df = pd.DataFrame(rows)
    cols = ["val_mae", "test_mae", "test_rmse", "test_r2"]
    grouped = df.groupby(["model", "split"])
    summary = grouped[cols].agg(["mean", "std"]).round(3)
    summary["n_runs"] = grouped.size()
    print(summary.sort_values(("test_mae", "mean")).to_string())


def cmd_explain(args) -> None:

    config = load_config(args.config, args.set)
    run_config = copy.deepcopy(config)

    explainability_pipeline = ExplainabilityPipeline(run_config)
    explainability_pipeline.run()

def cli(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="graphml-lichess")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p):
        """Helper method to add config arguments to cli, almost all sub-parsers use it"""
        p.add_argument("--config", type=Path, required=True, help="YAML config (may inherit via `base:`)")
        p.add_argument(
            "--set",
            nargs="*",
            default=[],
            metavar="KEY=VALUE",
            help="Override config values, e.g. --set training.epochs=50 model.num_layers=2",
        )

    # Parser for creating a new data-split
    p = sub.add_parser("make-split", help="Create the shared train/val/test split file")
    add_config(p)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overides an already existing data split file with this name. Watch out: will invalidate old comparisons.",
    )
    p.set_defaults(func=cmd_make_split)

    # Parser for training a model
    p = sub.add_parser("train", help="Train + evaluate one model on the shared split")
    add_config(p)
    p.add_argument("--seeds", type=int, nargs="+", help="Run once per model seed")
    p.set_defaults(func=cmd_train)

    # Parser for training with multiple alphas
    p = sub.add_parser("train-alphas", help="Train + evaluate multiple models with different alphas on the shared split")
    add_config(p)
    p.add_argument("--seeds", type=int, nargs="+", help="Run once per model seed")
    p.set_defaults(func=cmd_train_multiple_alphas)

    # Parser for tuning
    p = sub.add_parser("tune", help="Optuna search on the validation split; writes config/tuned/<model>.yaml")
    add_config(p)
    p.add_argument("--search", type=Path, help="Search space YAML (default: config/search/<config name>.yaml)")
    p.add_argument("--trials", type=int, help="Override n_trials from the search file")
    p.set_defaults(func=cmd_tune)

    # Parser for reporting results of already finished runs
    p = sub.add_parser("report", help="Aggregate finished runs into a comparison table")
    p.add_argument("--runs-dir", type=Path, default=Path("runs", "training"))
    p.set_defaults(func=cmd_report)

    # Parser for explainability
    p = sub.add_parser("explain", help="Run explainability experiments.")
    add_config(p)
    p.set_defaults(func=cmd_explain)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    cli()
