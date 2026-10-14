import copy
import json
import logging
import os
from pathlib import Path

import optuna
import torch
import yaml

from graphml_lichess.config import set_path
from graphml_lichess.training.pipeline import build_regressor, prepare_data
from graphml_lichess.utils.seeding import seed_everything


def _suggest(trial: optuna.Trial, key: str, spec: dict):
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "int":
        return trial.suggest_int(key, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(key, spec["choices"])
    raise ValueError(f"Unknown search type '{kind}' for '{key}'")


def tune(
    config: dict,
    search: dict,
    config_path: Path,
    logger: logging.Logger,
    n_trials: int | None = None,
    tuned_dir: Path = Path("configs/tuned"),
    work_dir: Path = Path("runs/tuning"),
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, masks, _, split = prepare_data(config)  # once, shared by every trial
    name = config["name"]
    n_trials = n_trials or int(search["n_trials"])

    work_dir = work_dir / name
    work_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(config)
        for key, spec in search["params"].items():
            set_path(cfg, key, _suggest(trial, key, spec))
        seed_everything(int(cfg["seed"]))  # same seed every trial, so differences come from params
        regressor = build_regressor(cfg, graph.data, device, logger)

        def on_epoch(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return regressor.fit(graph.data, masks, on_epoch=on_epoch).best_validation_score

    study = optuna.create_study(
        study_name=f"{name}-{split.fingerprint}",  # new split -> new study
        storage=f"sqlite:///{(work_dir / 'study.db').as_posix()}",
        load_if_exists=True,  # resume after a crash or Ctrl-C
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(config["seed"])),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=20),
    )
    remaining = max(0, n_trials - len(study.trials))
    logger.info("Study %s: %d trials done, running %d more", study.study_name, len(study.trials), remaining)
    study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    # Write the tuned config: inherits the model config, overrides only the winners
    best: dict = {}
    for key, value in study.best_params.items():
        set_path(best, key, value)
    tuned_dir.mkdir(parents=True, exist_ok=True)
    tuned_path = tuned_dir / f"{config_path.stem}.yaml"
    tuned = {
        "base": Path(os.path.relpath(config_path.resolve(), tuned_path.parent.resolve())).as_posix(),
        "name": f"{name}_tuned",
        **best,
    }
    tuned_path.write_text(yaml.safe_dump(tuned, sort_keys=False))

    study.trials_dataframe().to_csv(work_dir / "trials.csv", index=False)
    (work_dir / "best.json").write_text(
        json.dumps(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "split": split.fingerprint,
            },
            indent=2,
        )
    )
    logger.info("Best val %s=%.4f -> %s", config["training"].get("monitor", "mae"), study.best_value, tuned_path)
    return tuned_path
