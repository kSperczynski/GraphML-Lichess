# DSAIT4305 - Graph Machine Learning - Group 7

## Setup

- To activate pre-commit, run `uv run pre-commit install` once.
- To prepare your `.venv` run: `uv sync`

## To-do

- Create experiments notebook
- Run experiments

## Models

I split `GAT` up into two models (and configs etc.). One is using GATv2, and the other the original GAT. Since this changes the actual architecture I think we can view it as more than a hyperparameter.

## Configs

A model is defined by a YAML file in `configs/models/` that inherits the shared settings (data, graph, split, output) from `configs/base.yaml` through its `base:` key.
To add a model, write a new file there, register a builder for it in `models/model_factory.py`, and use its registry key as `model_type`. Any config value can be overridden from the command line with `--set`, for example `--set training.epochs=50`. Use `--set` as the last argument, because it consumes everything after it. Give every variant a distinct `name`, as the run folder and the report group runs by that name.

## Commands

You need to create a data split before you use it in a training / tuning run. If you want to use another one, update the name in the `base config` file. If you don't, you'll get an error that the data split already exists. You can overwrite it with `--force` if you really want, but keep in mind that this invalidates all comparisons with runs that used the split before you overwrite it.

```bash
graphml-lichess make-split --config config/base.yaml
graphml-lichess tune   --config config/models/torch_GAT.yaml
graphml-lichess train  --config config/tuned/torch_GAT.yaml --seeds 0 1 2 3 4
graphml-lichess report
```

`tune` runs an Optuna search over the ranges in `configs/search/<model>.yaml` and writes the best settings to `configs/tuned/`. `train` fits the model and stores metrics, predictions and the model in a new folder under `runs/`. `report` aggregates all finished runs into one table with mean and standard deviation per model.

For more options to pass to the CLI, look at the `src/graphml_lichess/cli.py` file.

The logs of a tuning run should automatically be stored at `runs/tuning`, a training run should automatically be stored at `runs/training`.

## Packages

Most is handled via `uv`. However, `uv` might be unable to resolve the `pyg-lib` dependency needed for the Dominant model in the PyGOD package. You can install it locally with:
```
uv pip install pyg-lib --find-links "https://data.pyg.org/whl/torch-2.14.0+cpu.html"
```