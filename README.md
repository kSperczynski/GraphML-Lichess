# DSAIT4305 - Graph Machine Learning - Group 7

## Setup

- To activate pre-commit, run `uv run pre-commit install` once.
- To prepare your `.venv` run: `uv sync`

## To-do

- Create experiments notebook
- Run experiments

## Shared result analysis

`graphml-lichess analyze --runs runs/training --output runs/analysis/first`
reads existing run artifacts, exports per-run/per-split summaries and creates
model or alpha/zero-edge plots. It does not start training. See
[the analysis guide](docs/experiment_analysis.md) for uncertainty definitions,
legacy provenance warnings, notebook use and split-level intervals.

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

## MLP baseline

The registered `mlp` model predicts ratings from each player's own features and
ignores graph edges. Input width follows the shared feature list. The initial
network uses hidden widths 64/32, ReLU, dropout 0.1 and one linear output.

Run from the repository root, after preparing the environment and the agreed
split. If that split is absent, create it once with `make-split`; do not use
`--force` to replace an existing shared split just to add the MLP.

```bash
graphml-lichess make-split --config config/base.yaml
graphml-lichess train --config config/models/mlp.yaml --seeds 42 43 44
graphml-lichess tune --config config/models/mlp.yaml --search config/search/mlp.yaml
graphml-lichess train --config config/tuned/mlp.yaml --seeds 42 43 44
```

MLP training uses the shared AdamW loop, MSE on raw ratings, and validation RMSE
for checkpoint selection. Feature scaling is fitted on training players only.
RMSE and MSE select the same checkpoint on a fixed validation set. Predictions
and metrics stay in rating units. The search covers dropout, learning rate and
weight decay; hidden widths are fixed. Freeze the experimental configuration
before final test evaluation.

This integration does not change DirGNN, GAT, Ridge, data cleaning or the shared
split defaults. The current cleaned data contains 7,181 players, so the previous
4,494-player MLP tuning results must not be presented as results on this data.
No old tuned configuration is carried over. The initial 200-epoch MLP config is
an integration starting point, not a validated optimal configuration.
