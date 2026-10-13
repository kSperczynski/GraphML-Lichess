# DSAIT4305 - Graph Machine Learning - Group 7

## Members

## Pre-commit

To activate pre-commit, run `uv run pre-commit install` once.

## To-do

- Handle the `inf` values in `avg_move_time` feature of the nodes
- Implement `CustomGATRegressor` in `src/graphml_lichess/models/gat.py` with a custom architecture that works well for our use case
- Add CLI option to load data split from files, instead of randomly splitting it
    - Make data split so we don't leak data. For example remove game n-2 but keep the latest game in training data.

## Run the model

Example on how to run the default torch_geometric GAT model:

```bash
uv run src/graphml_lichess/cli.py train -config config/torch_gat.yaml
```
