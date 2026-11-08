# Shared experiment analysis

This tool reads saved artifacts; it never launches training, picks a winning seed,
overwrites runs, or loads model/checkpoint pickle files. Use it from the repository
root after preparing the normal project environment.

## Quick start

```bash
graphml-lichess analyze --runs runs/training --output runs/analysis/first --metric rmse
```

Standalone entry points are also available:

```bash
graphml-experiments --runs runs/training --output runs/analysis/first --metric rmse
python -m graphml_lichess.analysis --runs runs/training --output runs/analysis/first --metric rmse
```

The output path must be new. Multiple directories and nested run folders are
supported. Each completed run needs `run_info.json`, `metrics.json`, and the fully
resolved `config.yaml` snapshot. `predictions.csv` is used to check test cohorts
and independently verify reported MAE/RMSE. Incomplete or malformed runs are
listed in `diagnostics.json`; they are not silently replaced with zero scores.

Output files:

- `runs.csv`: one row per run and evaluation phase, with IDs, seeds and metrics.
- `per_split.csv`: training seeds averaged within each configuration and split.
- `summary.csv`: the chosen level of aggregation and uncertainty.
- `diagnostics.json`: rejected duplicates, malformed artifacts and legacy warnings.
- `analysis.json`: units, interval definition, provenance status and caveats.
- `models_*.png/svg`: model variants with error bars, in separate plots for
  incompatible data/protocol/split sets.
- `alpha_*.png/svg`: alpha sweeps, comparing retained and removed zero edges,
  when matching experiment configurations are present.

Model names may differ between seeds; names are display labels, not grouping keys.
Grouping uses model parameters, training settings, graph settings, input protocol,
data identity and split fingerprints. Different learning rates, epoch budgets,
dropouts, GAT/GATv2 settings and graph variants are never averaged as seed repeats.
Alpha plots intentionally vary only `model.dir_alpha` and the zero-edge settings;
other changes create separate sweeps. DirGNN's two registered zero-edge variants
are recognized as one sweep family, while keeping their individual config IDs.

## Training seeds and data splits

Default `--unit training --interval sd` keeps every split separate. Its SD is
variation across model seeds on that fixed split, not sampling uncertainty.
Single runs retain missing SD; a missing error bar does not mean zero uncertainty.

To summarize genuinely different player splits:

```bash
graphml-lichess analyze --runs runs/training --output runs/analysis/splits --unit split --interval ci95
```

The tool first averages model seeds inside each split, then weights split means
equally. It reports a descriptive Student-t interval of those means, requiring at
least three split fingerprints for CI plots. Unequal seed counts do not give one
split greater weight. Three training seeds on ONE split never yield a split-level
confidence interval. `--unit split --interval sd` can be used without CI plots.

With three splits, intervals can be very wide. Random splits of the same dataset
overlap and are not guaranteed independent: these intervals are descriptive, not
a significance test or proof that one model is superior. Agree on the split
protocol and unit of inference before presenting confidence intervals in a paper.
Plots also separate different sets of split IDs; one model's three splits are not
silently compared as if identical to another model's different three splits.

Duplicated config/split/seed records are ambiguous reruns or copied artifacts,
not new observations. ALL such copies are excluded and diagnosed. Exact overlapping
input root paths are deduplicated. The tool does not choose the best duplicate.

## Input identity and legacy runs

New training runs now record SHA-256 hashes of node and edge inputs plus feature
schema, preprocessing identity and rating units under `metadata.data_provenance`.
This only adds metadata; model fitting, checkpoint selection and predictions are
unchanged. Data, shared split defaults and other model architectures are untouched.

Old runs did not record historical CSV hashes. They can still be read but are
marked `legacy_unverified`; grouping is provisional and scoped to their source
repository and recorded data paths. Today's CSV contents cannot prove what an old
run loaded. Model/feature/split checks do not replace a full provenance audit.
Conflicting test usernames or targets for the same claimed split are refused.
Without predictions.csv, cohort identity cannot be verified and a warning is saved.

## Metrics and normalization

`--metric` supports MAE, RMSE, MSE, NMSE and R2. `--phase` selects train, validation
or test. MAE/RMSE are rating points; MSE is rating points squared. MSE is derived
from RMSE squared. Each phase is summarized independently.

For new training outputs:

`NMSE = model MSE / MSE of the training-mean constant predictor on the same evaluation players`.

NMSE below 1 beats that baseline. This is explicitly a baseline-relative NMSE,
not an unspecified variance normalization. Training artifacts save the denominator
as `mean_baseline_mse`. Legacy test runs can use the recorded training target mean
and prediction CSV if available. Otherwise NMSE remains missing; it is never
reconstructed from test-set mean or assumed equal to `1 - R2`.

## Notebook / Python use

```python
from pathlib import Path
from graphml_lichess.analysis import ExperimentCollection
from graphml_lichess.analysis.plots import plot_summary

experiments = ExperimentCollection.from_roots([Path("runs/training")])
summary, per_split = experiments.summarize(unit="training")
print(summary[["model_type", "split_id", "n_units", "rmse_mean", "rmse_std"]])
destination = experiments.export(Path("runs/analysis/notebook_example"))
plot_summary(summary, destination, metric="rmse", phase="test", interval="sd")
```

The accompanying `notebooks/analyze_experiments.ipynb` is an unexecuted template.
Set its roots and new destination before running. It includes no simulated scores.

## Limits of this first version

- No automatic experiment scheduler, no automatic tuning, no test-based selection.
- No statistical significance claim, no bootstrap of correlated training runs.
- No arbitrary loading of notebook image outputs as if they were full run records.
- Outlier-group comparison is a future extension once a shared username-to-group
  definition is available. Current tables retain paths to per-player predictions.
- Karol's notebook contains saved alpha summaries, but the raw `runs/` files are
  not checked into the shared repository. Obtain those run folders to analyze them
  with full config/split checks; do not infer missing metadata from a screenshot.
