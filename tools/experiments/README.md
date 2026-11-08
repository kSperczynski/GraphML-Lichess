# Experiment analysis tools

Run from a prepared project environment. No machine-specific Python or dependency
paths are embedded in the launcher. Install the normal project dependencies first.
All relative run/output paths in this launcher refer to the repository root.

```powershell
.\tools\experiments\analyze.ps1 -Runs runs/training -Output runs/analysis/first
```

Use `-Python 'C:\path\to\.venv\Scripts\python.exe'` to choose an interpreter.
Use `-NoPlots` to export tables without importing matplotlib.

Read a selected set of runs without copying or changing them:

```powershell
$selectedRuns = Get-ChildItem runs/training -Directory -Filter 'mlp_convergence_5000_*' |
    Select-Object -ExpandProperty FullName
.\tools\experiments\analyze.ps1 -Runs $selectedRuns -Output runs/analysis/mlp_selected
```

For different data splits, average seeds within each split and request a
descriptive 95% interval across at least three splits:

```powershell
.\tools\experiments\analyze.ps1 -Runs runs/training -Output runs/analysis/split_summary -Unit split -Interval ci95
```

Three training seeds on one player split cannot supply a split-level interval.
Use a new output folder each time; existing outputs are never overwritten.

The implementation is in `src/graphml_lichess/analysis/`, shared by this launcher,
the `analyze` CLI subcommand and `notebooks/analyze_experiments.ipynb`.
See [the full guide](../../docs/experiment_analysis.md) for input requirements,
output columns, grouping rules and legacy-data limitations.
