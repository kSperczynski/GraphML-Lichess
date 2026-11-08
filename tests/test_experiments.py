import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from graphml_lichess.analysis.experiments import ExperimentCollection
from graphml_lichess.analysis.plots import plot_summary
from graphml_lichess.analysis.provenance import capture_data_provenance


def write_run(root, name, seed=42, split="split-a", rmse=2.0, alpha=None, zero=True, data="dataset-a", legacy=False):
    run = root / name
    run.mkdir(parents=True)
    config = {
        "name": name,
        "seed": seed,
        "model_type": "mlp" if alpha is None else "dirgnn",
        "data": {
            "nodes": "data/nodes.csv",
            "edges": "data/edges.csv",
            "id_column": "username",
            "target_column": "rating",
            "feature_columns": ["accuracy"],
            "edge_feature_columns": ["points"],
        },
        "model": {"dropout": 0.1},
        "training": {"epochs": 50, "loss": "mse"},
        "graph": {"add_zero_edges": zero},
        "split": {"seed": 21, "path": f"{split}.json"},
        "output": {"directory": "runs"},
    }
    if alpha is not None:
        config["model"]["dir_alpha"] = alpha
    provenance = {"nodes_sha256": data, "edges_sha256": "edges", "feature_columns": ["accuracy"]}
    meta = {"data_split_fingerprint": split, "seed": seed, "train_target_mean": 100.0}
    if not legacy:
        meta["data_provenance"] = provenance
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "run_info.json").write_text(json.dumps({"name": name, "model_seed": seed, "metadata": meta}))
    result = {"mae": rmse, "rmse": rmse, "r2": 0.5}
    (run / "metrics.json").write_text(json.dumps({"validation": result, "test": result}))
    pd.DataFrame({"username": ["a", "b"], "target": [90.0, 110.0], "prediction": [90 + rmse, 110 + rmse]}).to_csv(
        run / "predictions.csv", index=False
    )
    return run


def test_seed_names_do_not_split_identical_configs_and_nmse_is_mean_baseline_ratio(tmp_path):
    for seed, score in [(42, 1), (43, 2), (44, 3)]:
        write_run(tmp_path, f"mlp_seed_{seed}", seed=seed, rmse=score)
    collection = ExperimentCollection.from_roots([tmp_path, tmp_path])
    summary, _ = collection.summarize()
    test = summary.query('phase == "test"').iloc[0]
    assert test.n_units == 3 and test.rmse_mean == 2 and test.rmse_std == 1
    assert np.isnan(test.rmse_ci_low)
    assert test.nmse_mean == pytest.approx((1 + 4 + 9) / 300)


def test_different_data_hyperparameters_and_splits_are_not_pooled(tmp_path):
    write_run(tmp_path, "a", data="first")
    write_run(tmp_path, "b", data="second")
    run = write_run(tmp_path, "c", seed=43, data="first")
    config = yaml.safe_load((run / "config.yaml").read_text())
    config["model"]["dropout"] = 0.2
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    write_run(tmp_path, "d", data="first", split="another")
    summary, _ = ExperimentCollection.from_roots([tmp_path]).summarize()
    assert len(summary.query('phase == "test"')) == 4


def test_split_means_are_equally_weighted_not_pooled_training_seeds(tmp_path):
    write_run(tmp_path, "a", split="a", seed=0, rmse=1)
    for seed in range(3):
        write_run(tmp_path, f"b{seed}", split="b", seed=seed, rmse=3)
    write_run(tmp_path, "c", split="c", seed=0, rmse=5)
    summary, per_split = ExperimentCollection.from_roots([tmp_path]).summarize(unit="split")
    row = summary.query('phase == "test"').iloc[0]
    assert row.rmse_mean == 3
    assert row.rmse_std == 2
    assert row.n_units == 3 and row.n_training_seeds == 5
    assert row.rmse_ci_low < 3 < row.rmse_ci_high
    assert len(per_split.query('phase == "test"')) == 3


def test_ci_not_claimed_for_two_splits_or_one_training_seed(tmp_path):
    write_run(tmp_path, "a", split="a")
    write_run(tmp_path, "b", split="b")
    collection = ExperimentCollection.from_roots([tmp_path])
    summary, _ = collection.summarize("split")
    assert summary.rmse_ci_low.isna().all()
    training, _ = collection.summarize()
    assert training.rmse_std.isna().all()
    with pytest.raises(ValueError, match="Confidence intervals"):
        plot_summary(training, tmp_path, interval="ci95")


def test_duplicate_seeds_are_excluded_and_incomplete_runs_are_reported(tmp_path):
    write_run(tmp_path, "a", seed=1)
    write_run(tmp_path, "copy", seed=1)
    write_run(tmp_path, "good", seed=2)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "run_info.json").write_text("{}")
    collection = ExperimentCollection.from_roots([tmp_path])
    assert collection.runs.run_dir.nunique() == 1
    assert len([d for d in collection.diagnostics if d["status"] == "skipped"]) == 3


def test_metric_mismatch_is_rejected_without_affecting_good_runs(tmp_path):
    bad = write_run(tmp_path, "bad", seed=1)
    predictions = pd.read_csv(bad / "predictions.csv")
    predictions["prediction"] += 50
    predictions.to_csv(bad / "predictions.csv", index=False)
    write_run(tmp_path, "good", seed=2)
    collection = ExperimentCollection.from_roots([tmp_path])
    assert collection.runs.run_dir.nunique() == 1
    assert "disagrees" in collection.diagnostics[0]["reason"]


def test_inconsistent_player_cohort_for_same_split_is_refused(tmp_path):
    write_run(tmp_path, "a", seed=1)
    bad = write_run(tmp_path, "b", seed=2)
    predictions = pd.read_csv(bad / "predictions.csv")
    predictions.loc[0, "username"] = "different-player"
    predictions.to_csv(bad / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="Conflicting test cohorts"):
        ExperimentCollection.from_roots([tmp_path])


def test_legacy_identity_is_warned_and_unknown_nmse_stays_missing(tmp_path):
    run = write_run(tmp_path, "a", legacy=True)
    info = json.loads((run / "run_info.json").read_text())
    del info["metadata"]["train_target_mean"]
    (run / "run_info.json").write_text(json.dumps(info))
    collection = ExperimentCollection.from_roots([tmp_path])
    assert set(collection.runs.provenance_quality) == {"legacy_unverified"}
    assert collection.runs.nmse.isna().all()
    assert collection.diagnostics[0]["status"] == "warning"


def test_alpha_plot_and_read_only_export(tmp_path):
    roots = tmp_path / "input"
    for zero in [True, False]:
        for alpha in [0.0, 0.5, 1.0]:
            for seed in [1, 2]:
                run = write_run(roots, f"{zero}_{alpha}_{seed}", seed=seed, rmse=2 + alpha + seed, alpha=alpha, zero=zero)
                if not zero:
                    config = yaml.safe_load((run / "config.yaml").read_text())
                    config["model_type"] = "dirgnn_no_zero_edges"
                    config["graph"] = {"add_zero_edges": True, "remove_added_zero_edges": True}
                    (run / "config.yaml").write_text(yaml.safe_dump(config))
    before = {p: p.read_bytes() for p in roots.rglob("*") if p.is_file()}
    collection = ExperimentCollection.from_roots([roots])
    out = tmp_path / "analysis"
    collection.export(out)
    summary, _ = collection.summarize()
    paths = plot_summary(summary, out)
    assert len([p for p in paths if p.name.startswith("alpha_")]) == 2
    assert all(p.stat().st_size > 1000 for p in paths)
    assert before == {p: p.read_bytes() for p in roots.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        collection.export(out)


def test_ci_plot_on_three_distinct_splits(tmp_path):
    for i in range(3):
        write_run(tmp_path / "runs", f"split-{i}", split=f"split-{i}", rmse=i + 1)
    collection = ExperimentCollection.from_roots([tmp_path / "runs"])
    summary, _ = collection.summarize("split")
    output = collection.export(tmp_path / "ci", "split")
    assert len(plot_summary(summary, output, interval="ci95")) == 2


def test_provenance_changes_when_data_changes(tmp_path):
    nodes, edges = tmp_path / "nodes.csv", tmp_path / "edges.csv"
    nodes.write_text("player,accuracy\na,90\n")
    edges.write_text("from,to,points\na,a,0\n")
    config = {
        "data": {
            "nodes": str(nodes),
            "edges": str(edges),
            "feature_columns": ["accuracy"],
            "id_column": "player",
            "target_column": "rating",
            "edge_feature_columns": ["points"],
        }
    }
    first = capture_data_provenance(config)
    nodes.write_text("player,accuracy\na,91\n")
    second = capture_data_provenance(config)
    assert first["nodes_sha256"] != second["nodes_sha256"]
    assert first["edges_sha256"] == second["edges_sha256"]


def test_training_pipeline_records_provenance_and_normalized_error(tmp_path, monkeypatch):
    """Exercise the metadata additions through the real pipeline on toy data only."""
    from graphml_lichess.data_processing.data_splits import NodeSplit
    from graphml_lichess.training.pipeline import TrainingPipeline

    monkeypatch.chdir(tmp_path)
    names = [f"player-{i}" for i in range(30)]
    pd.DataFrame({"username": names, "accuracy": np.arange(30), "rating": 1000 + 4 * np.arange(30)}).to_csv(
        "nodes.csv", index=False
    )
    pd.DataFrame({"from": names[:-1], "to": names[1:], "points": 1.0}).to_csv("edges.csv", index=False)
    split = NodeSplit(names[:21], names[21:25], names[25:], {"data_split_seed": 21})
    split.save(Path("split.json"))
    config = {
        "name": "toy_metadata_check",
        "seed": 42,
        "model_type": "ridge",
        "model": {"alpha": 1.0},
        "data": {
            "nodes": "nodes.csv",
            "edges": "edges.csv",
            "id_column": "username",
            "target_column": "rating",
            "feature_columns": ["accuracy"],
            "edge_feature_columns": ["points"],
        },
        "graph": {"directed": True, "add_self_loops": False},
        "split": {"path": "split.json", "seed": 21},
        "output": {"directory": "runs"},
    }
    run = TrainingPipeline(config).run()
    info = json.loads((run / "run_info.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    assert info["metadata"]["data_provenance"]["nodes_sha256"]
    assert info["metadata"]["train_target_mean"] == pytest.approx(1040)
    for phase in ("train", "validation", "test"):
        assert metrics[phase]["nmse"] == pytest.approx(metrics[phase]["rmse"] ** 2 / metrics[phase]["mean_baseline_mse"])
    collection = ExperimentCollection.from_roots([Path("runs")])
    assert set(collection.runs.provenance_quality) == {"recorded"}
    assert collection.runs.nmse.notna().all()
