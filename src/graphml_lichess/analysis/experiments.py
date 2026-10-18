"""Conservative grouping of model runs, training seeds and data splits."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from graphml_lichess.analysis.provenance import fingerprint

METRICS = ("mae", "rmse", "mse", "nmse", "r2")
GROUP_KEYS = ["data_id", "protocol_id", "config_id", "model_type", "sweep_id", "alpha", "zero_edges"]


def _repository_scope(run: Path) -> str:
    for parent in run.parents:
        if (parent / ".git").exists():
            return str(parent)
    return str(run.parent)


def _configuration(config: dict) -> tuple[dict, dict, dict, bool]:
    model = copy.deepcopy(config.get("model", {}))
    model.pop("device", None)
    graph = copy.deepcopy(config.get("graph", {}))
    graph.pop("add_zero_egdes", None)  # The loader does not read the misspelled historical key.
    for name, default in [
        ("directed", True),
        ("add_self_loops", True),
        ("add_zero_edges", False),
        ("remove_added_zero_edges", False),
        ("replace_edge_features_with_attention", False),
    ]:
        graph.setdefault(name, default)
    zero_edges = bool(graph["add_zero_edges"] and not graph["remove_added_zero_edges"])
    variant = {"model_type": config["model_type"], "model": model, "training": config.get("training", {}), "graph": graph}
    sweep = copy.deepcopy(variant)
    sweep["model"].pop("dir_alpha", None)
    if sweep["model_type"] == "dirgnn_no_zero_edges":
        sweep["model_type"] = "dirgnn"
    sweep["graph"].pop("add_zero_edges", None)
    sweep["graph"].pop("remove_added_zero_edges", None)
    protocol = {
        key: config["data"].get(key) for key in ("id_column", "target_column", "feature_columns", "edge_feature_columns")
    }
    return variant, sweep, protocol, zero_edges


def _read_run(run: Path) -> tuple[list[dict], list[str]]:
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "base" in config:
        raise ValueError("Expected a resolved config snapshot, not a layered input config.")
    if info.get("model_type", config["model_type"]) != config["model_type"]:
        raise ValueError("Run metadata and configuration disagree on model type.")
    variant, sweep, protocol, zero_edges = _configuration(config)
    meta = info.get("metadata", {})
    split = info.get("data_split_fingerprint") or meta.get("data_split_fingerprint")
    if not split:
        raise ValueError("Missing split fingerprint; cannot safely identify repeated runs.")
    seed = info.get("model_seed", info.get("seed", meta.get("seed", config.get("seed"))))
    if seed is None:
        raise ValueError("Missing model seed.")
    notices = []
    provenance = meta.get("data_provenance")
    if provenance and provenance.get("nodes_sha256") and provenance.get("edges_sha256"):
        data_id = fingerprint(provenance)
        quality = "recorded"
        if provenance.get("feature_columns") != protocol["feature_columns"]:
            raise ValueError("Recorded feature schema disagrees with config.")
    else:
        # Existing runs cannot prove which historical CSV contents were loaded.
        data_id = "legacy-" + fingerprint({"scope": _repository_scope(run), "data": config["data"]})
        quality = "legacy_unverified"
        notices.append("Legacy run has no training-time data hashes; grouping is provisional within its source repository.")
    if config.get("graph", {}).get("add_zero_egdes"):
        notices.append("Misspelled add_zero_egdes is ignored by the training loader; verify the intended graph.")
    predictions = None
    cohort = "unavailable"
    prediction_path = run / "predictions.csv"
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path, dtype={"username": str})
        required = {"username", "target", "prediction"}
        if not required <= set(predictions.columns):
            raise ValueError("Prediction CSV lacks username, target or prediction.")
        if predictions["username"].isna().any() or predictions["username"].duplicated().any():
            raise ValueError("Prediction usernames must be unique and nonmissing.")
        values = predictions[["target", "prediction"]].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Predictions or targets are empty or nonfinite.")
        pairs = predictions.sort_values("username")[["username", "target"]].values.tolist()
        cohort = fingerprint(pairs)
        errors = values[:, 1] - values[:, 0]
        for key, computed in {"mae": np.abs(errors).mean(), "rmse": np.sqrt(np.mean(errors**2))}.items():
            if not np.isclose(computed, metrics["test"][key], rtol=1e-5, atol=1e-3):
                raise ValueError(f"Saved test {key} disagrees with predictions.")
    else:
        notices.append("No predictions.csv; test cohort identity cannot be checked.")
    rows = []
    for phase in ("train", "validation", "test"):
        if phase not in metrics:
            continue
        result = metrics[phase]
        if not all(np.isfinite(float(result[key])) for key in ("mae", "rmse")):
            raise ValueError(f"Nonfinite primary metrics in {phase}.")
        mse = float(result["rmse"]) ** 2
        nmse = result.get("nmse", np.nan)
        if "mean_baseline_mse" in result:
            denominator = float(result["mean_baseline_mse"])
            nmse = mse / denominator if denominator > 0 else np.nan
        elif phase == "test" and predictions is not None and "train_target_mean" in meta:
            denominator = float(np.mean((predictions["target"] - float(meta["train_target_mean"])) ** 2))
            nmse = mse / denominator if denominator > 0 else np.nan
        rows.append(
            {
                "run_dir": str(run),
                "run_name": info.get("name", config.get("name")),
                "model_type": config["model_type"],
                "model_seed": int(seed),
                "split_id": str(split),
                "data_split_seed": config.get("split", {}).get("seed"),
                "phase": phase,
                "data_id": data_id,
                "provenance_quality": quality,
                "protocol_id": fingerprint(protocol),
                "config_id": fingerprint(variant),
                "sweep_id": fingerprint(sweep),
                "alpha": config.get("model", {}).get("dir_alpha", np.nan),
                "zero_edges": zero_edges,
                "test_cohort_id": cohort,
                "n_test": len(predictions) if predictions is not None else np.nan,
                "mae": float(result["mae"]),
                "rmse": float(result["rmse"]),
                "mse": mse,
                "nmse": float(nmse),
                "r2": float(result.get("r2", np.nan)),
            }
        )
    if not rows:
        raise ValueError("No recognized split metrics.")
    return rows, notices


class ExperimentCollection:
    """Collect artifact snapshots without loading executable checkpoints."""

    def __init__(self, runs: pd.DataFrame, diagnostics: list[dict]):
        self.runs = runs
        self.diagnostics = diagnostics

    @classmethod
    def from_roots(cls, roots: list[Path | str]) -> ExperimentCollection:
        candidates = set()
        for root in roots:
            path = Path(root).resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Run directory not found: {path}")
            candidates.update(p.parent for p in path.rglob("run_info.json"))
        rows, diagnostics = [], []
        for run in sorted(candidates):
            try:
                records, notices = _read_run(run)
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
                diagnostics.append({"run_dir": str(run), "status": "skipped", "reason": str(error)})
                continue
            rows.extend(records)
            diagnostics.extend({"run_dir": str(run), "status": "warning", "reason": notice} for notice in notices)
        if not rows:
            raise ValueError(f"No readable completed runs. Diagnostics: {diagnostics}")
        frame = pd.DataFrame(rows)
        # Repeated copies of a seed are not independent observations.
        keys = ["data_id", "protocol_id", "config_id", "split_id", "model_seed", "phase"]
        duplicate = frame.duplicated(keys, keep=False)
        for run in frame.loc[duplicate, "run_dir"].unique():
            diagnostics.append(
                {
                    "run_dir": run,
                    "status": "skipped",
                    "reason": "Duplicate config/split/model-seed; excluded all ambiguous copies.",
                }
            )
        frame = frame[~duplicate].copy()
        # A claimed split must refer to the same held-out usernames and targets.
        identity_keys = ["data_id", "protocol_id", "split_id"]
        conflicts = frame.groupby(identity_keys, dropna=False)["test_cohort_id"].nunique()
        for key in conflicts[conflicts > 1].index:
            raise ValueError(f"Conflicting test cohorts for the same split: {key}")
        if frame.empty:
            raise ValueError("All runs were ambiguous duplicate copies.")
        return cls(frame.reset_index(drop=True), diagnostics)

    def summarize(self, unit: str = "training", confidence: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate seeds within each split before optionally aggregating split means."""
        if unit not in {"training", "split"}:
            raise ValueError("Unit must be training or split.")
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1.")
        per_split = []
        keys = GROUP_KEYS + ["phase", "split_id"]
        for values, group in self.runs.groupby(keys, dropna=False, sort=False):
            row = dict(zip(keys, values))
            row.update(n_training_seeds=len(group), provenance_quality=group["provenance_quality"].iloc[0])
            for metric in METRICS:
                scores = group[metric].dropna()
                row[f"{metric}_mean"] = scores.mean()
                row[f"{metric}_std"] = scores.std(ddof=1)
            per_split.append(row)
        per_split = pd.DataFrame(per_split)
        if unit == "training":
            summary = per_split.copy()
            summary["unit"] = "training_seed"
            summary["confidence"] = np.nan
            summary["n_units"] = summary["n_training_seeds"]
            summary["n_splits"] = 1
            summary["split_set_id"] = summary["split_id"].map(lambda value: fingerprint([value]))
            for metric in METRICS:
                summary[f"{metric}_ci_low"] = np.nan
                summary[f"{metric}_ci_high"] = np.nan
            return summary, per_split
        rows = []
        for values, group in per_split.groupby(GROUP_KEYS + ["phase"], dropna=False, sort=False):
            row = dict(zip(GROUP_KEYS + ["phase"], values))
            row.update(
                unit="split_mean",
                confidence=confidence,
                n_units=len(group),
                n_splits=len(group),
                split_set_id=fingerprint(sorted(group["split_id"].tolist())),
                n_training_seeds=int(group["n_training_seeds"].sum()),
                split_id="multiple" if len(group) > 1 else group["split_id"].iloc[0],
                provenance_quality=group["provenance_quality"].iloc[0],
            )
            for metric in METRICS:
                scores = group[f"{metric}_mean"].dropna()
                mean, std, n = scores.mean(), scores.std(ddof=1), len(scores)
                margin = t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n) if n >= 3 else np.nan
                row.update(
                    {
                        f"{metric}_mean": mean,
                        f"{metric}_std": std,
                        f"{metric}_ci_low": mean - margin,
                        f"{metric}_ci_high": mean + margin,
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows), per_split

    def export(self, destination: Path | str, unit: str = "training", confidence: float = 0.95) -> Path:
        """Export tables to a new directory; never overwrite existing analysis."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        summary, per_split = self.summarize(unit, confidence)
        self.runs.to_csv(destination / "runs.csv", index=False)
        per_split.to_csv(destination / "per_split.csv", index=False)
        summary.to_csv(destination / "summary.csv", index=False)
        (destination / "diagnostics.json").write_text(json.dumps(self.diagnostics, indent=2), encoding="utf-8")
        report = {
            "unit": unit,
            "confidence": confidence,
            "n_runs": int(self.runs["run_dir"].nunique()),
            "legacy_runs": int(self.runs.loc[self.runs.provenance_quality == "legacy_unverified", "run_dir"].nunique()),
            "ci_method": "Student-t interval of equally weighted split means; minimum 3 splits; descriptive only.",
            "caveats": [
                "Overlapping random splits are not guaranteed independent; intervals are not a significance test.",
                "Training-seed summaries describe optimization variation only and do not include confidence intervals.",
                "Legacy data grouping is unverified; historical input hashes cannot be inferred from today's files.",
                "NMSE uses a training-mean constant predictor on the same evaluation players; missing provenance stays NaN.",
            ],
        }
        (destination / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return destination
