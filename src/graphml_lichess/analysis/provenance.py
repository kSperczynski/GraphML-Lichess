"""Record input identity at training time, rather than guessing it later."""

import hashlib
import json
from pathlib import Path


def fingerprint(value: dict | list) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def capture_data_provenance(config: dict) -> dict:
    data = config["data"]
    return {
        "schema_version": 1,
        "nodes_sha256": hashlib.sha256(Path(data["nodes"]).read_bytes()).hexdigest(),
        "edges_sha256": hashlib.sha256(Path(data["edges"]).read_bytes()).hexdigest(),
        "feature_columns": list(data["feature_columns"]),
        "id_column": data["id_column"],
        "target_column": data["target_column"],
        "edge_feature_columns": list(data["edge_feature_columns"]),
        "preprocessing": "training_only_feature_standardization_v1",
        "target_units": "rating_points",
    }
