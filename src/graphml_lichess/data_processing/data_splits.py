from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Masks:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass
class NodeSplit:
    train: list[str]
    validation: list[str]
    test: list[str]
    meta: dict

    @property
    def fingerprint(self) -> str:
        payload = json.dumps([sorted(self.train), sorted(self.validation), sorted(self.test)])
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "meta": {**self.meta, "fingerprint": self.fingerprint},
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> NodeSplit:
        if not path.exists():
            raise FileNotFoundError(f"Split file {path} not found. Create it with: make-split --config <cfg>")
        d = json.loads(path.read_text())
        return cls(d["train"], d["validation"], d["test"], d["meta"])

    def to_masks(self, usernames) -> Masks:
        """Boolean masks aligned with the graph's node order. Fails loudly if the data changed."""
        names = np.asarray(usernames)
        listed = set(self.train) | set(self.validation) | set(self.test)
        if listed != set(names):
            raise ValueError(
                f"Split and graph disagree: {len(listed - set(names))} split users missing from graph, "
                f"{len(set(names) - listed)} graph users missing from split. Regenerate the split."
            )
        return Masks(np.isin(names, self.train), np.isin(names, self.validation), np.isin(names, self.test))


def make_split(
    usernames,
    targets,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool = True,
) -> NodeSplit:
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1")
    names = np.asarray(usernames)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate usernames in graph")

    strata = pd.qcut(np.asarray(targets), q=10, labels=False, duplicates="drop") if stratify else None
    train, rest, _, strata_rest = train_test_split(
        names,
        strata if strata is not None else np.zeros(len(names)),
        test_size=1 - train_ratio,
        stratify=strata,
        random_state=seed,
    )

    val, test = train_test_split(
        rest,
        test_size=test_ratio / (validation_ratio + test_ratio),
        stratify=strata_rest if stratify else None,
        random_state=seed,
    )

    meta = {
        "method": "random_stratified" if stratify else "random",
        "seed": seed,
        "ratios": [train_ratio, validation_ratio, test_ratio],
        "num_nodes": len(names),
    }
    return NodeSplit(train.tolist(), val.tolist(), test.tolist(), meta)
