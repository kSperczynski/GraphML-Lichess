import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GraphSplit:
    train: np.ndarray
    test: np.ndarray
    validation: np.ndarray

    def masks(self, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)

        train_mask[self.train] = True
        test_mask[self.test] = True
        val_mask[self.validation] = True
        return train_mask, test_mask, val_mask

    def as_json(self) -> dict[str, list[int]]:
        return {
            "train": self.train.tolist(),
            "test": self.test.tolist(),
            "validation": self.validation.tolist(),
        }


def random_node_split(
    num_nodes: int,
    train_ratio: float,
    test_ratio: float,
    seed: int,
) -> GraphSplit:
    """Randomly split the data, we provide the train and test ratio, the validation ration is 1 - (train_ration + test_ratio)"""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 < test_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1).")
    if train_ratio + test_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be < 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_nodes)

    train_end = int(num_nodes * train_ratio)
    test_end = train_end + int(num_nodes * test_ratio)

    return GraphSplit(
        train=indices[:train_end],
        test=indices[train_end:test_end],
        validation=indices[test_end:],
    )


def save_split(split: GraphSplit, path: Path) -> None:
    path.write_text(
        json.dumps(split.as_json(), indent=4),
        encoding="utf-8",
    )


def load_split(path: Path) -> GraphSplit:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GraphSplit(
        train=np.asarray(data["train"], dtype=np.int64),
        test=np.asarray(data["test"], dtype=np.int64),
        validation=np.asarray(data["validation"], dtype=np.int64),
    )
