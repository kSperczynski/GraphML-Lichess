from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """Standardize node features using training nodes"""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, x: np.ndarray) -> FeaturePreprocessor:
        self.scaler.fit(x)
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted first.")
        return self.scaler.transform(x).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> FeaturePreprocessor:
        return joblib.load(path)
