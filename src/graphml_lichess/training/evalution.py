import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def _corr(fn, target: np.ndarray, prediction: np.ndarray) -> float:
    """Correlation, or NaN when it is undefined (too few points / constant input)."""
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return float("nan")
    return float(fn(target, prediction).statistic)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(root_mean_squared_error(target, prediction)),
        "r2": float(r2_score(target, prediction)),
        "pearson": _corr(pearsonr, target, prediction),
        "spearman": _corr(spearmanr, target, prediction),
    }


def evaluate_predictions(
    target: np.ndarray,
    prediction: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Metrics per split, from full-graph target and prediction arrays."""
    return {name: regression_metrics(target[m], prediction[m]) for name, m in masks.items()}
