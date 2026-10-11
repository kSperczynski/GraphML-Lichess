import math
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _checked_correlation(
    target: np.ndarray,
    prediction: np.ndarray,
    method: str,
) -> float:
    if len(target) < 2:
        return float("nan")

    if np.std(target) == 0 or np.std(prediction) == 0:
        return float("nan")

    if method == "pearson":
        return float(np.corrcoef(target, prediction)[0, 1])

    if method == "spearman":
        return float(spearmanr(target, prediction).statistic)  # type: ignore

    raise ValueError(method)


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": math.sqrt(mean_squared_error(target, prediction)),
        "r2": float(r2_score(target, prediction)),
        "pearson": _checked_correlation(target, prediction, "pearson"),
        "spearman": _checked_correlation(target, prediction, "spearman"),
    }


@torch.no_grad()
def evaluate_mask(
    model: torch.nn.Module,
    data: Any,
    mask: np.ndarray,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()

    # forward pass
    predictions = model(
        data.x.to(device),
        data.edge_index.to(device),
        data.edge_attr.to(device) if data.edge_attr is not None else None,
    )

    # calc loss
    predictions = predictions.squeeze(-1)
    target = data.y.to(device).squeeze(-1)
    loss = loss_fn(predictions[mask], target[mask]).item()

    pred_np = predictions[mask].cpu().numpy()
    target_np = target[mask].cpu().numpy()

    return loss, regression_metrics(target_np, pred_np), target_np, pred_np
