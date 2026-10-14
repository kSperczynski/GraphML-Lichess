from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import torch
from torch import nn

from graphml_lichess.data_processing.data_splits import Masks
from graphml_lichess.training.evalution import regression_metrics


@dataclass
class TrainingResult:
    """Dataclass to store the best result so far"""

    best_epoch: int
    best_validation_score: float
    history: list[dict[str, float]]


@dataclass
class TrainConfig:
    """Class to store the training config"""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200
    patience: int = 20
    gradient_clip_norm: float | None = None
    monitor: str = "mae"
    loss: str = "huber"
    huber_delta: float = 1.0

    def __post_init__(self):
        """Called to make sure that things like 1e-3 get converted to float"""
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**d)

    def build_loss(self) -> nn.Module:
        if self.loss == "huber":
            return nn.HuberLoss(delta=float(self.huber_delta))
        if self.loss == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss: {self.loss}")


class Regressor(Protocol):
    """Protocol to unify the Deep learning models and others"""

    def fit(self, data: Any, masks: Masks, on_epoch: Callable[[int, float], None] | None = None) -> TrainingResult: ...
    def predict(self, data: Any) -> np.ndarray: ...  # predictions for all nodes
    def save(self, path: Path) -> None: ...


def _targets(data) -> np.ndarray:
    return data.y.squeeze(-1).cpu().numpy()


class TorchRegressor:
    """Regressor for deep learning models, uses a loop with epochs for training"""

    def __init__(self, model: nn.Module, cfg: TrainConfig, device: torch.device, logger: logging.Logger):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.loss_fn = cfg.build_loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _forward(self, data) -> torch.Tensor:
        return self.model(data.x, data.edge_index, data.edge_attr).squeeze(-1)

    def _train_step(self, data, train_mask: torch.Tensor) -> float:
        """Execute one training step. One forward and one backward pass"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass
        loss = self.loss_fn(self._forward(data)[train_mask], data.y.squeeze(-1)[train_mask])

        # Backward pass
        loss.backward()
        if self.cfg.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_norm)
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, data) -> np.ndarray:
        self.model.eval()
        return self._forward(data.to(self.device)).cpu().numpy()

    def fit(self, data, masks: Masks, on_epoch: Callable[[int, float], None] | None = None) -> TrainingResult:
        """Train the model"""
        data = data.to(self.device)
        y = _targets(data)
        train_mask = torch.as_tensor(masks.train, device=self.device)

        best_score, best_epoch, best_state, wait = float("inf"), 0, None, 0
        history: list[dict[str, float]] = []

        # Training loop
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_step(data, train_mask)
            val = regression_metrics(y[masks.validation], self.predict(data)[masks.validation])
            history.append({"epoch": float(epoch), "train_loss": train_loss, **val})

            # Every 10 epochs, we log the current performance
            if epoch == 1 or epoch % 10 == 0:
                self.logger.info(
                    "Epoch %03d | train_loss=%.4f | val_mae=%.4f | val_rmse=%.4f", epoch, train_loss, val["mae"], val["rmse"]
                )

            score = val[self.cfg.monitor]

            # If a callback is provided, we call it here
            if on_epoch is not None:
                on_epoch(epoch, score)

            if score < best_score:
                best_score, best_epoch, wait = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    self.logger.info("Early stopping at epoch %d; best epoch=%d.", epoch, best_epoch)
                    break

        # Leave the model in its best state, so no need to reload a checkpoint afterwards
        assert best_state is not None
        self.model.load_state_dict(best_state)
        return TrainingResult(best_epoch, best_score, history)

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)


class SklearnRegressor:
    """Regressor class for models that don't need a training loop"""

    def __init__(self, estimator, monitor: str = "mae"):
        self.estimator = estimator
        self.monitor = monitor

    def fit(self, data, masks: Masks, on_epoch: Callable[[int, float], None] | None = None) -> TrainingResult:
        X, y = data.x.cpu().numpy(), _targets(data)
        self.estimator.fit(X[masks.train], y[masks.train])
        val = regression_metrics(y[masks.validation], self.predict(data)[masks.validation])
        return TrainingResult(1, val[self.monitor], [{"epoch": 1.0, **val}])

    def predict(self, data) -> np.ndarray:
        return self.estimator.predict(data.x.cpu().numpy())

    def save(self, path: Path) -> None:
        joblib.dump(self.estimator, path)
