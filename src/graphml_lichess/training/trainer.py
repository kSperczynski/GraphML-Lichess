from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_score: float
    history: list[dict[str, float]]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int,
        patience: int,
        evaluator: Callable[..., tuple[float, dict[str, float], np.ndarray, np.ndarray]],
        gradient_clip_norm: float | None,
        logger: logging.Logger,
        trial: Any | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.evaluator = evaluator
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger
        self.trial = trial

    def fit(
        self,
        data: Any,
        train_mask: np.ndarray,
        validation_mask: np.ndarray,
        checkpoint_path: str | None = None,
    ) -> TrainingResult:
        data = data.to(self.device)
        best_val_score = float("inf")
        best_epoch = 0
        wait = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            prediction = self.model(
                data.x,
                data.edge_index,
                data.edge_attr if data.edge_attr is not None else None,
            )
            train_loss = self.loss_fn(
                prediction.squeeze(-1)[train_mask],
                data.y[train_mask],
            )
            train_loss.backward()

            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

            self.optimizer.step()

            val_score, val_metrics, _, _ = self.evaluator(self.model, data, validation_mask, self.device, self.loss_fn)

            row = {"epoch": float(epoch), "train_loss": float(train_loss.item()), **val_metrics}
            history.append(row)

            if epoch == 1 or epoch % 10 == 0:
                self.logger.info(
                    "Epoch %03d | train_loss=%.4f | val_mae=%.4f | val_rmse=%.4f",
                    epoch,
                    train_loss.item(),
                    val_metrics["mae"],
                    val_metrics["rmse"],
                )

            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                wait = 0

                if checkpoint_path:
                    torch.save(self.model.state_dict(), checkpoint_path)
            else:
                wait += 1

            if wait >= self.patience:
                self.logger.info(
                    "Early stopping at epoch %d; best epoch=%d.",
                    epoch,
                    best_epoch,
                )
                break

        return TrainingResult(
            best_epoch=best_epoch,
            best_validation_score=best_val_score,
            history=history,
        )
