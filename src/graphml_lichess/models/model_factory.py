import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch_geometric.nn.models import GAT

from graphml_lichess.models.custom_gat import CustomGATRegressor
from graphml_lichess.models.dirgnn import AnalyticalAttentionNetwork
from graphml_lichess.training.regressors import Regressor, SklearnRegressor, TorchRegressor, TrainConfig

MODEL_REGISTRY: dict[str, Callable[..., Regressor]] = {}


@dataclass
class BuildContext:
    train_cfg: TrainConfig
    device: torch.device
    logger: logging.Logger

    def wrap_torch(self, model: nn.Module) -> TorchRegressor:
        return TorchRegressor(model, self.train_cfg, self.device, self.logger)


def register(name: str):
    """Decorator method to easily add models"""

    def deco(fn):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' registered twice")
        MODEL_REGISTRY[name] = fn
        return fn

    return deco


@register("torch_gat")
def _torch_gat(input_dim, edge_dim, cfg, ctx):
    model = GAT(in_channels=input_dim, **cfg)
    return ctx.wrap_torch(model)


@register("custom_gat")
def _custom_gat(input_dim, edge_dim, cfg, ctx):
    return ctx.wrap_torch(CustomGATRegressor(**cfg))


@register("ridge")
def _ridge(input_dim, edge_dim, cfg, ctx):
    return SklearnRegressor(Ridge(**cfg))


@register("dirgnn")
def _dirgnn(input_dim, edge_dim, cfg, ctx):
    model = AnalyticalAttentionNetwork(cfg)
    dirgnn = TorchRegressor(model, ctx.train_cfg, ctx.device, ctx.logger)
    dirgnn.optimizer = torch.optim.SGD(
        model.parameters(), lr=ctx.train_cfg.learning_rate, weight_decay=ctx.train_cfg.weight_decay, momentum=0.9
    )
    return dirgnn


@register("dirgnn_no_zero_edges")
def _dirgnn_no_zero_edges(input_dim, edge_dim, cfg, ctx):
    model = AnalyticalAttentionNetwork(cfg)
    dirgnn = TorchRegressor(model, ctx.train_cfg, ctx.device, ctx.logger)
    dirgnn.optimizer = torch.optim.SGD(
        model.parameters(), lr=ctx.train_cfg.learning_rate, weight_decay=ctx.train_cfg.weight_decay, momentum=0.9
    )
    return dirgnn


def build_model(input_dim, edge_dim, config, ctx) -> Regressor:
    name = config["model_type"]
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}") from None
    return builder(input_dim, edge_dim, config.get("model", {}), ctx)
