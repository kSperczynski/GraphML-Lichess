from torch import nn
from torch_geometric.nn import GAT

from graphml_lichess.models.gat import CustomGATRegressor


def build_model(input_dim, edge_dim, config) -> nn.Module:
    model_cfg = config["model"]
    if config["name"] == "torch_gat":
        return GAT(
            in_channels=input_dim,
            out_channels=model_cfg["out_channels"],
            num_layers=model_cfg.get("num_layers", 1),
            hidden_channels=model_cfg["hidden_channels"],
            v2=model_cfg.get("v2", True),
            act=model_cfg.get("act"),
            act_first=model_cfg.get("act_first"),
            dropout=model_cfg.get("dropout"),
            heads=model_cfg.get("heads"),
        )

    elif config["name"] == "custom_gat":
        return CustomGATRegressor()

    else:
        raise ValueError(f"Doesn't know a model with name: {config['name']}")
