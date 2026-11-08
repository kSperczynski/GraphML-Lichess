import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graphml_lichess.cli import cmd_report
from graphml_lichess.config import load_config
from graphml_lichess.data_processing.data_splits import Masks
from graphml_lichess.models.mlp import MLP
from graphml_lichess.models.model_factory import MODEL_REGISTRY, BuildContext, build_model
from graphml_lichess.training.regressors import TrainConfig


def test_predictions_ignore_graph_and_other_players():
    """A non-graph baseline must not aggregate neighbor or batch information."""
    torch.manual_seed(42)
    model = MLP(14).eval()
    x = torch.randn(8, 14)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    expected = model(x, edges, torch.ones(3, 1))
    assert expected.shape == (8, 1)
    torch.testing.assert_close(expected, model(x, edges.flip(0), torch.zeros(3, 1)))
    changed = x.clone()
    changed[1:] += 100
    torch.testing.assert_close(expected[0], model(changed)[0])
    torch.testing.assert_close(expected[0], model(x[:1])[0])


def test_factory_trains_saves_and_reloads_without_using_held_out_targets(tmp_path):
    """Changing test labels must not change training or checkpoint selection."""
    torch.manual_seed(0)
    x = torch.randn(60, 3)
    data = Data(x=x, y=2 * x[:, 0] - x[:, 1], edge_index=torch.empty(2, 0, dtype=torch.long))
    masks = Masks(np.arange(60) < 40, (np.arange(60) >= 40) & (np.arange(60) < 50), np.arange(60) >= 50)
    altered = data.clone()
    altered.y[masks.test] += 10000
    cfg = {"model_type": "mlp", "model": {"hidden_dims": [16, 8], "dropout": 0.0}}
    ctx = BuildContext(
        TrainConfig(epochs=100, patience=20, learning_rate=0.02, loss="mse", monitor="rmse"),
        torch.device("cpu"),
        logging.getLogger("test-mlp"),
    )
    fitted = []
    for graph in (data, altered):
        torch.manual_seed(42)
        regressor = build_model(3, None, cfg, ctx)
        initial = regressor.predict(graph)
        result = regressor.fit(graph, masks)
        predictions = regressor.predict(graph)
        assert np.isfinite(predictions).all()
        initial_rmse = np.sqrt(np.mean((initial[masks.validation] - graph.y.numpy()[masks.validation]) ** 2))
        assert result.best_validation_score < initial_rmse / 2
        assert result.best_validation_score == pytest.approx(
            np.sqrt(np.mean((predictions[masks.validation] - graph.y.numpy()[masks.validation]) ** 2)), rel=1e-5
        )
        fitted.append((regressor, result, predictions))
    assert fitted[0][1].best_epoch == fitted[1][1].best_epoch
    np.testing.assert_allclose(fitted[0][2], fitted[1][2], rtol=0, atol=0)
    checkpoint = tmp_path / "model.pt"
    fitted[0][0].save(checkpoint)
    restored = build_model(3, None, cfg, ctx)
    restored.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    np.testing.assert_array_equal(fitted[0][2], restored.predict(data))


def test_model_config_inherits_shared_protocol():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config/models/mlp.yaml")
    base = load_config(root / "config/base.yaml")
    for key in ("data", "graph", "split", "output"):
        assert cfg[key] == base[key]
    assert cfg["training"]["loss"] == "mse"
    assert cfg["training"]["monitor"] == "rmse"


def test_other_models_remain_registered_and_buildable():
    """MLP registration must preserve the updated GAT, Ridge and DirGNN builders."""
    assert {"mlp", "ridge", "torch_gat", "dirgnn", "custom_gat"} <= MODEL_REGISTRY.keys()
    ctx = BuildContext(TrainConfig(), torch.device("cpu"), logging.getLogger("registry-test"))
    root = Path(__file__).resolve().parents[1]
    for relative in ["config/models/ridge.yaml", "config/models/torch_GAT.yaml", "config/models/dirGNN.yaml"]:
        config = load_config(root / relative)
        if config["model_type"] == "dirgnn":
            config["model"]["device"] = "cpu"
        regressor = build_model(14, 1, config, ctx)
        assert callable(regressor.fit) and callable(regressor.predict)


@pytest.mark.parametrize("kwargs", [{"hidden_dims": []}, {"hidden_dims": [0]}, {"dropout": 1.0}])
def test_invalid_architecture_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MLP(14, **kwargs)


def test_report_includes_runs_with_nested_split_metadata(tmp_path, capsys):
    """RunManager nests the split fingerprint; reporting must not silently drop it."""
    metrics = {"mae": 3.0, "rmse": 4.0, "r2": 0.5}
    for seed in (42, 43):
        run = tmp_path / str(seed)
        run.mkdir()
        (run / "run_info.json").write_text(
            json.dumps({"name": "mlp", "model_seed": seed, "metadata": {"data_split_fingerprint": "same-split"}})
        )
        (run / "metrics.json").write_text(json.dumps({"validation": metrics, "test": metrics}))
    cmd_report(SimpleNamespace(runs_dir=tmp_path))
    output = capsys.readouterr().out
    assert "mlp" in output and "same-split" in output
    assert "Empty DataFrame" not in output
