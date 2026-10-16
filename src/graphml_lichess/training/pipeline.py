from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graphml_lichess.data_processing.data_splits import Masks, NodeSplit
from graphml_lichess.data_processing.feature_preprocessing import FeaturePreprocessor
from graphml_lichess.data_processing.graph_data_loader import GraphData, LichessGraphLoader
from graphml_lichess.models.model_factory import BuildContext, build_model
from graphml_lichess.training.evalution import evaluate_predictions
from graphml_lichess.training.regressors import Regressor, TrainConfig, TrainingResult
from graphml_lichess.utils.logging import configure_logging
from graphml_lichess.utils.run_manager import RunManager
from graphml_lichess.utils.seeding import seed_everything


def load_graph(config: dict):
    cfg = config["data"]
    return LichessGraphLoader(
        nodes_path=Path(cfg["nodes"]),
        edges_path=Path(cfg["edges"]),
        id_column=cfg["id_column"],
        target_column=cfg["target_column"],
        feature_columns=list(cfg["feature_columns"]),
        edge_feature_columns=list(cfg["edge_feature_columns"]),
        directed=bool(config["graph"]["directed"]),
        add_self_loops=bool(config["graph"].get("add_self_loops", True)),
        add_zero_edges=bool(config["graph"].get("add_zero_edges", False)),
        replace_edge_features_with_attention=bool(config["graph"].get("replace_edge_features_with_attention", False)),
        chisquare_adjust_for_attention=int(config["graph"].get("chisquare_adjust", 0)),
    ).load()


def prepare_data(config: dict) -> tuple[GraphData, Masks, FeaturePreprocessor, NodeSplit]:
    """Load the graph, split nodes, scale features (scaler fit on train only)."""
    graph = load_graph(config)

    # Split the data
    split = NodeSplit.load(Path(config["split"]["path"]))
    masks = split.to_masks(graph.usernames)

    # Scale the features
    # TODO mb we want to do this in preprocessing instead, same as the data splitting?
    assert graph.data.x is not None
    x = graph.data.x.numpy()
    feature_processor = FeaturePreprocessor().fit(x[masks.train])
    graph.data.x = torch.tensor(feature_processor.transform(x), dtype=torch.float32)
    return graph, masks, feature_processor, split


def build_regressor(config: dict, data, device, logger) -> Regressor:
    ctx = BuildContext(
        train_cfg=TrainConfig.from_dict(config.get("training", {})),
        device=device,
        logger=logger,
    )
    edge_dim = data.edge_attr.shape[1] if data.edge_attr is not None else None
    return build_model(data.num_node_features, edge_dim, config, ctx)


class TrainingPipeline:
    """Runs one experiment: load data -> build regressor -> fit -> evaluate -> save."""

    def __init__(self, config: dict):
        self.config = config
        self.seed = int(self.config.get("seed", -1))
        seed_everything(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.run_manager = RunManager(
            base_dir=self.config["output"]["directory"],
            prefix=self.config.get("name", "john_doe"),
        )
        self.run_dir = self.run_manager.run_dir
        self.logger = configure_logging(self.run_dir / "train.log", self.run_manager.run_id)
        self.metadata = self.run_manager.collect_metadata(self.device, self.seed)

    def _save_results(self, graph, regressor: Regressor, masks: Masks, result: TrainingResult) -> None:
        data = graph.data
        target = data.y.squeeze(-1).cpu().numpy()
        prediction = regressor.predict(data)

        metrics = evaluate_predictions(target, prediction, masks.as_dict())
        summary = {
            **metrics,
            "best_epoch": result.best_epoch,
            "best_validation_score": result.best_validation_score,
        }

        test_idx = np.where(masks.test)[0]
        error = prediction[test_idx] - target[test_idx]
        pd.DataFrame(
            {
                "username": np.asarray(graph.usernames)[test_idx],
                "target": target[test_idx],
                "prediction": prediction[test_idx],
                "error": error,
                "absolute_error": np.abs(error),
            }
        ).to_csv(self.run_dir / "predictions.csv", index=False)

        regressor.save(self.run_dir / "model.pt")
        self.metadata["finished_at"] = datetime.now(UTC).isoformat()

        self.run_manager.write_json(summary, "metrics.json")
        self.run_manager.write_json({"epochs": result.history}, "history.json")
        self.run_manager.write_json(
            {
                "name": self.config.get("name"),
                "model_type": self.config.get("model_type"),
                "model_seed": self.seed,
                "metadata": self.metadata,
            },
            "run_info.json",
        )

        self.logger.info("Best validation score: %.4f", result.best_validation_score)
        self.logger.info(
            "Test MAE: %.4f | RMSE: %.4f | R2: %.4f", metrics["test"]["mae"], metrics["test"]["rmse"], metrics["test"]["r2"]
        )
        self.logger.info("Saved artifacts to %s", self.run_dir)

    def run(self) -> Path:
        """Main method of the Training pipeline. Its the entrypoint."""
        try:
            self.logger.info("Starting run %s on %s", self.run_manager.run_id, self.device)
            self.metadata["started_at"] = datetime.now(UTC).isoformat()
            self.run_manager.save_config(self.config)

            # Load the graph
            graph, masks, feature_processor, split = prepare_data(self.config)
            self.metadata["data_split_fingerprint"] = split.fingerprint
            feature_processor.save(self.run_dir / "scaler.joblib")

            # Create the regressor and the model
            regressor = build_regressor(self.config, graph.data, self.device, self.logger)
            if hasattr(regressor, "num_parameters"):
                self.metadata["parameter_count"] = regressor.num_parameters  # type: ignore

            result = regressor.fit(graph.data, masks)
            self._save_results(graph, regressor, masks, result)
            return self.run_dir
        except Exception:
            self.logger.exception("Training pipeline failed")
            raise
