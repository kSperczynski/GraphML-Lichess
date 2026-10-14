from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from graphml_lichess.data_processing.data_splits import random_node_split
from graphml_lichess.data_processing.feature_preprocessing import FeaturePreprocessor
from graphml_lichess.data_processing.graph_data_loader import LichessGraphLoader
from graphml_lichess.models.model_factory import build_model
from graphml_lichess.training.evaluator import evaluate_mask
from graphml_lichess.training.trainer import Trainer, TrainingResult
from graphml_lichess.utils.logging import configure_logging
from graphml_lichess.utils.run_manager import RunManager
from graphml_lichess.utils.seeding import seed_everything


class TrainingPipeline:
    """Orchestrates the entire lifecycle of a single graph training run"""

    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)

        self.seed = int(self.config.get("seed", 42))
        seed_everything(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Utility class to save data about the training run
        self.run_manager = RunManager(base_dir=self.config["output"]["directory"], prefix=self.config.get("name", "gat"))

        self.logger = configure_logging(self.run_manager.run_dir / "train.log", self.run_manager.run_id)
        self.metadata = self.run_manager.collect_metadata(self.device, self.seed)

    def _load_config(self, path: Path):
        """Load yaml configuration file"""
        assert path.exists(), f"Path to configuration does not exist, path was: {path}"
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _build_loss(self) -> nn.Module:
        """Get the method that will be used to compute loss"""
        loss_cfg = self.config.get("training", {})
        loss_name = loss_cfg.get("loss", "huber").lower()
        if loss_name == "huber":
            return nn.HuberLoss(delta=float(loss_cfg.get("huber_delta", 1.0)))
        if loss_name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss: {loss_name}")

    def _prepare_and_preprocess_data(self):
        """Loads graph, splits nodes, and applies scaling."""
        self.logger.info("Loading and preparing data...")

        # Load the data
        data_config = self.config["data"]
        self.graph = LichessGraphLoader(
            nodes_path=Path(data_config["nodes"]),
            edges_path=Path(data_config["edges"]),
            id_column=data_config["id_column"],
            target_column=data_config["target_column"],
            feature_columns=list(data_config["feature_columns"]),
            edge_feature_columns=list(data_config["edge_feature_columns"]),
            directed=bool(self.config["graph"]["directed"]),
            add_self_loops=bool(self.config["graph"]["directed"]),
        ).load()

        # Split the data
        assert self.graph.data.num_nodes
        split_cfg = self.config["split"]
        self.split = random_node_split(
            num_nodes=self.graph.data.num_nodes,
            train_ratio=float(split_cfg["train_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            seed=self.seed,
        )
        self.run_manager.save_split(self.split.as_json())

        # Preprocess features
        assert self.graph.data.x is not None
        train_data = self.graph.data.x.numpy()[self.split.train]
        feature_preprocessor = FeaturePreprocessor().fit(train_data)
        self.graph.data.x = torch.tensor(feature_preprocessor.transform(self.graph.data.x.numpy()), dtype=torch.float32)
        feature_preprocessor.save(self.run_manager.run_dir / "scaler.joblib")

    def _evaluate_and_save(
        self,
        result: TrainingResult,
        state_path: Path,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        loss_fn,
    ) -> None:
        self.logger.info("Evaluating best model and saving artifacts...")

        # Load the model
        self.model.load_state_dict(torch.load(state_path, map_location=self.device, weights_only=True))

        # Compute the metrics for all splits
        _, train_metrics, _, _ = evaluate_mask(self.model, self.graph.data, train_mask, self.device, loss_fn)
        test_loss, test_metrics, test_target, test_pred = evaluate_mask(
            self.model, self.graph.data, test_mask, self.device, loss_fn
        )
        _, val_metrics, _, _ = evaluate_mask(self.model, self.graph.data, val_mask, self.device, loss_fn)

        # Save all the metrics
        metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "best_epoch": result.best_epoch,
            "best_validation_score": result.best_validation_score,
        }
        self.run_manager._write_json(data=metrics, filename="metrics.json")
        self.run_manager._write_json(data={"epochs": result.history}, filename="metrics.json")

        # Save the raw predictions
        test_indices_aligned = np.where(test_mask)[0]
        predictions_df = pd.DataFrame(
            {
                "username": np.asarray(self.graph.usernames)[test_indices_aligned],
                "target": test_target,
                "prediction": test_pred,
                "error": test_pred - test_target,
                "absolute_error": np.abs(test_pred - test_target),
            }
        )
        predictions_df.to_csv(self.run_manager.run_dir / "predictions.csv", index=False)

        self.metadata["finished_at"] = datetime.now(UTC).isoformat()
        self.run_manager.save_checkpoint(
            model=self.model,
            model_config={
                **self.config["model"],
                "input_dim": self.graph.data.num_node_features,
                "edge_dim": self.graph.data.edge_attr.shape[1] if self.graph.data.edge_attr is not None else None,
            },
            feature_columns=self.graph.feature_columns,
            target_column=self.graph.target_column,
            edge_feature_columns=self.graph.edge_feature_columns,
            metadata=self.metadata,
            filename="model.pt",
        )

        info = {"metadata": self.metadata, "metrics": metrics}
        self.run_manager._write_json(info, filename="run_info.json")
        self.logger.info("Best validation score: %.4f", result.best_validation_score)
        self.logger.info("Test score: %.4f", test_loss)
        self.logger.info("Saved artifacts to %s", self.run_manager.run_dir)

    def execute(self) -> Path:
        """Execute the training pipeline"""
        try:
            self.logger.info(f"Starting training on {self.device}. Run ID: {self.run_manager.run_id}")
            self.metadata["started_at"] = datetime.now(UTC).isoformat()
            self.run_manager.save_config(self.config)

            # Prepare the data
            self._prepare_and_preprocess_data()
            assert self.graph.data.num_nodes
            train_mask, val_mask, test_mask = self.split.masks(self.graph.data.num_nodes)

            # Load the model
            self.model = build_model(
                input_dim=self.graph.data.num_node_features,
                edge_dim=self.graph.data.edge_attr.shape[1] if self.graph.data.edge_attr is not None else None,
                config=self.config,
            ).to(self.device)
            self.metadata["parameter_count"] = sum(p.numel() for p in self.model.parameters())

            # Train the model
            train_cfg = self.config["training"]
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(train_cfg["learning_rate"]),
                weight_decay=float(train_cfg["weight_decay"]),
            )

            trainer = Trainer(
                model=self.model,
                optimizer=optimizer,
                loss_fn=self._build_loss(),
                device=self.device,
                epochs=int(train_cfg["epochs"]),
                patience=int(train_cfg["patience"]),
                gradient_clip_norm=train_cfg.get("gradient_clip_norm"),
                logger=self.logger,
                evaluator=evaluate_mask,
            )

            state_path = self.run_manager.run_dir / "best_model_state.pt"
            result = trainer.fit(self.graph.data, train_mask, val_mask, str(state_path))

            # Evaluation
            self._evaluate_and_save(result, state_path, train_mask, val_mask, test_mask, loss_fn=self._build_loss())

            return self.run_manager.run_dir

        except Exception as e:
            self.logger.exception(f"Training pipeline failed. {e.__traceback__}")
            raise

    # ... (_evaluate_and_save and _get_masks_on_device implemented as separate private methods)
