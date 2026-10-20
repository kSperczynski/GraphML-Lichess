from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pygod.detector import DOMINANT
from torch_geometric.explain import Explainer, Explanation, GNNExplainer
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from graphml_lichess.data_processing.data_splits import Masks, NodeSplit
from graphml_lichess.data_processing.feature_preprocessing import FeaturePreprocessor
from graphml_lichess.data_processing.graph_data_loader import GraphData, LichessGraphLoader
from graphml_lichess.models.model_factory import BuildContext, build_model
from graphml_lichess.training.evalution import evaluate_predictions
from graphml_lichess.training.regressors import Regressor, TrainConfig, TrainingResult
from graphml_lichess.utils.logging import configure_logging
from graphml_lichess.utils.run_manager import RunManager
from graphml_lichess.utils.seeding import seed_everything

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
        remove_added_zero_edges=bool(config["graph"].get("remove_added_zero_edges", False)),
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


class InterfaceModel(torch.nn.Module):
    """
    Abstraction model that serves as an interface between the GNNExplainer and Dominant for compatibility.

    Notes:
    - Dominant's forward pass returns two tensors, X_pred [N x 14] and Adj [N x N]
    - Dominant exposes a `predict` function, but it appears non-differentiable somewhere in between.
    - Score computation according to eq. (3.11) from the paper "Deep Anomaly Detection on Attributed Networks"

    This interface model implements a forward pass that translates the results of Dominant to the Dominant score.
    """

    def __init__(self, dominant_backbone: DOMINANT, data, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dominant_backbone = dominant_backbone
        self.dominant_backbone.model.requires_grad_(True)
        self.data = data
        self.config = config

    def forward(self, x, edge_index):

        # forward pass of the full Dominant
        xs, A = self.dominant_backbone.model.forward(x, edge_index)

        # compute score ourselves -> while Dominant provides a `predict` function, it seems to drop gradients somewhere along the way, and I could not figure out how to make it compatible with GNNExplainer
        xs_component = ((xs - x)**2.0).sum(dim=-1)
        adj_component = ((A - to_dense_adj(edge_index))**2.0).sum(dim=-1).squeeze(0) # necessary squeeze, don't remove; otherwise [1, N]

        # double check, but to my understanding this is the alpha from (3.11) in the paper.
        alpha = self.dominant_backbone.weight

        score = alpha * xs_component + (1 - alpha) * adj_component ## WARN! if the shapes do not match, an arbitrary masking error will appear. Check the assert below.
        assert xs_component.shape == adj_component.shape, f'ERR: shapes do not match: {xs_component.shape} and {adj_component.shape}'

        ### Probability Computation
        # mirror computation of the probability via the linear algorithm of Dominant
        # for details, see `.venv/lib/python3.12/site-packages/pygod/detector/base.py:195`
        # or search for `prob = self._predict_prob(score, prob_method)` in VSCode in the full directory (incl. the venv)
        train_score = self.dominant_backbone.decision_score_
        prob = score - train_score.min()
        prob /= train_score.max() - train_score.min()
        prob = prob.clamp(0, 1)
        ###

        return prob
    

class ExplainabilityPipeline:
    """Runs explainability experiments."""

    def __init__(self, config: dict):
        self.config = config

        self.run_manager = RunManager(
            base_dir=self.config["output"]["directory"], # inherits from base yaml
            prefix=self.config.get("name", "john_doe"),
        )
        self.run_dir = self.run_manager.run_dir

    def _save_feature_reconstruction_errors(self, graph, dominant_model, outlier_mask, top_k: int = 5):
        """Per-outlier, per-feature squared reconstruction error."""
        feature_names = list(self.config["data"]["feature_columns"])
        model = dominant_model.model
        device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            x = graph.data.x.to(device)
            x_hat, _ = model(x, graph.data.edge_index.to(device))

        sq_err = ((x_hat - x) ** 2).cpu()                  # [N, F]
        outlier_idx = torch.nonzero(outlier_mask).squeeze(-1)
        err = sq_err[outlier_idx].numpy()                  # [n_outliers, F]
        usernames = np.asarray(graph.usernames)[outlier_idx.numpy()]

        df = pd.DataFrame(err, columns=feature_names)
        df.insert(0, "username", usernames)
        df.to_csv(self.run_dir / "outlier_feature_errors.csv", index=False)

        share = err / err.sum(axis=1, keepdims=True)
        pd.DataFrame(share, columns=feature_names).assign(username=usernames).to_csv(
            self.run_dir / "outlier_feature_error_share.csv", index=False
        )


    def _save_outlier_info(
            self, 
            graph,
            predicted_mask: torch.Tensor, 
            predicted_scores: torch.Tensor, 
            predicted_probabilities: torch.Tensor, 
            predicted_confidence: torch.Tensor
        ) -> None:
        pd.DataFrame(
            {
                "username": np.asarray(graph.usernames),
                "predicted_mask": predicted_mask,
                "predicted_scores": predicted_scores,
                "predicted_probabilities": predicted_probabilities,
                "predicted_confidence": predicted_confidence
            }
        ).to_csv(self.run_dir / "predicted_outliers.csv", index=False)

        print(f'Outlier info saved to {self.run_dir / "predicted_outliers.csv"}')


    def _save_explanation(self, explanation: Explanation):
        """
        TODO: save more info
        """

        # bar plot, feature importance
        explanation.visualize_feature_importance(self.run_dir / "feature_importance.png")

        # inform
        print(f'Feature importance saved to {self.run_dir / "feature_importance.png"}')


    def _cluster_outliers(self, graph, dominant_model, outlier_mask, k_range=range(2, 7)):
        feature_names = list(self.config["data"]["feature_columns"])
        model = dominant_model.model
        device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            x = graph.data.x.to(device)
            edge_index = graph.data.edge_index.to(device)
            x_hat, a_hat = model(x, edge_index)
            a = to_dense_adj(edge_index, max_num_nodes=x.shape[0]).squeeze(0)
            attr_err = ((x_hat - x) ** 2).cpu().numpy()
            struct_err = ((a_hat - a) ** 2).sum(dim=-1).cpu().numpy()

        def zlog(v):
            v = np.log1p(v)
            return (v - v.mean(0)) / (v.std(0) + 1e-12)

        profile = np.column_stack([zlog(attr_err), zlog(struct_err[:, None])])
        columns = feature_names + ["structure"]

        idx = torch.nonzero(outlier_mask).squeeze(-1).cpu().numpy()
        X = profile[idx]
        usernames = np.asarray(graph.usernames)[idx]

        if len(idx) < 4:
            print("Too few outliers to cluster.")
            return

        # pick k by silhouette score
        best_k, best_score = None, -1.0
        for k in k_range:
            if k >= len(idx):
                break
            labels = KMeans(n_clusters=k, n_init=20, random_state=self.config.get("seed", 0)).fit_predict(X)
            score = silhouette_score(X, labels)
            print(f"k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_k, best_score = k, score

        labels = KMeans(n_clusters=best_k, n_init=20, random_state=self.config.get("seed", 0)).fit_predict(X)
        print(f"Chose k={best_k} (silhouette={best_score:.3f})")

        pd.DataFrame({
            "username": usernames,
            "cluster": labels,
            "attr_error_total": attr_err[idx].sum(axis=1),
            "struct_error_total": struct_err[idx],
        }).to_csv(self.run_dir / "outlier_clusters.csv", index=False)

        prof = pd.DataFrame(X, columns=columns).assign(cluster=labels).groupby("cluster").mean()
        prof.insert(0, "size", np.bincount(labels))
        prof.to_csv(self.run_dir / "outlier_cluster_profiles.csv")


    def run(self):

        # load all available data
        graph, masks, feature_processor, split = prepare_data(self.config)

        # get the data
        data = graph.data

        # create the model
        dominant_model = DOMINANT(
            hid_dim=self.config["model"]["hid_dim"],
            num_layers=self.config["model"]["num_layers"],
            dropout=self.config["model"]["dropout"],
            weight_decay=self.config["training"]["weight_decay"],
            contamination=self.config["model"]["contamination"],
            lr=self.config["training"]["learning_rate"],
            epoch=self.config["training"]["epochs"]
        )

        # fit
        dominant_model.fit(data=data, label=None) # <-- no labels

        # (INFO) this is an alternative to the custom score calculation inside the forward() of the InterfaceModel; however, gradients are removed somewhere in between
        # we still want predictions: allows us to explain only the nodes that were flagged as outliers hence this line still is needed.
        pred, score, prob, conf = dominant_model.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True, prob_method=self.config["predictions"]["probability-type"])

        self._cluster_outliers(graph, dominant_model, pred)
        self._save_feature_reconstruction_errors(graph, dominant_model, pred)
        
        # debug, inform
        print(f'Predicted {pred.sum().item()} outliers.')

        # save outlier info
        self._save_outlier_info(graph, pred, score, prob, conf)

        # create interface model, see class docstring for details
        interface_model = InterfaceModel(dominant_backbone=dominant_model, data=data, config=self.config)
        # an alternative would be to overwrite the Dominant forward-pass function with the score computation similarly to:
        # dominant_model.model.forward = lambda x_edge_tuple: ...
        # but the interface model above might be a cleaner approach.

        explainer = Explainer(
            model=interface_model,
            algorithm=GNNExplainer(epochs=100), # TODO: add to config
            explanation_type='model',
            node_mask_type='attributes',
            # edge_mask_type='object', # -> ValueError: Could not compute gradients for edges. Please make sure that edges are used via message passing inside the model or disable it via `edge_mask_type=None`.
            edge_mask_type=None, 
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )

        # Explanation for all nodes detected as outliers:
        # node_index = torch.arange(start=0, end=pred.shape[0]-1, step=1)[pred]

        # Explanation for the first outlier:
        node_index = torch.arange(start=0, end=pred.shape[0]-1, step=1)[pred][0] # <- 0th index = single outlier

        explanation: Explanation = explainer(data.x, data.edge_index, index=node_index)

        # debug print
        print(explanation)

        # save results (TODO: extend this)
        self._save_explanation(explanation)

        

