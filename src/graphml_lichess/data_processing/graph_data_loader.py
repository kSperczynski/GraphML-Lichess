import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphData:
    data: Data
    usernames: list[str]
    feature_columns: list[str]
    target_column: str
    edge_feature_columns: list[str]


class LichessGraphLoader:
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        id_column: str,
        target_column: str,
        feature_columns: list[str],
        edge_feature_columns: list[str],
        directed: bool = True,
        add_self_loops: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.nodes_path = nodes_path
        self.edges_path = edges_path
        self.id_column = id_column
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.edge_feature_columns = edge_feature_columns
        self.directed = directed
        self.add_self_loops = add_self_loops
        self.logger = logger or logging.getLogger("lichess_gat")

    def load(self) -> GraphData:
        """Load the sepperate node and edge data files into an graph"""
        nodes = pd.read_csv(self.nodes_path)
        edges = pd.read_csv(self.edges_path)

        self._validate_nodes(nodes)
        self._validate_edges(edges)

        duplicated = nodes[nodes.duplicated()]
        if len(duplicated):
            self.logger.warning(f"There are {len(duplicated)} duplicated rows, they are: {duplicated}")

        nodes = nodes.drop_duplicates(self.id_column).reset_index(drop=True)
        usernames = nodes[self.id_column].astype(str).tolist()
        node_index = {name: i for i, name in enumerate(usernames)}

        x = nodes[self.feature_columns].to_numpy(dtype=np.float32)
        y = nodes[self.target_column].to_numpy(dtype=np.float32)

        duplicated_edges = edges[edges.duplicated()]
        if len(duplicated_edges):
            self.logger.warning(f"There are {len(duplicated)} duplicated edges, they are: {duplicated}")

        edges = edges.drop_duplicates(["from", "to"]).reset_index(drop=True)
        edge_rows = edges[edges["from"].astype(str).isin(node_index) & edges["to"].astype(str).isin(node_index)]

        missing = len(edges) - len(edge_rows)
        if missing:
            self.logger.warning(
                "Skipping %d edges because one or both usernames are absent from nodes.csv.",
                missing,
            )

        source = edge_rows["from"].astype(str).map(node_index).to_numpy()
        target = edge_rows["to"].astype(str).map(node_index).to_numpy()
        edge_index = torch.tensor(np.vstack([source, target]), dtype=torch.long)
        edge_attr = edge_rows[self.edge_feature_columns].to_numpy(dtype=np.float32)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)

        # If the graph would be undirected, we add edges both sides
        if not self.directed and edge_index.numel():
            reverse = edge_index.flip(0)
            edge_index = torch.cat([edge_index, reverse], dim=1)
            edge_attr_tensor = torch.cat([edge_attr_tensor, edge_attr_tensor], dim=0)

        # We can add self loops so the node can consider its own features
        if self.add_self_loops:
            loops = torch.arange(len(nodes), dtype=torch.long)
            loop_edges = torch.stack([loops, loops], dim=0)
            edge_index = torch.cat([edge_index, loop_edges], dim=1)

            if edge_attr_tensor.shape[1]:
                loop_attr = torch.zeros((len(nodes), edge_attr_tensor.shape[1]), dtype=torch.float32)
                edge_attr_tensor = torch.cat([edge_attr_tensor, loop_attr], dim=0)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr_tensor,
            y=torch.tensor(y, dtype=torch.float32),
        )

        self.logger.info(
            "Loaded graph: nodes=%d edges=%d features=%d",
            data.num_nodes,
            data.num_edges,
            data.num_node_features,
        )

        return GraphData(
            data=data,
            usernames=usernames,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            edge_feature_columns=self.edge_feature_columns,
        )

    def _check_infinities(self, df: pd.DataFrame, columns: list[str], context_label: str) -> None:
        data_matrix = df[columns].to_numpy(dtype=np.float32)

        # If there are infinites, show some of the rows where that is the case
        if np.isinf(data_matrix).any():
            bad_columns = [col for col in columns if np.isinf(df[col].to_numpy(dtype=np.float32)).any()]

            row_mask = np.isinf(data_matrix).any(axis=1)
            bad_rows = df[row_mask]
            sample_df = bad_rows.head(5)

            raise ValueError(
                f"Data error: Infinite values detected in {context_label} columns: {bad_columns}.\n"
                f"Found {len(bad_rows)} total corrupted rows. {sample_df.to_string()}"
            )

    def _validate_nodes(self, nodes: pd.DataFrame) -> None:
        """Validate the nodes. Check if:
        - They have all required columns
        - The target column is not also in the feature columns
        - There are no infinite values
        """
        required = {
            self.id_column,
            self.target_column,
            *self.feature_columns,
        }
        missing = required.difference(nodes.columns)
        if missing:
            raise ValueError(f"Missing node columns: {sorted(missing)}")

        if self.target_column in self.feature_columns:
            raise ValueError(f"Target column '{self.target_column}' must not be an input feature.")

        self._check_infinities(nodes, self.feature_columns + [self.target_column], "nodes")

    def _validate_edges(self, edges: pd.DataFrame) -> None:
        """Validate the edges. Check if:
        - All the required columns are present
        - There are no infinite values
        """
        required = {"from", "to", *self.edge_feature_columns}
        missing = required.difference(edges.columns)
        if missing:
            raise ValueError(f"Missing edge columns: {sorted(missing)}")

        if self.edge_feature_columns:
            self._check_infinities(edges, self.edge_feature_columns, "edge features")
