import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats.mstats import chisquare
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
        add_zero_edges: bool = False,
        replace_edge_features_with_attention: bool = False,
        chisquare_adjust_for_attention: int = 0,
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
        self.add_zero_edges = add_zero_edges
        self.replace_edge_features_with_attention = replace_edge_features_with_attention
        self.chisquare_adjust_for_attention = chisquare_adjust_for_attention
        self.logger = logger or logging.getLogger("lichess_gat")

    def load(self) -> GraphData:
        """Load the sepperate node and edge data files into an graph"""
        nodes = pd.read_csv(self.nodes_path)
        edges = pd.read_csv(self.edges_path)

        # adding zero edges
        if self.add_zero_edges:
            edges = self._add_zero_edges(edges)

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

        if self.replace_edge_features_with_attention:
            edge_attr_tensor = self._construct_attention_edge_features(
                torch.tensor(x, dtype=torch.float32), edge_index, edge_attr_tensor
            )

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

    def _construct_attention_edge_features(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_attrs: torch.Tensor):
        CHISQUARE_ADJUST = self.chisquare_adjust_for_attention
        new_edge_attrs = edge_attrs.clone()

        # for each node
        for node_idx in torch.arange(x.shape[0]):
            # Zero access after torch.where because the return type is (Tensor, optional)
            incoming_edges_indices = torch.where(edge_idx[1] == node_idx)[0]
            outgoing_edges_indices = torch.where(edge_idx[0] == node_idx)[0]

            # We need to ensure that there is always an edge pair between two nodes:
            assert incoming_edges_indices.shape == outgoing_edges_indices.shape, (
                f"ERR: incoming shape {incoming_edges_indices.shape}, outgoing: {outgoing_edges_indices.shape}"
            )

            # this is a (2, K) tensor where K is the number of neighbours (=incoming edges)
            mirrored_edges_connectivity = edge_idx[:, incoming_edges_indices].flip(
                0
            )  # outgoing but only to incoming neighbours

            outgoing_edges_indices = []
            for i in range(mirrored_edges_connectivity.shape[1]):
                outgoing_edges_indices.append(
                    torch.argwhere(
                        torch.logical_and(
                            edge_idx[0] == mirrored_edges_connectivity[0][i],
                            edge_idx[1] == mirrored_edges_connectivity[1][i],
                        )
                    )
                )

            outgoing_edges_indices = torch.tensor(outgoing_edges_indices).long()

            # Assume A is the current node
            # We now have the indices of the edges which are A->B,A->C in 'incoming_edge_indices'
            # And the indices of the edges B->A,C->A in 'outgoing_edge_indices'

            points_won_current = edge_attrs[incoming_edges_indices]
            points_won_neighbor = edge_attrs[outgoing_edges_indices]

            new_edge_attrs[incoming_edges_indices] = torch.softmax(
                torch.tensor(
                    [
                        chisquare([w.item() + CHISQUARE_ADJUST, l.item() + CHISQUARE_ADJUST])[1]
                        for (w, l) in zip(points_won_current, points_won_neighbor)
                    ]
                )
                .reshape(-1, 1)
                .float(),
                dim=0,
            )

        return new_edge_attrs

    def _add_zero_edges(self, edges: pd.DataFrame):

        num_edges_before = edges.shape[0]

        # rebuild missing zero edges
        edge_df_mirrored = edges.copy()
        edge_df_mirrored = edge_df_mirrored.rename({"from": "to", "to": "from"}, axis=1)
        edge_df_mirrored["points"] = 0

        # big dataframe with both
        concat_df = pd.concat([edges, edge_df_mirrored]).drop_duplicates(["from", "to"])
        concat_df = concat_df.reset_index().drop("index", axis=1)

        # mirror edges
        concat_edge_df = pd.concat([edges, edge_df_mirrored]).drop_duplicates(["from", "to"])
        concat_edge_df = concat_df.reset_index().drop("index", axis=1)

        # replace old edges with mirrored (zero-value edges)
        edges = concat_edge_df

        num_edges_after = edges.shape[0]

        # inform
        print(f"Adding zero edges, before: {num_edges_before} edges, after: {num_edges_after}.")

        return edges
