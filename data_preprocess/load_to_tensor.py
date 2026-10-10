import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler

# for graph info
from torch_geometric.utils import to_dense_adj
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

def load_data_to_tensors():
    """
    Just loads the data. Does not remove inf values, or do any scaling yet.

    NB! A few notes:
    - acpl feature has very high outliers (256 acpl)
    - move time is infinite sometimes
    """

    # load csv data
    edge_df = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'edges.csv'))
    node_df = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'node_features.csv'))

    # index nodes on unique username
    node_df_index = node_df.reset_index().set_index('username')

    # compute the indices of each edge
    edge_index_from = edge_df.apply(lambda row: node_df_index.loc[row['from']]['index'], axis=1).astype(int)
    edge_index_to = edge_df.apply(lambda row: node_df_index.loc[row['to']]['index'], axis=1).astype(int)

    # apply indices to data
    edge_df['idx_from'] = edge_index_from
    edge_df['idx_to'] = edge_index_to
    edge_df.drop(['from', 'to'], axis=1, inplace=True) # remove the from/to that were usernames

    # create the edge index in PyG format
    edge_index_df = edge_df.drop('points', axis=1)
    edge_index = torch.from_numpy(edge_index_df.to_numpy().T)

    # save the X data as a torch tensor
    x = torch.from_numpy(node_df.drop('username', axis=1).to_numpy())

    # create edge attributes
    edge_attrs = torch.from_numpy(edge_df.drop(['idx_from', 'idx_to'], axis=1).to_numpy().copy())

    # just check if data has expected shape
    assert edge_attrs.shape == (edge_df.shape[0], 1), f'ERR: wrong dimension: {edge_attrs.shape}. Expected (num_edges, 1) since one feature per edge (points)'
    assert edge_index.shape == (2, edge_df.shape[0]), f'ERR: wrong dimension: {edge_index.shape}. Expected (2, num_edges)'
    assert x.shape[0] == node_df.shape[0], f'ERR: wrong dimension: {x.shape}. Expected (num_nodes, any)'

    return x.to(torch.float32), edge_index.long(), edge_attrs.to(torch.float32)


def load_data_to_tensors_and_transform():
    """
    Returns 5-tuple of (X, y, edge_index, edge_attrs, scaler)

    Preprocessing by:
    - building reflected edges (missing 0's)
    - removing outliers
    - scaling the data
    """
    
    # load csv data
    edge_df = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'edges.csv'))
    node_df = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'node_features.csv'))
    node_df_before_preprocess = node_df.copy()

    # --- node preprocessing ---
    # remove outliers
    node_df = node_df[node_df['acpl'] < 150]
    node_df = node_df[node_df['avg_move_time'] < 0.050]
    node_df = node_df[node_df['win_streak'] < 30]
    node_df = node_df[node_df['loss_streak'] < 30]
    node_df = node_df[node_df['num_games'] < 40000]
    node_df = node_df[node_df['draw_rate'] < 0.20]
    node_df = node_df.reset_index(drop=True)

    # check the difference in usernames: needed to remove relevant edges
    diff = set(node_df_before_preprocess['username']) - set(node_df['username'])

    # remove edges that connect to the removed nodes
    edges_to_be_removed = (edge_df['from'].isin(diff) | edge_df['to'].isin(diff)) # = either of the two ends has a removed node
    edge_df = edge_df[~edges_to_be_removed]
    edge_df = edge_df.reset_index(drop=True)

    # rebuild missing zero edges
    edge_df_mirrored = edge_df.copy()
    edge_df_mirrored = edge_df_mirrored.rename({'from': 'to', 'to': 'from'}, axis=1)
    edge_df_mirrored['points'] = 0

    # big dataframe with both 
    concat_df = pd.concat([edge_df, edge_df_mirrored]).drop_duplicates(['from', 'to'])#.rename(columns={'points': 'values'})
    concat_df = concat_df.reset_index().drop('index', axis=1)

    # mirror edges
    concat_edge_df = pd.concat([edge_df, edge_df_mirrored]).drop_duplicates(['from', 'to'])#.rename(columns={'points': 'values'})
    concat_edge_df = concat_df.reset_index().drop('index', axis=1)


    # index nodes on unique username
    node_df_index = node_df.reset_index().set_index('username')

    # compute the indices of each edge
    edge_index_from = concat_edge_df.apply(lambda row: node_df_index.loc[row['from']]['index'], axis=1).astype(int)
    edge_index_to = concat_edge_df.apply(lambda row: node_df_index.loc[row['to']]['index'], axis=1).astype(int)

    # apply indices to data
    concat_edge_df['idx_from'] = edge_index_from
    concat_edge_df['idx_to'] = edge_index_to
    concat_edge_df.drop(['from', 'to'], axis=1, inplace=True) # remove the from/to that were usernames

    # create the edge index in PyG format
    edge_index_df = concat_edge_df.drop('points', axis=1)
    edge_index = torch.from_numpy(edge_index_df.to_numpy().T)

    # split node attributes into features and labels 
    x_numpy = node_df.drop(['username', 'target_rating'], axis=1).to_numpy()
    y_numpy = node_df['target_rating'].to_numpy().copy()

    # standard scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_numpy)

    # save to torch tensors
    x = torch.from_numpy(x_scaled)  # scaled X
    y = torch.from_numpy(y_numpy)   # unscaled y

    # create edge attributes
    edge_attrs = torch.from_numpy(concat_edge_df.drop(['idx_from', 'idx_to'], axis=1).to_numpy().copy())

    # just check if data has expected shape
    assert edge_attrs.shape == (concat_edge_df.shape[0], 1), f'ERR: wrong dimension: {edge_attrs.shape}. Expected (num_edges, 1) since one feature per edge (points)'
    assert edge_index.shape == (2, concat_edge_df.shape[0]), f'ERR: wrong dimension: {edge_index.shape}. Expected (2, num_edges)'
    assert x.shape[0] == node_df.shape[0], f'ERR: wrong X dimension: {x.shape}. Expected (num_nodes, any)'
    assert y.shape == (node_df.shape[0],), f'ERR: wrong y dimension: {y.shape}. Expected (num_nodes,)'
    assert y.min() >= 0 and 1000 <= y.mean() <= 2000, f'ERR: y label has min of {y.min()} and mean of {y.mean()}, which are likely out-of-distribution'

    return x.to(torch.float32), y.to(torch.float32), edge_index.long(), edge_attrs.to(torch.float32), scaler

def graph_details(x: torch.Tensor, edge_index: torch.Tensor):
    # conver to dense matrix
    adjacency = to_dense_adj(edge_index)[0] # 0th dim access as returns batched, e.g. size [1, 7571, 7571]
    # and to compressed sparse matrix (following https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html)
    graph = csr_array(adjacency)

    # get connected components
    n_components, _ = connected_components(graph)

    # compute sparsity
    sparsity = 1 - edge_index.shape[1] / (x.shape[0] ** 2)

    return {
        'num_connected_components': n_components,
        'sparsity': sparsity
    }


if __name__ == '__main__':

    x, edge_idx, edge_attrs = load_data_to_tensors()
    print(x)
    print(edge_attrs)
    print(edge_idx)