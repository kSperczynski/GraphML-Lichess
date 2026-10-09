import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler

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
    Preprocessing by:
    - make sure that the edges are reflected
    - removing outliers
    - scaling the data
    """
    
    # load csv data
    edge_df = pd.read_csv('edges.csv')
    node_df = pd.read_csv('node_features.csv')
    node_df_before_preprocess = node_df.copy()

    # --- node preprocessing ---
    # remove outliers
    node_df = pd.read_csv('node_features.csv')
    node_df = node_df[node_df['acpl'] < 150]
    node_df = node_df[node_df['avg_move_time'] < 0.050]
    node_df = node_df[node_df['win_streak'] < 30]
    node_df = node_df[node_df['loss_streak'] < 30]
    node_df = node_df[node_df['num_games'] < 40000]
    node_df = node_df[node_df['draw_rate'] < 0.20]

    diff = set(node_df_before_preprocess['username']) - set(node_df['username'])

    # remove (removed) nodes from the edge df
    edges_to_be_removed = (edge_df['from'].isin(diff) | edge_df['to'].isin(diff))
    edge_df = edge_df[~edges_to_be_removed]
    # print((edge_df['from'].isin(diff) | edge_df['to'].isin(diff)).sum())
    # return
    # print(edge_df[data.edge_df.isin(list1) == False])
    # edge_df.drop()

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

    x_numpy = node_df.drop('username', axis=1).to_numpy()
    # standard scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_numpy)

    # save the X data as a torch tensor
    x = torch.from_numpy(x_scaled)

    # create edge attributes
    edge_attrs = torch.from_numpy(concat_edge_df.drop(['idx_from', 'idx_to'], axis=1).to_numpy().copy())

    # just check if data has expected shape
    assert edge_attrs.shape == (concat_edge_df.shape[0], 1), f'ERR: wrong dimension: {edge_attrs.shape}. Expected (num_edges, 1) since one feature per edge (points)'
    assert edge_index.shape == (2, concat_edge_df.shape[0]), f'ERR: wrong dimension: {edge_index.shape}. Expected (2, num_edges)'
    assert x.shape[0] == node_df.shape[0], f'ERR: wrong dimension: {x.shape}. Expected (num_nodes, any)'

    return x.to(torch.float32), edge_index.long(), edge_attrs.to(torch.float32)


if __name__ == '__main__':

    x, edge_idx, edge_attrs = load_data_to_tensors()
    print(x)
    print(edge_attrs)
    print(edge_idx)