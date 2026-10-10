import torch
from scipy.stats.mstats import chisquare

def construct_attention_edge_features(x: torch.Tensor, edge_idx: torch.Tensor, edge_attrs: torch.Tensor,CHISQUARE_ADJUST:int=25):

    new_edge_attrs = edge_attrs.clone()

    # for each node
    for node_idx in torch.arange(x.shape[0]):

        # Zero access after torch.where because the return type is (Tensor, optional)
        incoming_edges_indices = torch.where(edge_idx[1] == node_idx)[0]
        outgoing_edges_indices = torch.where(edge_idx[0] == node_idx)[0]

        # We need to ensure that there is always an edge pair between two nodes:
        assert incoming_edges_indices.shape == outgoing_edges_indices.shape, f'ERR: incoming shape {incoming_edges_indices.shape}, outgoing: {outgoing_edges_indices.shape}'

        # this is a (2, K) tensor where K is the number of neighbours (=incoming edges)
        mirrored_edges_connectivity = edge_idx[:,incoming_edges_indices].flip(0) # outgoing but only to incoming neighbours

        outgoing_edges_indices = []
        for i in range(0,mirrored_edges_connectivity.shape[1]):
            outgoing_edges_indices.append(torch.argwhere(torch.logical_and(edge_idx[0]==mirrored_edges_connectivity[0][i], edge_idx[1]==mirrored_edges_connectivity[1][i])))
        
        outgoing_edges_indices = torch.tensor(outgoing_edges_indices).long()

        # Assume A is the current node
        # We now have the indices of the edges which are A->B,A->C in 'incoming_edge_indices'
        # And the indices of the edges B->A,C->A in 'outgoing_edge_indices'

        points_won_current = edge_attrs[incoming_edges_indices]
        points_won_neighbor = edge_attrs[outgoing_edges_indices]

        new_edge_attrs[incoming_edges_indices] = torch.softmax(torch.tensor([chisquare([w.item()+CHISQUARE_ADJUST,l.item()+CHISQUARE_ADJUST])[1] for (w,l) in zip(points_won_current, points_won_neighbor)]).reshape(-1,1).float(),dim=0)

    return new_edge_attrs

def test_graph_construction_example():
    x = torch.tensor([
        [1,2,3],# Node A
        [4,5,6],# Node B
        [7,8,9],# Node C
        [1,5,7],# Node D
        [7,4,1],# Node E
        [1,3,5],# Node F
    ])

    edge_index = torch.tensor([
        [1,2,1,3,1,4,1,5,2,6],
        [2,1,3,1,4,1,5,1,6,2]
    ])
    edge_index = edge_index - torch.ones_like(edge_index) # because zero-indexed

    # Shuffle edge index order
    permutation = torch.randperm(edge_index.shape[1])
    edge_index = edge_index[:,permutation]

    edge_attrs = torch.tensor([
        [1],
        [1],
        [0],
        [10],
        [2],
        [2],
        [1],
        [0],
        [3],
        [5]
    ]).float()

    edge_attrs = edge_attrs[permutation]

    # Same return as load_data_to_tensors_and_transform
    return x.float(), torch.ones(x.shape[0]), edge_index.long(), edge_attrs.float(), None