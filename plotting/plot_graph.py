import torch
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from torch_geometric.utils import to_dense_adj
import os

def plot_graph(x: torch.Tensor, edge_index: torch.Tensor, edge_attrs: torch.Tensor, config: dict):

    # config:
    #
    # {
    #   is_softmax: bool,
    #   figname: str
    # }
    #

    plt.figure(figsize=(10,10))

    if (not config['is_softmax']):
        # for the plotting: have zero-weight edges still plotted
        edge_attrs += 1e-7

    A = to_dense_adj(edge_index)
    A1 = to_dense_adj(edge_index, edge_attr=edge_attrs)
    # A = to_dense_adj(edge_index, edge_attr=edge_attrs)

    G = nx.from_numpy_array(A[0].numpy(), create_using=nx.MultiDiGraph)
    G1 = nx.from_numpy_array(A1[0,:,:,0].numpy(), create_using=nx.MultiDiGraph)

    # G = nx.from_numpy_array(A[0,:,:,0].numpy(), create_using=nx.MultiDiGraph)

    pos = nx.bfs_layout(G, start=0)

    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

    nx.draw_networkx_nodes(G, pos, node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=24)
    nx.draw_networkx_edges(
        G, pos, edge_color="#cccccc", connectionstyle=connectionstyle, width=2, arrowsize=30
    )

    if (config['is_softmax']):
        labels = {
            tuple(edge): f"{'atten_w'}={attrs['weight']:.4f}"
            for *edge, attrs in G1.edges(keys=True, data=True)
        }
    else:
        labels = {
            tuple(edge): f"{'points'}={attrs['weight']:.1f}"
            for *edge, attrs in G1.edges(keys=True, data=True)
        }
    
    nx.draw_networkx_edge_labels(
        G1,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.5,
        font_color="black",
        font_size=12,
        bbox={"alpha": 0}
    )
    plt.savefig(os.path.join(os.getcwd(), 'plotting', 'saved_figs', f'{config["figname"].removesuffix(".jpg")}.jpg'))
