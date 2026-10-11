from data_preprocess.load_to_tensor import load_data_to_tensors_and_transform, graph_details
from models.dirgnn.dirgnn import AnalyticalAttentionNetwork
from models.dirgnn.dirgnn_graph_construction import construct_attention_edge_features, test_graph_construction_example
from scipy.stats.mstats import chisquare
from plotting.plot_graph import plot_graph

# from models.label_propagation.label_propagation import evaluate_label_propagation, print_LP_evaluation_results, trivial_baseline

if __name__ == "__main__":

    # Test on test graph
    # x, y, edge_index, edge_attrs, scaler = test_graph_construction_example()
    # plot_graph(x,edge_index,edge_attrs,config={"is_softmax": False, "figname": "test_graph"})
    # new_edge_attrs = construct_attention_edge_features(x,edge_index,edge_attrs)
    # plot_graph(x,edge_index,new_edge_attrs,config={"is_softmax": True, "figname": "attention_graph"})
    
    x, y, edge_index, edge_attrs, scaler = load_data_to_tensors_and_transform()

    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')
    print(f'edge_index shape: {edge_index.shape}')
    print(f'edge_attrs shape: {edge_attrs.shape}')

    print(f'min edge index: {edge_index.min()}')

    # connected components, sparsity
    print(f'\nGraph details:\n{graph_details(x, edge_index)}')

    # res = evaluate_label_propagation(y, edge_index, verbose=True)
    # print_LP_evaluation_results(res)
    # trivial_baseline(y)