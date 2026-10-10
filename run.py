from data_preprocess.load_to_tensor import load_data_to_tensors_and_transform, graph_details
from models.dirgnn.dirgnn import AnalyticalAttentionNetwork
from models.dirgnn.dirgnn_graph_construction import construct_attention_edge_features, test_graph_construction_example
from scipy.stats.mstats import chisquare
from plotting.plot_graph import plot_graph
import argparse

if __name__ == "__main__":

    # Test on test graph
    # x, y, edge_index, edge_attrs, scaler = test_graph_construction_example()
    # plot_graph(x,edge_index,edge_attrs,config={"is_softmax": False, "figname": "test_graph"})
    # new_edge_attrs = construct_attention_edge_features(x,edge_index,edge_attrs)
    # plot_graph(x,edge_index,new_edge_attrs,config={"is_softmax": True, "figname": "attention_graph"})
    
    x, y, edge_index, edge_attrs, scaler = load_data_to_tensors_and_transform()
    # res = construct_attention_edge_features(x,edge_index,edge_attrs)

    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')
    print(f'edge_index shape: {edge_index.shape}')
    print(f'edge_attrs shape: {edge_attrs.shape}')

    print(f'min edge index: {edge_index.min()}')

    # connected components, sparsity
    print(f'\nGraph details:\n{graph_details(x, edge_index)}')

    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--num-epochs", type=int, default=10, required=False)
    parser.add_argument("--batch-size", type=int, default=32, required=False)
    parser.add_argument("--device", type=str, default='cuda', required=False)

    parser.add_argument("--mp_input_features", type=int, default=14, required=False)
    parser.add_argument("--mp_latent_features", type=int, default=32, required=False)
    parser.add_argument("--mp_output_features", type=int, default=16, required=False)

    parser.add_argument("--mp_bias", type=bool, default=True, required=False)
    parser.add_argument("--num_mp_layers", type=int, default=4, required=False)

    parser.add_argument("--num_nodewise_layers", type=int, default=3, required=False)
    parser.add_argument("--hid_dim_nodewise", type=int, default=16, required=False)

    parser.add_argument("--CHISQUARE_ADJUST", type=int, default=25, required=False)

    args = parser.parse_args()

    model = AnalyticalAttentionNetwork(args)

    # Construct attention graph by replacing edge features with analytical attention weights
    edge_features = construct_attention_edge_features(x, edge_index, edge_attrs, args.CHISQUARE_ADJUST)

    print(model)

    out = model.forward(x,edge_index,edge_features)

    print(out.shape)
