from data_preprocess.load_to_tensor import load_data_to_tensors_and_transform, graph_details

if __name__ == "__main__":
    
    x, y, edge_index, edge_attrs, scaler = load_data_to_tensors_and_transform()
    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')
    print(f'edge_index shape: {edge_index.shape}')
    print(f'edge_attrs shape: {edge_attrs.shape}')

    # connected components, sparsity
    print(f'\nGraph details:\n{graph_details(x, edge_index)}')
