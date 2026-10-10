import torch
from torch_geometric.data import InMemoryDataset
from data_preprocess.load_to_tensor import load_data_to_tensors_and_transform, graph_details
from models.dirgnn.dirgnn import AnalyticalAttentionNetwork
from models.dirgnn.dirgnn_graph_construction import construct_attention_edge_features
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import os, argparse
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import itertools
from collections import namedtuple
from tqdm import tqdm
import random

def train_dirgnn(args):
    # Load data and prepare data loader
    x, y, edge_index, edge_attr, scaler = load_data_to_tensors_and_transform()

    # Construct attention graph by replacing edge features with analytical attention weights
    edge_features = construct_attention_edge_features(x, edge_index, edge_attr, args.CHISQUARE_ADJUST)

    # Move to device and construct model
    x = x.to(args.device)
    y = y.to(args.device)
    edge_index = edge_index.to(args.device)
    edge_features = edge_features.to(args.device)

    # Create data loader
    data = Data(x,edge_index,edge_features,y)
    dataloader = DataLoader([data], batch_size=args.batch_size)

    # Train-Test split
    node_transform = RandomNodeSplit(num_val=0, num_test=0.3) #30% test, 70% train
    node_splits = node_transform(data)
    train_mask = node_splits.train_mask
    test_mask = node_splits.test_mask

    # Create model
    model = AnalyticalAttentionNetwork(args)
    
    # optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    
    loss_fn = torch.nn.MSELoss()
    
    epoch_losses = {}
    val_losses = {}
    all_losses = []
    for epoch_idx in range(args.num_epochs):

        loss_list = []
        val_loss_list = []
        print(f'='*10 + f'Epoch {epoch_idx}' + '='*10)
        for idx, data in enumerate(dataloader):

            optimizer.zero_grad()
            x, y, edge_attr, edge_index = data.x, data.y, data.edge_attr, data.edge_index # unpack

            output = model.forward(x, edge_index, edge_attr)
            loss = loss_fn.forward(output[train_mask,:], y.unsqueeze(-1)[train_mask,:])

            loss_list.append(loss.item())
            print(f'batch {idx} loss = {loss}')

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_loss = loss_fn.forward(output[test_mask,:], y.unsqueeze(-1)[test_mask,:])
                val_loss_list.append(val_loss.item())
                print(f'batch {idx} val loss = {val_loss}')
            
        epoch_losses[epoch_idx] = loss_list
        val_losses[epoch_idx] = val_loss_list
        all_losses = all_losses + loss_list

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")

    with open(os.path.join(os.getcwd(), 'experiment_results', 'dirgnn', 'losses.json'), 'w+') as f:
        json_str = json.dumps({
            'all_losses': all_losses,
            'epoch_losses': epoch_losses,
            'val_losses': val_losses
        }, indent=4)
        f.write(json_str)

    with open(os.path.join(os.getcwd(), 'experiment_results', 'dirgnn', 'outputs.json'), 'w+') as f:
        json_str = json.dumps({
            'final_outputs': output.detach().cpu().numpy().tolist()
        })
        f.write(json_str)

def train_baseline_regressor():
    x, y, edge_index, edge_features, _ = load_data_to_tensors_and_transform()

    data = Data(x,edge_index,edge_features,y)
    node_transform = RandomNodeSplit(num_val=0, num_test=0.3) #30% test, 70% train
    node_splits = node_transform(data)
    train_mask = node_splits.train_mask
    test_mask = node_splits.test_mask

    gb_regressor = GradientBoostingRegressor()

    x_train = x[train_mask].numpy()
    x_test = x[test_mask].numpy()
    y_train = y[train_mask].numpy()
    y_test = y[test_mask].numpy()

    gb_regressor.fit(X=x_train, y=y_train)

    preds = gb_regressor.predict(X=x_test)
    mse = mean_squared_error(y_pred=preds, y_true=y_test)

    print(f'MSE gb regressor: {mse:.3f}')

def hyperparameter_search():
    
    lr_options = (1e-4 * 5 ** np.linspace(start=-1, stop=1, num=3)).tolist()
    wd_options = (1e-5 * 5 ** np.linspace(start=-1, stop=1, num=3)).tolist()

    mp_latent_features_options = [8, 14, 20, 32, 64]
    mp_output_features = [8, 14, 20, 32, 64]
    num_mp_layers = [2,3,4,5]
    num_nodewise_layers = [1,2,3]
    mp_bias = [True, False]

    product = itertools.product(lr_options, wd_options, mp_latent_features_options, mp_output_features, num_mp_layers, num_nodewise_layers, mp_bias)
    product = list(product)
    print(f"Configurations to try: {len(product)}")
    # random permute, allow for "approximately anytime" stop
    random.shuffle(product)
    
    ArgsObj = namedtuple('ArgsObj', 'device, num_epochs, batch_size, lr, weight_decay, mp_input_features, mp_latent_features, mp_output_features, mp_bias, num_mp_layers, num_nodewise_layers')
    
    x, y, edge_index, edge_attr, scaler = load_data_to_tensors_and_transform()

    # Construct attention graph by replacing edge features with analytical attention weights
    CHISQUARE_ADJUST = 25
    edge_features = construct_attention_edge_features(x, edge_index, edge_attr, CHISQUARE_ADJUST)

    # Move to device and construct model
    x = x.to("cuda")
    y = y.to("cuda")
    edge_index = edge_index.to("cuda")
    edge_features = edge_features.to("cuda")

    # Create data loader
    data = Data(x,edge_index,edge_features,y)
    dataloader = DataLoader([data], batch_size=1)

    # Train-Test split
    node_transform = RandomNodeSplit(num_val=0, num_test=0.3) #30% test, 70% train
    node_splits = node_transform(data)
    train_mask = node_splits.train_mask
    test_mask = node_splits.test_mask

    # Init file
    with open(os.path.join(os.getcwd(), 'experiment_results', 'dirgnn', 'evaluation.csv'), 'w+') as f:
        f.write("lr, weight_decay, mp_latent_features, mp_output_features, num_mp_layers, num_nodewise_layers, mp_bias, all_training_losses, all_validation_losses")
    
    product_results = []
    for config in tqdm(product):
        lr, weight_decay, mp_latent_features, mp_output_features, num_mp_layers, num_nodewise_layers, mp_bias = config
    
        args = ArgsObj(device="cuda", num_epochs=200, batch_size=1, lr=lr, weight_decay=weight_decay, mp_input_features=14, mp_latent_features=mp_latent_features, mp_output_features=mp_output_features, mp_bias=mp_bias, num_mp_layers=num_mp_layers, num_nodewise_layers=num_nodewise_layers)

        # Create model
        model = AnalyticalAttentionNetwork(args)
        
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
        
        loss_fn = torch.nn.MSELoss()
        
        all_training_losses = []
        all_validation_losses = []
        for epoch_idx in range(args.num_epochs):

            loss_list = []
            val_loss_list = []
            # print(f'='*10 + f'Epoch {epoch_idx}' + '='*10)
            for idx, data in enumerate(dataloader):

                optimizer.zero_grad()
                x, y, edge_attr, edge_index = data.x, data.y, data.edge_attr, data.edge_index # unpack

                output = model.forward(x, edge_index, edge_attr)
                loss = loss_fn.forward(output[train_mask,:], y.unsqueeze(-1)[train_mask,:])

                loss_list.append(loss.item())

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    val_loss = loss_fn.forward(output[test_mask,:], y.unsqueeze(-1)[test_mask,:])
                    val_loss_list.append(val_loss.item())
                
            all_training_losses = all_training_losses + loss_list
            all_validation_losses = all_validation_losses+val_loss_list
        
        # Write down results
        # product_results.append({
        #     'config': {
        #         'lr': lr, 'weight_decay': weight_decay, 'mp_latent_features': mp_latent_features, 'mp_output_features': mp_output_features, 'num_mp_layers': num_mp_layers, 'num_nodewise_layers': num_nodewise_layers, 'mp_bias': mp_bias
        #     },
        #     'losses': all_losses, # useful for graphs
        #     'lowest_validation_loss': min(all_losses)
        # })

        # Append to file
        with open(os.path.join(os.getcwd(), 'experiment_results', 'dirgnn', 'evaluation.csv'), 'a') as f:
            f.write('\n' + ','.join([str(x) for x in [lr, weight_decay, mp_latent_features, mp_output_features, num_mp_layers, num_nodewise_layers, mp_bias, min(all_training_losses), min(all_validation_losses)]]))


    


if __name__ == "__main__":

    # Comment out for a regular training session
    hyperparameter_search()
    exit(0)

    parser = argparse.ArgumentParser(description="Train Model")

    # Attention graph hyperparameter
    parser.add_argument("--CHISQUARE_ADJUST", type=int, default=25, required=False)

    # Device for training
    parser.add_argument("--device", type=str, default='cuda', required=False)

    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=200, required=False)
    parser.add_argument("--batch-size", type=int, default=1, required=False)
    parser.add_argument("--lr", type=float, default=1e-4, required=False)
    parser.add_argument("--weight_decay", type=float, default=1e-5, required=False)

    # Message passing layers arguments
    parser.add_argument("--mp_input_features", type=int, default=14, required=False)
    parser.add_argument("--mp_latent_features", type=int, default=16, required=False)
    parser.add_argument("--mp_output_features", type=int, default=16, required=False)
    parser.add_argument("--mp_bias", type=bool, default=False, required=False)
    parser.add_argument("--num_mp_layers", type=int, default=4, required=False)

    # Decoder MLP arguments
    parser.add_argument("--num_nodewise_layers", type=int, default=1, required=False)

    args = parser.parse_args()

    train_dirgnn(args)

    train_baseline_regressor()