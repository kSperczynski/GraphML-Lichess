import torch
import torch.nn as nn
from collections import OrderedDict
import torch_geometric as pyg
from models.dirgnn.dirgnn_graph_construction import construct_attention_edge_features

class MessagePassingLayer(pyg.nn.MessagePassing):
    def __init__(self, in_features: int, latent_features: int, out_features: int, use_bias: bool, device:str):
        # Use sum aggregation because we used softmax in the message function
        super().__init__(aggr="sum") 
        self.feat_dim = in_features
        self.latent_dim = latent_features
        self.out_dim = out_features
        self.use_bias = use_bias
        self.device = device
        
        self.prepare_layer_mlps()

    def prepare_layer_mlps(self):
        # We have first an MLP to transform the input features to a latent space
        self.encode_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.latent_dim, bias=self.use_bias, device=self.device),
            nn.BatchNorm1d(self.latent_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=self.use_bias, device=self.device),
            nn.BatchNorm1d(self.latent_dim, device=self.device)
        )

        # We then have an MLP to transform from latent space (after aggregation) to output
        #TODO: Experiment whether decoder should take own features separately
        self.decode_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim, bias=self.use_bias, device=self.device),
            nn.BatchNorm1d(self.latent_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.out_dim, bias=self.use_bias, device=self.device),
            nn.BatchNorm1d(self.out_dim, device=self.device),
        )

    def forward(self, x, edge_index, edge_features):
        # Update node features using encoder MLP
        x = self.encode_mlp.forward(x)

        # Perform the message passing to update each node's embedding
        # This will implicitly call the message function for each node and do the aggregation
        encoded_and_aggregated = self.propagate(edge_index, x=x, edge_features=edge_features)

        # We also want to add the current node embedding to the sum
        # TODO: Consider decode MLP on concatenation
        encoded_and_aggregated = torch.concat([x, encoded_and_aggregated], dim=1)

        # We now apply the decode MLP
        out = self.decode_mlp.forward(encoded_and_aggregated)

        # Return the new node embeddings
        return out

    def message(self, x_j, edge_features):
        # Computes the message for each edge in the graph
        # The edge features are now the softmax weights, so we only need to element wise multiply

        x_j = edge_features * x_j

        return x_j

class AnalyticalAttentionNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Stack MP layers
        self.mp_layers = self.build_message_passing_layers()

        # Add Final MLP
        self.decode_mlp = self.build_nodewise_mlp()

    def build_message_passing_layers(self):

        layers = nn.ModuleList()

        layers.append(
            MessagePassingLayer(self.args.mp_input_features, self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_bias, self.args.device)
        )

        for i in range(1, self.args.num_mp_layers-1):
            layers.append(
                MessagePassingLayer(self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_bias, self.args.device)
            )

        layers.append(
            MessagePassingLayer(self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_output_features, self.args.mp_bias, self.args.device)
        )

        return layers

    def build_nodewise_mlp(self):

        layers = nn.ModuleList()

        for layer_idx in range(self.args.num_nodewise_layers-1):
            layers.append(nn.Linear(in_features=self.args.mp_output_features, out_features=self.args.mp_output_features, device=self.args.device))
            # layers.append(nn.BatchNorm1d(num_features=self.args.mp_output_features, device=self.args.device))
            layers.append(nn.ReLU())

        # add deproject layer -> 1 output
        layers.append(nn.Linear(in_features=self.args.mp_output_features, out_features=1, device=self.args.device))

        return layers
        

    def forward(self, x, edge_index, edge_features):
        # Forward MP layers
        for layer in self.mp_layers:
            x = layer.forward(x, edge_index, edge_features)

        # Node-wise MLP
        for layer in self.decode_mlp:
            x = layer.forward(x)

        return x
