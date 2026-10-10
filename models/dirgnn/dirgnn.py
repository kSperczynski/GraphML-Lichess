import torch.nn as nn
from collections import OrderedDict
import torch_geometric as pyg
from models.dirgnn.dirgnn_graph_construction import construct_attention_edge_features

class MessagePassingLayer(pyg.nn.MessagePassing):
    def __init__(self, in_features: int, latent_features: int, out_features: int, use_bias: bool):
        # Use sum aggregation because we used softmax in the message function
        super().__init__(aggr="sum") 
        self.feat_dim = in_features
        self.latent_dim = latent_features
        self.out_dim = out_features
        self.use_bias = use_bias
        
        self.prepare_layer_mlps()

    def prepare_layer_mlps(self):
        # We have first an MLP to transform the input features to a latent space
        self.encode_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, self.latent_dim, bias=self.use_bias),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim, bias=self.use_bias),
            nn.BatchNorm1d(self.latent_dim)
        )

        # We then have an MLP to transform from latent space (after aggregation) to output
        #TODO: Experiment whether decoder should take own features separately
        self.decode_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=self.use_bias),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.out_dim, bias=self.use_bias),
            nn.BatchNorm1d(self.out_dim),
        )

    def forward(self, x, edge_index, edge_features):
        # Update node features using encoder MLP
        x = self.encode_mlp.forward(x)

        # Perform the message passing to update each node's embedding
        # This will implicitly call the message function for each node and do the aggregation
        encoded_and_aggregated = self.propagate(edge_index, x=x, edge_features=edge_features)

        # We also want to add the current node embedding to the sum
        # TODO: Consider decode MLP on concatenation
        encoded_and_aggregated = x + encoded_and_aggregated

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

        layers = {}

        layers[f"{0}_mp"] = (
            MessagePassingLayer(self.args.mp_input_features, self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_bias)
        )

        for i in range(1, self.args.num_mp_layers-1):
            layers[f"{i}_mp"] = (
                MessagePassingLayer(self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_bias)
            )

        layers[f"{self.args.num_mp_layers-1}_mp"] = (
            MessagePassingLayer(self.args.mp_latent_features, self.args.mp_latent_features, self.args.mp_output_features, self.args.mp_bias)
        )

        ordered_dict = OrderedDict(
            [(key, layers[key]) for key in layers.keys()]
        )

        return nn.Sequential(ordered_dict)

    def build_nodewise_mlp(self):

        layers = {}

        for layer_idx in range(self.args.num_nodewise_layers-1):
            layers[f'{layer_idx}_linear'] = (nn.Linear(in_features=self.args.hid_dim_nodewise, out_features=self.args.hid_dim_nodewise))
            layers[f'{layer_idx}_bn'] = (nn.BatchNorm1d(num_features=self.args.hid_dim_nodewise))
            layers[f'{layer_idx}_relu'] = nn.ReLU()

        # add deproject layer -> 1 output
        layers[f"{self.args.num_nodewise_layers-1}_linear"] = nn.Linear(in_features=self.args.hid_dim_nodewise, out_features=1)

        ordered_dict = OrderedDict(
            [(key, layers[key]) for key in layers.keys()]
        )
        return nn.Sequential(ordered_dict)
        

    def forward(self, x, edge_index, edge_features):
        # Forward MP layers
        x = self.mp_layers.forward(x, edge_index, edge_features)

        # Node-wise MLP
        x = self.decode_mlp.forward(x)

        return x
