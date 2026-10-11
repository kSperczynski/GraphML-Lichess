from torch import nn


# TODO design own gat, we can come up with an architecture that works well for our usecase
class CustomGATRegressor(nn.Module):
    """Customizable GAT using GATv2 layers"""

    def __init__(
        self,
    ):
        raise NotImplementedError("Still has to implement own GAT")

    def forward(self):
        raise NotImplementedError("Still has to impelemnt own GAT")
