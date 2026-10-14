from torch import nn


# TODO If we deem it beneficial we can design our own gat, we can come up with an architecture that works well for our usecase
# Instead of using the default one from pytorch. We could for example do some extra things between layers etc.
class CustomGATRegressor(nn.Module):
    """Customizable GAT using GATv2Conv layers"""

    def __init__(
        self,
    ):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()
