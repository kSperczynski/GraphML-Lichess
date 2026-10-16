from collections.abc import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """Predict one rating per node from its own features, without using edges."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 32), dropout: float = 0.1):
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("Input and hidden dimensions must be positive, with at least one hidden layer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1).")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous_dim, width), nn.ReLU(), nn.Dropout(dropout)])
            previous_dim = width
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Tensor | None = None, edge_attr: Tensor | None = None) -> Tensor:
        """Accept the shared graph interface; predictions depend only on x."""
        return self.network(x)
