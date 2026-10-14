import random

import numpy as np
import torch
import torch_geometric


def seed_everything(seed: int) -> None:
    """Seed: 'random', 'torch' and 'torch_geometric'"""
    random.seed(seed)
    np.random.seed(seed)
    torch_geometric.seed_everything(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
