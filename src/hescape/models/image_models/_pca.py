# hescape/models/image_models/_pca.py
import torch, torch.nn as nn, numpy as np

class PCALinear(nn.Module):
    """Apply (x - mu) @ W where W: [d_in, d_out], mu: [d_in]."""
    def __init__(self, components_path: str, mean_path: str):
        super().__init__()
        W  = np.load(components_path).astype("float32")   # [d_in, d_out]
        mu = np.load(mean_path).astype("float32")         # [d_in]
        self.register_buffer("W",  torch.from_numpy(W))
        self.register_buffer("mu", torch.from_numpy(mu))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) @ self.W
