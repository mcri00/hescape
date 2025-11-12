# hescape/models/pos/spatial_positional_encoder.py
import torch
import torch.nn as nn

class SpatialPositionalEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, coords_3d: torch.Tensor):
        pe = self.net(coords_3d)
        return self.dropout(pe)


