# hescape/models/stitch/dual_head_mlp.py
import torch
import torch.nn as nn

class _MLP(nn.Module):
    def __init__(self, in_dim, hid, out, num_layers=2, p=0.5):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hid), nn.ReLU()]
            if p > 0: layers.append(nn.Dropout(p))
            d = hid
        layers.append(nn.Linear(hid, out))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DualHeadMLP(nn.Module):
    """
    Sola parte 'due MLP' che mancava:
    - image_mlp:  in_dim_img -> out_dim
    - gene_mlp:   in_dim_tx  -> out_dim
    La fusione posizionale la tieni nel tuo ST_MODEL attuale (già presente).
    """
    def __init__(self, in_dim_img, in_dim_tx, hidden, out_dim, num_layers=2, p=0.5):
        super().__init__()
        self.image_mlp = _MLP(in_dim_img, hidden, out_dim, num_layers, p)
        self.gene_mlp  = _MLP(in_dim_tx,  hidden, out_dim, num_layers, p)

    def forward(self, img_feats, tx_feats):
        return self.image_mlp(img_feats), self.gene_mlp(tx_feats)
