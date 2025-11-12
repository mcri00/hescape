import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from typing import Dict

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
'''

class mclSTExp_MLP(nn.Module):
    def __init__(self, temperature, image_embedding, spot_embedding, projection_dim, dropout=0.):
        super().__init__()
        self.x_embed = nn.Embedding(65536, spot_embedding)
        self.y_embed = nn.Embedding(65536, spot_embedding)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim)
        self.temperature = temperature

    def forward(self, batch):
        image_features = batch["image"]
        spot_features = batch["expression"]
        image_embeddings = self.image_projection(image_features)
        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_features + centers_x + centers_y

        spot_embeddings = self.spot_projection(spot_features)
        cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature
        #label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).cuda()
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1], device=cos_smi.device)

        spots_loss = F.cross_entropy(cos_smi, label)
        images_loss = F.cross_entropy(cos_smi.T, label.T)
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        return spot_embeddings, image_embeddings, loss.mean()
 '''   
class mclSTExp_Attention(nn.Module):
    def __init__(self, temperature, image_dim, spot_dim, projection_dim, heads_num, heads_dim, head_layers, dropout=0.):
        super().__init__()
        self.temperature = temperature

        self.x_embed = nn.Embedding(65536, spot_dim)
        self.y_embed = nn.Embedding(65536, spot_dim)

        self.spot_encoder = nn.Sequential(
            *[attn_block(spot_dim, heads=heads_num, dim_head=heads_dim, mlp_dim=spot_dim, dropout=dropout)
              for _ in range(head_layers)]
        )

        self.image_projection = ProjectionHead(embedding_dim=image_dim, projection_dim=projection_dim, dropout=dropout)
        self.spot_projection = ProjectionHead(embedding_dim=spot_dim, projection_dim=projection_dim, dropout=dropout)

    def forward(self, batch):
        image_features = batch["image"]  # directly pre-extracted features
        spot_features = batch["expression"]

        image_embeddings = self.image_projection(image_features)

        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_features + centers_x + centers_y
        spot_features = spot_features.unsqueeze(dim=0)

        spot_embeddings = self.spot_encoder(spot_features)
        spot_embeddings = self.spot_projection(spot_embeddings)
        spot_embeddings = spot_embeddings.squeeze(dim=0)

        sim_matrix = (spot_embeddings @ image_embeddings.T) / self.temperature
        labels = torch.eye(sim_matrix.shape[0], device=sim_matrix.device)

        loss = 0.5 * (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels.T))
        return spot_embeddings, image_embeddings, loss



# ===========================
#  mclSTExp_Attention Loader
# ===========================


def _strip_prefixes(sd: Dict[str, torch.Tensor], prefixes=("model.", "module.", "net.")):
    """Remove unwanted prefixes from checkpoint keys."""
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _load_any_ckpt(path: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch or safetensors checkpoint."""
    try:
        from safetensors.torch import load_file as safe_load
        if str(path).endswith(".safetensors"):
            return safe_load(path)
    except Exception:
        pass

    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint {path} is not a valid state_dict.")
    return sd


def load_mcl_from_cfg(cfg) -> nn.Module:
    """
    Build the mclSTExp_Attention model from a config dict and optionally load weights.
    Matches the style and behavior of load_stitch_from_cfg.
    """
    model = mclSTExp_Attention(
        temperature=float(cfg.get("temperature", 0.5)),
        image_dim=int(cfg["input_dim_image"]),
        spot_dim=int(cfg.get("input_dim_transcriptome") or cfg["hvg_k"]),
        projection_dim=int(cfg.get("mcl_projection_dim", cfg.get("output_dim", 128))),
        heads_num=int(cfg.get("mcl_heads_num", 8)),
        heads_dim=int(cfg.get("mcl_heads_dim", 64)),
        head_layers=int(cfg.get("mcl_heads_layers", 2)),
        dropout=float(cfg.get("dropout_prob", 0.0)),
    )

    ckpt = cfg.get("checkpoint", None)
    if ckpt:
        sd = _load_any_ckpt(ckpt)
        sd = _strip_prefixes(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[load] missing keys:", missing)
        print("[load] unexpected keys:", unexpected)

    return model
