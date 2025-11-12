# src/hescape/models/custom_stitch.py
import json
import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------- Inception feature extractor (frozen) ----------
class InceptionExtractor(nn.Module):
    def __init__(self, trainable: bool = False):
        super().__init__()
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                aux_logits=False)
        # keep backbone (till pool)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B,2048,1,1]
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):  # [B,3,H,W] normalized by DM
        f = self.backbone(x).squeeze(-1).squeeze(-1)  # [B,2048]
        return F.normalize(f, dim=-1)


# -------- Gene preprocessing (top-K + z-score) ----------
class GenePreprocess(nn.Module):
    def __init__(self, topk_idx, mu, sigma, zscore=True):
        super().__init__()
        topk_idx = torch.as_tensor(topk_idx, dtype=torch.long)
        self.register_buffer("topk_idx", topk_idx)
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float).clamp_min(1e-6))
        self.zscore = zscore

    def forward(self, g):  # [B, G_total]
        g = g[:, self.topk_idx]  # [B, K]
        if self.zscore:
            g = (g - self.mu) / self.sigma
        return g


# -------- STitch wrappers (image and gene encoders) ----------
class StitchImageEncoder(nn.Module):
    def __init__(self, stitch_model: nn.Module, proj_dim: int):
        super().__init__()
        self.stitch = stitch_model
        self.inception = InceptionExtractor()
        self.proj = nn.Linear(2048, proj_dim)

    @torch.no_grad()
    def forward(self, x):  # [B,3,H,W]
        f = self.inception(x)        # [B,2048]
        z = self.proj(f)             # [B,D]
        return F.normalize(z, dim=-1)


class StitchGeneEncoder(nn.Module):
    def __init__(self, stitch_model: nn.Module, proj_dim: int, topk_idx, mu, sigma):
        super().__init__()
        self.stitch = stitch_model
        self.prep = GenePreprocess(topk_idx, mu, sigma)
        self.proj = nn.Linear(len(topk_idx), proj_dim)

    @torch.no_grad()
    def forward(self, g):  # [B,G]
        g = self.prep(g)             # [B,K]
        z = self.proj(g)             # [B,D]
        return F.normalize(z, dim=-1)


# -------- Factory: builds encoders given Hydra cfg --------
def _load_stats(stats_dir: str, dataset_name: str):
    """
    Load per-dataset gene stats computed by scripts/compute_gene_stats.py.
    Expected file: {stats_dir}/{dataset_name}.json
    """
    path = os.path.join(stats_dir, f"{dataset_name}.json")
    with open(path, "r") as f:
        obj = json.load(f)
    return obj["topk_idx"], obj["mu"], obj["sigma"]


def build_custom_encoders(cfg) -> Tuple[nn.Module, nn.Module]:
    """
    Returns (img_encoder, gene_encoder) to be used by LitStitch.
    """
    # ---- import your STitch model (only for consistency with your codebase) ----
    # TODO: fix this import to your actual project path:
    # from your_repo.models.st_model import ST_MODEL
    from models.mlp import ST_MODEL  # <-- if your ST_MODEL is importable as in your code snippet

    stitch = ST_MODEL(
        input_dim_image=cfg.model.stitch.input_dim_image,           # 2048 (Inception)
        input_dim_transcriptome=cfg.model.stitch.input_dim_transcriptome,  # >=100
        hidden_dim=cfg.model.stitch.hidden_dim,
        output_dim=cfg.model.litmodule.embed_dim,
        num_layers=cfg.model.stitch.num_layers,
        dropout_prob=cfg.model.stitch.dropout,
        use_positional_embedding=True,
        use_positional_attention=False,
        use_feature_selector=False,
    )

    # gene stats
    topk_idx, mu, sigma = _load_stats(
        stats_dir=cfg.model.stitch.stats_dir,
        dataset_name=cfg.datamodule.dataset_name,
    )

    img_enc = StitchImageEncoder(stitch, proj_dim=cfg.model.litmodule.embed_dim)
    gene_enc = StitchGeneEncoder(stitch,
                                 proj_dim=cfg.model.litmodule.embed_dim,
                                 topk_idx=topk_idx, mu=mu, sigma=sigma)
    return img_enc, gene_enc
