from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from open_clip.loss import ClipLoss, SigLipLoss
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef

from hescape.constants import DatasetEnum
from hescape.models._utils import print_trainable_parameters
from hescape.models.gexp_models import GexpEncoder
from hescape.models.image_models import ImageEncoder

from functools import partial
from hescape.models.losses.stitch_loss import simclr_loss_func_spatial_reg
from hescape.models.pos.spatial_positional_encoder import SpatialPositionalEncoder
from hescape.models._utils import ForceFP32

from hescape.models.stitch.dual_head_mlp import DualHeadMLP

LOCAL_RANK = "LOCAL_RANK"

REGRESSION_METRICS = [
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
]

CLASSIFICATION_METRICS = [
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
]


class CLIPModel(nn.Module):
    """CLIP model."""

    def __init__(
        self,
        input_genes: int,
        embed_dim: int,
        img_enc_name: Literal["ctranspath", "densenet", "uni", "optimus", "conch", "gigapath"],
        gene_enc_name: Literal["drvi", "nicheformer", "scfoundation", "uce", "generic", "identity"],
        loss: Literal["CLIP", "SIGLIP", "SIMCLR_SPATIAL_REG"],
        img_finetune: bool = False,
        gene_finetune: bool = False,
        n_tissue: int | None = None,
        n_region: int | None = None,
        image_size: int = 224,
        temperature: float = 0.07,
        world_size: int = 1,
        rank: int = 0,
        cfg: DictConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        
        '''
        self.image_encoder = ImageEncoder(
            model_name=img_enc_name,
            embed_dim=embed_dim,
            finetune=img_finetune,
            checkpoint_path=kwargs.get("img_enc_path", None),
            proj=kwargs.get("img_proj", "linear"),
        )
        '''
        
        self.use_dual_mlp = bool(getattr(cfg.model, "dual_mlp", {}).get("enable", False))


        img_enc_kwargs = {
            "checkpoint_path": kwargs.get("img_enc_path", None),
        }
        if cfg.model.litmodule.img_enc_name == "inception_pca100":
            img_enc_kwargs.update({
                "pca_components_path": cfg.paths.pretrain_weights.img_pca_components_path,
                "pca_mean_path":       cfg.paths.pretrain_weights.img_pca_mean_path,
            })

        img_proj = kwargs.get("img_proj", "linear")
        gene_proj = kwargs.get("gene_proj", "linear")
        if self.use_dual_mlp:
            img_proj  = "identity"
            gene_proj = "identity"

        self.image_encoder = ImageEncoder(
            model_name=img_enc_name,
            finetune=img_finetune,
            embed_dim=embed_dim,
            proj=img_proj,
            **img_enc_kwargs,   # (come nel tuo codice)
        )

        self.gexp_encoder = GexpEncoder(
            input_genes=input_genes,
            model_name=gene_enc_name,
            checkpoint_path=kwargs.get("gene_enc_path", None),
            drvi_model_dir=kwargs.get("drvi_model_dir", None),
            n_region=n_region,
            n_tissue=n_tissue,
            finetune=gene_finetune,
            embed_dim=embed_dim,
            proj=gene_proj,
        )


        '''
        self.gene_enc_name = gene_enc_name
        self.gexp_encoder = GexpEncoder(
            input_genes=input_genes,
            model_name=gene_enc_name,
            checkpoint_path=kwargs.get("gene_enc_path", None),
            drvi_model_dir=kwargs.get("drvi_model_dir", None),
            n_region=n_region,
            n_tissue=n_tissue,
            finetune=gene_finetune,  # (Always fine-tune gene encoder)
            embed_dim=embed_dim,
            proj=kwargs.get("gene_proj", "linear"),
            # idx_genes_target=cfg.paths.idx_genes_target,
        )
        '''
        print_trainable_parameters(img_enc_name, self.image_encoder)
        print_trainable_parameters(gene_enc_name, self.gexp_encoder)
        
        
        if self.use_dual_mlp:
            in_dim_img = getattr(self.image_encoder.trunk, "num_features", None)
            if in_dim_img is None:
                in_dim_img = embed_dim  # fallback

            # per il gene trunk: prova a leggere una dim esplicita, altrimenti fallback
            in_dim_tx = getattr(self.gexp_encoder.trunk, "num_features", None)
            if in_dim_tx is None:
                # DRVI tipicamente 128; HVG generic/identity = #HVG; tieni un fallback robusto
                in_dim_tx = getattr(self.gexp_encoder, "input_genes", embed_dim)

            dcfg = cfg.model.dual_mlp
            self.dual_mlp = DualHeadMLP(
                in_dim_img=in_dim_img,
                in_dim_tx=in_dim_tx,
                hidden=256,
                out_dim=64,     
                num_layers=2,
                p=0.7,
            )
            
            pe_dim=64
        else:
            pe_dim=embed_dim
        

        
        # --------------------------
        # ✅ Default positional encoder
        # --------------------------
        # toggled from config: model.posenc.enable
        pe_cfg = getattr(cfg, "posenc", None)
        self.use_posenc = bool(getattr(pe_cfg, "enable", False)) if pe_cfg is not None else False

        if self.use_posenc:
            #pe_dim = 64
            pe_pdrop = 0.2
            self.pos_encoder = SpatialPositionalEncoder(in_dim=3, out_dim=pe_dim, p_drop=pe_pdrop)
            #self.pe_to_img = nn.Identity() if pe_dim == embed_dim else nn.Linear(pe_dim, embed_dim)
            #self.pe_to_rna = nn.Identity() if pe_dim == embed_dim else nn.Linear(pe_dim, embed_dim)
            self.ln_img = nn.LayerNorm(pe_dim)
            self.ln_rna = nn.LayerNorm(pe_dim)
            self.ln_pos = nn.LayerNorm(pe_dim)
            
            '''
            self.ln_img = nn.LayerNorm(embed_dim)
            self.ln_rna = nn.LayerNorm(embed_dim)
            self.ln_pos = nn.LayerNorm(embed_dim)
            '''
        
        # ------------------------
        # 2) CLIP Loss Setup
        # ------------------------
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.loss_kind = loss
        
        if loss == "CLIP":
            self.logit_bias = None
            loss_fn = ClipLoss(world_size=world_size, rank=rank)
        elif loss == "SIGLIP":
            self.logit_bias = nn.Parameter(torch.ones([]) * -10)
            loss_fn = SigLipLoss(world_size=world_size, rank=rank)
        elif loss == "SIMCLR_SPATIAL_REG":
            self.logit_bias = None  # not used
            # default params (overridden in module)
            loss_fn = partial(
                simclr_loss_func_spatial_reg,
                temperature=0.4261,
                w_pair=1.0, w_anchor=0.7, w_spatial=0.2,
                sigma=0.01, k_anchor=10, anchor_metric="cosine",
                residualize_for_anchors=True,
            )
        else:
            raise ValueError(f"Loss {loss} not supported.")
        self.loss = loss_fn

    def forward(self, batch: dict[str, torch.Tensor], norm: bool = True, coords3d: torch.Tensor | None = None):
        """Forward pass: returns (img_embed, gexp_embed, logit_scale.exp())."""
        '''
        gexp_encoder_input = batch[DatasetEnum.GEXP]
        region, tissue = None, None

        # Encode gene expressions
        gexp_embed = self.gexp_encoder(gexp_encoder_input, tissue, region)
        # Encode images
        img_embed = self.image_encoder(batch[DatasetEnum.IMG])
        '''
        gexp_in = batch[DatasetEnum.GEXP]

        # 1) feature grezze dagli encoder
        img_raw  = self.image_encoder(batch[DatasetEnum.IMG])    # [B, d_img]
        tx_raw   = self.gexp_encoder(gexp_in, None, None)        # [B, d_tx]

        # 2) i tuoi due MLP
        if self.use_dual_mlp:
            img_embed, gexp_embed = self.dual_mlp(img_raw, tx_raw)
        else:
            img_embed, gexp_embed = img_raw, tx_raw

        
        # === Positional embedding (always active)
        if self.use_posenc and coords3d is not None:
            pe = self.pos_encoder(coords3d)
            #pe_img = self.pe_to_img(pe)
            #pe_rna = self.pe_to_rna(pe)
            
            #print("[DEBUG] dims:", img_embed.shape, gexp_embed.shape, pe.shape, pe_img.shape, pe_rna.shape)
            '''
            img_embed = self.ln_img(img_embed) + self.ln_pos(pe_img)
            gexp_embed = self.ln_rna(gexp_embed) + self.ln_pos(pe_rna)
            '''
            img_embed = self.ln_img(img_embed) + self.ln_pos(pe)
            gexp_embed = self.ln_rna(gexp_embed) + self.ln_pos(pe)

        if norm:
            return (
                F.normalize(img_embed, p=2, dim=-1),
                F.normalize(gexp_embed, p=2, dim=-1),
                self.logit_scale.exp(),
            )
        return img_embed, gexp_embed, self.logit_scale.exp()

    def compute_loss(self, img_embed, gexp_embed, coords3d: torch.Tensor | None = None):  # default 1.0
        """
        Compute the total loss comprising:
          - Contrastive CLIP loss.
        """
        if self.loss_kind in ("CLIP", "SIGLIP"):
            if self.logit_bias is not None:
                contrastive_loss = self.loss(
                    img_embed, gexp_embed, logit_scale=self.logit_scale.exp(), logit_bias=self.logit_bias
                )
            else:
                contrastive_loss = self.loss(img_embed, gexp_embed, logit_scale=self.logit_scale.exp())
        else:
            assert coords3d is not None, "coords3d required for SIMCLR_SPATIAL_REG"
            contrastive_loss = self.loss(img_embed, gexp_embed, coords3d)
            
        return contrastive_loss
