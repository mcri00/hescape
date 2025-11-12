from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Literal

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from hescape._logging import logger
from hescape.models import CLIPModel


from collections import defaultdict

from hescape.constants import DatasetEnum

import pandas as pd
import numpy as np

def world_info_from_env():
    # from https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


class ClampCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.model.logit_scale.clamp_(min=0, max=math.log(100))

def extract_xy_from_cell_coords(coords) -> torch.Tensor:
    """
    coords: tipicamente tensor/list di shape [B, 1, 2] o [B,2]
    Ritorna tensor [B,2] su stesso device.
    """
    if isinstance(coords, torch.Tensor):
        if coords.dim() == 3:   # [B, 1, 2]
            xy = coords[:, 0, :2]
        else:                   # [B, 2]
            xy = coords[:, :2]
        return xy.to(dtype=torch.float32)
    else:
        # lista di [[x,y]] → tensor
        xy = torch.tensor([c[0][:2] if isinstance(c[0], (list, tuple)) else c[:2] for c in coords], dtype=torch.float32)
        return xy


'''
@torch.no_grad()
def build_coords3d_no_patient(xy: torch.Tensor, slice_ids) -> torch.Tensor:
    """
    xy:        [B,2] float tensor con (x,y) grezze
    slice_ids: lista/array/torch.Tensor di id-slice (qui = 'name'), usata solo per il grouping
    Ritorna:   [B,3] con (x,y) z-score per slice e z=0
    """
    device, dtype = xy.device, xy.dtype
    B = xy.size(0)

    # rende slice_ids indicizzabili
    if isinstance(slice_ids, torch.Tensor):
        # ok usare l'int “codificato” come chiave di gruppo
        sl = slice_ids.detach().cpu().tolist()
    else:
        sl = list(slice_ids)

    xy_norm = xy.clone()
    by_slice = defaultdict(list)
    for i, s in enumerate(sl):
        by_slice[s].append(i)

    eps = 1e-6
    for _, idxs in by_slice.items():
        ii = torch.as_tensor(idxs, device=device)
        mu = xy_norm.index_select(0, ii).mean(0, keepdim=True)
        sd = xy_norm.index_select(0, ii).std(0, keepdim=True).clamp_min(eps)
        xy_norm[ii] = (xy_norm.index_select(0, ii) - mu) / sd

    z = torch.zeros((B, 1), device=device, dtype=dtype)
    return torch.cat([xy_norm, z], dim=1)  # [B,3]
'''



@torch.no_grad()
def build_3d_coords_safe(
    xy_coords: torch.Tensor,                  # [B,2]
    slice_ids_str,                            # list/array of per-panel ids (e.g., name)
    patient_ids_str=None,                     # list/array of per-patient ids (or None → infer from slice)
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    if dtype is None:
        dtype = xy_coords.dtype
    B = xy_coords.shape[0]

    # Normalize (x,y) per slice/panel
    df = pd.DataFrame({"slice": np.asarray(slice_ids_str)})
    if patient_ids_str is None:
        df["patient"] = df["slice"].astype(str).str.split("_").str[0]
    else:
        df["patient"] = np.asarray(patient_ids_str)

    xy_np = xy_coords.detach().to("cpu").numpy().astype(np.float32)
    for s in df["slice"].unique():
        idx = (df["slice"] == s).values
        mu = xy_np[idx].mean(axis=0)
        sd = xy_np[idx].std(axis=0) + 1e-6
        xy_np[idx] = (xy_np[idx] - mu) / sd
    xy_norm = torch.from_numpy(xy_np).to(device=device, dtype=dtype)

    # z = relative slice order within each patient, zero-mean per patient
    z_rel_np = np.zeros(B, dtype=np.float32)
    for p in df["patient"].unique():
        m = (df["patient"] == p).values
        slices_p = sorted(df.loc[m, "slice"].unique().tolist())
        s2i = {s: i for i, s in enumerate(slices_p)}
        z_vals = df.loc[m, "slice"].map(s2i).to_numpy(dtype=float)
        if len(slices_p) > 1:
            z_vals = z_vals / (len(slices_p) - 1)
        z_vals = z_vals - z_vals.mean()
        z_rel_np[m] = z_vals.astype(np.float32)

    z_rel = torch.from_numpy(z_rel_np).to(device=device, dtype=dtype).unsqueeze(1)
    return torch.cat([xy_norm, z_rel], dim=1)  # [B,3]


def _to_str_list(x):
    # x may be a torch.Tensor of ints or a python list
    if isinstance(x, torch.Tensor):
        return [str(v) for v in x.detach().cpu().tolist()]
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v) for v in x]
    # fallback
    return [str(x)]


class PretrainModule(LightningModule):
    def __init__(
        self,
        input_genes: int,
        embed_dim: int,
        img_enc_name: Literal["ctranspath", "uni", "conch", "optimus", "densenet", "gigapath"],
        gene_enc_name: Literal["drvi", "nicheformer", "uce", "scfoundation", "generic"],
        loss: Literal["CLIP", "SIGLIP", "SIMCLR_SPATIAL_REG"],
        img_finetune: bool,
        gene_finetune: bool,
        img_proj: Literal["linear", "mlp", "transformer"],
        gene_proj: Literal["linear", "mlp", "identity"],
        n_tissue: int,
        n_region: int,
        image_size: int,
        temperature: float,
        lr: float,
        weight_decay: float,
        cfg: DictConfig,
        lambda_scheduler: Callable | None,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        local_rank, global_rank, world_size = world_info_from_env()

        logger.info(f"LOCAL RANK: {local_rank}")
        logger.info(f"GLOBAL RANK: {global_rank}")
        logger.info(f"WORLD SIZE: {world_size}")

        if torch.cuda.is_available():
            logger.info(f"CUDA DEVICE NAME: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"CUDA DEVICE INDEX: {torch.cuda.current_device()}")
            logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")
        else:
            logger.info("CUDA is not available.")

        self.model = CLIPModel(
            input_genes=input_genes,
            embed_dim=embed_dim,
            img_enc_name=img_enc_name,
            loss=loss,
            img_finetune=img_finetune,
            gene_finetune=gene_finetune,
            gene_enc_name=gene_enc_name,
            image_size=image_size,
            n_tissue=n_tissue,
            n_region=n_region,
            temperature=temperature,
            world_size=world_size,
            rank=local_rank,
            img_enc_path=self.cfg.paths.pretrain_weights.img_enc_path,
            gene_enc_path=self.cfg.paths.pretrain_weights.gene_enc_path,
            drvi_model_dir=self.cfg.paths.anatomy.pretrain_weights.drvi_model_dir,
            cfg=cfg,
            img_proj=img_proj,
            gene_proj=gene_proj,
        )
        # === debug print
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print("TRAINABLE:", n, p.shape)
        self.lambda_scheduler: Callable[[int], float] | None = lambda_scheduler
        self.lr = lr
        self.weight_decay = weight_decay

        # evaluations
        self.eval_outputs = {"val": [], "test": []}
        self.strategy = self.cfg.training.lightning.trainer.strategy
        self.eval_batch_key = self.cfg.training.evaluations.batch_key
        self.eval_label_key = self.cfg.training.evaluations.label_key

    def configure_optimizers(self):
        params = []
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)

        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), lr=self.lr, weight_decay=self.weight_decay)
        if self.lambda_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lambda_scheduler)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        loss, metrics = self.shared_step(batch, "train")

        loss_params = {}
        logit_bias = getattr(self.model, "logit_bias", None)
        if logit_bias is not None:
            loss_params["logit_bias"] = logit_bias.item()
        loss_params["logit_scale"] = self.model.logit_scale.item()
        metrics.update(loss_params)
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        _, metrics = self.shared_step(batch, "val")
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return metrics

    def test_step(self, batch, batch_idx):
        bs = batch["image"].shape[0]
        _, metrics = self.shared_step(batch, "test")
        self.log_dict(metrics, sync_dist=True, batch_size=bs)
        return metrics
    '''
    #original
    def shared_step(self, batch, stage: str):
        img_embed, gexp_embed, logit_scale = self.model(batch, norm=False)
        # For batch "ID" and "GEXP"
        # source_ids = batch[DatasetEnum.ID].to(torch.int64)
        # source_exp = batch[DatasetEnum.GEXP].to(img_embed.dtype)

        contrastive_loss = self.model.compute_loss(  # recon_loss, cls_loss
            img_embed=img_embed, gexp_embed=gexp_embed
        )

        metrics = {f"{stage}_loss": contrastive_loss}

        knn_recall_i2g = get_clip_metrics(img_embed, gexp_embed, logit_scale=logit_scale, stage=stage)
        metrics.update(knn_recall_i2g)

        return contrastive_loss, metrics
    '''
    # new for stitch loss
    def shared_step(self, batch, stage: str):
        #img_embed, gexp_embed, logit_scale = self.model(batch, norm=False)

        # pick a device (prefer the image tensor; fallback to model params)
        if isinstance(batch[DatasetEnum.IMG], torch.Tensor):
            dev = batch[DatasetEnum.IMG].device
        else:
            dev = next(self.model.parameters()).device

        coords3d = None
        if self.model.loss_kind == "SIMCLR_SPATIAL_REG":
            
            '''
            #old version working but not performing the same way
            names  = batch.get(DatasetEnum.NAME)
            coords = batch.get("cell_coords")
            
            if (names is not None) and (coords is not None):
                xy = extract_xy_from_cell_coords(coords).to(dev)
                coords3d = build_coords3d_no_patient(xy, names).to(dev)
                
                #prova vs explosion
                # robustify + clamp range
                coords3d = torch.nan_to_num(coords3d, nan=0.0, posinf=0.0, neginf=0.0)
                coords3d = torch.tanh(coords3d / 3.0) * 3.0   # squash extremes to [-3, 3]

                # 🔍 print debug for first batch only
                if (not hasattr(self, "_printed_coords_debug")):
                    self._printed_coords_debug = True
                    print("\n=== DEBUG COORDS SAMPLE ===")
                    print(f"Stage: {stage}")
                    print(f"XY coords shape: {xy.shape}")
                    print(f"Coords3D shape: {coords3d.shape}")
                    print(f"First 5 coords3D:\n{coords3d[:5].detach().cpu().numpy()}")
                    if isinstance(names, torch.Tensor):
                        print(f"Slice IDs (names) sample: {names[:5].detach().cpu().numpy()}")
                    else:
                        print(f"Slice IDs (names) sample: {names[:5]}")
                    print("============================\n")
            '''
            names  = batch.get(DatasetEnum.NAME)
            coords = batch.get("cell_coords")

            coords3d = None
            if (names is not None) and (coords is not None):
                # 1) XY from Hescape: take the first pair [[x,y]]
                xy = extract_xy_from_cell_coords(coords).to(dev)      # [B,2], float32

                # 2) Use panel "name" as the SLICE/PANEL id
                slice_ids = _to_str_list(names)                       # list[str] like ["1","1",...]
                # 3) No real patient id in panel → make patient == slice (z will be 0)
                patient_ids = slice_ids

                # 4) Build robust coords
                coords3d = build_3d_coords_safe(
                    xy_coords=xy,
                    slice_ids_str=slice_ids,
                    patient_ids_str=patient_ids,
                    device=dev
                )

                # (optional) robustify & clamp extremes
                coords3d = torch.nan_to_num(coords3d, nan=0.0, posinf=0.0, neginf=0.0)
                coords3d = torch.tanh(coords3d / 3.0) * 3.0

                if (not hasattr(self, "_printed_coords_debug")):
                    self._printed_coords_debug = True
                    print("\n=== DEBUG COORDS SAMPLE ===")
                    print(f"Stage: {stage}")
                    print(f"XY coords shape: {xy.shape}")
                    print(f"Coords3D shape: {coords3d.shape}")
                    print(f"First 5 coords3D:\n{coords3d[:5].detach().cpu().numpy()}")
                    print(f"Slice IDs (panel names) sample: {slice_ids[:5]}")
                    print("============================\n")


        img_embed, gexp_embed, logit_scale = self.model(batch, norm=False, coords3d=coords3d)
        
        for name, t in {"img_embed": img_embed, "gexp_embed": gexp_embed}.items():
            if not torch.isfinite(t).all():
                bad = ~torch.isfinite(t)
                print(f"[{name}] non-finite at step {self.global_step}: "
                    f"{bad.sum().item()} / {t.numel()} values")
                t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)

        #prova vs explosion
        '''
        img_embed = torch.nn.functional.normalize(img_embed, p=2, dim=-1, eps=1e-6)
        gexp_embed  = torch.nn.functional.normalize(gexp_embed, p=2, dim=-1, eps=1e-6)
        '''
        
        # --- compute loss ---
        if self.model.loss_kind == "SIMCLR_SPATIAL_REG":
            assert coords3d is not None, "coords3d required for SIMCLR_SPATIAL_REG"
            contrastive_loss = self.model.compute_loss(img_embed, gexp_embed, coords3d=coords3d)
        else:
            contrastive_loss = self.model.compute_loss(img_embed, gexp_embed)

        metrics = {f"{stage}_loss": contrastive_loss}
        knn_recall_i2g = get_clip_metrics(img_embed, gexp_embed, logit_scale=logit_scale, stage=stage)
        metrics.update(knn_recall_i2g)
        return contrastive_loss, metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch)


def get_clip_metrics(image_features, gene_features, logit_scale, stage: str):
    metrics = {}
    logits_per_image = logit_scale * image_features @ gene_features.T
    logits_per_gene = logits_per_image.T

    logits = {"image_to_gene": logits_per_image, "gene_to_image": logits_per_gene}
    ground_truth = torch.arange(len(gene_features), device=gene_features.device).contiguous().view(-1, 1)

    # metrics for +ve and -ve pairs
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.float()
        metrics[f"{stage}/{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{stage}/{name}_median_rank"] = torch.floor(torch.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{stage}/{name}_R@{k}"] = torch.mean((preds < k).float())

    return metrics
