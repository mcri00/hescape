#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- import path bootstrap (script or module) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT   = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT  = os.path.dirname(PKG_ROOT)
for p in (SCRIPT_DIR, PKG_ROOT, PROJ_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Try package-style first, fallback to local files.
try:
    from .muse_model import load_muse_from_cfg
    from .cfg_bridge import load_model_train_cfgs
except Exception:
    from muse_model import load_muse_from_cfg
    from cfg_bridge import load_model_train_cfgs

# ==== HESCAPE metrics (import) with graceful fallbacks ====
HAS_HESCAPE = True
try:
    from hescape.constants import DatasetEnum
    from hescape.metrics import EvalMetrics  # original implementation
except Exception:
    HAS_HESCAPE = False
    # Minimal fallback with the same R@k semantics as HESCAPE
    class _DE:
        IMG_EMBED = "img_embedding"
        GEXP_EMBED = "gexp_embedding"
    DatasetEnum = _DE  # type: ignore

    import torch

    class EvalMetrics:
        def __init__(self, stage: str, strategy: str,
                     recall_range=(1,5,10),
                     knn_recall_metrics=True,
                     knn_gexp_metrics=False,
                     classif_metrics=False,
                     scib_metrics=False,
                     **kwargs):
            self.stage = stage
            self.recall_range = recall_range
            self.knn_recall_metrics = knn_recall_metrics

        def knn_recall(self, left_embedding: torch.Tensor, right_embedding: torch.Tensor, embedding_type: str):
            # identical logic: dot-product on (optionally) L2-normalized embeds, argsort desc, same-index match
            metrics = {}
            group_idx = torch.arange(left_embedding.shape[0], device=left_embedding.device)
            emb_sim = left_embedding @ right_embedding.T
            left_gid = group_idx.unsqueeze(1).expand_as(emb_sim)
            right_gid = group_idx.unsqueeze(0).expand_as(emb_sim)
            right_simrank = torch.argsort(emb_sim, dim=1, descending=True)
            rightgid_sorted = torch.gather(right_gid, 1, right_simrank)
            rightgid_matched = (rightgid_sorted == left_gid)
            leftgid_hasmatch, leftgid_firstmatch = torch.max(rightgid_matched, dim=1)
            leftmatch_rank = leftgid_firstmatch[leftgid_hasmatch]
            for k in self.recall_range:
                match_count = (leftmatch_rank < k).sum()
                totcal_count = leftgid_hasmatch.sum()
                metrics[f"{self.stage}_R@{k}_{embedding_type}"] = (match_count / totcal_count).item()
            return metrics

        def __call__(self, batch):
            metrics = {}
            if self.knn_recall_metrics:
                L = batch[DatasetEnum.IMG_EMBED]
                R = batch[DatasetEnum.GEXP_EMBED]
                metrics.update(self.knn_recall(L, R, "I2G"))
                metrics.update(self.knn_recall(R, L, "G2I"))
            return metrics


# ----------------- config helpers -----------------

def _load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f) or {}
    else:
        raise ValueError("Use .yaml/.yml or .json for --cfg")

def _merge(overrides: dict, base: dict) -> dict:
    out = dict(base or {})
    out.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return out


# --- image prefix resolver (choose which image PCA CSV to read) ---
_IMAGE_PREFIX_BY_ENCODER = {
    "inception_v3": "img_inception_pca",
    "densenet121":  "img_densenet121_pca",
    "resnet18":     "img_resnet18_pca",
}
def resolve_image_prefix_from_cfg(cfg: dict) -> str:
    if "image_prefix" in cfg and cfg["image_prefix"]:
        return str(cfg["image_prefix"])
    enc = str(cfg.get("image_encoder", "")).lower()
    if enc in _IMAGE_PREFIX_BY_ENCODER:
        return _IMAGE_PREFIX_BY_ENCODER[enc]
    return "img_inception_pca"


# ----------------- I/O helpers -----------------

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_precomputed_bundle(
    root_dir: str,
    hvg_k: int,
    pca_dim: int,
    image_prefix: str = "img_inception_pca",
    gene_prefix: str  = "gene_hvg",
):
    root = Path(root_dir)
    g_path = root / f"{gene_prefix}_{int(hvg_k)}.csv"
    i_path = root / f"{image_prefix}_{int(pca_dim)}.csv"
    m_path = root / "labels_meta.csv"

    df_g = _read_csv(g_path)
    df_i = _read_csv(i_path)
    df_m = _read_csv(m_path)

    for df, n in [(df_g, "gene"), (df_i, "image"), (df_m, "meta")]:
        if "Image Name" not in df.columns:
            raise ValueError(f"{n} file lacks 'Image Name' column.")

    if not {"x", "y"}.issubset(df_m.columns):
        raise ValueError("labels_meta.csv must contain 'x' and 'y' columns.")

    df = df_m.merge(df_g, on="Image Name", how="inner").merge(df_i, on="Image Name", how="inner")
    if df.empty:
        raise RuntimeError("Join produced 0 rows. Check that the three files belong to the same export.")

    gene_cols = [c for c in df.columns if c.startswith("g_e")]
    img_cols  = [c for c in df.columns if c.startswith(f"{image_prefix}_e")]
    if len(gene_cols) != int(hvg_k):
        raise ValueError(f"Expected {hvg_k} gene columns, got {len(gene_cols)}.")
    if len(img_cols) != int(pca_dim):
        raise ValueError(f"Expected {pca_dim} image columns, got {len(img_cols)}.")

    Xgene  = df[gene_cols].to_numpy(dtype=np.float32)
    Ximg   = df[img_cols].to_numpy(dtype=np.float32)
    names  = df["Image Name"].astype(str).tolist()
    coords = df[["x", "y"]].to_numpy(dtype=np.float32)  # not used by MUSE, but kept for parity/debug

    rows_meta = df[df_m.columns].to_dict(orient="records")
    return Ximg, Xgene, names, coords, rows_meta


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser("Evaluate MUSE on precomputed CSV features (config-driven)")
    ap.add_argument("--cfg", required=True, help="YAML/JSON with paths and options")
    args = ap.parse_args()

    cfg_user = _load_cfg(args.cfg)

    # 1) Optionally derive defaults from training JSONs (dims / hypers)
    derived = {}
    if cfg_user.get("train_cfg_data") and cfg_user.get("train_cfg_model"):
        derived = load_model_train_cfgs(cfg_user["train_cfg_data"], cfg_user["train_cfg_model"])

    # 2) Merge: user overrides > derived
    cfg = _merge(cfg_user, derived)

    # Required paths
    pre_dir   = cfg.get("precomputed_dir")
    checkpoint= cfg.get("checkpoint")
    out_dir   = cfg.get("out_dir")
    if not pre_dir or not checkpoint or not out_dir:
        raise ValueError("Config must include: precomputed_dir, checkpoint, out_dir")

    # Required dims
    hvg_k   = int(cfg.get("hvg_k"))
    pca_dim = int(cfg.get("image_pca_dim") or cfg.get("pca_dim"))
    if hvg_k is None or pca_dim is None:
        raise ValueError("Config must specify hvg_k and (image_pca_dim or pca_dim) to match CSV filenames.")

    # Optional hypers
    n_hidden_muse = int(cfg.get("n_hidden_muse", 512))
    dim_z_muse    = int(cfg.get("dim_z_muse", 256))
    gene_log1p    = bool(cfg.get("gene_log1p", False))
    batch_size    = int(cfg.get("batch_size", 1024))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3) Load precomputed features
    image_prefix = resolve_image_prefix_from_cfg(cfg)
    print(f"[cfg] using image_prefix='{image_prefix}' (pca_dim={pca_dim})")
    Ximg, Xgene, names_all, coords_xy, rows_meta = load_precomputed_bundle(
        pre_dir, hvg_k=hvg_k, pca_dim=pca_dim, image_prefix=image_prefix, gene_prefix="gene_hvg"
    )
    if gene_log1p:
        Xgene = np.log1p(Xgene, dtype=np.float32)

    # 4) Build MUSE model (dims from loaded arrays) + load weights
    muse_cfg = {
        "checkpoint": cfg["checkpoint"],
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        "n_hidden_muse": n_hidden_muse,
        "dim_z_muse": dim_z_muse,
    }
    model = load_muse_from_cfg(muse_cfg).to(device).eval()
    print("[model] MUSE ready.")

    # 5) Inference on precomputed batches
    bs = int(batch_size)
    IMG_list, TX_list = [], []
    with torch.no_grad():
        for i in range(0, len(names_all), bs):
            Xi = torch.tensor(Ximg[i:i+bs],  dtype=torch.float32, device=device)  # images
            Xg = torch.tensor(Xgene[i:i+bs], dtype=torch.float32, device=device)  # genes
            # MUSE returns: x_hat, y_hat, z, encode_x, encode_y
            _x, _y, _z, encode_x, encode_y = model(Xg, Xi)
            TX_list.append(encode_x.detach().cpu().numpy())   # genes
            IMG_list.append(encode_y.detach().cpu().numpy())  # images

    IMG = np.concatenate(IMG_list, axis=0)
    TX  = np.concatenate(TX_list,  axis=0)
    print(f"[eval] encoded N={IMG.shape[0]} samples")

    # 6) Retrieval metrics (HESCAPE R@k)
    # L2-normalize like CLIP/HESCAPE
    IMG = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-9)
    TX  = TX  / (np.linalg.norm(TX,  axis=1, keepdims=True) + 1e-9)

    # Build batch for EvalMetrics
    EVAL_DEVICE = torch.device("cpu")
    batch = {
        DatasetEnum.IMG_EMBED: torch.tensor(IMG, dtype=torch.float32, device=EVAL_DEVICE),
        DatasetEnum.GEXP_EMBED: torch.tensor(TX,  dtype=torch.float32, device=EVAL_DEVICE),
    }
    em = EvalMetrics(
        stage="test",
        strategy="single",
        recall_range=(1,5,10),
        knn_recall_metrics=True,
        knn_gexp_metrics=False,
        classif_metrics=False,
        scib_metrics=False,
    )
    m = em(batch)  # keys: test_R@k_I2G, test_R@k_G2I

    # 7) Export embeddings + labels + metrics
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    img_df = pd.DataFrame(IMG, columns=[f"img_e{i}" for i in range(IMG.shape[1])]); img_df.insert(0, "Image Name", names_all)
    tx_df  = pd.DataFrame(TX,  columns=[f"tx_e{i}"  for i in range(TX.shape[1])]);  tx_df.insert(0,  "Image Name", names_all)
    labels_df = pd.DataFrame(rows_meta)

    img_df.to_csv(out_path / "image_embeddings.csv", index=False)
    tx_df.to_csv (out_path / "transcriptome_embeddings.csv", index=False)
    labels_df.to_csv(out_path / "labels.csv", index=False)

    metrics_row = {
        "n_samples": int(IMG.shape[0]),
        "hvg_k": int(Xgene.shape[1]),
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        "test_R@1_I2G": float(m["test_R@1_I2G"]),
        "test_R@5_I2G": float(m["test_R@5_I2G"]),
        "test_R@10_I2G": float(m["test_R@10_I2G"]),
        "test_R@1_G2I": float(m["test_R@1_G2I"]),
        "test_R@5_G2I": float(m["test_R@5_G2I"]),
        "test_R@10_G2I": float(m["test_R@10_G2I"]),
    }
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics_row, f, indent=2)
    write_header = not (out_path / "metrics.csv").exists()
    pd.DataFrame([metrics_row]).to_csv(out_path / "metrics.csv", index=False, mode="a", header=write_header)

    print("\n=== Retrieval (precomputed CSV) — MUSE (HESCAPE R@k) ===")
    print("I2G  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        metrics_row["test_R@1_I2G"], metrics_row["test_R@5_I2G"], metrics_row["test_R@10_I2G"]
    ))
    print("G2I  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        metrics_row["test_R@1_G2I"], metrics_row["test_R@5_G2I"], metrics_row["test_R@10_G2I"]
    ))

if __name__ == "__main__":
    main()
