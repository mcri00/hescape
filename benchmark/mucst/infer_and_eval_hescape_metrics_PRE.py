#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- import-path bootstrap ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT   = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT  = os.path.dirname(PKG_ROOT)
for p in (SCRIPT_DIR, PKG_ROOT, PROJ_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# === MuCST model (riusa le tue classi) ===
# Se le classi sono in file separati, importa da lì; qui le reimportiamo se già nel PYTHONPATH.
try:
    from .mucst_model import load_mucst_from_cfg
    from .cfg_bridge import load_model_train_cfgs
except Exception:
    from mucst_model import load_mucst_from_cfg
    from cfg_bridge import load_model_train_cfgs
# ----------------- cfg helpers -----------------
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

_IMAGE_PREFIX_BY_ENCODER = {
    "inception_v3": "img_inception_pca",
    "densenet121":  "img_densenet121_pca",
    "resnet18":     "img_resnet18_pca",
}
def resolve_image_prefix_from_cfg(cfg: dict) -> str:
    if cfg.get("image_prefix"): return str(cfg["image_prefix"])
    enc = str(cfg.get("image_encoder", "")).lower()
    return _IMAGE_PREFIX_BY_ENCODER.get(enc, "img_inception_pca")

# ----------------- I/O CSV -----------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_precomputed_bundle(root_dir: str, hvg_k: int, pca_dim: int,
                            image_prefix: str = "img_inception_pca",
                            gene_prefix: str  = "gene_hvg"):
    """
    Ritorna:
      Ximg (N,D_img), Xgene (N,D_gene), names (N), coords_xy (N,2), slice_ids (N), rows_meta(df_m columns)
    """
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
        raise RuntimeError("Join produced 0 rows. Check exports belong to the same subset.")

    gene_cols = [c for c in df.columns if c.startswith("g_e")]
    img_cols  = [c for c in df.columns if c.startswith(f"{image_prefix}_e")]
    if len(gene_cols) != int(hvg_k):  raise ValueError(f"Expected {hvg_k} gene columns, got {len(gene_cols)}.")
    if len(img_cols)  != int(pca_dim): raise ValueError(f"Expected {pca_dim} image columns, got {len(img_cols)}.")

    Xgene  = df[gene_cols].to_numpy(dtype=np.float32)
    Ximg   = df[img_cols].to_numpy(dtype=np.float32)
    names  = df["Image Name"].astype(str).tolist()
    coords = df[["x", "y"]].to_numpy(dtype=np.float32)

    # “slice id” per la normalizzazione per-slice (usa atlas/name_category se presenti)
    atlas_series   = df["atlas"].fillna("").astype(str) if "atlas" in df.columns else pd.Series([""] * len(df))
    namecat_series = df["name_category"].fillna("").astype(str) if "name_category" in df.columns else pd.Series([""] * len(df))
    slice_ids = [a if (a not in ("", "NA", "None")) else ncat for a, ncat in zip(atlas_series.tolist(), namecat_series.tolist())]

    rows_meta = df[df_m.columns].copy()
    return Ximg, Xgene, names, coords, slice_ids, rows_meta

# ----------------- piccoli utilities MuCST-like -----------------
def _safe_knn_adj_from_xy(xy: np.ndarray, k: int = 15) -> np.ndarray:
    """Costruisce un grafo simmetrico con self-loop, clampando k a [1, N-1]."""
    from sklearn.metrics import pairwise_distances
    N = xy.shape[0]
    kk = min(max(1, k), max(1, N-1))
    D = pairwise_distances(xy, metric="euclidean")
    neigh = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        order = np.argsort(D[i])
        for j in order[1:kk+1]:
            neigh[i, j] = 1.0
        neigh[i, i] = 1.0
    A = neigh + neigh.T
    A[A > 1] = 1.0
    return A

def _normalize_adj_with_self_loops(A: np.ndarray) -> np.ndarray:
    """D^{-1/2}(A+I)D^{-1/2} con gestione righe-zero."""
    import scipy.sparse as sp
    A = sp.coo_matrix(A) + sp.eye(A.shape[0])
    row_sum = np.array(A.sum(1)).flatten()
    row_sum[row_sum == 0] = 1.0
    d_inv_sqrt = np.power(row_sum, -0.5)
    D_inv = sp.diags(d_inv_sqrt)
    return (D_inv @ A @ D_inv).toarray().astype(np.float32)

def _cosine_sim(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return np.clip(Xn @ Xn.T, 0.0, 1.0).astype(np.float32)

def _build_adata_like_training(Ximg, Xgene, coords_xy, names, rows_meta, k_graph=15):
    """
    Replicazione minima di ciò che usavi:
      - obsm['spatial'], ['image_feature'], ['aug_image_feature1/2'] (qui copie o +noise),
      - obsm['feat'], ['feat_fake'] (perm shuffle),
      - obsm['mor_adj'] = adj_spatial * cos_sim(img),
      - obsm['label_CSL'] (positivi/negativi),
      - obs: Tumor (se presente) + idx + names
    """
    N = len(names)
    # grafo spaziale
    A_sp = _safe_knn_adj_from_xy(coords_xy, k=k_graph)
    # similarità morfologica sulle features immagine
    morph_sim = _cosine_sim(Ximg)
    mor_adj = (A_sp * morph_sim).astype(np.float32)

    # features “gene” e “fake”
    feat = Xgene.astype(np.float32)
    idx = np.arange(N)
    fake = feat.copy()
    np.random.shuffle(fake)

    # augment immagini: copia (oppure piccola noise gaussian)
    aug1 = Ximg.copy()
    aug2 = Ximg.copy()
    # esempio noise lieve (disattivo di default):
    # aug1 += 0.01 * np.random.randn(*aug1.shape).astype(np.float32)
    # aug2 += 0.01 * np.random.randn(*aug2.shape).astype(np.float32)

    # label_CSL shape [N,2]: [1,0] pos, [0,1] neg (come nel tuo training)
    one = np.ones((N,1), dtype=np.float32)
    zero= np.zeros((N,1), dtype=np.float32)
    label_CSL = np.concatenate([one, zero], axis=1)

    # labels se presenti in rows_meta
    tumor = rows_meta["Tumor"].to_numpy(dtype=np.int64) if "Tumor" in rows_meta.columns else np.zeros(N, dtype=np.int64)

    pack = {
        "feat": feat,
        "feat_fake": fake,
        "image_feature": Ximg.astype(np.float32),
        "aug_image_feature1": aug1.astype(np.float32),
        "aug_image_feature2": aug2.astype(np.float32),
        "mor_adj": mor_adj,           # grezzo (prima della normalizzazione)
        "names": names,
        "idx": idx.astype(np.int64),
        "tumor": tumor,
    }
    return pack

# ----------------- Recall@k (paired) -----------------
@torch.no_grad()
def paired_recall_at_k(left_emb: torch.Tensor, right_emb: torch.Tensor, ks=(1,5,10)) -> dict:
    sim = left_emb @ right_emb.T
    rank = torch.argsort(sim, dim=1, descending=True)
    idx = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1)
    eq = (rank == idx)
    has_match, pos = torch.max(eq, dim=1)
    assert bool(has_match.all()), "Ogni query deve avere il suo match (paired)."
    out = {}
    for k in ks:
        out[f"R@{k}"] = float((pos < k).float().mean().item())
    return out



# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Evaluate MuCST on precomputed CSV features (paired Recall@k)")
    ap.add_argument("--cfg", required=True, help="YAML/JSON with paths and options")
    args = ap.parse_args()

    cfg_user = _load_cfg(args.cfg)

    # 1) deriva impostazioni dai JSON di training, poi sovrascrivi con cfg_user
    derived = {}
    if cfg_user.get("train_cfg_data") and cfg_user.get("train_cfg_model"):
        derived = load_model_train_cfgs(cfg_user["train_cfg_data"], cfg_user["train_cfg_model"])
    cfg = _merge(cfg_user, derived)

    pre_dir    = cfg.get("precomputed_dir")
    checkpoint = cfg.get("checkpoint")
    out_dir    = cfg.get("out_dir")
    if not pre_dir or not checkpoint or not out_dir:
        raise ValueError("Config must include: precomputed_dir, checkpoint, out_dir")

    hvg_k      = int(cfg.get("hvg_k"))
    pca_dim    = int(cfg.get("image_pca_dim") or cfg.get("pca_dim"))
    batch_size = int(cfg.get("batch_size", 4096))
    gene_log1p = bool(cfg.get("gene_log1p", True))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) Carica CSV
    image_prefix = resolve_image_prefix_from_cfg(cfg)
    Ximg, Xgene, names_all, coords_xy, slice_ids, rows_meta = load_precomputed_bundle(
        pre_dir, hvg_k=hvg_k, pca_dim=pca_dim, image_prefix=image_prefix, gene_prefix="gene_hvg"
    )
    if gene_log1p:
        Xgene = np.log1p(Xgene).astype(np.float32)

    # 3) Ricostruisci “adata-like” + adjacency normalizzato
    pack = _build_adata_like_training(
        Ximg=Ximg, Xgene=Xgene, coords_xy=coords_xy, names=names_all, rows_meta=rows_meta, k_graph=15
    )
    adj_norm = _normalize_adj_with_self_loops(pack["mor_adj"])

    # 4) Istanzia modello + checkpoint
    model = load_mucst_from_cfg(cfg, input_dims_img=Ximg.shape[1], input_dims_gene=Xgene.shape[1],
                                adj_norm_np=adj_norm, device=device)

    
    # 5) Forward FULL-BATCH (niente mini-batch, richiesto dalla GCN)
    def _to_t(x): 
        return torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        features      = _to_t(pack["feat"])            # [N, Dg]
        features_fake = _to_t(pack["feat_fake"])       # [N, Dg]
        image_feature = _to_t(pack["image_feature"])   # [N, Di]
        aug1          = _to_t(pack["aug_image_feature1"])
        aug2          = _to_t(pack["aug_image_feature2"])
        graph_full    = torch.tensor(adj_norm, dtype=torch.float32, device=device)  # [N, N]

        # forward una sola volta su TUTTI i nodi
        zg, zi, aug_zi1, aug_zi2, hg, hi, aug_hi1, aug_hi2, _, _, _, _ = model(
            features, features_fake, image_feature, aug1, aug2, graph_full
        )

    # Passa a NumPy
    IMG = F.normalize(zi, p=2, dim=1).cpu().numpy()
    TX  = F.normalize(zg, p=2, dim=1).cpu().numpy()


    # L2-norm (coerente con CLIP/HESCAPE)
    IMG = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-9)
    TX  = TX  / (np.linalg.norm(TX,  axis=1, keepdims=True) + 1e-9)

    # 6) Recall@k
    EVAL_DEVICE = torch.device("cpu")
    IMG_t = torch.tensor(IMG, dtype=torch.float32, device=EVAL_DEVICE)
    TX_t  = torch.tensor(TX,  dtype=torch.float32, device=EVAL_DEVICE)
    i2g = paired_recall_at_k(IMG_t, TX_t, ks=(1,5,10))
    g2i = paired_recall_at_k(TX_t, IMG_t, ks=(1,5,10))

    metrics = {
        "test_R@1_I2G": i2g["R@1"], "test_R@5_I2G": i2g["R@5"], "test_R@10_I2G": i2g["R@10"],
        "test_R@1_G2I": g2i["R@1"], "test_R@5_G2I": g2i["R@5"], "test_R@10_G2I": g2i["R@10"],
    }

    # 7) Export
    out_path = Path(cfg.get("out_dir")); out_path.mkdir(parents=True, exist_ok=True)
    img_df = pd.DataFrame(IMG, columns=[f"img_e{i}" for i in range(IMG.shape[1])]); img_df.insert(0, "Image Name", names_all)
    tx_df  = pd.DataFrame(TX,  columns=[f"tx_e{i}"  for i in range(TX.shape[1])]);  tx_df.insert(0,  "Image Name", names_all)
    img_df.to_csv(out_path / "image_embeddings.csv", index=False)
    tx_df.to_csv (out_path / "transcriptome_embeddings.csv", index=False)

    metrics_row = {
        "n_samples": int(IMG.shape[0]),
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        **{k: float(v) for k, v in metrics.items()}
    }
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics_row, f, indent=2)
    write_header = not (out_path / "metrics.csv").exists()
    pd.DataFrame([metrics_row]).to_csv(out_path / "metrics.csv", index=False, mode="a", header=write_header)

    print(f"[eval] encoded N={IMG.shape[0]} samples")
    print("\n=== Recall@k (paired) — MuCST ===")
    print("I2G  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(metrics["test_R@1_I2G"], metrics["test_R@5_I2G"], metrics["test_R@10_I2G"]))
    print("G2I  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(metrics["test_R@1_G2I"], metrics["test_R@5_G2I"], metrics["test_R@10_G2I"]))

if __name__ == "__main__":
    main()
