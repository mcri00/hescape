'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

# --- import path bootstrap ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT   = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT  = os.path.dirname(PKG_ROOT)
for p in (SCRIPT_DIR, PKG_ROOT, PROJ_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# ==== Your modules ====
try:
    from .stitch_model import load_stitch_from_cfg
    from .coords import coords3d_from_xy_groups
    from .cfg_bridge import load_model_train_cfgs
except Exception:
    from stitch_model import load_stitch_from_cfg
    from coords import coords3d_from_xy_groups
    from cfg_bridge import load_model_train_cfgs

# ==== HESCAPE metrics (import) with graceful fallbacks ====
HAS_HESCAPE = True
try:
    from hescape.constants import DatasetEnum
    from hescape.metrics import EvalMetrics  # if your fork exposes it
except Exception:
    HAS_HESCAPE = False
    # Minimal fallback DatasetEnum with the string keys EvalMetrics expects
    class _DE:
        IMG_EMBED = "img_embedding"
        GEXP_EMBED = "gexp_embedding"
        IMG = "IMG"
        GEXP = "GEXP"
        CLUSTER = "cluster"
        REGION = "region"
    DatasetEnum = _DE  # type: ignore

    # Inline EvalMetrics (estratto dal repo – semplificato a ciò che serve qui)
    from typing import Any, Tuple, List, Dict
    from sklearn.metrics import r2_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def r2_eval(true: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
        true_mean, pred_mean = np.nanmean(true, axis=0), np.nanmean(predicted, axis=0)
        true_var, pred_var = np.nanvar(true, axis=0), np.nanvar(predicted, axis=0)
        return r2_score(true_mean, pred_mean), r2_score(true_var, pred_var)

    class EvalMetrics:  # minimal clone; SCIB part only if packages are present
        def __init__(
            self,
            stage: str,
            strategy: str,
            label_key: str | None = None,
            batch_key: str | None = None,
            eval_labels: List[str] | None = None,
            recall_range: Tuple[int, ...] = (1, 5, 10),
            knn_recall_metrics: bool = False,
            knn_gexp_metrics: bool = False,
            classif_metrics: bool = False,
            scib_metrics: bool = False,
        ):
            self.stage = stage
            self.strategy = strategy
            self.label_key = label_key
            self.batch_key = batch_key
            self.eval_labels = eval_labels or []
            self.recall_range = recall_range
            self.knn_recall_metrics = knn_recall_metrics
            self.knn_gexp_metrics = knn_gexp_metrics
            self.classif_metrics = classif_metrics
            self.scib_metrics = scib_metrics

        def knn_recall(self, left_embedding: torch.Tensor, right_embedding: torch.Tensor, embedding_type: str):
            metrics = {}
            group_idx = torch.arange(left_embedding.shape[0], device=left_embedding.device)
            emb_sim = left_embedding @ right_embedding.T
            left_gid = group_idx.unsqueeze(1).expand_as(emb_sim)
            right_gid = group_idx.unsqueeze(0).expand_as(emb_sim)
            right_simrank = torch.argsort(emb_sim, dim=1, descending=True)
            rightgid_sorted = torch.gather(right_gid, 1, right_simrank)
            rightgid_matched = rightgid_sorted == left_gid
            leftgid_hasmatch, leftgid_firstmatch = torch.max(rightgid_matched, dim=1)
            leftmatch_rank = leftgid_firstmatch[leftgid_hasmatch]
            assert leftmatch_rank.shape[0] > 0
            for k in self.recall_range:
                match_count = (leftmatch_rank < k).sum()
                totcal_count = leftgid_hasmatch.sum()
                metrics[f"{self.stage}_R@{k}_{embedding_type}"] = (match_count / totcal_count).item()
            return metrics

        def knn_gexp(
            self,
            left_embedding: torch.Tensor,   # (N, D)  IMG
            right_embedding: torch.Tensor,  # (N, D)  GEXP (embed)
            true_gexp: torch.Tensor,        # (N, G)  HVG reali
            stratify: torch.Tensor,         # (N,)
            groups: torch.Tensor,           # (N,)
            embedding_type: str,            # "I2G" o "G2I"
        ):
            """
            Tutto su CPU/NumPy per compatibilità con scikit-learn:
            - split stratificato per 'groups' (atlas/slice/region)
            - predizione gexp media dai top-k vicini nel dominio opposto
            - R2 mean/var come in HESCAPE
            """
            # --- to CPU/NumPy ---
            L = left_embedding.detach().cpu().numpy()
            R = right_embedding.detach().cpu().numpy()
            G = true_gexp.detach().cpu().numpy()
            y = stratify.detach().cpu().numpy()
            grp = groups.detach().cpu().numpy()

            metrics = {}
            # cicla sui k richiesti
            for k in self.recall_range:
                r2_mean_l, r2_var_l = [], []

                for g in np.unique(grp):
                    mask = (grp == g)
                    # serve sufficiente cardinalità per stratificare in train/test
                    if np.sum(mask) <= k * 2:
                        continue

                    try:
                        from sklearn.model_selection import train_test_split
                        # split su L (query), R (database), G (truth) e y (stratify)
                        L_tr, L_te, R_tr, R_te, G_tr, G_te, y_tr, y_te = train_test_split(
                            L[mask], L[mask], R[mask], R[mask], G[mask], G[mask], y[mask],
                            test_size=0.5, stratify=y[mask], random_state=42
                        )
                    except ValueError:
                        # classe con 1 solo membro, o stratify non fattibile
                        continue

                    # Similarità (dot product) tra query e database
                    # se embedding_type == "I2G" vogliamo IMG→GEXP: usa L_tr (db) vs R_te (query)
                    # se "G2I" invertiremo fuori con una seconda chiamata
                    # Qui manteniamo la convenzione del chiamante:
                    if embedding_type == "I2G":
                        S = np.matmul(R_te, L_tr.T)   # (Nt, Ntr)  query=R_te, db=L_tr
                        # top-k indici nei DB (asse 1)
                        topk_idx = np.argpartition(-S, kth=min(k-1, S.shape[1]-1), axis=1)[:, :k]
                        # pred gexp = media dei G_tr dei vicini scelti nel DB
                        G_pred = np.stack([G_tr[idx].mean(axis=0) for idx in topk_idx], axis=0)
                    else:  # "G2I"
                        S = np.matmul(L_te, R_tr.T)   # query = L_te, db=R_tr
                        topk_idx = np.argpartition(-S, kth=min(k-1, S.shape[1]-1), axis=1)[:, :k]
                        G_pred = np.stack([G_tr[idx].mean(axis=0) for idx in topk_idx], axis=0)

                    # R² su mean/var globali (come nel codice HESCAPE)
                    from sklearn.metrics import r2_score
                    true_mean, pred_mean = np.nanmean(G_te, axis=0), np.nanmean(G_pred, axis=0)
                    true_var,  pred_var  = np.nanvar(G_te,  axis=0), np.nanvar(G_pred,  axis=0)
                    r2m = r2_score(true_mean, pred_mean)
                    r2v = r2_score(true_var,  pred_var)

                    r2_mean_l.append(r2m)
                    r2_var_l.append(r2v)

                metrics[f"{self.stage}_r2mean@{k}_{embedding_type}"] = float(np.mean(r2_mean_l)) if r2_mean_l else np.nan
                metrics[f"{self.stage}_r2var@{k}_{embedding_type}"]  = float(np.mean(r2_var_l))  if r2_var_l  else np.nan

            return metrics


        def svc_eval(self, X: np.ndarray, y: np.ndarray, label: str, embedding_type: str):
            metrics = {}
            if len(np.unique(y)) <= 1:
                metrics[f"{self.stage}_acc_{label}_{embedding_type}"] = np.nan
                return metrics
            clf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear"))])
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            metrics[f"{self.stage}_acc_{label}_{embedding_type}"] = float(cross_val_score(X, y, cv=skf, estimator=clf).mean())
            return metrics

        def scib_eval(self, X: np.ndarray, batch_values: np.ndarray, label_values: np.ndarray, embedding_type: str):
            # optional; uses scib-metrics if available
            try:
                from anndata import AnnData
                from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
                Xs = StandardScaler().fit_transform(X)
                adata = AnnData(
                    Xs,
                    obs={self.batch_key: batch_values.flatten(), self.label_key: label_values.flatten()},
                )
                adata.obsm["pre"] = Xs.copy()
                adata.obsm[embedding_type] = Xs.copy()
                bm = Benchmarker(
                    adata,
                    batch_key=self.batch_key,
                    label_key=self.label_key,
                    embedding_obsm_keys=[embedding_type],
                    bio_conservation_metrics=BioConservation(
                        clisi_knn=False, isolated_labels=False,
                        nmi_ari_cluster_labels_kmeans=True, nmi_ari_cluster_labels_leiden=False,
                        silhouette_label=True,
                    ),
                    batch_correction_metrics=BatchCorrection(
                        silhouette_batch=True, kbet_per_label=False,
                        graph_connectivity=True, ilisi_knn=True, pcr_comparison=False,
                    ),
                    pre_integrated_embedding_obsm_key="pre",
                    n_jobs=10,
                )
                bm.benchmark()
                bm_d = bm.get_results(min_max_scale=False).to_dict()
                return {f"{self.stage}_{k}_{bm_d[k]['Metric Type']}": bm_d[k][embedding_type] for k in bm_d}
            except Exception:
                return {}

        def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            metrics = {}
            img_emb = batch[DatasetEnum.IMG_EMBED]
            gexp_emb = batch[DatasetEnum.GEXP_EMBED]

            if self.knn_recall_metrics:
                metrics.update(self.knn_recall(img_emb, gexp_emb, "I2G"))
                metrics.update(self.knn_recall(gexp_emb, img_emb, "G2I"))

            if self.knn_gexp_metrics:
                true_gexp = batch.get(DatasetEnum.GEXP, None)
                stratify = batch.get(DatasetEnum.CLUSTER, None)
                groups   = batch.get(DatasetEnum.REGION, None)
                if true_gexp is not None and stratify is not None and groups is not None:
                    metrics.update(self.knn_gexp(img_emb, gexp_emb, true_gexp, stratify, groups, "I2G"))
                    metrics.update(self.knn_gexp(img_emb, gexp_emb, true_gexp, stratify, groups, "G2I"))

            if self.classif_metrics:
                for k in self.eval_labels:
                    if k in batch:
                        y = batch[k].cpu().numpy()
                        metrics.update(self.svc_eval(img_emb.cpu().numpy(), y, k, DatasetEnum.IMG))
                        metrics.update(self.svc_eval(gexp_emb.cpu().numpy(), y, k, DatasetEnum.GEXP))

            if self.scib_metrics and (self.label_key in batch) and (self.batch_key in batch):
                metrics.update(self.scib_eval(
                    img_emb.cpu().numpy(),
                    batch[self.batch_key].cpu().numpy(),
                    batch[self.label_key].cpu().numpy(),
                    DatasetEnum.IMG,
                ))
                metrics.update(self.scib_eval(
                    gexp_emb.cpu().numpy(),
                    batch[self.batch_key].cpu().numpy(),
                    batch[self.label_key].cpu().numpy(),
                    DatasetEnum.GEXP,
                ))
            return metrics

# ----------------- basic cfg I/O -----------------
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

# ----------------- CSV loader -----------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_precomputed_bundle(root_dir: str, hvg_k: int, pca_dim: int,
                            image_prefix: str = "img_inception_pca",
                            gene_prefix: str  = "gene_hvg"):
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

    # features
    gene_cols = [c for c in df.columns if c.startswith("g_e")]
    img_cols  = [c for c in df.columns if c.startswith(f"{image_prefix}_e")]
    if len(gene_cols) != int(hvg_k):  raise ValueError(f"Expected {hvg_k} gene columns, got {len(gene_cols)}.")
    if len(img_cols)  != int(pca_dim): raise ValueError(f"Expected {pca_dim} image columns, got {len(img_cols)}.")

    Xgene  = df[gene_cols].to_numpy(dtype=np.float32)              # (N, G) true gexp (HVG_k)
    Ximg   = df[img_cols].to_numpy(dtype=np.float32)               # (N, D_img_in)
    names  = df["Image Name"].astype(str).tolist()
    coords = df[["x", "y"]].to_numpy(dtype=np.float32)

    # slice ids
    atlas_series   = df["atlas"].fillna("").astype(str) if "atlas" in df.columns else pd.Series([""] * len(df))
    namecat_series = df["name_category"].fillna("").astype(str) if "name_category" in df.columns else pd.Series([""] * len(df))
    slice_ids = [a if (a not in ("", "NA", "None")) else ncat for a, ncat in zip(atlas_series.tolist(), namecat_series.tolist())]

    rows_meta = df[df_m.columns].copy()

    return Ximg, Xgene, names, coords, slice_ids, rows_meta

# ----------------- helpers to detect label sets by panel -----------------
def infer_eval_labels_and_keys(pre_dir: str, rows_meta: pd.DataFrame):
    """
    Decide quali metriche attivare in base ai campi realmente presenti.
    - classification labels (CRC/BRCA) solo se le colonne esistono
    - altrimenti: label_key = 'tissue'  ; batch_key = 'atlas' o 'name_category'
    """
    path_lower = str(pre_dir).lower()
    eval_labels = []

    # CRC (colon) → MSI, BRAF, KRAS se presenti
    if ("colon" in path_lower) and all(c in rows_meta.columns for c in ["MSI","BRAF","KRAS"]):
        eval_labels = ["MSI", "BRAF", "KRAS"]

    # BRCA (breast) → ER, PR, HER2 se presenti
    if ("breast" in path_lower) and all(c in rows_meta.columns for c in ["ER","PR","HER2"]):
        eval_labels = ["ER", "PR", "HER2"]

    # Label per SCIB/stratify knn-gexp:
    # priorità: 'cluster' (se già calcolato), altrimenti 'tissue', poi 'diagnosis' o 'cancer'
    if "cluster" in rows_meta.columns:
        label_key = "cluster"
    elif "tissue" in rows_meta.columns:
        label_key = "tissue"
    elif "diagnosis" in rows_meta.columns:
        label_key = "diagnosis"
    elif "cancer" in rows_meta.columns:
        label_key = "cancer"
    else:
        label_key = None

    # Batch key (regioni/sorgenti): preferisci 'atlas', poi 'name_category', poi 'source'
    if "atlas" in rows_meta.columns and rows_meta["atlas"].notna().any():
        batch_key = "atlas"
    elif "name_category" in rows_meta.columns:
        batch_key = "name_category"
    elif "source" in rows_meta.columns:
        batch_key = "source"
    else:
        batch_key = None

    return eval_labels, label_key, batch_key


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Evaluate ST model on precomputed CSV features using HESCAPE metrics")
    ap.add_argument("--cfg", required=True, help="YAML/JSON with paths and options")
    args = ap.parse_args()

    cfg_user = _load_cfg(args.cfg)
    derived = {}
    if cfg_user.get("train_cfg_data") and cfg_user.get("train_cfg_model"):
        derived = load_model_train_cfgs(cfg_user["train_cfg_data"], cfg_user["train_cfg_model"])
    cfg = _merge(cfg_user, derived)

    pre_dir    = cfg.get("precomputed_dir")
    checkpoint = cfg.get("checkpoint")
    out_dir    = cfg.get("out_dir")
    if not pre_dir or not checkpoint or not out_dir:
        raise ValueError("Config must include: precomputed_dir, checkpoint, out_dir")

    hvg_k   = int(cfg.get("hvg_k"))
    pca_dim = int(cfg.get("image_pca_dim") or cfg.get("pca_dim"))
    batch_size  = int(cfg.get("batch_size", 1024))
    gene_log1p  = bool(cfg.get("gene_log1p", False))
    hidden_dim  = int(cfg.get("hidden_dim", 512))
    output_dim  = int(cfg.get("output_dim", 256))
    num_layers  = int(cfg.get("num_layers", 2))
    dropout_prob= float(cfg.get("dropout_prob", 0.5))
    use_pos_emb = bool(cfg.get("use_positional_embedding", False))
    pos_emb_dim = int(cfg.get("pos_embedding_dim", 128))
    use_pos_attn = bool(cfg.get("use_positional_attention", False))
    use_feat_sel = bool(cfg.get("use_feature_selector", False))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load precomputed
    image_prefix = resolve_image_prefix_from_cfg(cfg)
    Ximg, Xgene, names_all, coords_xy, slice_ids, rows_meta = load_precomputed_bundle(
        pre_dir, hvg_k=hvg_k, pca_dim=pca_dim, image_prefix=image_prefix, gene_prefix="gene_hvg"
    )
    if gene_log1p:
        Xgene = np.log1p(Xgene, dtype=np.float32)
        
    # === DERIVE REGION & CLUSTER (runtime, senza toccare labels_meta.csv) ===
    # region = atlas se non vuoto, altrimenti name_category
    atlas_series   = rows_meta.get("atlas", pd.Series([""] * len(rows_meta))).astype(str).fillna("")
    namecat_series = rows_meta.get("name_category", pd.Series([""] * len(rows_meta))).astype(str).fillna("")
    region_series  = np.where(~atlas_series.isin(["", "NA", "None"]), atlas_series, namecat_series)
    region_series  = pd.Series(region_series, index=rows_meta.index).astype(str)

    # cluster = tissue (baseline raccomandata per un confronto fair)
    if "tissue" in rows_meta.columns:
        cluster_series = rows_meta["tissue"].astype(str).fillna("")
    elif "diagnosis" in rows_meta.columns:
        cluster_series = rows_meta["diagnosis"].astype(str).fillna("")
    elif "cancer" in rows_meta.columns:
        # bool -> '0'/'1'
        cluster_series = rows_meta["cancer"].map(lambda v: str(int(bool(v)))).astype(str)
    else:
        cluster_series = pd.Series(["_unk_"] * len(rows_meta), index=rows_meta.index)


    # 2) 3D coords
    coords3d = coords3d_from_xy_groups(
        xy_coords=torch.tensor(coords_xy, dtype=torch.float32, device=device),
        slice_ids_str=[str(s) for s in slice_ids],
        device=device,
        dtype=torch.float32,
    )

    # 3) Build model & load ckpt
    model_cfg = {
        "checkpoint": checkpoint,
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "dropout_prob": dropout_prob,
        "use_positional_embedding": use_pos_emb,
        "use_positional_attention": use_pos_attn,
        "use_feature_selector": use_feat_sel,
        "pos_embedding_dim": pos_emb_dim,
    }
    model = load_stitch_from_cfg(model_cfg).to(device).eval()

    # 4) Forward → embeddings
    bs = int(batch_size)
    IMG_list, TX_list = [], []
    with torch.no_grad():
        for i in range(0, len(names_all), bs):
            Xi = torch.tensor(Ximg[i:i+bs],  dtype=torch.float32, device=device)
            Xg = torch.tensor(Xgene[i:i+bs], dtype=torch.float32, device=device)
            C3 = coords3d[i:i+bs]
            img_out, tx_out, _ = model(Xi, Xg, coords_3d=C3)
            IMG_list.append(img_out.detach().cpu().numpy())
            TX_list.append(tx_out.detach().cpu().numpy())

    IMG = np.concatenate(IMG_list, axis=0)
    TX  = np.concatenate(TX_list,  axis=0)
    # L2-normalize (coerente con CLIP/HESCAPE)
    IMG = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-9)
    TX  = TX  / (np.linalg.norm(TX,  axis=1, keepdims=True) + 1e-9)
    print(f"[eval] encoded N={IMG.shape[0]} samples")

    # 5) Costruisci 'batch' per EvalMetrics
    # decide eval_labels, label_key, batch_key in base ai campi disponibili
    eval_labels, label_key, batch_key = infer_eval_labels_and_keys(pre_dir, rows_meta)

    # colonne d'appoggio per i gruppi
    EVAL_DEVICE = torch.device("cpu")

    # costruisci tensori base con region/cluster derivati qui sopra
    batch = {
        DatasetEnum.IMG_EMBED: torch.tensor(IMG, dtype=torch.float32, device=EVAL_DEVICE),
        DatasetEnum.GEXP_EMBED: torch.tensor(TX,  dtype=torch.float32, device=EVAL_DEVICE),
        DatasetEnum.GEXP:       torch.tensor(Xgene, dtype=torch.float32, device=EVAL_DEVICE),

        # derived (sempre presenti)
        DatasetEnum.REGION:  torch.tensor(pd.Categorical(region_series).codes,  dtype=torch.long, device=EVAL_DEVICE),
        DatasetEnum.CLUSTER: torch.tensor(pd.Categorical(cluster_series).codes, dtype=torch.long, device=EVAL_DEVICE),
    }

    # per SCIB servono entrambi: label_key & batch_key
    if label_key and label_key in rows_meta.columns:
        v = rows_meta[label_key]
        if v.dtype == bool:
            v = v.astype(int)
        batch[label_key] = torch.tensor(pd.Categorical(v.astype(str)).codes, dtype=torch.long, device=EVAL_DEVICE)
    if batch_key and batch_key in rows_meta.columns:
        batch[batch_key] = torch.tensor(pd.Categorical(rows_meta[batch_key].astype(str)).codes, dtype=torch.long, device=EVAL_DEVICE)

    # classificazioni cliniche (solo se colonne presenti ⇒ CRC/BRCA)
    for lab in eval_labels:
        batch[lab] = torch.tensor(pd.Categorical(rows_meta[lab]).codes.values, dtype=torch.long, device=EVAL_DEVICE)

    # 6) EvalMetrics — attiva tutto ciò che è disponibile
    eval_kwargs = dict(
        stage="test",
        strategy="single",
        label_key=label_key,
        batch_key=batch_key,
        eval_labels=eval_labels,
        recall_range=(1,5,10),
        knn_recall_metrics=True,
        knn_gexp_metrics=True,
        classif_metrics=len(eval_labels) > 0,
        scib_metrics=(label_key is not None and batch_key is not None),
    )
    
    # Usa le stesse definizioni del benchmark
    eval_kwargs["label_key"] = "tissue" if "tissue" in rows_meta.columns else None
    eval_kwargs["batch_key"] = "region"

    # === SCIB: aggiungi le chiavi stringa attese e forza l'attivazione ===
    EVAL_DEVICE = torch.device("cpu")

    # 1) region esplicita (stringa)
    batch["region"] = torch.tensor(
        pd.Categorical(region_series).codes, dtype=torch.long, device=EVAL_DEVICE
    )

    # 2) label esplicita: preferisci 'tissue' se ha >=2 classi, altrimenti fallback a un cluster derivato
    if "tissue" in rows_meta.columns and rows_meta["tissue"].nunique() > 1:
        batch["tissue"] = torch.tensor(
            pd.Categorical(rows_meta["tissue"].astype(str)).codes,
            dtype=torch.long, device=EVAL_DEVICE
        )
        eval_kwargs["label_key"] = "tissue"
    else:
        # Fallback: cluster_derived (se non l’hai già fatto sopra va bene anche tissue→diagnosis→cancer)
        # Se vuoi forzare >=2 classi per regione, fai un KMeans per-region (consigliato per far partire SCIB sempre)
        from sklearn.cluster import KMeans
        codes = np.full(len(rows_meta), -1, dtype=int)
        idx_by_reg = pd.Series(np.arange(len(rows_meta))).groupby(region_series).groups
        for reg, idx_list in idx_by_reg.items():
            idx = np.fromiter(idx_list, dtype=int)
            if idx.size < 6:
                continue
            k = max(2, min(10, idx.size // 50))  # k adattivo
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            codes[idx] = km.fit_predict(Xgene[idx])
        # dopo aver creato `codes` (np.int)
        cluster_derived_series = pd.Series(codes, index=rows_meta.index)

        batch["cluster_derived"] = torch.tensor(
            cluster_derived_series.to_numpy(),  # già int, senza Categorical
            dtype=torch.long, device=EVAL_DEVICE
        )
        eval_kwargs["label_key"] = "cluster_derived"


    # 3) batch_key = "region" (coerente col benchmark)
    eval_kwargs["batch_key"] = "region"

    # 4) attiva SCIB esplicitamente
    eval_kwargs["scib_metrics"] = True
    
    # Subito dopo aver deciso il label per SCIB:
    if eval_kwargs["label_key"] == "tissue":
        batch[DatasetEnum.CLUSTER] = batch["tissue"]        # stratify coerente
    else:
        batch[DatasetEnum.CLUSTER] = batch["cluster_derived"]


    print("[SCIB] region uniques =", pd.Series(region_series).nunique())
    if "tissue" in rows_meta.columns:
        print("[SCIB] tissue uniques =", rows_meta["tissue"].nunique())
    if "cluster_derived" in batch:
        # se usi il fallback
        print("[SCIB] cluster_derived set =", True)


    em = EvalMetrics(**eval_kwargs)
    metrics = em(batch)

    # 7) Export embeddings + labels + metrics
    out_path = Path(cfg.get("out_dir")); out_path.mkdir(parents=True, exist_ok=True)
    img_df = pd.DataFrame(IMG, columns=[f"img_e{i}" for i in range(IMG.shape[1])]); img_df.insert(0, "Image Name", names_all)
    tx_df  = pd.DataFrame(TX,  columns=[f"tx_e{i}"  for i in range(TX.shape[1])]);  tx_df.insert(0,  "Image Name", names_all)
    rows_meta = rows_meta.copy()
    rows_meta["region"]  = region_series.values
    if "cluster_derived" in batch:
        rows_meta["cluster_derived"] = cluster_derived_series.values


    labels_df = rows_meta.copy()

    img_df.to_csv(out_path / "image_embeddings.csv", index=False)
    tx_df.to_csv (out_path / "transcriptome_embeddings.csv", index=False)
    labels_df.to_csv(out_path / "labels.csv", index=False)

    # metrics.json + metrics.csv append
    metrics_row = {
        "n_samples": int(IMG.shape[0]),
        "hvg_k": int(Xgene.shape[1]),
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        **{k: (float(v) if isinstance(v, (np.floating,)) else (float(v) if torch.is_tensor(v) else v)) for k, v in metrics.items()}
    }
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics_row, f, indent=2)
    write_header = not (out_path / "metrics.csv").exists()
    pd.DataFrame([metrics_row]).to_csv(out_path / "metrics.csv", index=False, mode="a", header=write_header)

    # stampa essenziale (tabella paper → R@5 I2G/G2I)
    def _get(key, default=np.nan): return metrics_row.get(key, default)
    print("\n=== HESCAPE-style metrics ===")
    print("I2G  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        _get("test_R@1_I2G"), _get("test_R@5_I2G"), _get("test_R@10_I2G")
    ))
    print("G2I  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        _get("test_R@1_G2I"), _get("test_R@5_G2I"), _get("test_R@10_G2I")
    ))
    if eval_labels:
        for lab in eval_labels:
            print(f"{lab}  acc_IMG={_get(f'test_acc_{lab}_{DatasetEnum.IMG}'):0.3f}  "
                  f"acc_GEXP={_get(f'test_acc_{lab}_{DatasetEnum.GEXP}'):0.3f}")
    if eval_kwargs["scib_metrics"]:
        keys_pretty = [k for k in metrics_row.keys() if k.startswith("test_") and ("Bio conservation" in k or "Batch correction" in k)]
        if keys_pretty:
            print("\n[SCIB] (principali)")
            for k in sorted(keys_pretty):
                print(f" - {k}: {metrics_row[k]:.3f}")

if __name__ == "__main__":
    main()
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# --- import path bootstrap ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT   = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT  = os.path.dirname(PKG_ROOT)
for p in (SCRIPT_DIR, PKG_ROOT, PROJ_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# ==== Your modules ====
try:
    from .stitch_model import load_stitch_from_cfg
    from .coords import coords3d_from_xy_groups
    from .cfg_bridge import load_model_train_cfgs
except Exception:
    from stitch_model import load_stitch_from_cfg
    from coords import coords3d_from_xy_groups
    from cfg_bridge import load_model_train_cfgs

# ----------------- basic cfg I/O -----------------
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

# ----------------- CSV loader -----------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def load_precomputed_bundle(root_dir: str, hvg_k: int, pca_dim: int,
                            image_prefix: str = "img_inception_pca",
                            gene_prefix: str  = "gene_hvg"):
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

    # slice ids (per coords3d)
    atlas_series   = df["atlas"].fillna("").astype(str) if "atlas" in df.columns else pd.Series([""] * len(df))
    namecat_series = df["name_category"].fillna("").astype(str) if "name_category" in df.columns else pd.Series([""] * len(df))
    slice_ids = [a if (a not in ("", "NA", "None")) else ncat for a, ncat in zip(atlas_series.tolist(), namecat_series.tolist())]

    rows_meta = df[df_m.columns].copy()
    return Ximg, Xgene, names, coords, slice_ids, rows_meta

# ----------------- Recall@k (paired retrieval) -----------------
@torch.no_grad()
def paired_recall_at_k(left_emb: torch.Tensor, right_emb: torch.Tensor, ks=(1,5,10)) -> dict:
    """
    left_emb:  (N, D)  query
    right_emb: (N, D)  database
    Assunzione: elementi corrispondenti sono allo stesso indice.
    """
    # Similarità dot-product (coerente con normalizzazione L2 fatta a monte)
    sim = left_emb @ right_emb.T  # (N, N)
    # per ogni query prendi l'ordine decrescente di similarità
    rank = torch.argsort(sim, dim=1, descending=True)  # (N, N)
    # posizione (rank) del vero match (stesso indice)
    idx = torch.arange(sim.shape[0], device=sim.device).unsqueeze(1)       # (N,1)
    eq = (rank == idx)                                                     # (N,N) True solo alla colonna del match
    # posizione del True lungo la riga
    has_match, pos = torch.max(eq, dim=1)  # pos ∈ [0..N-1]
    assert bool(has_match.all()), "Ogni query deve avere il suo match 1-1 (paired)."
    out = {}
    for k in ks:
        out[f"R@{k}"] = float((pos < k).float().mean().item())
    return out

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Evaluate ST model on precomputed CSV features (Recall@k only)")
    ap.add_argument("--cfg", required=True, help="YAML/JSON with paths and options")
    args = ap.parse_args()

    cfg_user = _load_cfg(args.cfg)
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
    batch_size = int(cfg.get("batch_size", 1024))
    gene_log1p = bool(cfg.get("gene_log1p", False))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    output_dim = int(cfg.get("output_dim", 256))
    num_layers = int(cfg.get("num_layers", 2))
    dropout_prob = float(cfg.get("dropout_prob", 0.5))
    use_pos_emb  = bool(cfg.get("use_positional_embedding", False))
    pos_emb_dim  = int(cfg.get("pos_embedding_dim", 128))
    use_pos_attn = bool(cfg.get("use_positional_attention", False))
    use_feat_sel = bool(cfg.get("use_feature_selector", False))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load precomputed
    image_prefix = resolve_image_prefix_from_cfg(cfg)
    Ximg, Xgene, names_all, coords_xy, slice_ids, rows_meta = load_precomputed_bundle(
        pre_dir, hvg_k=hvg_k, pca_dim=pca_dim, image_prefix=image_prefix, gene_prefix="gene_hvg"
    )
    if gene_log1p:
        Xgene = np.log1p(Xgene).astype(np.float32)

    # 2) 3D coords (se il tuo modello le usa)
    coords3d = coords3d_from_xy_groups(
        xy_coords=torch.tensor(coords_xy, dtype=torch.float32, device=device),
        slice_ids_str=[str(s) for s in slice_ids],
        device=device,
        dtype=torch.float32,
    )

    # 3) Build model & load ckpt
    model_cfg = {
        "checkpoint": checkpoint,
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "dropout_prob": dropout_prob,
        "use_positional_embedding": use_pos_emb,
        "use_positional_attention": use_pos_attn,
        "use_feature_selector": use_feat_sel,
        "pos_embedding_dim": pos_emb_dim,
    }
    model = load_stitch_from_cfg(model_cfg).to(device).eval()

    # 4) Forward → embeddings
    bs = int(batch_size)
    IMG_list, TX_list = [], []
    with torch.no_grad():
        for i in range(0, len(names_all), bs):
            Xi = torch.tensor(Ximg[i:i+bs],  dtype=torch.float32, device=device)
            Xg = torch.tensor(Xgene[i:i+bs], dtype=torch.float32, device=device)
            C3 = coords3d[i:i+bs]
            img_out, tx_out, _ = model(Xi, Xg, coords_3d=C3)
            IMG_list.append(img_out.detach().cpu().numpy())
            TX_list.append(tx_out.detach().cpu().numpy())

    IMG = np.concatenate(IMG_list, axis=0)
    TX  = np.concatenate(TX_list,  axis=0)

    # L2-normalize (coerente con CLIP/HESCAPE)
    IMG = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-9)
    TX  = TX  / (np.linalg.norm(TX,  axis=1, keepdims=True) + 1e-9)

    # 5) Recall@k (I2G e G2I) — SOLO QUESTO
    EVAL_DEVICE = torch.device("cpu")
    IMG_t = torch.tensor(IMG, dtype=torch.float32, device=EVAL_DEVICE)
    TX_t  = torch.tensor(TX,  dtype=torch.float32, device=EVAL_DEVICE)

    i2g = paired_recall_at_k(IMG_t, TX_t, ks=(1,5,10))
    g2i = paired_recall_at_k(TX_t, IMG_t, ks=(1,5,10))

    metrics = {
        "test_R@1_I2G": i2g["R@1"], "test_R@5_I2G": i2g["R@5"], "test_R@10_I2G": i2g["R@10"],
        "test_R@1_G2I": g2i["R@1"], "test_R@5_G2I": g2i["R@5"], "test_R@10_G2I": g2i["R@10"],
    }

    print(f"[eval] encoded N={IMG.shape[0]} samples")
    print("\n=== Recall@k (paired) ===")
    print("I2G  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(metrics["test_R@1_I2G"], metrics["test_R@5_I2G"], metrics["test_R@10_I2G"]))
    print("G2I  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(metrics["test_R@1_G2I"], metrics["test_R@5_G2I"], metrics["test_R@10_G2I"]))

    # 6) Export embeddings + (solo) metrics
    out_path = Path(cfg.get("out_dir")); out_path.mkdir(parents=True, exist_ok=True)
    img_df = pd.DataFrame(IMG, columns=[f"img_e{i}" for i in range(IMG.shape[1])]); img_df.insert(0, "Image Name", names_all)
    tx_df  = pd.DataFrame(TX,  columns=[f"tx_e{i}"  for i in range(TX.shape[1])]);  tx_df.insert(0,  "Image Name", names_all)
    img_df.to_csv(out_path / "image_embeddings.csv", index=False)
    tx_df.to_csv (out_path / "transcriptome_embeddings.csv", index=False)

    metrics_row = {
        "n_samples": int(IMG.shape[0]),
        "hvg_k": int(Xgene.shape[1]),
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        **{k: float(v) for k, v in metrics.items()}
    }
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics_row, f, indent=2)
    write_header = not (out_path / "metrics.csv").exists()
    pd.DataFrame([metrics_row]).to_csv(out_path / "metrics.csv", index=False, mode="a", header=write_header)

if __name__ == "__main__":
    main()
