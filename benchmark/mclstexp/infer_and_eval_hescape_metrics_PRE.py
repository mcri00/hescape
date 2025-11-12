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
    from .mclstexp_model import load_mcl_from_cfg
    from .cfg_bridge import load_model_train_cfgs
except Exception:
    from mclstexp_model import load_mcl_from_cfg
    from cfg_bridge import load_model_train_cfgs

# ==== HESCAPE metrics (import) with graceful fallbacks ====
HAS_HESCAPE = True
try:
    from hescape.constants import DatasetEnum
    from hescape.metrics import EvalMetrics  # original implementation
except Exception:
    HAS_HESCAPE = False
    class _DE:
        IMG_EMBED = "img_embedding"
        GEXP_EMBED = "gexp_embedding"
    DatasetEnum = _DE  # type: ignore

    class EvalMetrics:
        def __init__(self, stage: str, strategy: str,
                     recall_range=(1,5,10),
                     knn_recall_metrics=True,
                     **kwargs):
            self.stage = stage
            self.recall_range = recall_range
            self.knn_recall_metrics = knn_recall_metrics

        def knn_recall(self, left_embedding: torch.Tensor, right_embedding: torch.Tensor, embedding_type: str):
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

# --- image prefix resolver ---
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
def _maybe_getattr(obj, dotted: str):
    cur = obj
    for p in dotted.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def resolve_encode_functions(model):
    """
    Cerca di ottenere due callable:
      encode_expr(Xg, P2) -> (B,D)
      encode_img (Xi, P2) -> (B,D)
    restituendo (encode_expr, encode_img) oppure (None, None) se non trovate.
    """
    # 0) API esplicita
    if hasattr(model, "encode_expression") and callable(model.encode_expression) and \
       hasattr(model, "encode_image") and callable(model.encode_image):
        return model.encode_expression, model.encode_image

    # 1) API unica tipo encode(dict)->{"spot":..., "image":...}
    enc = getattr(model, "encode", None)
    if callable(enc):
        def e_expr(xg, p2):
            out = enc({"expression": xg, "position": p2})
            return out.get("spot", out.get("expression", None))
        def e_img(xi, p2):
            out = enc({"image": xi, "position": p2})
            return out.get("image", None)
        # Verifica veloce: le funzioni devono restituire tensori
        return e_expr, e_img

    # 2) Heuristics su sottoclassi/attributi frequenti
    #    Proviamo catene tipiche: .expr_encoder/.gene_encoder/.spot_encoder e .img_encoder/.image_encoder
    expr_cands = [
        "expr_encoder", "gene_encoder", "spot_encoder", "g_encoder", "encoder_x", "encoder_gene",
        "backbone.expression", "module.expression_encoder", "module.expr_encoder",
    ]
    img_cands = [
        "img_encoder", "image_encoder", "i_encoder", "encoder_y", "encoder_image",
        "backbone.image", "module.image_encoder", "module.img_encoder",
    ]
    proj_cands = ["projector", "proj", "head", "projection", "module.projection"]

    def chain_call(f_enc, f_proj, x):
        z = f_enc(x)
        return f_proj(z) if f_proj is not None else z

    # Trova funzioni/nn.Module chiamabili
    expr_enc = next((m for m in ( _maybe_getattr(model, c) for c in expr_cands ) if callable(m) or hasattr(m, "forward")), None)
    img_enc  = next((m for m in ( _maybe_getattr(model, c) for c in img_cands )  if callable(m) or hasattr(m, "forward")), None)

    # projector opzionale (se esiste)
    proj = next((m for m in ( _maybe_getattr(model, c) for c in proj_cands ) if m is not None), None)
    expr_proj = getattr(proj, "expr", None) or getattr(proj, "x", None) or getattr(proj, "spot", None) or proj
    img_proj  = getattr(proj, "img",  None) or getattr(proj, "y", None) or getattr(proj, "image", None) or proj

    def wrap(module):
        if module is None:
            return None
        if callable(module):
            return module
        if hasattr(module, "forward"):
            return module.forward
        return None

    expr_enc_f = wrap(expr_enc)
    img_enc_f  = wrap(img_enc)

    expr_proj_f = wrap(expr_proj)
    img_proj_f  = wrap(img_proj)

    if expr_enc_f is not None and img_enc_f is not None:
        def e_expr(xg, p2):
            # se l’encoder vuole anche posizioni, prova (xg,p2) altrimenti solo xg
            try:
                z = expr_enc_f(xg, p2)
            except TypeError:
                z = expr_enc_f(xg)
            return expr_proj_f(z) if expr_proj_f is not None else z

        def e_img(xi, p2):
            try:
                z = img_enc_f(xi, p2)
            except TypeError:
                z = img_enc_f(xi)
            return img_proj_f(z) if img_proj_f is not None else z

        return e_expr, e_img

    # Fallimento: non trovate funzioni encode sicure
    return None, None

def infer_knn_k(model, default=32):
    # prova una lista ampia di attributi/percorsi
    candidates = [
        "knn_k", "num_neighbors", "k_neighbors", "neighbor_k",
        "retrieval_k", "loss_k",
        "head.knn_k", "module.knn_k", "mcl.knn_k",
        "head.k", "module.k", "mcl.k",
    ]
    for attr in candidates:
        obj = model
        ok = True
        for part in attr.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, (int, float)):
            return int(obj)
    return int(default)

def build_batches(N, base_bs, min_needed):
    """
    Spezza [0..N) in intervalli [i,j) garantendo che l'ULTIMO batch sia >= min_needed.
    Se l'ultimo spezzone sarebbe piccolo, lo fonde col precedente.
    """
    if base_bs >= N:
        return [(0, N)]
    idx = list(range(0, N, base_bs))
    if idx[-1] != N:
        idx.append(N)
    # se l'ultimo blocco è troppo piccolo, fondilo col precedente
    starts = idx[:-1]
    ends   = idx[1:]
    if (ends[-1] - starts[-1]) < min_needed and len(starts) >= 2:
        # fondi ultimo con penultimo
        starts[-2] = starts[-2]
        ends[-2]   = ends[-1]
        starts = starts[:-1]
        ends   = ends[:-1]
    return list(zip(starts, ends))


def clamp_knn_k_on_model(model, max_k):
    """
    Imposta (se presenti) gli attributi *k* usati in topk/gather ad un valore
    sicuro: max(1, min(attuale, max_k)).
    """
    cand_attrs = [
        "knn_k", "num_neighbors", "k_neighbors", "neighbor_k",
        "retrieval_k", "loss_k",
        "head.knn_k", "module.knn_k", "mcl.knn_k",
        "head.k", "module.k", "mcl.k",
    ]
    for attr in cand_attrs:
        obj = model
        parts = attr.split(".")
        try:
            for p in parts[:-1]:
                obj = getattr(obj, p)
            leaf = parts[-1]
            if hasattr(obj, leaf):
                v = getattr(obj, leaf)
                if isinstance(v, (int, float)):
                    safe = int(max(1, min(int(v), max_k)))
                    setattr(obj, leaf, safe)
        except Exception:
            pass


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
    coords = df[["x", "y"]].to_numpy(dtype=np.float32)

    rows_meta = df[df_m.columns].to_dict(orient="records")
    return Ximg, Xgene, names, coords, rows_meta

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser("Evaluate mclSTExp on precomputed CSV features (config-driven)")
    ap.add_argument("--cfg", required=True, help="YAML/JSON with paths and options")
    args = ap.parse_args()

    cfg_user = _load_cfg(args.cfg)

    # 1) Derive defaults from training JSONs if provided (dims / hypers)
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

    # Optional hypers (fallbacks if not in derived)
    temperature       = float(cfg.get("temperature", 0.5))
    mcl_projection_dim= int(cfg.get("mcl_projection_dim", cfg.get("output_dim", 128)))
    mcl_heads_num     = int(cfg.get("mcl_heads_num", 8))
    mcl_heads_dim     = int(cfg.get("mcl_heads_dim", 64))
    mcl_heads_layers  = int(cfg.get("mcl_heads_layers", 2))
    dropout_prob      = float(cfg.get("dropout_prob", 0.0))
    gene_log1p        = bool(cfg.get("gene_log1p", False))
    batch_size        = int(cfg.get("batch_size", 1024))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3) Load precomputed features
    image_prefix = resolve_image_prefix_from_cfg(cfg)
    print(f"[cfg] using image_prefix='{image_prefix}' (pca_dim={pca_dim})")
    Ximg, Xgene, names_all, coords_xy, rows_meta = load_precomputed_bundle(
        pre_dir, hvg_k=hvg_k, pca_dim=pca_dim, image_prefix=image_prefix, gene_prefix="gene_hvg"
    )
    
    # --- Normalize coords to [0,1] per 'atlas' (fallback: name_category, else global) ---
    def _pick_group_key(rows_meta_records):
        # records = list of dicts
        if any(("atlas" in r and r["atlas"] not in (None, "", "NA", "None")) for r in rows_meta_records):
            return "atlas"
        if any(("name_category" in r and r["name_category"] not in (None, "", "NA", "None")) for r in rows_meta_records):
            return "name_category"
        return None  # global

    def _normalize_coords_per_group(coords_xy_np, rows_meta_records, group_key):
        xy = coords_xy_np.copy().astype(np.float32)
        N = xy.shape[0]
        if group_key is None:
            # global min-max
            x_min, y_min = np.nanmin(xy[:,0]), np.nanmin(xy[:,1])
            x_max, y_max = np.nanmax(xy[:,0]), np.nanmax(xy[:,1])
            dx = max(1e-6, float(x_max - x_min))
            dy = max(1e-6, float(y_max - y_min))
            xy[:,0] = (xy[:,0] - x_min) / dx
            xy[:,1] = (xy[:,1] - y_min) / dy
        else:
            # per-group min-max
            # build group arrays
            groups = np.array([str(r.get(group_key, "")) for r in rows_meta_records])
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                if idx.size == 0: 
                    continue
                xg = xy[idx, 0]; yg = xy[idx, 1]
                x_min, y_min = np.nanmin(xg), np.nanmin(yg)
                x_max, y_max = np.nanmax(xg), np.nanmax(yg)
                dx = max(1e-6, float(x_max - x_min))
                dy = max(1e-6, float(y_max - y_min))
                xy[idx, 0] = (xg - x_min) / dx
                xy[idx, 1] = (yg - y_min) / dy
        # clamp hard to [0,1]
        xy = np.clip(xy, 0.0, 1.0, out=xy)
        return xy

    group_key = _pick_group_key(rows_meta)  # 'atlas' | 'name_category' | None
    coords_xy_norm = _normalize_coords_per_group(coords_xy, rows_meta, group_key)

    # (Opzionale) prova a inferire numero di bin dal modello per un check extra
    def _infer_pos_bins(model, default_bins=64):
        cand = ["pos_bins", "num_pos_embeddings", "grid_size", "spatial_bins",
                "position_bins", "pos_emb.num_embeddings", "pos_embedding.num_embeddings"]
        for attr in cand:
            obj, ok = model, True
            for part in attr.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False; break
            if ok and isinstance(obj, (int, float)):
                return int(obj)
        return int(default_bins)

    

    if gene_log1p:
        Xgene = np.log1p(Xgene, dtype=np.float32)

    # 4) Build mclSTExp model (dims from loaded arrays)
    mcl_cfg = {
        "checkpoint": checkpoint,
        "temperature": temperature,
        "input_dim_image": int(Ximg.shape[1]),
        "input_dim_transcriptome": int(Xgene.shape[1]),
        "mcl_projection_dim": mcl_projection_dim,
        "mcl_heads_num": mcl_heads_num,
        "mcl_heads_dim": mcl_heads_dim,
        "mcl_heads_layers": mcl_heads_layers,
        "dropout_prob": dropout_prob,
        "hvg_k": hvg_k,
        "gene_log1p": gene_log1p,
        "image_pca_dim": pca_dim,
    }
    model = load_mcl_from_cfg(mcl_cfg).to(device).eval()
    print("[model] mclSTExp-Attention ready.")
    
    _pos_bins = _infer_pos_bins(model, default_bins=64)
    # Sanity check: se il modello fa dentro idx = floor(norm * (bins-1)), i tuoi idx resteranno in range
    # (non lo usiamo per creare indici qui, ma è utile per debuggare)
    print(f"[dbg] coords normalized per-group='{group_key}', pos_bins≈{_pos_bins}")
    
    # PRIMA del loop di inference, subito dopo model = ... .eval()
    encode_expr = None
    encode_img  = None
    if hasattr(model, "encode_expression") and hasattr(model, "encode_image"):
        encode_expr = model.encode_expression
        encode_img  = model.encode_image
    elif hasattr(model, "encode") and callable(getattr(model, "encode")):
        # a volte un unico encode(dict) -> {"spot":..., "image":...}
        encode_expr = lambda xg, p2: model.encode({"expression": xg, "position": p2})["spot"]
        encode_img  = lambda xi, p2: model.encode({"image": xi, "position": p2})["image"]


    
    # 5) Inference on precomputed batches (safe w/ last-batch padding)
    # 5) Inference on precomputed batches — evita ultimo batch corto
    bs = int(batch_size)
    IMG_list, TX_list = [], []

    # 1) prova a usare encode-only (se disponibile)
    encode_expr = getattr(model, "encode_expression", None)
    encode_img  = getattr(model, "encode_image", None)
    if not (callable(encode_expr) and callable(encode_img)):
        # tenta encode(dict)->{"spot","image"}
        enc = getattr(model, "encode", None)
        if callable(enc):
            def encode_expr(xg, p2): return enc({"expression": xg, "position": p2})["spot"]
            def encode_img (xi, p2): return enc({"image": xi, "position": p2})["image"]
        else:
            encode_expr = None
            encode_img  = None

    # 2) deduci un k minimo ragionevole dal modello (fallback 32)
    inferred_k = infer_knn_k(model, default=32)
    min_needed = max(inferred_k + 1, 2*inferred_k + 1, 65)  # prudente

    with torch.no_grad():
        N = len(names_all)
        # costruisci slicing che EVITA ultimo batch corto
        for i, j in build_batches(N, bs, min_needed=min_needed):
            cur_bs = j - i
            Xi = torch.tensor(Ximg[i:j],  dtype=torch.float32, device=device)
            Xg = torch.tensor(Xgene[i:j], dtype=torch.float32, device=device)
            P2 = torch.tensor(coords_xy_norm[i:j], dtype=torch.float32, device=device)


            if callable(encode_expr) and callable(encode_img):
                # percorso "puro": nessun knn/gather interno se le encode sono davvero pure
                spot_emb = encode_expr(Xg, P2)
                img_emb  = encode_img(Xi, P2)
            else:
                # fallback: usa forward completo MA il batch ora è >= min_needed
                batch_dict = {"image": Xi, "expression": Xg, "position": P2}
                spot_emb, img_emb, _ = model(batch_dict)

            IMG_list.append(img_emb.detach().cpu().numpy())
            TX_list.append(spot_emb.detach().cpu().numpy())




    IMG = np.concatenate(IMG_list, axis=0)
    TX  = np.concatenate(TX_list,  axis=0)
    print(f"[eval] encoded N={IMG.shape[0]} samples")

    # 6) Retrieval metrics (HESCAPE R@k)
    IMG = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-9)
    TX  = TX  / (np.linalg.norm(TX,  axis=1, keepdims=True) + 1e-9)

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
    m = em(batch)  # -> keys: test_R@k_I2G, test_R@k_G2I

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

    print("\n=== Retrieval (precomputed CSV) — mclSTExp (HESCAPE R@k) ===")
    print("I2G  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        metrics_row["test_R@1_I2G"], metrics_row["test_R@5_I2G"], metrics_row["test_R@10_I2G"]
    ))
    print("G2I  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        metrics_row["test_R@1_G2I"], metrics_row["test_R@5_G2I"], metrics_row["test_R@10_G2I"]
    ))

if __name__ == "__main__":
    main()
