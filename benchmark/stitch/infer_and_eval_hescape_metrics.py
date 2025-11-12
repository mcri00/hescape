import os, sys, json, numpy as np, torch, yaml, traceback, gc, math
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

# --- robust import path setup: works for both "script" and "module" execution ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT   = os.path.dirname(SCRIPT_DIR)          # .../benchmark
PROJ_ROOT  = os.path.dirname(PKG_ROOT)            # .../hescape

for p in (SCRIPT_DIR, PKG_ROOT, PROJ_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Try relative-import style first (when running with -m), then fallback to local files.
try:
    from .cfg_bridge import load_model_train_cfgs            # module mode
    from .stitch_model import load_stitch_from_cfg
    from .preprocess import (
        HescapeTorchDataset,
        make_img_transform,
        make_inception_extractor,
        select_hvg_indices,
        collate_batch,
        encode_images,
    )
    from .coords import coords3d_from_xy_atlas, coords3d_from_xy_groups
    from .metrics_hescape import get_clip_metrics
except Exception:
    from cfg_bridge import load_model_train_cfgs             # script mode
    from stitch_model import load_stitch_from_cfg
    from preprocess import (
        HescapeTorchDataset,
        make_img_transform,
        make_inception_extractor,
        select_hvg_indices,
        collate_batch,
        encode_images,
    )
    from coords import coords3d_from_xy_atlas, coords3d_from_xy_groups
    from metrics_hescape import get_clip_metrics

from sklearn.decomposition import PCA
from datasets import ClassLabel

import csv

from collections import Counter
import pandas as pd
from PIL import Image


def _load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)

def _merge(over, base):
    out = dict(over)
    out.update({k: v for k, v in base.items() if v is not None})
    return out

def decode_field(fname, val, features):
    feat = features.get(fname)
    if isinstance(feat, ClassLabel):
        try:
            return feat.int2str(int(val))
        except Exception:
            return str(val)
    return "" if val is None else val

#new for test
def _load_id_whitelist(cfg):
    """
    Returns a set of sample IDs to keep, or None if no whitelist is provided.
    Accepts either:
      - cfg['test_ids']: list of strings in YAML
      - cfg['test_ids_file']: CSV/TSV with a column named 'id'
    """
    if cfg.get("test_ids"):
        return set(map(str, cfg["test_ids"]))

    p = cfg.get("test_ids_file")
    if not p:
        return None

    wl = set()
    with open(p, "r", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
        except Exception:
            reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"'id' column not found in {p}. Columns={reader.fieldnames}")
        for row in reader:
            if row.get("id"):
                wl.add(str(row["id"]).strip())
    if not wl:
        raise ValueError(f"No IDs loaded from {p}")
    return wl

def _detect_id_field(features, prefer=None):
    """Pick which HF field carries the sample ID."""
    if prefer:
        return prefer
    for c in ("id", "dataset_id", "name"):
        if c in features:
            return c
    return None

def _get_sample_id_from_row(row, id_field, features):
    """Extract a comparable string sample-id from a HF row."""
    if id_field and (id_field in row):
        val = row.get(id_field)
        try:
            from datasets import ClassLabel
            feat = features.get(id_field)
            if isinstance(feat, ClassLabel):
                return feat.int2str(int(val))
        except Exception:
            pass
        return "" if val is None else str(val)
    for k in ("id", "dataset_id", "name"):
        if k in row and row[k] is not None:
            return str(row[k])
    return ""
# end new for test

def run_from_cfg(user_cfg: dict):
    print(f"[boot] python: {sys.executable}", flush=True)

    runtime_defaults = {
        "panel": "human-5k-panel",
        "split": "train",
        "batch_size": 64,
        "num_workers": 8,
        "pin_memory": True,
        "img_size": 299,
        "img_mean": [0.5, 0.5, 0.5],
        "img_std":  [0.5, 0.5, 0.5],
        "slice_field": "atlas",
        "out_dir": None,
        # streaming controls
        "streaming": True,
        "hvg_fit_samples": 30000,
        "eval_max_samples": 30000,
        "stream_shuffle_buffer": 0,
        #new for test
        "id_field": None,      # "id", "dataset_id", or "name" (auto if None)
        "test_ids": None,      # optional list in YAML
        "test_ids_file": None, # optional CSV/TSV path with column 'id'
    }

    assert "train_cfg_data" in user_cfg and "train_cfg_model" in user_cfg, \
        "Il YAML deve contenere 'train_cfg_data' e 'train_cfg_model'."
    print("[stage] loading training JSONs…", flush=True)
    derived = load_model_train_cfgs(user_cfg["train_cfg_data"], user_cfg["train_cfg_model"])
    cfg = _merge(derived, _merge(runtime_defaults, user_cfg))
        
    #new for test
    id_whitelist = _load_id_whitelist(cfg)  # set[str] or None
    print(f"[filter] using {len(id_whitelist) if id_whitelist else 0} whitelisted IDs", flush=True)
    remaining = set(id_whitelist) if id_whitelist else None
    STOP_AFTER_MISSES = 1000   # tune: 1000–10000; depends on typical rows per sample

    # trackers
    seen_ids_hvg  = Counter()
    seen_ids_eval = Counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n[FINAL CONFIG]", flush=True)
    print(json.dumps(cfg, indent=2, default=str), flush=True)

    # STREAMING dataset
    if bool(cfg["streaming"]):
        print("[stage] opening HF dataset stream…", flush=True)
        ds_stream = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"], streaming=True)

        #new for test
        features = ds_stream.features
        id_field = _detect_id_field(features, cfg.get("id_field"))

        # HVG sui primi N
        #old version fit_n = int(cfg["hvg_fit_samples"])
        #new for test
        # HVG sui primi N (0 → use all matching rows)
        fit_n = int(cfg.get("hvg_fit_samples", 30000))
        if fit_n <= 0:
            fit_n = math.inf
        #end new for test

        '''
        #old version
        print(f"[hvg] collecting first {fit_n} rows for HVG…", flush=True)
        g_list, seen = [], 0
        
        for r in ds_stream:
            g = np.asarray(r["gexp"], dtype=np.float32)
            g = np.squeeze(g)                 # (G,) — rimuove eventuale (1, G)
            if g.ndim != 1:
                raise ValueError(f"gexp shape inattesa durante HVG fit: {g.shape}")
            g_list.append(g)
            seen += 1
            if seen % 2000 == 0:
                print(f"  - collected {seen}", flush=True)
            if seen >= fit_n:
                break
        '''
        #new for test
        miss_streak = 0

        print(f"[hvg] collecting first {fit_n} rows for HVG…", flush=True)
        g_list, seen = [], 0

        for r in ds_stream:
            sid = _get_sample_id_from_row(r, id_field, features)

            if id_whitelist:
                if sid not in id_whitelist:
                    # only start counting misses *after* we've seen all whitelist IDs at least once
                    if remaining is not None and len(remaining) == 0:
                        miss_streak += 1
                        if miss_streak == 1:
                            print("[hvg] all whitelist IDs seen; starting miss-streak…", flush=True)
                        if miss_streak >= STOP_AFTER_MISSES:
                            print(f"[hvg] early-stop after {STOP_AFTER_MISSES} consecutive non-whitelist rows.", flush=True)
                            break
                    continue
                # whitelist HIT
                seen_ids_hvg[sid] += 1
                if remaining is not None and sid in remaining:
                    remaining.remove(sid)
                    miss_streak = 0  # reset on hit
            else:
                # no whitelist: count everything
                seen_ids_hvg[sid] += 1

            g = np.asarray(r["gexp"], dtype=np.float32).squeeze()
            if g.ndim != 1:
                raise ValueError(f"gexp shape inattesa durante HVG fit: {g.shape}")
            g_list.append(g)
            seen += 1

            if seen % 2000 == 0:
                print(f"  - collected {seen}", flush=True)
            if seen >= fit_n:
                break

            
        # >>> HVG SUMMARY <<<
        print(f"[hvg] collected {seen} rows after ID filter.", flush=True)
        if id_whitelist:
            missing = set(id_whitelist) - set(seen_ids_hvg.keys())
            if missing:
                print(
                    f"[hvg][warn] whitelist IDs with 0 rows: {sorted(list(missing))[:10]}"
                    f"{' ...' if len(missing) > 10 else ''}",
                    flush=True
                )
        top_hvg = seen_ids_hvg.most_common(10)
        print(f"[hvg] per-ID counts (top 10): {top_hvg}", flush=True)
        

        if not g_list:
            raise RuntimeError("Stream vuoto: impossibile stimare HVG.")

        G_all = np.stack(g_list, axis=0)      # (N, G)
        if G_all.ndim != 2:
            raise ValueError(f"G_all deve essere 2D, invece è {G_all.shape}")

        print(f"[hvg] fitting HVG on {G_all.shape}…", flush=True)
        hvg_idx = select_hvg_indices(G_all, int(cfg["hvg_k"]))
        assert len(hvg_idx) == int(cfg["hvg_k"]), f"HVG idx len {len(hvg_idx)} != hvg_k {cfg['hvg_k']}"
        print("[hvg] done.", flush=True)


        # nuova stream per inference
        print("[stage] opening second stream for inference…", flush=True)
        '''
        #old version
        ds_eval_stream = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"], streaming=True)
        if int(cfg.get("stream_shuffle_buffer", 0)) > 0:
            ds_eval_stream = ds_eval_stream.shuffle(seed=42, buffer_size=int(cfg["stream_shuffle_buffer"]))
        loader = None
        features = ds_eval_stream.features
        '''
        #new for test
        ds_eval_stream = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"], streaming=True)
        features = ds_eval_stream.features
        id_field = _detect_id_field(features, cfg.get("id_field"))
        if int(cfg.get("stream_shuffle_buffer", 0)) > 0:
            ds_eval_stream = ds_eval_stream.shuffle(seed=42, buffer_size=int(cfg["stream_shuffle_buffer"]))
        loader = None
        
        remaining_eval = set(id_whitelist) if id_whitelist else None
        miss_streak_eval = 0


    else:
        '''
        #old version
        # fallback non-streaming (serve molto spazio)
        ds = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"])
        tds = HescapeTorchDataset(ds)
        G_all = np.stack([np.array(r["gexp"], dtype=np.float32) for r in ds], axis=0)
        hvg_idx = select_hvg_indices(G_all, int(cfg["hvg_k"]))
        loader = DataLoader(tds, batch_size=int(cfg["batch_size"]), shuffle=False,
                            num_workers=int(cfg["num_workers"]), pin_memory=bool(cfg["pin_memory"]))
        ds_eval_stream = None
        features = ds.features
        '''
        #new for test
         # fallback non-streaming (serve molto spazio)
        ds_full = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"])
        features = ds_full.features
        id_field = _detect_id_field(features, cfg.get("id_field"))

        if id_whitelist:
            print(f"[filter] applying whitelist to non-streaming dataset…", flush=True)
            def _keep(rec):
                sid = _get_sample_id_from_row(rec, id_field, features)
                return sid in id_whitelist
            ds = ds_full.filter(_keep)
        else:
            ds = ds_full

        if len(ds) == 0:
            raise RuntimeError("Dataset vuoto dopo il filtro ID — controlla 'test_ids' / 'test_ids_file' e 'id_field'.")

        # proceed on the filtered set only
        tds = HescapeTorchDataset(ds)
        G_all = np.stack([np.array(r["gexp"], dtype=np.float32).squeeze() for r in ds], axis=0)
        hvg_idx = select_hvg_indices(G_all, int(cfg["hvg_k"]))
        loader = DataLoader(tds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=bool(cfg["pin_memory"]))
        ds_eval_stream = None
           
        #end new for test


    extractor = make_inception_extractor(device=device) if bool(cfg["use_inception"]) else None
    img_tf = make_img_transform(int(cfg["img_size"]), cfg["img_mean"], cfg["img_std"])
    
    pca_model = None
    if bool(cfg.get("use_inception", False)) and int(cfg.get("image_pca_dim", 0)) > 0:
        pca_fit_n = int(cfg.get("pca_fit_samples", max(4*int(cfg["image_pca_dim"]), int(cfg["batch_size"]))))
        pca_fit_n = max(pca_fit_n, int(cfg["image_pca_dim"]))  # safety
        print(f"[pca] warm-up: fitting PCA on {pca_fit_n} image features...", flush=True)

        # apri una stream separata (perché quella precedente è già stata consumata dall'HVG)
        pca_stream = load_dataset(
            "Peng-AI/hescape-pyarrow",
            name=cfg["panel"],
            split=cfg["split"],
            streaming=True,
        )

        feats_buf = []
        bs = int(cfg["batch_size"])
        seen = 0
        buffer = []
        for rec in pca_stream:
            buffer.append(rec)
            if len(buffer) == bs:
                # estrai solo le feature immagine, senza PCA
                imgs = torch.stack([img_tf(b["image"] if hasattr(b["image"], "size") else
                                        Image.fromarray(np.array(b["image"])).convert("RGB"))
                                    for b in buffer], dim=0)
                with torch.no_grad():
                    img_feats = encode_images(extractor, imgs, device=device)  # (B, 2048)
                feats_buf.append(img_feats.cpu().numpy())
                buffer.clear()
                seen += img_feats.shape[0]
                if seen >= pca_fit_n:
                    break
        if buffer and seen < pca_fit_n:
            imgs = torch.stack([img_tf(b["image"] if hasattr(b["image"], "size") else
                                    Image.fromarray(np.array(b["image"])).convert("RGB"))
                                for b in buffer], dim=0)
            with torch.no_grad():
                img_feats = encode_images(extractor, imgs, device=device)
            feats_buf.append(img_feats.cpu().numpy())
            seen += img_feats.shape[0]

        if not feats_buf:
            raise RuntimeError("Nessuna feature immagine raccolta per il fit della PCA.")
        X_fit = np.concatenate(feats_buf, axis=0)[:pca_fit_n]  # (Nfit, 2048)
        if X_fit.shape[0] < int(cfg["image_pca_dim"]):
            raise ValueError(f"pca_fit_samples={X_fit.shape[0]} < image_pca_dim={cfg['image_pca_dim']}. "
                            f"Aumenta pca_fit_samples o riduci image_pca_dim.")
        print(f"[pca] fitting PCA with X_fit={X_fit.shape} -> n_components={cfg['image_pca_dim']}", flush=True)
        pca_model = PCA(n_components=int(cfg["image_pca_dim"])).fit(X_fit)
        print("[pca] done.", flush=True)

        # (facoltativo) libera RAM
        del feats_buf, X_fit; gc.collect()
    else:
        print("[pca] skipped (use_inception=False o image_pca_dim=0).", flush=True)

    print("[stage] building model + loading checkpoint…", flush=True)
    model_cfg = {
        "checkpoint": cfg["checkpoint"],
        "input_dim_image": int(cfg["input_dim_image"]),
        "input_dim_transcriptome": int(cfg["input_dim_transcriptome"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "output_dim": int(cfg["output_dim"]),
        "num_layers": int(cfg["num_layers"]),
        "dropout_prob": float(cfg["dropout_prob"]),
        "use_positional_embedding": bool(cfg["use_positional_embedding"]),
        "use_positional_attention": bool(cfg.get("use_positional_attention", False)),
        "use_feature_selector": bool(cfg.get("use_feature_selector", False)),
        "pos_embedding_dim": int(cfg.get("pos_embedding_dim", 128)),
        "hvg_k": int(cfg["hvg_k"]),
        "gene_log1p": bool(cfg["gene_log1p"]),
        "image_pca_dim": cfg.get("image_pca_dim", None),
    }
    model = load_stitch_from_cfg(model_cfg).to(device).eval()

    first_img_linear = next(m for m in model.image_encoder.model if isinstance(m, torch.nn.Linear))
    first_tx_linear  = next(m for m in model.transcriptome_encoder.model if isinstance(m, torch.nn.Linear))
    assert first_img_linear.in_features == int(cfg["input_dim_image"])
    assert first_tx_linear.in_features  == int(cfg["input_dim_transcriptome"])
    print("[stage] model ready.", flush=True)

    print("[stage] inference…", flush=True)
    names_all, IMG_list, TX_list = [], [], []
    coords_all, atlas_all, source_all = [], [], []
    tissue_all, diagnosis_all, gttype_all = [], [], []
    rows_meta = []

    with torch.no_grad():
        #pca_model = None
        if bool(cfg["streaming"]):
            #old version eval_max_n = int(cfg["eval_max_samples"])
            #new for test
            # evaluation cap (0 → no cap)
            eval_max_n = cfg.get("eval_max_samples", 0)
            eval_max_n = float("inf") if (eval_max_n in (None, 0, -1)) else int(eval_max_n)
            #end new for test
            bs = int(cfg["batch_size"])
            processed, buffer = 0, []
            for rec in ds_eval_stream:
                #new for test
                sid = _get_sample_id_from_row(rec, id_field, features)

                if id_whitelist:
                    if sid not in id_whitelist:
                        if remaining_eval is not None and len(remaining_eval) == 0:
                            miss_streak_eval += 1
                            if miss_streak_eval == 1:
                                print("[eval] all whitelist IDs seen; starting miss-streak…", flush=True)
                            if miss_streak_eval >= STOP_AFTER_MISSES:
                                print(f"[eval] early-stop after {STOP_AFTER_MISSES} consecutive non-whitelist rows.", flush=True)
                                break
                        continue
                    # whitelist HIT
                    seen_ids_eval[sid] += 1
                    if remaining_eval is not None and sid in remaining_eval:
                        remaining_eval.remove(sid)
                        miss_streak_eval = 0  # reset on hit
                else:
                    seen_ids_eval[sid] += 1
                #end new for test
                
                buffer.append(rec)
                if len(buffer) == bs:
                    #new for test
                    sid_list = [ _get_sample_id_from_row(b, id_field, features) for b in buffer ]
                    #end new for test
                    
                    img_feats, g_tensor, names, coords, atlas, sources, gt, pca_model = collate_batch(
                        buffer, img_tf, extractor, cfg, hvg_idx, pca_model=pca_model, device=device
                    )
                    src_list = buffer
                    # Build unique_id and collect all labels/meta
                    for i in range(len(names)):
                        rec = src_list[i]  # HF row dict
                        # coords: coords is a tensor/array [B,2]; turn into floats
                        x_val = float(coords[i][0])
                        y_val = float(coords[i][1])

                        # slice key: prefer atlas if present, otherwise fall back to name category
                        atlas_str = rec.get("atlas", "") or ""
                        name_cat  = decode_field("name", rec.get("name", ""), features)
                        slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat

                        unique_id = f"{slice_key}_{int(x_val)}x{int(y_val)}"
                        names_all.append(unique_id)

                        # collect ALL metadata (decode ClassLabel where needed)
                        rows_meta.append({
                            "Image Name":          unique_id,
                            "sample_id":  sid_list[i], 
                            "name_category":       name_cat,
                            "atlas":               atlas_str,
                            "source":              decode_field("source",             rec.get("source",""), features),
                            "x":                   x_val,
                            "y":                   y_val,
                            "tissue":              decode_field("tissue",             rec.get("tissue",""), features),
                            "diagnosis":           rec.get("diagnosis",""),
                            "cancer":              rec.get("cancer",""),
                            "oncotree_code":       rec.get("oncotree_code",""),
                            "tumor_tissue_type":   rec.get("tumor_tissue_type",""),
                            "tumor_grade":         rec.get("tumor_grade",""),
                            "age":                 rec.get("age",""),
                            "gender":              rec.get("gender",""),
                            "race":                rec.get("race",""),
                            "treatment_type":      rec.get("treatment_type",""),
                            "therapeutic_agents":  rec.get("therapeutic_agents",""),
                            "assay":               rec.get("assay",""),
                            "preservation_method": rec.get("preservation_method",""),
                            "stain":               rec.get("stain",""),
                            "spaceranger":         rec.get("spaceranger",""),
                            "species":             rec.get("species",""),
                            "cytassist":           rec.get("cytassist",""),
                        })
                    
                    slice_ids = []
                    for i in range(len(names)):
                        rec = src_list[i]
                        atlas_str = rec.get("atlas", "") or ""
                        name_cat  = decode_field("name", rec.get("name",""), features)
                        slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat
                        slice_ids.append(slice_key)

                    coords3d = coords3d_from_xy_groups(
                        xy_coords=coords,
                        slice_ids_str=slice_ids,
                        device=device,
                        dtype=torch.float32,
                    )
                    # subito prima di: img_out, tx_out, _ = model(...)
                    if img_feats.shape[1] != int(cfg["input_dim_image"]):
                        raise RuntimeError(
                            f"Dimensione feature immagine {img_feats.shape[1]} "
                            f"≠ input_dim_image atteso {cfg['input_dim_image']}. "
                            f"Controlla use_inception / image_pca_dim / PCA warm-up."
                        )

                    img_out, tx_out, _ = model(img_feats.to(device), g_tensor.to(device), coords_3d=coords3d)
                    IMG_list.append(img_out.cpu().numpy())
                    TX_list.append(tx_out.cpu().numpy())
                    
                    processed += len(names)
                    if processed % (bs * 5) == 0:
                        print(f"  - processed {processed}", flush=True)
                    buffer.clear()
                    if processed >= eval_max_n:
                        break
            # coda
            if buffer and processed < eval_max_n:
                #new for test
                sid_list = [ _get_sample_id_from_row(b, id_field, features) for b in buffer ]
                #end for test
                
                img_feats, g_tensor, names, coords, atlas, sources, gt, pca_model = collate_batch(
                    buffer, img_tf, extractor, cfg, hvg_idx, pca_model=pca_model, device=device
                )
                # Use the raw HF rows to read metadata (buffer contains the original dicts)
                src_list = buffer  # raw HF rows for this mini-batch

                # Build unique_id and collect all labels/meta
                for i in range(len(names)):
                    rec = src_list[i]  # HF row dict
                    # coords: coords is a tensor/array [B,2]; turn into floats
                    x_val = float(coords[i][0])
                    y_val = float(coords[i][1])

                    # slice key: prefer atlas if present, otherwise fall back to name category
                    atlas_str = rec.get("atlas", "") or ""
                    name_cat  = decode_field("name", rec.get("name", ""), features)
                    slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat

                    unique_id = f"{slice_key}_{int(x_val)}x{int(y_val)}"
                    names_all.append(unique_id)

                    # collect ALL metadata (decode ClassLabel where needed)
                    rows_meta.append({
                        "Image Name":          unique_id,
                        "sample_id":  sid_list[i], 
                        "name_category":       name_cat,
                        "atlas":               atlas_str,
                        "source":              decode_field("source",             rec.get("source",""), features),
                        "x":                   x_val,
                        "y":                   y_val,
                        "tissue":              decode_field("tissue",             rec.get("tissue",""), features),
                        "diagnosis":           rec.get("diagnosis",""),
                        "cancer":              rec.get("cancer",""),
                        "oncotree_code":       rec.get("oncotree_code",""),
                        "tumor_tissue_type":   rec.get("tumor_tissue_type",""),
                        "tumor_grade":         rec.get("tumor_grade",""),
                        "age":                 rec.get("age",""),
                        "gender":              rec.get("gender",""),
                        "race":                rec.get("race",""),
                        "treatment_type":      rec.get("treatment_type",""),
                        "therapeutic_agents":  rec.get("therapeutic_agents",""),
                        "assay":               rec.get("assay",""),
                        "preservation_method": rec.get("preservation_method",""),
                        "stain":               rec.get("stain",""),
                        "spaceranger":         rec.get("spaceranger",""),
                        "species":             rec.get("species",""),
                        "cytassist":           rec.get("cytassist",""),
                    })


                slice_ids = []
                for i in range(len(names)):
                    rec = src_list[i]
                    atlas_str = rec.get("atlas", "") or ""
                    name_cat  = decode_field("name", rec.get("name",""), features)
                    slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat
                    slice_ids.append(slice_key)

                coords3d = coords3d_from_xy_groups(
                    xy_coords=coords,
                    slice_ids_str=slice_ids,
                    device=device,
                    dtype=torch.float32,
                )
                # subito prima di: img_out, tx_out, _ = model(...)
                if img_feats.shape[1] != int(cfg["input_dim_image"]):
                    raise RuntimeError(
                        f"Dimensione feature immagine {img_feats.shape[1]} "
                        f"≠ input_dim_image atteso {cfg['input_dim_image']}. "
                        f"Controlla use_inception / image_pca_dim / PCA warm-up."
                    )

                img_out, tx_out, _ = model(img_feats.to(device), g_tensor.to(device), coords_3d=coords3d)
                IMG_list.append(img_out.cpu().numpy())
                TX_list.append(tx_out.cpu().numpy())
            
            # >>> EVAL SUMMARY <<<
            print(f"[eval] processed {processed} rows after ID filter.", flush=True)
            if id_whitelist:
                bad = set(seen_ids_eval.keys()) - set(id_whitelist)
                if bad:
                    print(
                        f"[eval][ERROR] found rows with IDs NOT in whitelist: {sorted(list(bad))[:10]}"
                        f"{' ...' if len(bad) > 10 else ''}",
                        flush=True
                    )
            top_eval = seen_ids_eval.most_common(10)
            print(f"[eval] per-ID counts (top 10): {top_eval}", flush=True)
                
        else:
            for batch in tqdm(loader, desc="Encode"):
                
                
                img_feats, g_tensor, names, coords, atlas, sources, gt, pca_model = collate_batch(
                    batch, img_tf, extractor, cfg, hvg_idx, pca_model=pca_model, device=device
                )
                # If your HescapeTorchDataset returns original HF rows inside the batch,
                # use them. Otherwise, plumb the raw rows through your dataset/collate.
                src_list = batch  # list of original HF rows for this mini-batch
                
                #new for test
                sid_list = [_get_sample_id_from_row(b, id_field, features) for b in src_list]
                #end for test
                
                for i in range(len(names)):
                    rec = src_list[i]
                    x_val = float(coords[i][0])
                    y_val = float(coords[i][1])

                    atlas_str = rec.get("atlas", "") or ""
                    name_cat  = decode_field("name", rec.get("name",""), features)
                    slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat

                    unique_id = f"{slice_key}_{int(x_val)}x{int(y_val)}"
                    names_all.append(unique_id)

                    rows_meta.append({
                        "Image Name":          unique_id,
                        "sample_id":  sid_list[i], 
                        "name_category":       name_cat,
                        "atlas":               atlas_str,
                        "source":              decode_field("source",             rec.get("source",""), features),
                        "x":                   x_val,
                        "y":                   y_val,
                        "tissue":              decode_field("tissue",             rec.get("tissue",""), features),
                        "diagnosis":           rec.get("diagnosis",""),
                        "cancer":              rec.get("cancer",""),
                        "oncotree_code":       rec.get("oncotree_code",""),
                        "tumor_tissue_type":   rec.get("tumor_tissue_type",""),
                        "tumor_grade":         rec.get("tumor_grade",""),
                        "age":                 rec.get("age",""),
                        "gender":              rec.get("gender",""),
                        "race":                rec.get("race",""),
                        "treatment_type":      rec.get("treatment_type",""),
                        "therapeutic_agents":  rec.get("therapeutic_agents",""),
                        "assay":               rec.get("assay",""),
                        "preservation_method": rec.get("preservation_method",""),
                        "stain":               rec.get("stain",""),
                        "spaceranger":         rec.get("spaceranger",""),
                        "species":             rec.get("species",""),
                        "cytassist":           rec.get("cytassist",""),
                    })

                slice_ids = []
                for i in range(len(names)):
                    rec = src_list[i]
                    atlas_str = rec.get("atlas", "") or ""
                    name_cat  = decode_field("name", rec.get("name",""), features)
                    slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat
                    slice_ids.append(slice_key)

                coords3d = coords3d_from_xy_groups(
                    xy_coords=coords,
                    slice_ids_str=slice_ids,
                    device=device,
                    dtype=torch.float32,
                )
                # subito prima di: img_out, tx_out, _ = model(...)
                if img_feats.shape[1] != int(cfg["input_dim_image"]):
                    raise RuntimeError(
                        f"Dimensione feature immagine {img_feats.shape[1]} "
                        f"≠ input_dim_image atteso {cfg['input_dim_image']}. "
                        f"Controlla use_inception / image_pca_dim / PCA warm-up."
                    )

                img_out, tx_out, _ = model(img_feats.to(device), g_tensor.to(device), coords_3d=coords3d)
                IMG_list.append(img_out.cpu().numpy())
                TX_list.append(tx_out.cpu().numpy())
                

    IMG = np.concatenate(IMG_list, axis=0) if IMG_list else np.empty((0, int(cfg["output_dim"])), dtype=np.float32)
    TX  = np.concatenate(TX_list,  axis=0) if TX_list  else np.empty((0, int(cfg["output_dim"])), dtype=np.float32)
    print(f"[eval] encoded {IMG.shape[0]} samples", flush=True)
    if IMG.shape[0] == 0:
        raise RuntimeError("0 campioni encodati: controlla streaming e permessi HF.")


    IMG = IMG / (np.linalg.norm(IMG, axis=1, keepdims=True) + 1e-9)
    TX  = TX  / (np.linalg.norm(TX,  axis=1, keepdims=True) + 1e-9)
    m = get_clip_metrics(torch.tensor(IMG), torch.tensor(TX), logit_scale=torch.tensor(1.0), stage="test")
    print("\n=== HESCAPE Retrieval ({} / {}) ===".format(cfg["panel"], cfg["split"]), flush=True)
    print("I2G  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        m["test/image_to_gene_R@1"].item(),
        m["test/image_to_gene_R@5"].item(),
        m["test/image_to_gene_R@10"].item()
    ), flush=True)
    print("G2I  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}".format(
        m["test/gene_to_image_R@1"].item(),
        m["test/gene_to_image_R@5"].item(),
        m["test/gene_to_image_R@10"].item()
    ), flush=True)

    if cfg.get("out_dir"):
        os.makedirs(cfg["out_dir"], exist_ok=True)
        
        
        # HVG and eval per-ID counts
        if len(seen_ids_hvg):
            pd.DataFrame(
                [{"id": k, "n_rows_hvg": v} for k, v in seen_ids_hvg.items()]
            ).to_csv(os.path.join(cfg["out_dir"], "hvg_id_counts.csv"), index=False)

        if len(seen_ids_eval):
            pd.DataFrame(
                [{"id": k, "n_rows_eval": v} for k, v in seen_ids_eval.items()]
            ).to_csv(os.path.join(cfg["out_dir"], "eval_id_counts.csv"), index=False)

        IMG = np.concatenate(IMG_list, axis=0)
        TX  = np.concatenate(TX_list,  axis=0)
        
        if id_whitelist:
            # Nothing outside the whitelist?
            assert set(seen_ids_eval.keys()).issubset(id_whitelist), \
                "[assert] Some evaluated IDs are NOT in the whitelist."


        # Embeddings with stable, readable column names
        img_df = pd.DataFrame(IMG, columns=[f"img_e{i}" for i in range(IMG.shape[1])])
        tx_df  = pd.DataFrame(TX,  columns=[f"tx_e{i}"  for i in range(TX.shape[1])])
        img_df.insert(0, "Image Name", names_all)
        tx_df.insert(0,  "Image Name", names_all)

        # All labels/metadata (decoded), 1:1 with names_all
        labels_df = pd.DataFrame(rows_meta)
        
        if "sample_id" in labels_df.columns and id_whitelist:
            bad_ids = set(labels_df["sample_id"].unique()) - set(id_whitelist)
            if bad_ids:
                print(f"[labels][ERROR] labels.csv contains non-whitelisted IDs: {sorted(list(bad_ids))[:10]}", flush=True)

        img_df.to_csv(os.path.join(cfg["out_dir"], "image_embeddings.csv"), index=False)
        tx_df.to_csv (os.path.join(cfg["out_dir"], "transcriptome_embeddings.csv"), index=False)
        labels_df.to_csv(os.path.join(cfg["out_dir"], "labels.csv"), index=False)

    print("[sanity]", dict(
        n_names=len(names_all),
        n_img=IMG.shape[0],
        n_tx=TX.shape[0],
        n_labels=len(labels_df)
    ))


    # --- metrics.csv/json come già fai ---
    metrics_row = {
        "panel": cfg["panel"],
        "split": cfg["split"],
        "checkpoint": os.path.basename(str(cfg.get("checkpoint", ""))),
        "n_samples": int(IMG.shape[0]),
        "hvg_k": int(cfg["hvg_k"]),
        "input_dim_image": int(cfg["input_dim_image"]),
        "input_dim_transcriptome": int(cfg["input_dim_transcriptome"]),
        "i2g_R@1": float(m["test/image_to_gene_R@1"].item()),
        "i2g_R@5": float(m["test/image_to_gene_R@5"].item()),
        "i2g_R@10": float(m["test/image_to_gene_R@10"].item()),
        "g2i_R@1": float(m["test/gene_to_image_R@1"].item()),
        "g2i_R@5": float(m["test/gene_to_image_R@5"].item()),
        "g2i_R@10": float(m["test/gene_to_image_R@10"].item()),
    }
    metrics_csv = os.path.join(cfg["out_dir"], "metrics.csv")
    write_header = not os.path.exists(metrics_csv)
    pd.DataFrame([metrics_row]).to_csv(metrics_csv, index=False, mode="a", header=write_header)

    metrics_json = os.path.join(cfg["out_dir"], "metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics_row, f, indent=2)

    print(f"[export] embeddings/labels salvati in: {cfg['out_dir']}")

    
        

if __name__ == "__main__":
    try:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", required=True, help="Path al YAML")
        args = ap.parse_args()
        cfg = _load_yaml(args.cfg)
        run_from_cfg(cfg)
    except Exception:
        print("\n[FATAL] eccezione non gestita:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        # workaround contro il crash di teardown (threads streaming/aiohttp/pyarrow)
        gc.collect()
        os._exit(0)
