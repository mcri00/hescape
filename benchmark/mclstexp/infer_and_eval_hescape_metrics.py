import os, sys, json, numpy as np, torch, yaml, traceback, gc
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
    from .mclstexp_model import load_mcl_from_cfg
    from .preprocess import (
        HescapeTorchDataset,
        make_img_transform,
        make_densenet_extractor,
        select_hvg_indices,
        collate_batch,
        encode_images,
    )
    
    from .metrics_hescape import get_clip_metrics
except Exception:
    from cfg_bridge import load_model_train_cfgs             # script mode
    from mclstexp_model import load_mcl_from_cfg
    from preprocess import (
        HescapeTorchDataset,
        make_img_transform,
        make_densenet_extractor,
        select_hvg_indices,
        collate_batch,
        encode_images,
    )
    
    from metrics_hescape import get_clip_metrics

from sklearn.decomposition import PCA
from datasets import ClassLabel

import torchvision.models as tvm
import torch.nn as nn

# === helper: YAML & merge ===
def _load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _merge(over, base):
    out = dict(over)
    out.update({k: v for k, v in base.items() if v is not None})
    return out

# === adattatore: mantieni la tua signature di prepare_features ===
def prepare_features_for_benchmark(
    images_feats: torch.Tensor,
    transcriptomes_np: np.ndarray,
    use_embeddings: bool,
    device: torch.device,
    encoding_model=None,                 # ignorato: arrivano già features
    normalize_transcriptomes: bool=True,
    pca_dim: int = 0
):
    """
    Compatibile con: image_features, transcriptome_features, _, _ = prepare_features(...)
    - images_feats: torch.FloatTensor [B, D_img] (DenseNet/PCA già fatte)
    - transcriptomes_np: np.ndarray [B, G] (HVG già applicato)
    """
    # immagini: sono già features
    if isinstance(images_feats, np.ndarray):
        images_feats = torch.tensor(images_feats, dtype=torch.float32)
    image_features = images_feats.to(device)
    feature_size_image = image_features.shape[1]

    # trascrittomi
    if isinstance(transcriptomes_np, torch.Tensor):
        transcriptomes_np = transcriptomes_np.cpu().numpy()
    X = transcriptomes_np.astype(np.float32, copy=False)
    if normalize_transcriptomes:
        X = np.log1p(X)
    transcriptome_features = torch.tensor(X, dtype=torch.float32, device=device)
    feature_size_transcriptome = transcriptome_features.shape[1]

    return image_features, transcriptome_features, feature_size_image, feature_size_transcriptome

# === utility: decode ClassLabel in stringhe (per eventuali export) ===
def decode_field(fname, val, features):
    feat = features.get(fname)
    if isinstance(feat, ClassLabel):
        try:
            return feat.int2str(int(val))
        except Exception:
            return str(val)
    return "" if val is None else val

# === MAIN ===
def run_from_cfg(user_cfg: dict):
    runtime_defaults = {
        # data
        "panel": "human-multi-tissue-panel",
        "split": "train",
        # loader
        "batch_size": 64,
        "num_workers": 8,
        "pin_memory": True,
        # images (DenseNet-121)
        "img_size": 224,
        "img_mean": [0.485, 0.456, 0.406],
        "img_std":  [0.229, 0.224, 0.225],
        "use_densenet": True,
        # streaming
        "streaming": True,
        "hvg_fit_samples": 30000,
        "eval_max_samples": 30000,
        "stream_shuffle_buffer": 0,
        # PCA on image features (DenseNet-121 → 1024 -> PCA_dim if >0)
        "image_pca_dim": 0,
        "pca_fit_samples": 4000,
        # outputs
        "out_dir": None,
    }

    assert "train_cfg_data" in user_cfg and "train_cfg_model" in user_cfg, \
        "cfg must contain 'train_cfg_data' and 'train_cfg_model' paths"
    derived = load_model_train_cfgs(user_cfg["train_cfg_data"], user_cfg["train_cfg_model"])
    cfg = _merge(derived, _merge(runtime_defaults, user_cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[FINAL CONFIG]")
    print(json.dumps(cfg, indent=2, default=str), flush=True)

    # === dataset & HVG ===
    if bool(cfg["streaming"]):
        print("[stage] opening HF dataset stream…", flush=True)
        ds_stream = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"], streaming=True)
        g_list, n = [], 0
        for r in ds_stream:
            g = np.asarray(r["gexp"], dtype=np.float32).squeeze()
            g_list.append(g)
            n += 1
            if n % 2000 == 0:
                print(f"  - collected {n}", flush=True)
            if n >= int(cfg["hvg_fit_samples"]):
                break
        if not g_list:
            raise RuntimeError("No rows read for HVG fit.")
        G_all = np.stack(g_list, axis=0)
        hvg_idx = select_hvg_indices(G_all, int(cfg["hvg_k"]))

        # fresh stream for inference
        ds_eval = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"], streaming=True)
        if int(cfg.get("stream_shuffle_buffer", 0)) > 0:
            ds_eval = ds_eval.shuffle(seed=42, buffer_size=int(cfg["stream_shuffle_buffer"]))
        loader = None
        features = ds_eval.features
    else:
        ds = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"])
        tds = HescapeTorchDataset(ds)
        G_all = np.stack([np.asarray(r["gexp"], dtype=np.float32) for r in ds], axis=0)
        hvg_idx = select_hvg_indices(G_all, int(cfg["hvg_k"]))
        loader = DataLoader(
            tds, batch_size=int(cfg["batch_size"]), shuffle=False,
            num_workers=int(cfg["num_workers"]), pin_memory=bool(cfg["pin_memory"])
        )
        ds_eval = None
        features = ds.features

    # === image extractor (DenseNet-121) & transform ===
    img_tf = make_img_transform(int(cfg["img_size"]), cfg["img_mean"], cfg["img_std"])
    extractor = make_densenet_extractor(device=device) if bool(cfg["use_densenet"]) else None
    raw_img_feat_dim = 1024  # DenseNet-121 pooled features

    # === optional PCA warm-up on image features ===
    pca_model = None
    if int(cfg.get("image_pca_dim", 0)) > 0:
        want = int(cfg["image_pca_dim"])
        fit_n = int(cfg.get("pca_fit_samples", max(4 * want, int(cfg["batch_size"]))))
        pca_stream = load_dataset("Peng-AI/hescape-pyarrow", name=cfg["panel"], split=cfg["split"], streaming=True)

        feats, bs, buf, seen = [], int(cfg["batch_size"]), [], 0
        with torch.no_grad():
            for rec in pca_stream:
                buf.append(rec)
                if len(buf) == bs:
                    imgs = torch.stack([img_tf(r["image"]) for r in buf], dim=0)
                    f = encode_images(extractor, imgs, device=device)  # (B, 1024)
                    feats.append(f.cpu().numpy()); seen += f.size(0)
                    buf.clear()
                    if seen >= fit_n:
                        break
            if buf and seen < fit_n:
                imgs = torch.stack([img_tf(r["image"]) for r in buf], dim=0)
                f = encode_images(extractor, imgs, device=device)
                feats.append(f.cpu().numpy())

        X_fit = np.concatenate(feats, axis=0)[:fit_n]
        if X_fit.shape[0] < want:
            raise ValueError(f"pca_fit_samples={X_fit.shape[0]} < image_pca_dim={want}")
        pca_model = PCA(n_components=want).fit(X_fit)
        print(f"[pca] DenseNet features {raw_img_feat_dim} → PCA {want}")

    # === build mclSTExp model (ATTENTION variant) ===
    assert "checkpoint" in cfg, "cfg must contain 'checkpoint'"
    mcl_cfg = {
        # required dims for Attention variant
        "input_dim_image":            int(cfg["input_dim_image"]),          # must match (1024 or PCA dim)
        "input_dim_transcriptome":             int(cfg["input_dim_transcriptome"]),  # == hvg_k
        "mcl_projection_dim":       int(cfg["mcl_projection_dim"]),
        "mcl_heads_num":            int(cfg.get("mcl_heads_num", 8)),
        "mcl_heads_dim":            int(cfg.get("mcl_heads_dim", 64)),
        "mcl_heads_layers":         int(cfg.get("mcl_heads_layers", 2)),
        "temperature":          float(cfg.get("temperature", 0.5)),
        "dropout_prob":              float(cfg.get("dropout_prob", 0.0)),
        "checkpoint":           cfg["checkpoint"],
        "hvg_k": int(cfg["hvg_k"]),
        "gene_log1p": bool(cfg["gene_log1p"]),
        "image_pca_dim": cfg.get("image_pca_dim", None),
    }
    model = load_mcl_from_cfg(mcl_cfg).to(device).eval()

    # quick sanity
    print("[model] mclSTExp-Attention ready.", flush=True)

    # === INFERENCE (use collate_batch exactly like your STitch path) ===
    names_all, IMG_list, TX_list = [], [], []
    processed, max_n = 0, int(cfg["eval_max_samples"])
    coords_all, atlas_all, source_all = [], [], []
    tissue_all, diagnosis_all, gttype_all = [], [], []
    rows_meta = []

    with torch.no_grad():
        if bool(cfg["streaming"]):
            
            bs, buffer = int(cfg["batch_size"]), []
            for rec in ds_eval:
                buffer.append(rec)
                if len(buffer) == bs:
                    # collate_batch handles: transform, extractor(img→feat), HVG + log1p, optional PCA (if pca_model)
                    img_feats, g_tensor, names, coords, atlas, sources, gt, pca_model = collate_batch(
                        buffer, img_tf, extractor, cfg, hvg_idx, pca_model=pca_model, device=device
                    )

                    # raw 2D coords → position tensor (no preprocessing)
                    position = torch.tensor(coords, dtype=torch.float32, device=device)

                    batch_dict = {
                        "image":      img_feats.to(device).float(),
                        "expression": g_tensor.to(device).float(),
                        "position":   position,   # 2D
                    }

                    
                    
                    
                    # buffer = lista di record HF originali usata per collate_batch
                    src_list = buffer
                    for i in range(len(names)):
                        rec = src_list[i]
                        # coords è (B,2) float (x,y)
                        x_val = float(coords[i][0])
                        y_val = float(coords[i][1])

                        # slice key: prova atlas, altrimenti il campo 'name' (se categoriale va decodificato)
                        atlas_str = rec.get("atlas", "") or ""
                        name_cat  = decode_field("name", rec.get("name", ""), features)
                        slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat
                        unique_id = f"{slice_key}_{int(x_val)}x{int(y_val)}"
                        names_all.append(unique_id)

                        # allinea a quanto fai nello script STitch
                        rows_meta.append({
                            "Image Name":          unique_id,
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

                    spot_emb, img_emb, _ = model(batch_dict)
                    IMG_list.append(img_emb.detach().cpu().numpy())
                    TX_list.append(spot_emb.detach().cpu().numpy())
                    
                    processed += len(names)
                    if processed % (bs * 5) == 0:
                        print(f"  - processed {processed}", flush=True)
                    buffer.clear()
                    if processed >= max_n:
                        break

            # tail
            if buffer and processed < max_n:
                img_feats, g_tensor, names, coords, atlas, sources, gt, pca_model = collate_batch(
                    buffer, img_tf, extractor, cfg, hvg_idx, pca_model=pca_model, device=device
                )
                position = torch.tensor(coords, dtype=torch.float32, device=device)
                batch_dict = {
                    "image":      img_feats.to(device).float(),
                    "expression": g_tensor.to(device).float(),
                    "position":   position,
                }
                
                # buffer = lista di record HF originali usata per collate_batch
                src_list = buffer
                for i in range(len(names)):
                    rec = src_list[i]
                    # coords è (B,2) float (x,y)
                    x_val = float(coords[i][0])
                    y_val = float(coords[i][1])

                    # slice key: prova atlas, altrimenti il campo 'name' (se categoriale va decodificato)
                    atlas_str = rec.get("atlas", "") or ""
                    name_cat  = decode_field("name", rec.get("name", ""), features)
                    slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat
                    unique_id = f"{slice_key}_{int(x_val)}x{int(y_val)}"
                    names_all.append(unique_id)

                    # allinea a quanto fai nello script STitch
                    rows_meta.append({
                        "Image Name":          unique_id,
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
                        
                spot_emb, img_emb, _ = model(batch_dict)
                IMG_list.append(img_emb.detach().cpu().numpy())
                TX_list.append(spot_emb.detach().cpu().numpy())


        else:
            # non-streaming
            for batch in tqdm(loader, desc="Encode"):
                img_feats, g_tensor, names, coords, atlas, sources, gt, pca_model = collate_batch(
                    batch, img_tf, extractor, cfg, hvg_idx, pca_model=pca_model, device=device
                )
                position = torch.tensor(coords, dtype=torch.float32, device=device)
                batch_dict = {
                    "image":      img_feats.to(device).float(),
                    "expression": g_tensor.to(device).float(),
                    "position":   position,
                }
                # buffer = lista di record HF originali usata per collate_batch
                src_list = batch
                for i in range(len(names)):
                    rec = src_list[i]
                    # coords è (B,2) float (x,y)
                    x_val = float(coords[i][0])
                    y_val = float(coords[i][1])

                    # slice key: prova atlas, altrimenti il campo 'name' (se categoriale va decodificato)
                    atlas_str = rec.get("atlas", "") or ""
                    name_cat  = decode_field("name", rec.get("name", ""), features)
                    slice_key = atlas_str if atlas_str not in (None, "", "NA") else name_cat
                    unique_id = f"{slice_key}_{int(x_val)}x{int(y_val)}"
                    names_all.append(unique_id)   

                    # allinea a quanto fai nello script STitch
                    rows_meta.append({
                        "Image Name":          unique_id,
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
                        
                spot_emb, img_emb, _ = model(batch_dict)
                IMG_list.append(img_emb.detach().cpu().numpy())
                TX_list.append(spot_emb.detach().cpu().numpy())
                    

    # === metrics ===
    if not IMG_list:
        raise RuntimeError("No samples were encoded — check streaming settings and HF permissions.")
    IMG = np.concatenate(IMG_list, axis=0)
    TX  = np.concatenate(TX_list,  axis=0)
    
    

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
        import pandas as pd

        IMG = np.concatenate(IMG_list, axis=0)
        TX  = np.concatenate(TX_list,  axis=0)

        # Embeddings with stable, readable column names
        img_df = pd.DataFrame(IMG, columns=[f"img_e{i}" for i in range(IMG.shape[1])])
        tx_df  = pd.DataFrame(TX,  columns=[f"tx_e{i}"  for i in range(TX.shape[1])])
        img_df.insert(0, "Image Name", names_all)
        tx_df.insert(0,  "Image Name", names_all)

        # All labels/metadata (decoded), 1:1 with names_all
        labels_df = pd.DataFrame(rows_meta)

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
