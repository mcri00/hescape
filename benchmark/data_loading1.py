#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json, argparse, hashlib, time
from pathlib import Path
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from datasets import load_dataset, ClassLabel

from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

# ========== Config helpers ==========

def _load_cfg_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f) or {}
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix} (use .yaml/.yml or .json)")

def _merge_cfg_with_cli(cfg: dict, args_cli: argparse.Namespace) -> argparse.Namespace:
    merged = dict(cfg or {})
    for k, v in vars(args_cli).items():
        if k in ("cfg",):
            continue
        if v is not None and v is not False:
            merged[k] = v
        elif k not in merged:
            merged[k] = v
    return argparse.Namespace(**merged)

# ========== Utils ==========

def sha10(xs):
    s = "\n".join(sorted(map(str, xs)))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def load_id_whitelist(csv_path: str, col: str = "id") -> set[str]:
    wl = set()
    with open(csv_path, "r", newline="") as f:
        sample = f.read(2048); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            reader  = csv.DictReader(f, dialect=dialect)
        except Exception:
            reader  = csv.DictReader(f)
        if col not in (reader.fieldnames or []):
            raise ValueError(f"Column '{col}' not found in {csv_path}. Columns={reader.fieldnames}")
        for row in reader:
            v = row.get(col)
            if v is not None:
                wl.add(str(v).strip())
    if not wl:
        raise ValueError(f"No IDs loaded from {csv_path}")
    return wl

def detect_id_field(features, prefer=None):
    if prefer:
        return prefer
    for c in ("id", "dataset_id", "name"):
        if c in features:
            return c
    return None

def decode_classlabel(features, fname, val):
    feat = features.get(fname)
    if isinstance(feat, ClassLabel):
        try:
            return feat.int2str(int(val))
        except Exception:
            return str(val)
    return "" if val is None else val

def get_sample_id(row, id_field, features):
    if id_field and (id_field in row):
        v = row[id_field]
        feat = features.get(id_field)
        if isinstance(feat, ClassLabel):
            try:
                return feat.int2str(int(v))
            except Exception:
                return str(v)
        return "" if v is None else str(v)
    for k in ("id", "dataset_id", "name"):
        if k in row and row[k] is not None:
            return str(row[k])
    return ""

def to_pil(img_obj):
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, dict) and "bytes" in img_obj:
        return Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
    return Image.fromarray(np.array(img_obj)).convert("RGB")

def unique_slice_key(rec, features):
    atlas = rec.get("atlas", "") or ""
    namec = decode_classlabel(features, "name", rec.get("name", ""))
    return atlas if atlas not in (None, "", "NA") else namec

def get_decoded(features, rec, key):
    if key not in rec:
        return ""
    return decode_classlabel(features, key, rec.get(key, ""))

# -------- XY detection & reading --------
_CANDIDATE_XY_KEYS = [
    ("x", "y"), ("spot_x", "spot_y"), ("x_coord", "y_coord"),
    ("array_col", "array_row"), ("col", "row"),
    ("px", "py"), ("x_px", "y_px"), ("x_um", "y_um"),
]

def detect_xy_keys(features):
    keys = set(features.keys())
    for vec in ("cell_coords", "coords", "xy", "spot_xy", "position"):
        if vec in keys:
            return (vec, None)
    for xk, yk in _CANDIDATE_XY_KEYS:
        if xk in keys and yk in keys:
            return (xk, yk)
    raise ValueError(f"Could not detect XY fields. Keys={sorted(list(keys))}")

def _to_xy_tuple(obj):
    arr = np.asarray(obj, dtype=np.float32).squeeze()
    if arr.ndim != 1 or arr.shape[0] < 2:
        raise ValueError(f"XY vector must be length>=2, got {arr.shape}")
    return float(arr[0]), float(arr[1])

def read_xy(rec, x_key, y_key):
    if y_key is None:
        return _to_xy_tuple(rec[x_key])
    else:
        return float(rec[x_key]), float(rec[y_key])

# ========== Inception-only backbone (logits, coherent with your training) ==========

def build_inception_logits(pretrained: bool = True, device="cpu"):
    # Keep classifier head -> 1000-D logits
    weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    model = inception_v3(weights=weights, aux_logits=True)
    model.AuxLogits = None  # disable aux head, keep main classifier
    model.eval().to(device)

    # EXACT transforms used in your feature extractor:
    tf = transforms.Compose([
        transforms.Resize((229, 229)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return model, tf

@torch.no_grad()
def encode_batch(model, tf, pil_list, device="cpu", bs=64):
    out = []
    for i in range(0, len(pil_list), bs):
        chunk = pil_list[i:i+bs]
        tens = torch.stack([tf(im) for im in chunk], dim=0).to(device, non_blocking=True)
        feats = model(tens)
        if hasattr(feats, "logits"):
            feats = feats.logits
        elif isinstance(feats, (list, tuple)):
            feats = feats[0]
        out.append(feats.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0,))

# ========== Core runner ==========

LABEL_KEYS = [
    "name_category", "atlas", "source", "x", "y",
    "tissue", "diagnosis", "cancer", "oncotree_code",
    "tumor_tissue_type", "tumor_grade", "age", "gender", "race",
    "treatment_type", "therapeutic_agents", "assay",
    "preservation_method", "stain", "spaceranger", "species", "cytassist"
]

def run_one_panel(args: argparse.Namespace):
    # defaults
    if args.device in (None, ""):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.batch_size is None:
        args.batch_size = 64
    if args.pca_dim is None:
        args.pca_dim = 100

    wl = load_id_whitelist(args.test_ids_csv, col="id")
    wl_hash = sha10(wl)

    out_dir = Path(args.out_root) / args.panel / f"test_subset__{wl_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] panel={args.panel} split={args.split} | whitelist={len(wl)} | out={out_dir}")
    print(f"[cfg] device={args.device} pretrained={not args.no_pretrained} pca_dim={args.pca_dim}")

    ds_stream = load_dataset("Peng-AI/hescape-pyarrow", name=args.panel, split=args.split, streaming=True)
    features  = ds_stream.features
    id_field  = detect_id_field(features, getattr(args, "id_field", None))
    print(f"[detect] id_field={id_field}")

    x_key, y_key = detect_xy_keys(features)
    print(f"[detect] coord keys: x_key={x_key}, y_key={y_key}")

    kept_meta = []
    imgs_pil  = []

    counts = Counter()
    STOP_AFTER_MISSES = int(getattr(args, "stop_after_misses", 1000))
    MIN_ROWS_PER_ID   = int(getattr(args, "min_rows_per_id", 0))
    MAX_ROWS_PER_ID   = int(getattr(args, "max_rows_per_id", 0))

    per_id_counts = Counter()
    remaining = set(wl)
    miss_streak = 0
    all_seen_once = False

    t0 = time.time()
    hits = 0

    pbar = tqdm(ds_stream, desc="[stream] scanning", unit="rows")
    for r in pbar:
        sid = get_sample_id(r, id_field, features)

        if sid not in wl:
            if all_seen_once:
                miss_streak += 1
                if miss_streak >= STOP_AFTER_MISSES:
                    print(f"[stream] early-stop after {STOP_AFTER_MISSES} consecutive non-whitelist rows.")
                    break
            # progress
            elapsed = max(1e-6, time.time() - t0)
            hit_rate = hits / elapsed
            need_more = 0 if MIN_ROWS_PER_ID == 0 else sum(
                max(0, MIN_ROWS_PER_ID - per_id_counts[_id]) for _id in wl
            )
            eta = (need_more / hit_rate) if hit_rate > 0 else float("inf")
            pbar.set_postfix({
                "hits": hits,
                "ids_ok": sum(int(per_id_counts[_id] >= max(1, MIN_ROWS_PER_ID)) for _id in wl) if MIN_ROWS_PER_ID > 0 else len(wl) - len(remaining),
                "miss_streak": miss_streak,
                "hit_rate/s": f"{hit_rate:.2f}",
                "ETA(s)": "∞" if eta == float("inf") else f"{eta:.0f}"
            })
            continue

        # whitelist hit
        miss_streak = 0

        if MAX_ROWS_PER_ID > 0 and per_id_counts[sid] >= MAX_ROWS_PER_ID:
            pass
        else:
            per_id_counts[sid] += 1
            counts[sid] += 1
            hits += 1

            if hits % 2000 == 0:
                print(f"[collect] {hits} samples collected so far...")

            x_val, y_val = read_xy(r, x_key, y_key)
            slice_key = unique_slice_key(r, features)
            unique_id = f"{slice_key}_{int(x_val)}x{int(y_val)}"

            row_meta = {
                "Image Name": unique_id,
                "name_category": decode_classlabel(features, "name", r.get("name", "")),
                "atlas": r.get("atlas", "") or "",
                "source": get_decoded(features, r, "source"),
                "x": x_val, "y": y_val,
                "tissue": get_decoded(features, r, "tissue"),
                "diagnosis": r.get("diagnosis", ""),
                "cancer": r.get("cancer", ""),
                "oncotree_code": r.get("oncotree_code", ""),
                "tumor_tissue_type": r.get("tumor_tissue_type", ""),
                "tumor_grade": r.get("tumor_grade", ""),
                "age": r.get("age", ""),
                "gender": r.get("gender", ""),
                "race": r.get("race", ""),
                "treatment_type": r.get("treatment_type", ""),
                "therapeutic_agents": r.get("therapeutic_agents", ""),
                "assay": r.get("assay", ""),
                "preservation_method": r.get("preservation_method", ""),
                "stain": r.get("stain", ""),
                "spaceranger": r.get("spaceranger", ""),
                "species": r.get("species", ""),
                "cytassist": r.get("cytassist", ""),
            }
            kept_meta.append(row_meta)

            pil = to_pil(r["image"]); imgs_pil.append(pil)

        if not all_seen_once and sid in remaining:
            remaining.remove(sid)
            if not remaining:
                all_seen_once = True
                print(f"[stream] all whitelist IDs seen at least once.")

        elapsed = max(1e-6, time.time() - t0)
        hit_rate = hits / elapsed
        need_more = 0 if MIN_ROWS_PER_ID == 0 else sum(
            max(0, MIN_ROWS_PER_ID - per_id_counts[_id]) for _id in wl
        )
        eta = (need_more / hit_rate) if hit_rate > 0 else float("inf")
        pbar.set_postfix({
            "hits": hits,
            "ids_ok": sum(int(per_id_counts[_id] >= max(1, MIN_ROWS_PER_ID)) for _id in wl) if MIN_ROWS_PER_ID > 0 else len(wl) - len(remaining),
            "miss_streak": miss_streak,
            "hit_rate/s": f"{hit_rate:.2f}",
            "ETA(s)": "∞" if eta == float("inf") else f"{eta:.0f}"
        })

    n_rows = len(kept_meta)
    if n_rows == 0:
        raise RuntimeError("No rows collected from whitelist IDs. Check id_field/CSV.")
    print(f"[collected] rows={n_rows}, distinct IDs={len(counts)}; top counts: {counts.most_common(5)}")

    # 2) IMAGE FEATURES: Inception-v3 logits only → PCA → export
    device = args.device
    use_pretrained = not bool(args.no_pretrained)

    print(f"[image] backbone=inception_v3 (logits)")
    model, tf = build_inception_logits(pretrained=use_pretrained, device=device)

    raw_feats = encode_batch(model, tf, imgs_pil, device=device, bs=int(args.batch_size))
    if raw_feats.ndim != 2:
        raise RuntimeError(f"inception_v3 output is not 2D: {raw_feats.shape}")
    print(f"[image] feats shape {raw_feats.shape} → PCA({int(args.pca_dim)})")

    if raw_feats.shape[0] < int(args.pca_dim):
        raise ValueError(f"PCA dim {args.pca_dim} > N samples {raw_feats.shape[0]}")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=int(args.pca_dim), svd_solver="auto", random_state=42).fit(raw_feats)
    Xr = pca.transform(raw_feats)

    df = pd.DataFrame(Xr, columns=[f"img_inception_pca_e{i}" for i in range(int(args.pca_dim))])
    df.insert(0, "Image Name", [m["Image Name"] for m in kept_meta])
    csv_path = out_dir / f"img_inception_pca_{int(args.pca_dim)}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[image] saved: {csv_path}")

    try:
        import joblib
        joblib.dump(pca, out_dir / f"img_inception_pca_{int(args.pca_dim)}.joblib")
    except Exception:
        pass

    # 3) meta + manifest
    cols = ["Image Name"] + LABEL_KEYS
    meta_df = pd.DataFrame(kept_meta)[cols]
    meta_df.to_csv(out_dir / "labels_meta.csv", index=False)

    manifest = {
        "panel": args.panel,
        "split": args.split,
        "whitelist_csv": os.path.abspath(args.test_ids_csv),
        "whitelist_hash": sha10(load_id_whitelist(args.test_ids_csv, "id")),
        "rows": n_rows,
        "pca_dim": int(args.pca_dim),
        "id_field": detect_id_field(features, getattr(args, "id_field", None)),
        "backbone": "inception_v3_logits_only",
        "device": args.device,
        "pretrained": not bool(args.no_pretrained),
        "stop_after_misses": int(getattr(args, "stop_after_misses", 1000)),
        "min_rows_per_id": int(getattr(args, "min_rows_per_id", 0)),
        "max_rows_per_id": int(getattr(args, "max_rows_per_id", 0)),
        "coord_keys": {"x_key": x_key, "y_key": y_key},
        "transforms_inception": {
            "resize": [229, 229],
            "centercrop": 299,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std":  [0.229, 0.224, 0.225],
        }
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n[done]")
    print(f"Outputs in: {out_dir}")
    print(" - img_inception_pca_{pca_dim}.csv (+ joblib)")
    print(" - labels_meta.csv, manifest.json")

# ========== CLI entrypoint ==========

def main():
    ap = argparse.ArgumentParser(
        description="Build test-subset features (Inception-v3 logits + PCA only) from HESCAPE panel using a whitelist of IDs."
    )
    ap.add_argument("--cfg", help="YAML/JSON config file with all arguments")

    # minimal args (no HVG)
    ap.add_argument("--panel", help="HESCAPE panel, e.g. human-5k-panel")
    ap.add_argument("--split", default="train", help="HF split to stream from (default: train)")
    ap.add_argument("--test_ids_csv", help="CSV with a column named 'id'")
    ap.add_argument("--out_root", help="Directory where to save outputs")
    ap.add_argument("--id_field", default=None, help="Force ID field: id|dataset_id|name (auto if not set)")
    ap.add_argument("--pca_dim", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_pretrained", action="store_true", help="Do not load ImageNet pretrained weights")

    # early-stop knobs
    ap.add_argument("--stop_after_misses", type=int, default=1000)
    ap.add_argument("--min_rows_per_id", type=int, default=0)
    ap.add_argument("--max_rows_per_id", type=int, default=0)

    args_cli = ap.parse_args()

    if args_cli.cfg:
        cfg_data = _load_cfg_file(args_cli.cfg)

        if isinstance(cfg_data, dict) and "panels" in cfg_data and isinstance(cfg_data["panels"], list):
            base = dict(cfg_data)
            panels = base.pop("panels")
            for i, sub in enumerate(panels, 1):
                merged_ns = _merge_cfg_with_cli({**base, **(sub or {})}, args_cli)
                for req in ("panel", "test_ids_csv", "out_root"):
                    if not getattr(merged_ns, req, None):
                        raise ValueError(f"Missing required '{req}' for panel #{i} in config.")
                print(f"\n=== Running panel {i}/{len(panels)}: {merged_ns.panel} ===")
                run_one_panel(merged_ns)
            return

        merged_ns = _merge_cfg_with_cli(cfg_data, args_cli)
        for req in ("panel", "test_ids_csv", "out_root"):
            if not getattr(merged_ns, req, None):
                raise ValueError(f"Missing required '{req}' in config/CLI.")
        run_one_panel(merged_ns)
        return

    for req in ("panel", "test_ids_csv", "out_root"):
        if not getattr(args_cli, req, None):
            raise ValueError(f"Missing required CLI arg: --{req}")
    run_one_panel(args_cli)

if __name__ == "__main__":
    main()

