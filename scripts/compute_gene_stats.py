# scripts/compute_gene_stats.py
import os, json, argparse, numpy as np
from datasets import load_dataset

PANELS = [
    "human-5k-panel",
    "human-multi-tissue-panel",
    "human-immuno-oncology-panel",
    "human-colon-panel",
    "human-breast-panel",
    "human-lung-healthy-panel",
]

def compute_one(name: str, topk: int, out_dir: str):
    ds = load_dataset("Peng-AI/hescape-pyarrow", name=name, split="train")
    X = np.stack([r["genes"] for r in ds])  # [N, G]
    var = X.var(axis=0)
    topk_idx = np.argsort(var)[-topk:].tolist()
    Xk = X[:, topk_idx]
    mu = Xk.mean(axis=0).tolist()
    sigma = Xk.std(axis=0).tolist()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump({"topk_idx": topk_idx, "mu": mu, "sigma": sigma}, f)
    print(f"[OK] {name} -> {out_dir}/{name}.json  (K={topk})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/stitch_stats")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--name", default=None,
                    help="one dataset name (default: run all panels)")
    args = ap.parse_args()

    if args.name:
        compute_one(args.name, args.topk, args.out_dir)
    else:
        for n in PANELS:
            compute_one(n, args.topk, args.out_dir)
