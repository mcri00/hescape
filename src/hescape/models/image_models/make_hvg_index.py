#!/usr/bin/env python3
import os, math, argparse, numpy as np, torch
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser("Compute HVG indices (fast, batched, with progress).")
    p.add_argument("--dataset_path", type=str, default="Peng-AI/hescape-pyarrow")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--gene_key", type=str, default="gexp")
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--tops", type=int, nargs="+", default=[100])
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--log1p", action="store_true", default=True)
    p.add_argument("--no-log1p", dest="log1p", action="store_false")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = os.path.join(args.out_root, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load *index* only; we will slice in small chunks (no giant tensor).
    ds = load_dataset(args.dataset_path, name=args.dataset_name, split=args.split)
    N = len(ds)

    # Peek G safely without pulling the column:
    sample = ds[0][args.gene_key]
    G = np.asarray(sample).shape[-1] if isinstance(sample, (list, np.ndarray)) else int(sample.shape[-1])
    print(f"Split='{args.split}': N={N}, G={G}")

    # Running stats (float64 for stability)
    mean = torch.zeros(G, dtype=torch.float64)
    M2   = torch.zeros(G, dtype=torch.float64)
    n_obs = 0

    bs = int(args.batch_size)
    # Batched Welford (parallel across genes)
    # Combine formula for batches:
    # delta = mb_mean - mean
    # mean  = mean + delta * (mb_n / (n_obs + mb_n))
    # M2    = M2 + mb_M2 + delta^2 * n_obs * mb_n / (n_obs + mb_n)
    for i in tqdm(range(0, N, bs), desc="Accumulating mean/variance", total=math.ceil(N/bs)):
        batch = ds[i : min(i+bs, N)][args.gene_key]          # (B, G) as list/np
        xb = torch.as_tensor(np.asarray(batch), dtype=torch.float32)  # CPU tensor [B,G]
        if xb.ndim == 3:  # (B,1,G) → (B,G)
            xb = xb.squeeze(1)

        if args.log1p:
            xb = torch.clamp(xb, min=0).log1p()

        mb_n = xb.shape[0]
        if mb_n == 0:
            continue

        # batch stats
        mb_mean = xb.mean(dim=0, dtype=torch.float64)
        # centered sum of squares for the batch
        mb_M2   = (xb.to(torch.float64) - mb_mean).pow(2).sum(dim=0)

        # combine
        total = n_obs + mb_n
        delta = mb_mean - mean
        mean  = mean + delta * (mb_n / total)
        M2    = M2 + mb_M2 + (delta * delta) * (n_obs * mb_n / total)
        n_obs = total

    if n_obs < 2:
        raise RuntimeError("Not enough samples to compute variance.")

    var = (M2 / (n_obs - 1)).to(torch.float32).cpu().numpy()

    # Save all requested top-Ks
    for K in args.tops:
        if not (0 < K <= G):
            print(f"[warn] skip invalid K={K} (G={G})")
            continue
        idx = np.argpartition(-var, K-1)[:K]
        idx = idx[np.argsort(var[idx])[::-1]]  # sort for determinism
        out_file = os.path.join(out_dir, f"top{K}_idx.npy")
        np.save(out_file, idx.astype(np.int64))
        print(f"✅ Saved HVG indices K={K} → {out_file}")

    print("Done.")

if __name__ == "__main__":
    main()
