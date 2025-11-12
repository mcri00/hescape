#!/usr/bin/env python3
import os, sys, fnmatch, shutil, hashlib
from pathlib import Path
import argparse, yaml, requests
from tqdm import tqdm
from huggingface_hub import snapshot_download, HfApi

MIN_BYTES = 10 * 1024 * 1024   # 10 MB: rifiuta file più piccoli (placeholder HTML, ecc.)

api = HfApi()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_direct(url: str, out_path: Path, timeout=60):
    """
    Scarica file da URL diretti. Se l'URL è Google Drive, usa gdown.
    Rifiuta file < MIN_BYTES (di solito HTML/placeholder).
    """
    # Google Drive?
    if "drive.google.com" in url or "uc?export=download" in url:
        try:
            import gdown
        except ImportError as e:
            raise RuntimeError("gdown non installato. Esegui: uv add gdown") from e

        tmp = str(out_path) + ".part"
        print(f"   - (gdown) downloading: {url} -> {out_path.name}")
        gdown.download(url, tmp, quiet=False)
        Path(tmp).replace(out_path)
    else:
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"GET {out_path.name}")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024*256):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            pbar.close()
        tmp.replace(out_path)

    # Guardrail: rifiuta file troppo piccoli
    size = out_path.stat().st_size
    if size < MIN_BYTES:
        print(f"   ! downloaded file too small ({size} bytes) — likely a placeholder; removing")
        out_path.unlink(missing_ok=True)
        raise RuntimeError("Small file placeholder; please check URL/ID or permissions")

def copy_selected(src_dir: Path, dst_dir: Path, patterns):
    ensure_dir(dst_dir)
    copied = []
    for root, _, files in os.walk(src_dir):
        for name in files:
            if any(fnmatch.fnmatch(name, pat) for pat in patterns):
                src = Path(root) / name
                dst = dst_dir / name
                if dst.exists():
                    try:
                        if sha256_of(dst) == sha256_of(src):
                            continue
                    except Exception:
                        pass
                shutil.copy2(src, dst)
                copied.append(dst)
    return copied

def pull_hf(repo_id: str, cache_dir: Path) -> Path:
    return Path(snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir), local_files_only=False))

def process_block(root_dir: Path, family: str, name: str, spec: dict, cache_dir: Path):
    target_dir = ensure_dir(root_dir / family / name)
    hf_repo = spec.get("hf_repo")
    include = spec.get("include") or []
    direct_urls = spec.get("direct_urls") or []

    print(f"\n==> {family}/{name}")
    # 1) Hugging Face
    if hf_repo:
        try:
            # pre-check access (evita messaggi criptici)
            _ = api.model_info(hf_repo)
            print(f"   - pulling HF repo: {hf_repo}")
            local = pull_hf(hf_repo, cache_dir)
            copied = copy_selected(local, target_dir, include if include else ["*"])
            if copied:
                print(f"   - copied {len(copied)} files to {target_dir}")
            else:
                print(f"   - no files matched patterns {include}, check 'include' or repo contents")
        except Exception as e:
            print(f"   ! HF access or pull failed for {hf_repo}: {e}")
            print(f"     (Tip: request/accept access on the model page if gated.)")

    # 2) Direct URLs (supporta stringa o dict {url, filename})
    for item in direct_urls:
        if isinstance(item, str):
            url = item
            fname = url.split("/")[-1].split("?")[0] or "downloaded_file"
        else:
            # dict come: {url: "...", filename: "ctranspath_base.pth"}
            url = item.get("url")
            fname = item.get("filename") or (url.split("/")[-1].split("?")[0] if url else "downloaded_file")
        if not url:
            print("   ! direct_url item without 'url' — skipping")
            continue
        outp = target_dir / fname
        if outp.exists():
            print(f"   - exists, skip: {outp.name}")
            continue
        try:
            print(f"   - downloading: {url}")
            download_direct(url, outp)
        except Exception as e:
            print(f"   ! direct download failed: {url} -> {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tools/weights_map.yaml")
    ap.add_argument("--cache", default="~/.cache/hf_weights")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root_dir = ensure_dir(Path(cfg.get("root_dir","pretrain_weights")))
    cache_dir = ensure_dir(Path(os.path.expanduser(args.cache)))

    for family in ["image","gene"]:
        block = (cfg.get(family) or {})
        for name, spec in block.items():
            process_block(root_dir, family, name, spec, cache_dir)

    print("\nAll done ✅")

if __name__ == "__main__":
    main()
