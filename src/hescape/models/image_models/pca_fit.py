#!/usr/bin/env python3
import os, numpy as np, torch, tqdm
from sklearn.decomposition import IncrementalPCA, PCA
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

# ===== Inception preprocessing =====
inception_tfm = T.Compose([
    T.ToImage(),
    T.Resize(299, antialias=True),
    T.CenterCrop(299),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def build_inception():
    m = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    m.eval()
    return m.cuda() if torch.cuda.is_available() else m

class HFImageOnly(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_key="image", tfm=None):
        self.ds, self.image_key, self.tfm = hf_dataset, image_key, tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img = self.ds[self.image_key][idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        return self.tfm(img) if self.tfm is not None else img

def collate_imgs(batch): return torch.stack(batch, dim=0)

def main():
    # --- config ---
    dataset_path = "Peng-AI/hescape-pyarrow"
    dataset_name = "human-lung-healthy-panel"   # match your run
    split = "train"                              # FIT ON TRAIN ONLY
    image_key = "image"
    out_dir = "/home/meteli/cristina/pretrain_weights/image/pca"
    os.makedirs(out_dir, exist_ok=True)
    comp_path = os.path.join(out_dir, "inception_logits_pca100_components.npy")
    mean_path = os.path.join(out_dir, "inception_logits_pca100_mean.npy")

    # choose mode
    USE_INCREMENTAL = True       # set False to keep your original single-shot PCA
    N_COMPONENTS    = 100
    IPCA_BATCH      = 4096       # how many samples per partial_fit step

    # --- load split ---
    hf = load_dataset(dataset_path, name=dataset_name, split=split)
    ds = HFImageOnly(hf, image_key=image_key, tfm=inception_tfm)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8,
                        pin_memory=True, collate_fn=collate_imgs)

    # --- logits extraction ---
    model = build_inception()
    feats = []  # only used if USE_INCREMENTAL=False
    with torch.no_grad():
        for xb in tqdm.tqdm(loader, desc="Extracting Inception logits"):
            xb = xb.cuda(non_blocking=True) if torch.cuda.is_available() else xb
            z = model(xb)                         # [B, 1000]
            if USE_INCREMENTAL:
                # stream to IncrementalPCA later; store as float32 to CPU now
                feats.append(z.detach().cpu().numpy().astype("float32"))
            else:
                feats.append(z.detach().cpu().numpy())

    if USE_INCREMENTAL:
        # --- IncrementalPCA with visible progress ---
        ipca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=IPCA_BATCH)
        # partial_fit loop (can reuse feats list; memory-light enough and shows progress)
        for chunk in tqdm.tqdm(feats, desc="PCA partial_fit"):
            ipca.partial_fit(chunk)
        W  = ipca.components_.T.astype("float32")  # [1000, 100]
        mu = ipca.mean_.astype("float32")          # [1000]
    else:
        # --- single-shot PCA (can be heavy); we still show a progress bar enter/exit ---
        X = np.concatenate(feats, axis=0).astype("float32")  # [N,1000]
        tqdm.tqdm.write(f"Fitting PCA on X.shape={X.shape} …")
        pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
        pca.fit(X)
        W  = pca.components_.T.astype("float32")
        mu = pca.mean_.astype("float32")

    # --- save ---
    np.save(comp_path, W)
    np.save(mean_path, mu)
    print(f"Saved:\n  components: {comp_path}\n  mean:       {mean_path}")

if __name__ == "__main__":
    main()
