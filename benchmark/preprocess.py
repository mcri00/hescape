# preprocess.py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm
from sklearn.decomposition import PCA

class HescapeTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]

        # --- immagine ---
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img)).convert("RGB")

        # --- gexp: normalizza a vettore 1-D ---
        g = np.asarray(row["gexp"], dtype=np.float32)
        # squeeze una dimensione superflua come (1, G) -> (G,)
        g = np.squeeze(g)
        if g.ndim != 1:
            raise ValueError(f"gexp shape inattesa: {g.shape}")

        # --- coords: assicurati che sia (2,) ---
        coords = np.asarray(row["cell_coords"], dtype=np.float32)
        coords = np.squeeze(coords)
        if coords.ndim != 1 or coords.shape[-1] != 2:
            raise ValueError(f"coords shape inattesa: {coords.shape}")

        return {
            "name": row["name"],
            "image": img,
            "gexp": g,
            "coords": coords,                # [x, y]
            "atlas": row.get("atlas", None), # slice-id
            "source": row.get("source", None),  # patient-id se presente
            "gt_label": row.get("tumor_tissue_type", None),
        }


def make_img_transform(img_size, mean, std):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def make_inception_extractor(device="cuda"):
    import torchvision.models as tvm
    # Alcune versioni richiedono aux_logits=True quando si passano i pesi
    try:
        m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.DEFAULT, aux_logits=True)
    except TypeError:
        # fallback per versioni più vecchie di torchvision
        m = tvm.inception_v3(pretrained=True, aux_logits=True)
    m.fc = torch.nn.Identity()
    m.eval().to(device)
    return m  # output: vettori 2048-d dopo avgpool


def make_densenet_extractor(device="cuda"):
    # torchvision >=0.13
    try:
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
    except AttributeError:  # older torchvision
        m = tvm.densenet121(pretrained=True)

    # replace classifier with Identity so forward returns pooled features
    m.classifier = torch.nn.Identity()
    m.eval().to(device)
    return m  # output is (B, 1024) after global avg pool

def make_resnet_extractor(device="cuda"):
    """
    Crea un estrattore ResNet-18 pre-addestrato su ImageNet.
    Rimuove il classificatore finale per ottenere i vettori di feature (512-d).
    Output shape: (B, 512)
    """
    try:
        # torchvision >= 0.13
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    except AttributeError:
        # compatibilità per versioni precedenti
        m = tvm.resnet18(pretrained=True)

    # sostituisci il classificatore con Identity per ottenere le feature
    m.fc = torch.nn.Identity()
    m.eval().to(device)
    return m  # output: (B, 512) dopo global avg pool


@torch.no_grad()
def encode_images(extractor, img_batch: torch.Tensor, device="cuda"):
    feats = extractor(img_batch.to(device))
    if isinstance(feats, tuple): feats = feats[0]
    return feats.detach()

def select_hvg_indices(X: np.ndarray, k: int) -> np.ndarray:
    var = X.var(axis=0)
    idx = np.argsort(var)[::-1][:k]
    return np.sort(idx)



@torch.no_grad()
def collate_batch(batch, img_tf, extractor, cfg, hvg_idx, pca_model=None, device="cuda"):
    # --- helper: immagine in PIL ---
    def _to_pil(x):
        if isinstance(x, Image.Image):
            return x
        return Image.fromarray(np.array(x)).convert("RGB")

    # --- IMMAGINI ---
    imgs = torch.stack([img_tf(_to_pil(b["image"])) for b in batch], dim=0)  # (B,3,H,W)
    if extractor is not None:
        img_feats = extractor(imgs.to(device))
        if isinstance(img_feats, tuple):  # inception può restituire (main, aux)
            img_feats = img_feats[0]
        # se rimane 4D (per qualche modello), fai GAP
        if img_feats.ndim > 2:
            img_feats = torch.nn.functional.adaptive_avg_pool2d(img_feats, (1, 1)).view(img_feats.size(0), -1)
        img_feats = img_feats.detach().cpu()
    else:
        # niente extractor -> userai direttamente i pixel (evitabile, ma ok)
        img_feats = imgs.view(imgs.size(0), -1).cpu()

    # --- PCA opzionale: SOLO transform (il fit lo fai fuori) ---
    # PCA opzionale (solo se definita)
    if cfg.get("image_pca_dim", None):
        X = img_feats.detach().cpu().numpy()
        if pca_model is None:
            raise RuntimeError(
                "cfg.image_pca_dim è impostato, ma pca_model è None. "
                "Devi fit- tare la PCA (warm-up) prima dell'inferenza."
            )
        Xp = pca_model.transform(X)
        img_feats = torch.tensor(Xp, dtype=torch.float32)


    # --- GENE EXP ---
    g_list = []
    for b in batch:
        g = np.asarray(b["gexp"], dtype=np.float32)
        g = np.squeeze(g)                              # (G,)
        if g.ndim != 1:
            raise ValueError(f"gexp shape inattesa in batch: {g.shape}")
        g_list.append(g)

    G = np.stack(g_list, axis=0)                       # (B, G_tot)
    if G.ndim != 2:
        raise ValueError(f"G deve essere 2D, invece è {G.shape}")

    G = G[:, hvg_idx]                                  # (B, hvg_k) — HVG applicato UNA SOLA VOLTA
    if cfg.get("gene_log1p", True):
        G = np.log1p(G)

    g_tensor = torch.tensor(G, dtype=torch.float32)


    # --- METADATI ---
    def _get_coords(b):
        c = b.get("coords", None)
        if c is None:
            c = b.get("cell_coords", None)  # <- righe grezze HF
        if c is None:
            raise KeyError("coords/cell_coords mancante in una riga del batch")
        c = np.asarray(c, dtype=np.float32).squeeze()
        if c.ndim != 1 or c.shape[0] < 2:
            raise ValueError(f"coords shape inattesa: {c.shape}")
        return c[:2]
    
    def _get_gt(b):
        # non-streaming: il dataset wrapper espone "gt_label"
        # streaming grezzo: il record HF ha "tumor_tissue_type"
        return b.get("gt_label", b.get("tumor_tissue_type", None))

    names   = [str(b.get("name", "")) for b in batch]
    coords  = np.stack([_get_coords(b) for b in batch], axis=0).astype(np.float32)
    atlas   = [b.get("atlas", None) for b in batch]    # slice-id
    sources = [b.get("source", None) for b in batch]   # patient-id (se presente)
    gt      = [_get_gt(b) for b in batch]

    # --- sanity ---
    assert img_feats.shape[0] == g_tensor.shape[0] == len(names), "Batch misallineato."

    return img_feats, g_tensor, names, coords, atlas, sources, gt, pca_model
