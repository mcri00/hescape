# coords.py
import numpy as np
import pandas as pd
import torch

def coords3d_from_xy_atlas(xy: np.ndarray, atlas_tokens, device="cpu"):
    """
    Build 3D coords [x_norm, y_norm, z_norm] where z is the factorized 'atlas'.
    """
    x = xy[:, 0]; y = xy[:, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y = (y - y.min()) / (y.max() - y.min() + 1e-9)
    s = pd.Series([a if a is not None else "NA" for a in atlas_tokens])
    z_idx, _ = pd.factorize(s, sort=True)
    z = z_idx.astype(np.float32)
    if z.max() > 0: z = z / z.max()
    coords3d = np.stack([x, y, z], axis=1).astype(np.float32)
    return torch.tensor(coords3d, dtype=torch.float32, device=device)

'''

def coords3d_from_xy_groups(
    xy,                    # torch.Tensor [B,2] o np.ndarray [B,2]
    slice_tokens,          # list/array di id slice (es. HESCAPE 'atlas')
    patient_tokens=None,   # list/array di id paziente (es. HESCAPE 'source'); se None -> tutto un paziente
    device="cuda",
    dtype=torch.float32,
):
    # to torch
    if isinstance(xy, np.ndarray):
        xy = torch.tensor(xy, dtype=dtype, device=device)
    else:
        xy = xy.to(device=device, dtype=dtype)

    B = xy.shape[0]
    if patient_tokens is None:
        patient_tokens = ["_one_patient_"] * B

    df = pd.DataFrame({
        "slice":   np.asarray(slice_tokens,   dtype=object),
        "patient": np.asarray(patient_tokens, dtype=object),
    })

    # --- (x,y) normalizzate per-slice (zero-mean, unit-std) ---
    xy_np = xy.detach().cpu().numpy().astype(np.float32)
    for s in df["slice"].unique():
        idx = (df["slice"] == s).values
        mu = xy_np[idx].mean(axis=0)
        sd = xy_np[idx].std(axis=0) + 1e-6
        xy_np[idx] = (xy_np[idx] - mu) / sd
    xy_norm = torch.from_numpy(xy_np).to(device=device, dtype=dtype)

    # --- z relativo per-paziente: ordinal dei slice dentro al paziente, 0..1 poi zero-mean ---
    z_rel_np = np.zeros(B, dtype=np.float32)
    for p in df["patient"].unique():
        m = (df["patient"] == p).values
        slices_p = sorted(df.loc[m, "slice"].unique().tolist())
        s2i = {s: i for i, s in enumerate(slices_p)}
        z_vals = df.loc[m, "slice"].map(s2i).to_numpy(dtype=float)
        if len(slices_p) > 1:
            z_vals = z_vals / (len(slices_p) - 1)
        z_vals = z_vals - z_vals.mean()
        z_rel_np[m] = z_vals.astype(np.float32)

    z_rel = torch.from_numpy(z_rel_np).to(device=device, dtype=dtype).unsqueeze(1)  # [B,1]
    return torch.cat([xy_norm, z_rel], dim=1)  # [B,3]
'''


def coords3d_from_xy_groups(
    xy_coords,                 # torch.Tensor [B,2] or np.ndarray [B,2]
    slice_ids_str,             # list[str] len B (e.g. HESCAPE 'atlas' or decoded 'name')
    device="cuda",
    dtype=torch.float32,
):
    """
    Build (x,y,z) coordinates without patient IDs.

    - (x,y): normalized per slice (zero-mean, unit-std)
    - z: always 0 (fully patient-agnostic)

    Returns
    -------
    torch.Tensor [B,3] on `device` with dtype `dtype`.
    """

    # --- ensure torch tensor ---
    if isinstance(xy_coords, np.ndarray):
        xy = torch.tensor(xy_coords, dtype=dtype, device=device)
    else:
        xy = xy_coords.to(device=device, dtype=dtype)

    B = xy.shape[0]
    df = pd.DataFrame({"slice": np.asarray(slice_ids_str, dtype=object)})

    # --- normalize (x,y) within each slice ---
    xy_np = xy.detach().cpu().numpy().astype(np.float32)
    for s in df["slice"].unique():
        idx = (df["slice"] == s).values
        mu = xy_np[idx].mean(axis=0)
        sd = xy_np[idx].std(axis=0) + 1e-6
        xy_np[idx] = (xy_np[idx] - mu) / sd

    xy_norm = torch.from_numpy(xy_np).to(device=device, dtype=dtype)

    # --- z = 0 for all spots ---
    z_rel = torch.zeros((B, 1), device=device, dtype=dtype)

    return torch.cat([xy_norm, z_rel], dim=1)  # [B,3]
