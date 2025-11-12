# === NEW: SimCLR with spatial smoothness + unsupervised cross-view MNN anchors + coord residualization ===
import torch
import torch.nn.functional as F

def _gaussian_weight(dist_sq: torch.Tensor, sigma: float):
    return torch.exp(-dist_sq / (2.0 * (sigma**2)))

@torch.no_grad()
def _orthonormal_basis_from_coords(coords3d: torch.Tensor):
    """
    Build an orthonormal basis Q from coords.
    Basis = [1, x, y, z] (constant + linear terms).
    Returns Q: [B, r] with orthonormal columns in R^B.
    If coords are degenerate (single unique value), falls back gracefully.
    """
    B = coords3d.size(0)
    device = coords3d.device
    # Columns: 1, x, y, z  -> shape [B, 4]
    ones = torch.ones((B, 1), device=device, dtype=coords3d.dtype)
    X = torch.cat([ones, coords3d], dim=1)  # [B, 4]

    # Center columns (except the bias) to improve numerical stability
    X_center = X.clone()
    X_center[:, 1:] = X_center[:, 1:] - X_center[:, 1:].mean(dim=0, keepdim=True)

    # Orthonormalize (QR on Bx4, economic)
    # When B < 4 or rank-deficient, qr still returns fewer valid columns.
    Q, R = torch.linalg.qr(X_center, mode="reduced")
    # Remove near-zero columns (rank deficiency)
    diag = torch.abs(torch.diag(R))
    keep = diag > 1e-8
    if keep.sum() == 0:
        # degenerate case: use only the normalized bias
        q = ones / torch.linalg.norm(ones)
        return q  # [B,1]
    return Q[:, keep]

@torch.no_grad()
def _residualize_against_coords(Z: torch.Tensor, coords3d: torch.Tensor):
    """
    Remove the component of Z explained by coords basis.
    Z: [B, D], coords3d: [B, 3]
    Returns Z_resid with same shape.
    """
    # Build orthonormal basis in sample-space
    Q = _orthonormal_basis_from_coords(coords3d)      # [B, r], Q^T Q = I_r
    # Project Z onto span(Q):  (Q Q^T) Z
    proj = Q @ (Q.T @ Z)                              # [B, D]
    return Z - proj                                   # residuals

@torch.no_grad()
def _mutual_topk_cross_view(Z1: torch.Tensor, Z2: torch.Tensor, k: int = 5, metric: str = "cosine"):
    """
    Cross-view mutual top-k neighbor pairs:
    - Z1: [B, D]  (image)
    - Z2: [B, D]  (transcriptome)
    Returns a boolean mask M of shape [2B, 2B] with True at (i, i+B) if mutual-NN,
    and symmetric positions for both views.
    """
    if metric == "cosine":
        Z1n = F.normalize(Z1, dim=1)
        Z2n = F.normalize(Z2, dim=1)
        S12 = Z1n @ Z2n.T  # [B,B], larger is better
        topk_12 = torch.topk(S12, k=min(int(k), Z2.size(0)), dim=1).indices
        topk_21 = torch.topk(S12.T, k=min(int(k), Z1.size(0)), dim=1).indices
        B = Z1.size(0)
        M = torch.zeros((2*B, 2*B), dtype=torch.bool, device=Z1.device)
        for i in range(B):
            js = topk_12[i].tolist()
            # mutual check
            for j in js:
                if i in topk_21[j].tolist():
                    M[i, j+B] = True
                    M[j+B, i] = True
        return M
    elif metric == "euclidean":
        d = torch.cdist(Z1, Z2, p=2.0)   # [B,B], smaller is better
        neg_d = -d
        topk_12 = torch.topk(neg_d, k=min(int(k), Z2.size(0)), dim=1).indices
        topk_21 = torch.topk(neg_d.T, k=min(int(k), Z1.size(0)), dim=1).indices
        B = Z1.size(0)
        M = torch.zeros((2*B, 2*B), dtype=torch.bool, device=Z1.device)
        for i in range(B):
            js = topk_12[i].tolist()
            for j in js:
                if i in topk_21[j].tolist():
                    M[i, j+B] = True
                    M[j+B, i] = True
        return M
    else:
        raise ValueError("anchor_metric must be 'cosine' or 'euclidean'.")

def simclr_loss_func_spatial_reg(
    z_img: torch.Tensor, z_tx: torch.Tensor,
    coords3d: torch.Tensor,
    temperature: float = 0.1,
    w_pair: float = 1.0,
    w_anchor: float = 0.5,
    w_spatial: float = 0.2,
    sigma: float = 0.8,
    k_anchor: int = 5,
    anchor_metric: str = "cosine",
    residualize_for_anchors: bool = True,
):
    """
    InfoNCE with:
      - strong positives: image↔transcriptome same spot (w_pair)
      - anchor positives: cross-view mutual top-k neighbors on *residualized* features (w_anchor)
      - spatial positives: Gaussian kernel on 3D coords (w_spatial)
    Negatives: everything else. No patient/slice labels used.
    """
    B = z_img.size(0)
    device = z_img.device

    # Normalize for the InfoNCE similarity
    Z = torch.cat([z_img, z_tx], dim=0).float()
    Z = F.normalize(Z, dim=1)
    N = 2 * B

    # Similarity matrix for the denominator
    S = (Z @ Z.T) / temperature
    eye = torch.eye(N, device=device, dtype=torch.bool)

    # Weight matrix
    W = torch.zeros((N, N), device=device, dtype=torch.float32)

    # (1) strong positives (one-to-one pairs)
    idx = torch.arange(B, device=device)
    W[idx, idx + B] = w_pair
    W[idx + B, idx] = w_pair

    # === NEW: residualize wrt coords before mining anchors ===
    if residualize_for_anchors:
        with torch.no_grad():
            z_img_anchor = _residualize_against_coords(z_img.detach(), coords3d.detach())
            z_tx_anchor  = _residualize_against_coords(z_tx.detach(),  coords3d.detach())
    else:
        z_img_anchor, z_tx_anchor = z_img.detach(), z_tx.detach()

    # (2) unsupervised cross-view MNN anchors on residuals
    M = _mutual_topk_cross_view(z_img_anchor, z_tx_anchor, k=int(k_anchor), metric=anchor_metric)
    # don’t double count the strong pair
    pair_mask = torch.zeros_like(W, dtype=torch.bool)
    pair_mask[idx, idx + B] = True
    pair_mask[idx + B, idx] = True
    M = M & (~pair_mask) & (~eye)
    W = W + w_anchor * M.float()

    # (3) spatial positives (smoothness) — keeps your spatial term
    C = torch.cat([coords3d, coords3d], dim=0).float()
    with torch.no_grad():
        d2 = torch.cdist(C, C, p=2.0) ** 2
        G = _gaussian_weight(d2, sigma)
    W = W + w_spatial * G * (~eye).float()

    # InfoNCE
    S = S - eye.float() * 1e9
    expS = torch.exp(S)
    denom = expS.sum(dim=1) + 1e-12
    num = (expS * W).sum(dim=1) + 1e-12

    valid = (W.sum(dim=1) > 0)
    if valid.sum() == 0:
        # fall back to pure SimCLR pairs if no positives were formed
        return (-torch.log(num / denom)).mean()

    loss_i = -torch.log(num[valid] / denom[valid])
    return loss_i.mean()
