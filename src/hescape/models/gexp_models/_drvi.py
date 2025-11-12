from pathlib import Path
import anndata as ad
import drvi
import torch


from torch.cuda.amp import autocast
import torch
from contextlib import nullcontext
import torch.nn.functional as F
# ---- autocast-off helper (portable across torch versions) ----

def _autocast_off_ctx():
    if hasattr(torch, "autocast"):  # new API
        return torch.autocast("cuda", enabled=False)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):  # legacy
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()

def _build_drvi_model(path: Path):
    return DRVIModel(path)

'''
def _build_drvi_model(path: str):
    return DRVIModel(path)
    
class DRVIModel(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()
        adata = ad.read_h5ad(path / "drvi_reference.h5ad")
        self.trunk = drvi.model.DRVI.load(path / "drvi", adata=adata).module.z_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_m, q_v, latent = self.trunk(x, cat_full_tensor=None)

        return q_m.squeeze()
'''
class DRVIModel(torch.nn.Module):
    def __init__(self, path: Path):
        super().__init__()
        adata = ad.read_h5ad(path / "drvi_reference.h5ad")
        self.trunk = drvi.model.DRVI.load(path / "drvi", adata=adata).module.z_encoder
        self._printed_once = False

        # --- Scoped patch: sanitize Normal inside DRVI base components only ---
        import torch.distributions as _D
        _ORIG_Normal = _D.Normal

        class _SafeNormal(_ORIG_Normal):
            __doc__ = _ORIG_Normal.__doc__
            __module__ = _ORIG_Normal.__module__

            def __init__(self, loc, scale, *args, **kwargs):
                loc = torch.nan_to_num(loc.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
                scale = torch.nan_to_num(scale.to(torch.float32), nan=1.0, posinf=1e6, neginf=1e-6)
                scale = F.softplus(scale) + 1e-6
                super().__init__(loc, scale, *args, **kwargs)

        # Patch only the symbol DRVI uses
        import drvi.scvi_tools_based.nn._base_components as _drvi_base
        _drvi_base.Normal = _SafeNormal
        print("[patch] DRVI base_components.Normal -> _SafeNormal (scoped)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # counts/log1p inputs must be non-negative; ensure fp32
        x = torch.clamp(x, min=0).to(torch.float32)

        # run DRVI trunk strictly in fp32
        with _autocast_off_ctx():
            q_m, q_v, latent = self.trunk(x, cat_full_tensor=None)

            # one-time debug
            if not self._printed_once:
                self._printed_once = True
                print(
                    f"[DRVI wrapper] q_m dtype={q_m.dtype} device={q_m.device} "
                    f"finite={torch.isfinite(q_m).all().item()}"
                )
                if not torch.isfinite(q_m).all():
                    print(
                        "[DRVI wrapper] q_m stats:",
                        "min", torch.nanmin(q_m).item(), "max", torch.nanmax(q_m).item()
                    )
                if not torch.isfinite(q_v).all():
                    print(
                        "[DRVI wrapper] q_v stats:",
                        "min", torch.nanmin(q_v).item(), "max", torch.nanmax(q_v).item()
                    )

        # drop only the channel dim, keep batch even if B==1
        q_m = q_m.squeeze(1).to(torch.float32)
        return q_m