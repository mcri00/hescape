# takes all different image encoders in a single class function
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from timm.layers import Mlp

from hescape.models._utils import print_trainable_parameters
from hescape.models.image_models._conch import _build_conch_model
from hescape.models.image_models._ctranspath import _build_ctranspath_model
from hescape.models.image_models._h0_mini import _build_h0_mini_model
from hescape.models.image_models._utils import freeze_batch_norm_2d

from hescape.models.image_models._pca import PCALinear

class ImageEncoder(nn.Module):
    """ImageEncoder that wraps timm models."""

    def __init__(
        self,
        model_name: Literal["ctranspath", "densenet", "uni", "optimus", "conch", "gigapath", "h0-mini", "inception", "inception_pca100"] | str,
        finetune: bool = False,
        embed_dim: int = -1,
        proj: str = "mlp",
        **kwargs: Any,
    ):
        super().__init__()
        self.model_name = model_name

        # Build trunk model
        checkpoint_root = Path(kwargs.get("checkpoint_path", ""))
        self.trunk, self.total_blocks = self._build_trunk(self.model_name, checkpoint_root, **kwargs)

        if not finetune:  # i.e if finetune is false, we freeze the trunk
            self.freeze()

        self.trunk = self.get_ft_model(model_name, self.trunk, lora=finetune)

        # Build projection head
        self.proj = proj
        self.head = self._build_head(proj, self.trunk.num_features, embed_dim)

        # return hook

    def _build_trunk(self, model_name: str, checkpoint_root: Path, **kwargs: Any) -> tuple[nn.Module, int]:
        """
        Build the trunk (backbone) model for image encoding.
        Returns (trunk_module, total_blocks).
        """
        if model_name == "densenet":
            trunk = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
            print(f"Successfully loaded weights for {model_name}")
            total_blocks = 4  # Fine-tune up to 2

        elif model_name == "ctranspath":
            trunk = _build_ctranspath_model()
            checkpoint_path = checkpoint_root / model_name / "ctranspath.pth"
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
            print(f"Successfully loaded weights for {model_name}")

            total_blocks = 4  # Fine-tune up to 2

        elif model_name == "uni":
            trunk = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
            )
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
            print(f"Successfully loaded weights for {model_name}")

            total_blocks = 24  # Fine-tune up to 8

        elif model_name == "optimus":
            trunk = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0", pretrained=False, init_values=1e-5, dynamic_img_size=False
            )
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
            print(f"Successfully loaded weights for {model_name}")

            total_blocks = 40  # Fine-tune up to 12

        elif model_name == "h0-mini":  # to be refined
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            model = _build_h0_mini_model(str(checkpoint_path))
            print(f"Successfully loaded weights for {model_name}")

            trunk = model.trunk

            total_blocks = 12  # Fine-tune up to 12

        elif model_name == "conch":
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            model = _build_conch_model(str(checkpoint_path))
            print(f"Successfully loaded weights for {model_name}")

            trunk = model.visual.trunk

            total_blocks = 12

        elif model_name == "gigapath":
            checkpoint_path = checkpoint_root / model_name / "pytorch_model.bin"
            trunk = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
            trunk.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
            print(f"Successfully loaded weights for {model_name}")

            # total_blocks may differ, set it according to your needs
            total_blocks = 12  # Example
        # === NEW: your hub Inception-v3 with 1000-class logits ===
        elif model_name == "inception":
            # mirror your script: keep classifier head (1000 logits)
            trunk = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            trunk.eval()
            # expose feature dim to the head builder
            trunk.num_features = 1000
            total_blocks = 11
            return trunk, total_blocks

        # === NEW: same as above, but compress logits with PCA→100 before head ===
        elif model_name == "inception_pca100":
            base = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            base.eval()
            pca = PCALinear(
                components_path=kwargs["pca_components_path"],
                mean_path=kwargs["pca_mean_path"],
            )
            class InceptionLogitsWithPCA(nn.Module):
                def __init__(self, b, p):
                    super().__init__()
                    self.base = b
                    self.pca  = p
                    self.num_features = 100
                def forward(self, x):
                    z = self.base(x)   # [B, 1000] logits
                    return self.pca(z) # [B, 100]
            trunk = InceptionLogitsWithPCA(base, pca)
            total_blocks = 11
            return trunk, total_blocks


        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return trunk, total_blocks

    def get_ft_model(self, model_name: str, trunk, lora: bool = False) -> object:
        """
        Returns a fine-tuned model based on the given model name and trunk.

        Args:
            model_name (str): The name of the model.
            trunk: The trunk model.
            lora (bool, optional): Whether to use LoRA. Defaults to False.

        Returns
        -------
            object: The fine-tuned model.
        """
        # Define LoRA configurations for each model
        lora_configs = {
            "ctranspath": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "attn.proj"]},
            "uni": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "conch": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "optimus": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "h0-mini": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
            "gigapath": {"r": 8, "lora_alpha": 16, "target_modules": ["qkv", "proj"]},
        }

        if lora:
            # Get the LoRA configuration for the given model
            if model_name in lora_configs.keys():
                config = lora_configs.get(model_name)
                if config:
                    # Create a LoRA configuration object
                    lora_config = LoraConfig(
                        r=config["r"],
                        lora_alpha=config["lora_alpha"],
                        target_modules=config["target_modules"],
                        lora_dropout=0.1,
                        bias="none",
                    )
                    # Return the fine-tuned model with LoRA
                    return get_peft_model(trunk, lora_config)
                else:
                    # Handle unknown model names
                    raise ValueError(f"Unknown model name: {model_name}")

        # If LoRA is not enabled, return the original trunk
        return trunk  # it simply returns the trunk for densenet

    def _build_head(self, proj: str, in_features: int, embed_dim: int) -> nn.Sequential:
        """Build a projection head (Linear or MLP)."""
        head_layers = OrderedDict()
        if proj == "linear":
            head_layers["linear"] = nn.Linear(in_features, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(in_features, 2 * embed_dim, embed_dim, drop=0.2, norm_layer=nn.LayerNorm)
        elif proj == "transformer":
            # Add transformer specific layers here (From torch maybe)
            encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=8, dim_feedforward=embed_dim)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            head_layers["transformer"] = transformer_encoder
            head_layers["linear"] = nn.Linear(in_features, embed_dim)
            # TBD
        elif proj == "identity":
            head_layers["identity"] = nn.Identity()
        else:
            raise ValueError(f"Unknown projection type: {proj}")

        return nn.Sequential(head_layers)

    def freeze(self, freeze_bn_stats=True):
        """Freeze model params."""
        for param in self.trunk.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self.trunk)
    '''
    #original version
    def forward(self, x):
        """Forward pass."""
        features = {}

        if self.proj in ["mlp", "linear"]:
            x = self.trunk(x)
            if self.model_name in ["conch", "h0-mini"]:
                x = x[:, 0, :]

            x = self.head(x)

        elif self.proj == "transformer":
            tokens = self.trunk.forward_features(x)
            x = self.head(tokens)
            x = x[:, 0, :]

        return x.contiguous()  # Ensure contiguous memory layout
    '''
    # hescape/models/image_models/__init__.py (your ImageEncoder)
    def forward(self, x):
        """Forward pass."""
        # 1) ALWAYS extract features with the trunk
        x = self.trunk(x)                      # <- ensures we never pass raw images forward

        # models that output tokens
        if self.model_name in ["conch", "h0-mini"]:
            x = x[:, 0, :]

        if self.proj == "transformer":
            # if you genuinely need a transformer head, use forward_features here,
            # but given trunk already ran, most cases can just do:
            # tokens = self.trunk.forward_features(orig_x)  # if needed
            # x = self.head(tokens); x = x[:, 0, :]
            raise NotImplementedError("transformer head path needs tokens; not used here.")
        else:
            # 2) Apply head (Linear/MLP/Identity)
            x = self.head(x)

        return x.contiguous()


if __name__ == "__main__":
    # Create an instance of the ImageEncoder class

    for model_name in ["optimus"]:  # , "uni", "ctranspath", "optimus", "conch", "gigapath"]:
        encoder = ImageEncoder(
            model_name=model_name,
            finetune=True,
            embed_dim=128,
            proj="mlp",
            checkpoint_path="/p/project1/hai_spatial_clip/pretrain_weights/image",
        )

        # encoder.freeze()

        encoder = encoder.to("cuda")
        dummy_input = torch.Tensor(1, 3, 224, 224).uniform_().to("cuda")
        output = encoder(dummy_input)
        print(output.shape)  # Output shape: [batch_size, num_features]

        # print parameter names which have gradient set to True
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                print(name)

        print_trainable_parameters(model_name, encoder)
