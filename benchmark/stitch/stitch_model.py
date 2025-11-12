# stitch_model.py
import torch
import torch.nn as nn
from typing import Dict

# Try importing your real helpers; otherwise fall back to stubs.
try:
    from models.dynamic_selector import DynamicFeatureSelector
except Exception:
    class DynamicFeatureSelector(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.selector = nn.Parameter(torch.randn(input_dim))

        def forward(self, x):
            weights = F.softmax(self.selector, dim=0)  # [input_dim]
            return x * weights  # [batch_size, input_dim]


try:
    from models.spatialPositionalEncoder import SpatialPositionalEncoder
except Exception:
    class SpatialPositionalEncoder(nn.Module):
        def __init__(self, in_dim=3, out_dim=128, p_drop=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
                nn.LayerNorm(out_dim)
            )
            self.dropout = nn.Dropout(p_drop)

        def forward(self, coords_3d):
            pe = self.net(coords_3d)
            return self.dropout(pe)

try:
    from models.attention_fusion import CrossAttentionFusion
except Exception:
    class CrossAttentionFusion(nn.Module):
        def __init__(self, embed_dim, num_heads=4, pos_emb_scale=1.0):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.pos_emb_scale = pos_emb_scale

            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, query: torch.Tensor, key_value: torch.Tensor):
            """
            query: [B, 1, D]         (e.g. from pos_emb.unsqueeze(1))
            key_value: [B, N, D]     (e.g. stack of [image_emb, transcriptome_emb])
            """
            B, _, D = query.shape
            N = key_value.shape[1]

            Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)      # [B, H, 1, d_k]
            K = self.k_proj(key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_k]
            V = self.v_proj(key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_k]

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale                       # [B, H, 1, N]
            attn_weights = F.softmax(attn_scores, dim=-1)                                         # [B, H, 1, N]
            attn_output = torch.matmul(attn_weights, V)                                           # [B, H, 1, d_k]

            attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, D)  # [B, 1, D]
            fused = self.out_proj(attn_output).squeeze(1)                         # [B, D]

            return fused, attn_weights.squeeze(2)  # [B, D], [B, H, N]


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_prob=0.5
    ):
        """
        Initialize the MLP model with configurable layers.

        Parameters:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
            num_layers (int): Number of hidden layers.
            dropout_prob (float): Dropout probability.
        """
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ST_MODEL(nn.Module):
    def __init__(
        self,
        input_dim_image,
        input_dim_transcriptome,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout_prob=0.5,
        use_positional_embedding=False,
        use_positional_attention=False,
        use_feature_selector=False,
        pos_embedding_dim=128,
        num_attention_tokens=4,
    ):
        super(ST_MODEL, self).__init__()

        self.use_positional_embedding = use_positional_embedding
        self.use_positional_attention = use_positional_attention
        self.use_feature_selector = use_feature_selector
        
        
        self.image_encoder = MLP(input_dim_image, hidden_dim, output_dim, num_layers, dropout_prob)
        if self.use_feature_selector:
            self.feature_selector = DynamicFeatureSelector(input_dim_transcriptome)
        else:
            self.feature_selector = nn.Identity()  # non modifica il transcriptoma
        self.transcriptome_encoder = MLP(input_dim_transcriptome, hidden_dim, output_dim, num_layers, dropout_prob)

        
        
        if self.use_positional_embedding:
            self.pos_encoder = SpatialPositionalEncoder(in_dim=3, out_dim=output_dim)
            self.ln_img = nn.LayerNorm(output_dim)
            self.ln_rna = nn.LayerNorm(output_dim)
            self.ln_pos = nn.LayerNorm(output_dim)
            self.alpha = nn.Parameter(torch.tensor(0.05))
                
            if self.use_positional_attention:
                #self.attention_fusion = CrossAttentionFusion(embed_dim=output_dim, num_heads=4)
                #self.attention_fusion = CrossAttentionFusion(embed_dim=output_dim, num_heads=4, pos_emb_scale=2.0)
                self.num_attention_tokens = num_attention_tokens
                self.image_attention_fusion = CrossAttentionFusion(embed_dim=output_dim, num_heads=4)
                self.transcriptome_attention_fusion = CrossAttentionFusion(embed_dim=output_dim, num_heads=4)
                # 🔁 Optional learnable tokens per modality for richer attention
                self.image_tokens = nn.Parameter(torch.randn(1, num_attention_tokens, output_dim))
                self.transcriptome_tokens = nn.Parameter(torch.randn(1, num_attention_tokens, output_dim))
                self.warmup_epochs = 20  # oppure passalo come argomento
                #self.fusion_beta = nn.Parameter(torch.tensor(0.5))  # learnable bilanciamento
                self.alpha_pos = nn.Parameter(torch.tensor(0.5))
            
                
                
                '''
                # α warm-up (not learnable; scheduled each forward)
                self.alpha_img_init = 0.0
                self.alpha_img_final = 0.08   # tune 0.05–0.12
                self.alpha_img = nn.Parameter(torch.tensor(self.alpha_img_init), requires_grad=False)
                self.alpha_warmup_epochs = 10
                '''


    def forward(self, image, transcriptome, coords_3d=None, epoch=0, batch_ids=None):
        
        B = image.size(0)
        image_features = self.image_encoder(image)
        transcriptome_selected = self.feature_selector(transcriptome)
        transcriptome_features = self.transcriptome_encoder(transcriptome_selected)
        
        
        
        attn_weights = {}

        ''' old version of attention
        if self.use_positional_embedding and coords_3d is not None:
            pos_emb = self.pos_encoder(coords_3d)
            if self.use_positional_attention:
                # Variante 2: usa pos_emb come guida per attenzione
                query = image_features + pos_emb
                transcriptome_features = self.attention_fusion(query, transcriptome_features)
            else:
                # Variante 1: somma diretta
                image_features = image_features + pos_emb
                transcriptome_features = transcriptome_features + pos_emb
        '''
        if self.use_positional_embedding and coords_3d is not None:
            pos_emb = self.pos_encoder(coords_3d)

            if self.use_positional_attention:
                
                #Variante con alpha learnable e layer norm
                image_features = self.ln_img(image_features)
                transcriptome_features = self.ln_rna(transcriptome_features)
                pos_emb = self.ln_pos(pos_emb)

                image_features = image_features + self.alpha * pos_emb
                transcriptome_features = transcriptome_features + self.alpha * pos_emb

                return image_features, transcriptome_features, attn_weights
            else:
                
                #Variante con alpha learnable e layer norm
                image_features = self.ln_img(image_features)
                transcriptome_features = self.ln_rna(transcriptome_features)
                pos_emb = self.ln_pos(pos_emb)

                image_features = image_features + pos_emb
                transcriptome_features = transcriptome_features + pos_emb
                
                return image_features, transcriptome_features, attn_weights


        return image_features, transcriptome_features, attn_weights


def _strip_prefixes(sd: Dict[str, torch.Tensor], prefixes=("model.", "module.", "net.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p): nk = nk[len(p):]
        out[nk] = v
    return out

def _load_any_ckpt(path: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file as safe_load
        if path.endswith(".safetensors"):
            return safe_load(path)
    except Exception:
        pass
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint {path} is not a dict-like state_dict.")
    return sd

def load_stitch_from_cfg(cfg) -> nn.Module:
    model = ST_MODEL(
        input_dim_image=cfg["input_dim_image"],
        input_dim_transcriptome=cfg.get("input_dim_transcriptome") or cfg["hvg_k"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"],
        num_layers=cfg["num_layers"],
        dropout_prob=cfg["dropout_prob"],
        use_positional_embedding=cfg["use_positional_embedding"],
        use_positional_attention=cfg["use_positional_attention"],
        use_feature_selector=cfg["use_feature_selector"],
        pos_embedding_dim=cfg["pos_embedding_dim"],
    )
    ckpt = cfg.get("checkpoint", None)
    if ckpt:
        sd = _load_any_ckpt(ckpt)
        sd = _strip_prefixes(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[load] missing keys:", missing)
        print("[load] unexpected keys:", unexpected)
    return model
