import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.encoder_model import Encoder
except Exception:
    class Encoder(nn.Module):
        def __init__(self, input_dim, n_hidden):
            """
            Encoder for single modality

            Parameters:
                input_dim:  feature dimension of input data
                n_hidden:   hidden node number
            """
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(input_dim, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)

        def forward(self, x):
            """
            Forward pass through the encoder

            Parameters:
                x: input data

            Returns:
                latent representation for single modality
            """
            h1 = F.elu(self.fc1(x))
            h2 = torch.tanh(self.fc2(h1))
            return h2

try:
    from models.encoder_model import EncoderStep
except Exception:
    class EncoderStep(nn.Module):
        def __init__(self, input_dim_x, input_dim_y, n_hidden):
            """
            Initialize the EncoderStep with two encoders, one for each modality.

            Parameters:
                input_dim_x: feature dimension of x modality
                input_dim_y: feature dimension of y modality
                n_hidden: hidden node number for encoder layers
            """
            super(EncoderStep, self).__init__()
            self.encoder_x = Encoder(input_dim_x, n_hidden)
            self.encoder_y = Encoder(input_dim_y, n_hidden)

        def forward(self, x, y):
            """
            Forward pass through the EncoderStep

            Parameters:
                x: input batches for transcript modality; matrix of n * p
                y: input batches for morphological modality; matrix of n * q

            Returns:
                encode_x: latent representation for modality x
                encode_y: latent representation for modality y
            """
            encode_x = self.encoder_x(x)
            encode_y = self.encoder_y(y)
            return encode_x, encode_y

  
try:
    from models.jointEmbed_model import JointEmbeddingStep
except Exception:
    class JointEmbeddingStep(nn.Module):
        def __init__(self, input_dim, output_dim):
            """
            Combines encoded features into a joint latent representation

            Parameters:
                input_dim:  combined feature dimension of both modalities
                output_dim: dimension of joint latent representation
            """
            super(JointEmbeddingStep, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, encode_x, encode_y):
            """
            Forward pass to combine encoded features

            Parameters:
                encode_x: encoded features from modality x
                encode_y: encoded features from modality y

            Returns:
                z: Joint latent representation
            """
            h = torch.cat([encode_x, encode_y], dim=1)
            z = self.fc(h)
            return z
        


    
    
class MUSE(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, dim_z, n_hidden):
        """
        Construct structure and loss function of MUSE

        Parameters:
            input_dim_x:      feature dimension of x modality
            input_dim_y:      feature dimension of y modality
            dim_z:            dimension of joint latent representation
            n_hidden:         hidden node number for encoder and decoder layers
            triplet_margin:   margin for triplet loss
            weight_penalty:   weight for sparse penalty
            triplet_lambda:   weight for triplet loss
        """
        super(MUSE, self).__init__()
        self.encoder_step = EncoderStep(input_dim_x, input_dim_y, n_hidden)
        self.joint_embedding_step = JointEmbeddingStep(2 * n_hidden, dim_z)

    def forward(self, x, y):
        """
        Forward pass through the MUSE model

        Parameters:
            x:          input batches for transcript modality; matrix of n * p
            y:          input batches for morphological modality; matrix of n * q

        Returns:
            z:              joint latent representations (n * dim_z)
            x_hat:          reconstructed x (same shape as x)
            y_hat:          reconstructed y (same shape as y)
            encode_x:       latent representation for modality x
            encode_y:       latent representation for modality y
            loss:           total loss
            reconstruct_loss: reconstruction loss
            sparse_penalty: sparse penalty
            trip_loss_x:    triplet loss for x
            trip_loss_y:    triplet loss for y
        """
        # Encode inputs
        encode_x, encode_y = self.encoder_step(x, y)
        
        # Obtain joint latent representation
        z = self.joint_embedding_step(encode_x, encode_y)
        
        return x, y, z, encode_x, encode_y


def _strip_prefixes(sd, prefixes=("model.", "module.", "net.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def _load_any_ckpt(path):
    try:
        from safetensors.torch import load_file as safe_load
        if str(path).endswith(".safetensors"):
            return safe_load(path)
    except Exception:
        pass
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint {path} is not a valid state_dict.")
    return sd

def load_muse_from_cfg(cfg) -> nn.Module:
    model = MUSE(
        input_dim_x = int(cfg["input_dim_transcriptome"]),
        input_dim_y = int(cfg["input_dim_image"]),
        dim_z       = int(cfg.get("dim_z_muse", 256)),  # your dim_z
        n_hidden    = int(cfg.get("n_hidden_muse", 512)),  # your n_hidden
    )
    ckpt = cfg.get("checkpoint")
    if ckpt:
        sd = _load_any_ckpt(ckpt)
        sd = _strip_prefixes(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[MUSE load] missing keys:", missing)
        print("[MUSE load] unexpected keys:", unexpected)
    return model
