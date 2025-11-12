'''
# cfg_bridge.py
import json, re, os

def _int_in_string(s, default=None):
    m = re.findall(r"\d+", os.path.basename(str(s)))
    return int(m[0]) if m else default

def infer_hvg_k_from_filename(path, default=100):
    """Guess HVG count from filename like 'df_100_most_variable_genes.csv'."""
    return _int_in_string(path, default=default)

def load_model_train_cfgs(data_json_path: str, model_json_path: str):
    """
    Map your TRAINING configs to the exact inference settings your checkpoint needs.
    Returns a dict with keys that the main script applies to argparse `args`.
    """
    with open(data_json_path) as f:
        dcfg = json.load(f)
    with open(model_json_path) as f:
        mcfg = json.load(f)

    # ==== DATA PIPELINE ====
    # If you didn't pass precomputed image embeddings, you trained with an image encoder (+ optional PCA)
    use_inception = not dcfg.get("use_embeddings", False)

    # Try to recover image PCA dim from model config; otherwise from image filename; else None
    image_pca_dim = mcfg.get("pca_dim", None)
    if image_pca_dim in (None, 0):
        image_pca_dim = _int_in_string(dcfg.get("images_file", ""), default=None)

    # Transcriptomes: log1p normalization + HVG count from filename (fallback 100)
    gene_log1p = bool(dcfg.get("normalize_transcriptomes", True))
    hvg_k = infer_hvg_k_from_filename(dcfg.get("transcriptoms_file", ""), default=100)

    # ==== MODEL ARCH ====
    dropout_prob = float(mcfg.get("dropout_prob", 0.5))
    hidden_dim   = int(mcfg.get("hidden_dim", 512))
    output_dim   = int(mcfg.get("output_dim", 256))
    num_layers   = int(mcfg.get("mlp_layers", 2))
    use_pos_emb  = bool(mcfg.get("use_positional_embedding", False))
    use_feat_sel = bool(mcfg.get("use_feature_selector", False))
    use_pos_attn = bool(mcfg.get("use_positional_attention", False))  # may be absent
    temperature=float(mcfg.get("temperature", 0.5))
    mcl_projection_dim=int(mcfg.get("mcl_projection_dim", mcfg.get("output_dim", 128)))
    mcl_heads_num=int(mcfg.get("mcl_heads_num", 8))
    mcl_heads_dim=int(mcfg.get("mcl_heads_dim", 64))
    mcl_heads_layers=int(mcfg.get("mcl_heads_layers", 2))
    n_hidden_muse= int(mcfg.get("n_hidden_muse", 128))
    dim_z_muse=int(mcfg.get("dim_z_muse", 256)) 
    
       
    

    # First Linear input sizes must match what you trained with:
    #  - images: post-PCA dim (or 2048 if no PCA)
    #  - transcriptome: HVG_k
    input_dim_image = int(image_pca_dim) if image_pca_dim not in (None, 0) else 2048
    input_dim_transcriptome = int(hvg_k)

    return {
        # preprocessing
        "use_inception": use_inception,
        "image_pca_dim": image_pca_dim,
        "hvg_k": hvg_k,
        "gene_log1p": gene_log1p,

        # model
        "input_dim_image": input_dim_image,
        "input_dim_transcriptome": input_dim_transcriptome,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "dropout_prob": dropout_prob,
        "use_positional_embedding": use_pos_emb,
        "use_positional_attention": use_pos_attn,
        "use_feature_selector": use_feat_sel,
        "pos_embedding_dim": 128,  # adjust if you trained with a different value
        "temperature": temperature,
        "mcl_projection_dim": mcl_projection_dim,
        "mcl_heads_num": mcl_heads_num,
        "mcl_heads_dim": mcl_heads_dim,
        "mcl_heads_layers": mcl_heads_layers,
        "dropout_prob": dropout_prob,
        "n_hidden_muse": n_hidden_muse,
        "dim_z_muse": dim_z_muse
    }
'''


# new version for MUCST integration
# cfg_bridge.py
import json, re, os

def _int_in_string(s, default=None):
    m = re.findall(r"\d+", os.path.basename(str(s)))
    return int(m[0]) if m else default

def infer_hvg_k_from_filename(path, default=100):
    """Guess HVG count from filename like 'df_100_most_variable_genes.csv'."""
    return _int_in_string(path, default=default)

def _coerce_int_list(val, default):
    """
    Rende robusta la lettura di liste di interi da JSON/CLI:
    - se è None -> default
    - se è int -> [int]
    - se è list/tuple -> [int(x) ...]
    - se è stringa tipo "64,32,16" -> [64,32,16]
    """
    if val is None:
        return list(default)
    if isinstance(val, int):
        return [int(val)]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(",") if p.strip() != ""]
        if parts:
            return [int(p) for p in parts]
    return list(default)

def load_model_train_cfgs(data_json_path: str, model_json_path: str):
    """
    Map your TRAINING configs to the exact inference settings your checkpoint needs.
    Returns a dict with keys that the main script applies to argparse `args`.
    """
    with open(data_json_path) as f:
        dcfg = json.load(f)
    with open(model_json_path) as f:
        mcfg = json.load(f)

    # ==== DATA PIPELINE ====
    # If you didn't pass precomputed image embeddings, you trained with an image encoder (+ optional PCA)
    use_inception = not dcfg.get("use_embeddings", False)

    # Try to recover image PCA dim from model config; otherwise from image filename; else None
    image_pca_dim = mcfg.get("pca_dim", None)
    if image_pca_dim in (None, 0):
        image_pca_dim = _int_in_string(dcfg.get("images_file", ""), default=None)

    # Transcriptomes: log1p normalization + HVG count from filename (fallback 100)
    gene_log1p = bool(dcfg.get("normalize_transcriptomes", True))
    hvg_k = infer_hvg_k_from_filename(dcfg.get("transcriptoms_file", ""), default=100)

    # ==== MODEL ARCH (comuni) ====
    dropout_prob = float(mcfg.get("dropout_prob", 0.5))
    hidden_dim   = int(mcfg.get("hidden_dim", 512))
    output_dim   = int(mcfg.get("output_dim", 256))
    num_layers   = int(mcfg.get("mlp_layers", 2))
    use_pos_emb  = bool(mcfg.get("use_positional_embedding", False))
    use_feat_sel = bool(mcfg.get("use_feature_selector", False))
    use_pos_attn = bool(mcfg.get("use_positional_attention", False))  # may be absent
    temperature  = float(mcfg.get("temperature", 0.5))

    # mclSTExp
    mcl_projection_dim = int(mcfg.get("mcl_projection_dim", mcfg.get("output_dim", 128)))
    mcl_heads_num      = int(mcfg.get("mcl_heads_num", 8))
    mcl_heads_dim      = int(mcfg.get("mcl_heads_dim", 64))
    mcl_heads_layers   = int(mcfg.get("mcl_heads_layers", 2))

    # MUSE
    n_hidden_muse = int(mcfg.get("n_hidden_muse", 128))
    dim_z_muse    = int(mcfg.get("dim_z_muse", 256))

    # ==== MuCST-specific ====
    # proiezioni (lista di int), intermediate dim per ImageEncoder, pesi delle loss
    mucst_proj_dims   = _coerce_int_list(mcfg.get("mucst_proj_dims", [64, 64]), default=[64, 64])
    intermediate_dim  = int(mcfg.get("intermediate_dim", 512))
    lamb1             = float(mcfg.get("lamb1", 0.3))  # img recon
    lamb2             = float(mcfg.get("lamb2", 0.1))  # CSL (g2g)
    lamb3             = float(mcfg.get("lamb3", 1.0))  # i2g
    gamma             = float(mcfg.get("gamma", 1.0))  # i2i

    # First Linear input sizes must match what you trained with:
    #  - images: post-PCA dim (or 2048 if no PCA)
    #  - transcriptome: HVG_k
    input_dim_image = int(image_pca_dim) if image_pca_dim not in (None, 0) else 2048
    input_dim_transcriptome = int(hvg_k)

    # Se hidden_dim non è impostato ma hai mucst_proj_dims, allinea hidden al primo stadio del projector
    if "hidden_dim" not in mcfg or mcfg.get("hidden_dim") in (None, 0):
        # mantieni il tuo default 512 per retro-compatibilità, ma se MuCST è usato
        # conviene sincronizzare hidden_dim al primo elemento del projector
        hidden_dim = int(mucst_proj_dims[0]) if len(mucst_proj_dims) > 0 else hidden_dim

    return {
        # preprocessing
        "use_inception": use_inception,
        "image_pca_dim": image_pca_dim,
        "hvg_k": hvg_k,
        "gene_log1p": gene_log1p,

        # comuni
        "input_dim_image": input_dim_image,
        "input_dim_transcriptome": input_dim_transcriptome,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "dropout_prob": dropout_prob,
        "use_positional_embedding": use_pos_emb,
        "use_positional_attention": use_pos_attn,
        "use_feature_selector": use_feat_sel,
        "pos_embedding_dim": 128,  # cambia se in training era diverso
        "temperature": temperature,

        # mclSTExp
        "mcl_projection_dim": mcl_projection_dim,
        "mcl_heads_num": mcl_heads_num,
        "mcl_heads_dim": mcl_heads_dim,
        "mcl_heads_layers": mcl_heads_layers,

        # MUSE
        "n_hidden_muse": n_hidden_muse,
        "dim_z_muse": dim_z_muse,

        # MuCST
        "mucst_proj_dims": mucst_proj_dims,
        "intermediate_dim": intermediate_dim,
        "lamb1": lamb1,
        "lamb2": lamb2,
        "lamb3": lamb3,
        "gamma": gamma,
    }
