import torch

def get_clip_metrics(image_features, gene_features, logit_scale, stage: str):
    metrics = {}
    logits_per_image = logit_scale * image_features @ gene_features.T
    logits_per_gene = logits_per_image.T

    logits = {"image_to_gene": logits_per_image, "gene_to_image": logits_per_gene}
    ground_truth = torch.arange(len(gene_features), device=gene_features.device).contiguous().view(-1, 1)

    # metrics for +ve and -ve pairs
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.float()
        metrics[f"{stage}/{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{stage}/{name}_median_rank"] = torch.floor(torch.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{stage}/{name}_R@{k}"] = torch.mean((preds < k).float())

    return metrics