import torch
from torch import nn
from torch.nn import functional as F


def _masked_study_mean(element_loss, known):
    mask = known.to(element_loss.dtype)
    counts = mask.sum(dim=1)
    per_study = (element_loss * mask).sum(dim=1) / counts.clamp_min(1.0)
    eligible = counts > 0
    if not eligible.any():
        return element_loss.sum() * 0.0
    return per_study[eligible].mean()


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE with optional per-class positive weights and one equal vote per Study."""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets, known):
        element = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        return _masked_study_mean(element, known)


class AsymmetricLoss(nn.Module):
    """Mask-aware asymmetric focal loss, exposed as an ablation candidate."""
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits, targets, known):
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos
        if self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)
        log_likelihood = (targets * torch.log(prob_pos.clamp_min(self.eps)) +
                          (1.0 - targets) * torch.log(prob_neg.clamp_min(self.eps)))
        prob_true = prob_pos * targets + prob_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal_weight = (1.0 - prob_true).clamp_min(0.0).pow(gamma)
        return _masked_study_mean(-log_likelihood * focal_weight, known)


def estimate_pos_weight(targets, known, strategy="sqrt", cap=5.0):
    """Estimate weights from unique, confirmed training-Study labels only.

    Degenerate classes with no positive or no negative valid Studies get weight 1;
    their class counts must still be logged and the validation metric may be undefined.
    """
    targets, known = targets.bool(), known.bool()
    positives = ((targets) & known).sum(dim=0).float()
    negatives = ((~targets) & known).sum(dim=0).float()
    usable = (positives > 0) & (negatives > 0)
    raw = negatives / positives.clamp_min(1.0)
    if strategy == "raw":
        weights = raw
    elif strategy == "sqrt":
        weights = raw.sqrt()
    elif strategy == "cap":
        weights = raw.clamp(max=float(cap))
    elif strategy == "none":
        weights = torch.ones_like(raw)
    else:
        raise ValueError("strategy must be one of: none, raw, sqrt, cap")
    return torch.where(usable, weights, torch.ones_like(weights))
