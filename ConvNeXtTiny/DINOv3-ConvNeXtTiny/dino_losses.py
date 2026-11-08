"""Study-level classification and label-safe Study-aware contrastive loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def study_logit_mean(image_logits: torch.Tensor, study_index: torch.Tensor,
                     num_studies: int, image_counts: torch.Tensor) -> torch.Tensor:
    if image_logits.ndim != 2 or study_index.ndim != 1:
        raise ValueError("Expected image logits [images, labels] and Study indices [images]")
    if image_logits.shape[0] != study_index.shape[0]:
        raise ValueError("Image logits and Study index counts differ")
    image_logits = image_logits.float()
    logits_sum = image_logits.new_zeros((int(num_studies), image_logits.shape[1]))
    logits_sum = logits_sum.index_add(0, study_index, image_logits)
    counts = image_counts.to(device=image_logits.device, dtype=image_logits.dtype)
    if bool((counts <= 0).any()):
        raise ValueError("Every Study must contribute at least one image")
    return logits_sum / counts.unsqueeze(1)


def study_bce_per_study(study_logits: torch.Tensor, targets: torch.Tensor,
                        known: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if study_logits.shape != targets.shape or known.shape != targets.shape:
        raise ValueError("Study logits, targets and known-label mask must have matching shapes")
    elementwise = F.binary_cross_entropy_with_logits(
        study_logits.float(), targets.float(), reduction="none")
    known_float = known.to(device=elementwise.device, dtype=elementwise.dtype)
    known_count = known_float.sum(dim=1)
    eligible = known_count > 0
    per_study = (elementwise * known_float).sum(dim=1) / known_count.clamp_min(1)
    return per_study, eligible


def contrastive_relation_masks(batch: dict) -> dict:
    """Build symmetric positive/negative image-pair masks from Study metadata.

    Positive pairs are different images within one Study. Negative Studies must
    have different Subject IDs, complete known labels, disjoint positive disease
    labels (classes 1-9), and may not both assert No Finding. No Finding is kept
    separate from pathology labels. Uncertain/incomplete relationships are masked.
    """
    study_index = batch["study_index"].detach().cpu().long()
    labels = batch["labels"].detach().cpu().float()
    known = batch["known"].detach().cpu().bool()
    subjects = [str(value) for value in batch["subject_id"]]
    num_images = int(study_index.numel())
    num_studies = int(labels.shape[0])
    if known.shape != labels.shape or known.shape[1] != 10:
        raise ValueError("Contrastive filtering requires ten label columns and matching masks")
    if num_images == 0 or len(subjects) != num_studies:
        raise ValueError("Study/image metadata is inconsistent")

    same_study = study_index[:, None].eq(study_index[None, :])
    positive = same_study
    positive.fill_diagonal_(False)

    disease_positive = labels[:, :9] > 0.5
    overlap = (disease_positive[:, None, :] & disease_positive[None, :, :]).any(dim=-1)
    fully_known = known.all(dim=1)
    same_subject = torch.tensor(
        [[subjects[i] == subjects[j] for j in range(num_studies)]
         for i in range(num_studies)], dtype=torch.bool)
    no_finding = (known[:, 9] & (labels[:, 9] > 0.5))
    both_no_finding = no_finding[:, None] & no_finding[None, :]
    inconsistent_no_finding = no_finding & disease_positive.any(dim=1)
    unreliable_no_finding = (inconsistent_no_finding[:, None] |
                             inconsistent_no_finding[None, :])
    different_study = ~torch.eye(num_studies, dtype=torch.bool)
    negative_by_study = (
        different_study
        & ~same_subject
        & fully_known[:, None]
        & fully_known[None, :]
        & ~overlap
        & ~both_no_finding
        & ~unreliable_no_finding
    )
    negative = negative_by_study[study_index[:, None], study_index[None, :]]
    negative.fill_diagonal_(False)

    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    valid_anchor = (positive_count > 0) & (negative_count > 0)
    skipped_no_negative = (positive_count > 0) & (negative_count == 0)
    no_positive = positive_count == 0
    upper = torch.triu(torch.ones((num_images, num_images), dtype=torch.bool), diagonal=1)
    return {
        "positive": positive,
        "negative": negative,
        "candidates": positive | negative,
        "valid_anchor": valid_anchor,
        "valid_anchor_count": int(valid_anchor.sum()),
        "positive_pairs": int((positive & upper).sum()),
        "negative_pairs": int((negative & upper).sum()),
        "skipped_no_negative": int(skipped_no_negative.sum()),
        "anchors_without_positive": int(no_positive.sum()),
        "image_count": num_images,
    }


def multi_positive_info_nce(projected: torch.Tensor, relations: dict,
                            temperature: float = 0.1) -> tuple[torch.Tensor, int]:
    if projected.ndim != 2 or projected.shape[1] != 128:
        raise ValueError(f"Expected normalized 128-D projected features, got {tuple(projected.shape)}")
    if not temperature > 0:
        raise ValueError("Contrastive temperature must be positive")
    valid_anchor = relations["valid_anchor"].to(device=projected.device)
    anchor_count = int(valid_anchor.sum())
    if anchor_count == 0:
        return projected.sum() * 0.0, 0

    positive = relations["positive"].to(device=projected.device)
    candidates = relations["candidates"].to(device=projected.device)
    similarity = (projected.float() @ projected.float().transpose(0, 1)) / float(temperature)
    valid_similarity = similarity[valid_anchor]
    valid_candidates = candidates[valid_anchor]
    valid_positive = positive[valid_anchor]
    denominator = torch.logsumexp(
        valid_similarity.masked_fill(~valid_candidates, -torch.inf), dim=1)
    positive_average = (
        valid_similarity.masked_fill(~valid_positive, 0.0).sum(dim=1)
        / valid_positive.sum(dim=1).clamp_min(1)
    )
    per_anchor = denominator - positive_average
    return per_anchor.sum(), anchor_count
