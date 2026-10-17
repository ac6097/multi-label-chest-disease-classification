import numpy as np
import torch

from utils.metrics import multilabel_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, amp_enabled=False, amp_dtype=torch.bfloat16):
    model.eval()
    loss_total, valid_study_total = 0.0, 0
    logits_all, labels_all, known_all = [], [], []
    subject_ids, study_ids = [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        study_index = batch["study_index"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        known = batch["known"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images, study_index, batch["num_studies"])
        logits = logits.float()
        eligible_studies = int(known.any(dim=1).sum().item())
        loss_total += float(criterion(logits, labels, known)) * eligible_studies
        valid_study_total += eligible_studies
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
        known_all.append(known.cpu())
        subject_ids.extend(batch["subject_id"])
        study_ids.extend(batch["study_id"])
    if not logits_all:
        raise ValueError("Validation split contains no Study samples")
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    known = torch.cat(known_all).numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    metrics = multilabel_metrics(labels, probabilities, known)
    metrics.update(loss=(loss_total / valid_study_total if valid_study_total else float("nan")),
                   probabilities=probabilities,
                   targets=labels, known=known, subject_ids=subject_ids, study_ids=study_ids)
    return metrics
