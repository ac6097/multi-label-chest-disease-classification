import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .validator import validate
from utils.checkpoint import load_checkpoint, save_checkpoint


def _forward_studies(model, batch, device, amp_enabled, amp_dtype):
    images = batch["images"].to(device, non_blocking=True)
    study_index = batch["study_index"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(images, study_index, batch["num_studies"])
    return logits.float()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_enabled,
                    amp_dtype, accumulation, clip_norm, freeze_batchnorm_stats=False):
    model.train()
    if not next(model.backbone.parameters()).requires_grad:
        model.backbone.eval()
    elif freeze_batchnorm_stats:
        for module in model.backbone.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_valid_studies, total_norm, steps, updates = 0.0, 0, 0.0, 0, 0
    batch_sampler = getattr(loader, "batch_sampler", None)
    planned_batches = batch_sampler._batches() if hasattr(batch_sampler, "_batches") else None
    if planned_batches is not None and hasattr(loader.dataset, "studies"):
        study_is_valid = [bool(item["valid"].any()) for item in loader.dataset.studies]
        accumulation_counts = []
        for start in range(0, len(planned_batches), accumulation):
            window = planned_batches[start:start + accumulation]
            accumulation_counts.append(sum(study_is_valid[index]
                                           for batch_indices in window for index in batch_indices))
    else:
        accumulation_counts = None
    for index, batch in enumerate(loader):
        labels = batch["labels"].to(device, non_blocking=True)
        known = batch["known"].to(device, non_blocking=True)
        logits = _forward_studies(model, batch, device, amp_enabled, amp_dtype)
        loss = criterion(logits, labels, known)
        eligible_studies = int(known.any(dim=1).sum().item())
        if accumulation_counts is not None:
            window_valid_studies = max(accumulation_counts[index // accumulation], 1)
            scaled_loss = loss * (eligible_studies / window_valid_studies)
        else:
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            scaled_loss = loss / window_size
        scaler.scale(scaled_loss).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            total_norm += float(norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        total_loss += float(loss.detach()) * eligible_studies
        total_valid_studies += eligible_studies
        steps += 1
    return {"loss": total_loss / max(total_valid_studies, 1),
            "gradient_norm": total_norm / max(updates, 1)}


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device,
        logger, class_names=None):
    amp_dtype = torch.bfloat16 if config["amp_dtype"] == "bfloat16" else torch.float16
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    if amp_dtype == torch.bfloat16 and amp_enabled and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 is unsupported on this GPU; using FP16 autocast")
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    output_dir = Path(config["output_dir"])
    best, stale, start_epoch = float("-inf"), 0, 0
    resume = config.get("resume")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(state.get("epoch", 0))
        stored_best = state.get("best_metric")
        best = float("-inf") if stored_best is None else float(stored_best)
        stale = int(state.get("stale_epochs", 0))
        logger.info("resumed checkpoint=%s epoch=%d best_macro_auroc=%s", resume, start_epoch, best)

    for epoch in range(start_epoch, int(config["epochs"])):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        frozen = epoch < int(config.get("freeze_backbone_epochs", 0))
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            model.backbone.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
            amp_enabled, amp_dtype, int(config["grad_accumulation"]),
            float(config["gradient_clip_norm"]),
            bool(config.get("freeze_batchnorm_stats", False)))
        train_elapsed = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation = validate(model, val_loader, criterion, device, amp_enabled, amp_dtype)
        validation_elapsed = time.perf_counter() - validation_started
        if scheduler:
            scheduler.step()
        peak_memory = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                       if device.type == "cuda" else 0.0)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        logger.info("epoch=%d/%d train_loss=%.5f val_loss=%.5f macro_auroc=%s macro_f1=%s "
                    "gradient_norm=%.4f lr=%s peak_gpu_gb=%.3f train_s=%.1f val_s=%.1f",
                    epoch + 1, config["epochs"], train_stats["loss"], validation["loss"],
                    validation["macro_auroc"], validation["macro_f1"],
                    train_stats["gradient_norm"], current_lrs, peak_memory,
                    train_elapsed, validation_elapsed)
        for index, item in enumerate(validation["per_label"]):
            name = class_names[index] if class_names else index + 1
            logger.info("label[%s] valid=%d positive=%d negative=%d auroc=%s auprc=%s f1=%s",
                        name, item["n_valid"], item["n_positive"], item["n_negative"],
                        item["auroc"], item["auprc"], item["f1"])

        score = validation["macro_auroc"]
        improved = math.isfinite(score) and score > best
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        save_args = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                         epoch=epoch + 1, best_metric=best, config=config,
                         labels=class_names, stale_epochs=stale)
        save_checkpoint(output_dir / "checkpoint/last/last.pth", **save_args)
        if improved:
            # Keep the deployable best model weights-only so it stays below the submission size cap.
            save_checkpoint(output_dir / "checkpoint/best/best_by_macro_auc.pth",
                            model=model, epoch=epoch + 1, best_metric=best, config=config,
                            labels=class_names, stale_epochs=stale)
            rows = []
            for row_index, (subject_id, study_id) in enumerate(
                    zip(validation["subject_ids"], validation["study_ids"])):
                row = {"Subject_id": subject_id, "Study_id": study_id}
                for label_index, name in enumerate(class_names or range(1, 11)):
                    row[f"true_{name}"] = (float(validation["targets"][row_index, label_index])
                                            if validation["known"][row_index, label_index] else np.nan)
                    row[f"known_{name}"] = bool(validation["known"][row_index, label_index])
                    row[f"prob_{name}"] = float(validation["probabilities"][row_index, label_index])
                rows.append(row)
            predictions_path = output_dir / "outputs/predictions/val_best.csv"
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(predictions_path, index=False)
        if stale >= int(config["patience"]):
            logger.info("early stopping after %d epochs without Macro-AUROC improvement", stale)
            break
