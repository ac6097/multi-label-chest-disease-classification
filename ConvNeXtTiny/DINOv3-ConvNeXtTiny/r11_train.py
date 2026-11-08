"""R11 training: frozen H2 base + ten disease-specific sparse routing branches."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import genopt_analysis as base_analysis
from genopt_common import (
    EXPERIMENT_ROOT,
    LABEL_NAMES,
    SEED,
    build_h_datasets,
    class_bce_breakdown,
    load_r5_reference,
    read_json,
    study_logit_mean,
    write_json,
)
from dino_data import make_loader, shutdown_loader
from dino_feature_models import collate_masked_studies
from dino_losses import study_bce_per_study
from hybrid.common import write_csv
from stage_fusion_analysis import study_metrics
from r11_model import R11Router, build_r11_router, count_branch_parameters, load_h2_base

R11_DIR = EXPERIMENT_ROOT / "R11_DiseaseSpecific_SparseFeatureRouting_512"
R11_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = R11_DIR
BATCHES = 4
BASE_LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_EPOCHS = 20
WARMUP_EPOCHS = 2
EARLY = {"train_decrease": 3, "valid_increase": 3, "auroc_no_improve": 3,
         "epsilon": 0.0}


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(dataset, shuffle: bool, workers: int):
    loader, sampler, _ = make_loader(
        dataset,
        image_budget=8,
        shuffle=shuffle,
        seed=SEED,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_masked_studies,
    )
    return loader, sampler


def _image_logits(router: R11Router, batch, device):
    images = batch["images"].to(device, non_blocking=True)
    if True:
        images = images.contiguous(memory_format=torch.channels_last)
    logits, base_logits, residual = router(images)
    return logits, base_logits, residual, images


def _weighted_window_loss(router, window, device, weight_single=1.25):
    eligible_window = sum(
        int(batch["known"].any(dim=1).sum()) for batch in window)
    weight_total = 0.0
    for batch in window:
        counts = batch["image_counts"].to(device)
        known = batch["known"].to(device)
        eligible = known.any(dim=1)
        weights = torch.where(counts == 1, weight_single, 1.0)
        weight_total += float(weights[eligible].sum().cpu())
    loss_total = torch.zeros((), device=device)
    raw_sum = 0.0
    eligible_total = 0
    residual_stats = []
    for batch in window:
        images = batch["images"].to(device, non_blocking=True)
        images = images.contiguous(memory_format=torch.channels_last)
        logits, base_logits, residual = router(images)
        study_logits = study_logit_mean(
            logits, batch["study_index"].to(device), batch["num_studies"],
            batch["image_counts"].to(device))
        labels = batch["labels"].to(device)
        known = batch["known"].to(device)
        per_study, eligible = study_bce_per_study(study_logits, labels, known)
        counts = batch["image_counts"].to(device)
        weights = torch.where(counts == 1, weight_single, 1.0)
        loss = (per_study[eligible] * weights[eligible]).sum() / max(
            weight_total, 1e-12)
        loss_total = loss_total + loss
        raw_sum += float(per_study[eligible].sum().detach().float().cpu())
        eligible_total += int(eligible.sum())
        with torch.no_grad():
            residual_stats.append(residual.detach().float())
    return (loss_total, raw_sum / max(eligible_total, 1),
            torch.cat(residual_stats, dim=0))


def _scheduler_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return BASE_LR * float(step + 1) / float(max(warmup_steps, 1))
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _evaluate(router: R11Router, loader, device) -> dict:
    router.eval()
    router.base.eval()
    study_logits = []
    study_residual = []
    study_labels = []
    study_subjects = []
    study_ids = []
    all_channel_masks = []
    all_spatial_masks = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        images = images.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            logits, base_logits, residual, diagnostics = router(
                images, return_diagnostics=True)
            study_index = batch["study_index"].to(device)
            counts = batch["image_counts"].to(device)
            image_counts = batch["image_counts"].to(device)
            study = study_logit_mean(logits, study_index,
                                     batch["num_studies"], image_counts)
            study_res = study_logit_mean(residual, study_index,
                                         batch["num_studies"], image_counts)
        study_logits.append(study.float().cpu())
        study_residual.append(study_res.float().cpu())
        study_labels.append(batch["labels"].cpu())
        study_subjects.extend([str(value) for value in batch["subject_id"]])
        study_ids.extend([str(value) for value in batch["study_id"]])
        all_channel_masks.append(torch.stack(
            [info["channel_mask"].detach().cpu() for info in diagnostics]))
        all_spatial_masks.append(torch.stack(
            [info["spatial_mask"].detach().cpu() for info in diagnostics]))
    logits = torch.cat(study_logits, dim=0).numpy().astype(np.float64)
    residual = torch.cat(study_residual, dim=0).numpy().astype(np.float64)
    labels = torch.cat(study_labels, dim=0).numpy().astype(np.int64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    metrics = study_metrics(labels, probability)
    class_bce = class_bce_breakdown(logits, labels)
    overall_bce = float(np.mean([class_bce[name]["bce"] for name in LABEL_NAMES]))
    # Branch-off target-disease metrics are derived exactly from the residual.
    branch_off_metrics = {}
    for disease_index, disease in enumerate(LABEL_NAMES):
        off_logits = logits.copy()
        off_logits[:, disease_index] -= residual[:, disease_index]
        off_probability = 1.0 / (1.0 + np.exp(-np.clip(off_logits, -80, 80)))
        off = study_metrics(labels, off_probability)["per_class"][disease]
        branch_off_metrics[disease] = off
    channel_masks = torch.cat(all_channel_masks, dim=1)  # [10, N, C]
    spatial_masks = torch.cat(all_spatial_masks, dim=1)  # [10, N, H, W]
    residual_stats = {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_abs_mean": float(np.mean(np.abs(residual))),
        "residual_abs_p95": float(np.percentile(np.abs(residual), 95)),
        "residual_to_base_ratio": float(
            np.mean(np.abs(residual)) / max(np.mean(np.abs(logits)), 1e-8)),
    }
    per_class = {}
    for name in LABEL_NAMES:
        per_class[name] = {
            "auroc": metrics["per_class"][name]["auroc"],
            "auprc": metrics["per_class"][name]["auprc"],
            "bce": class_bce[name]["bce"],
            "positive_bce": class_bce[name]["positive_bce"],
            "negative_bce": class_bce[name]["negative_bce"],
            "branch_off_auroc": branch_off_metrics[name]["auroc"],
            "branch_off_auprc": branch_off_metrics[name]["auprc"],
        }
    return {
        "study_logits": logits,
        "study_residual": residual,
        "labels": labels,
        "subjects": study_subjects,
        "study_ids": study_ids,
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "overall_bce": overall_bce,
        "per_class": per_class,
        "residual_stats": residual_stats,
        "channel_masks": channel_masks,
        "spatial_masks": spatial_masks,
    }


def _mask_diagnostics(channel_masks, spatial_masks, sample_limit=64) -> dict:
    channel = channel_masks[:, :sample_limit]  # [10, n, C]
    spatial = spatial_masks[:, :sample_limit].float()  # [10, n, H, W]
    channel_float = channel.float()
    spatial_flat = spatial.flatten(2)  # [10,n,HW]
    same_channel = []
    same_spatial = []
    for disease in range(10):
        c = channel_float[disease]
        s = spatial_flat[disease]
        inter = c @ c.T
        union = c.sum(1, keepdim=True) + c.sum(1, keepdim=True).T - inter
        same_channel.append(float((inter / union.clamp_min(1.0)).mean()))
        inter_s = s @ s.T
        union_s = s.sum(1, keepdim=True) + s.sum(1, keepdim=True).T - inter_s
        same_spatial.append(float((inter_s / union_s.clamp_min(1.0)).mean()))
    cross_channel = []
    cross_spatial = []
    for a in range(10):
        for b in range(a + 1, 10):
            ca, cb = channel_float[a].mean(0), channel_float[b].mean(0)
            cross_channel.append(float(
                (ca * cb).sum() / ((ca + cb) > 0).float().sum()))
            sa = spatial_flat[a].mean(0)
            sb = spatial_flat[b].mean(0)
            cross_spatial.append(float(
                (sa * sb).sum() / ((sa + sb) > 0).float().sum()))
    return {
        "channel_retention": float(channel_float.mean()),
        "spatial_retention": float(spatial_flat.mean()),
        "same_disease_channel_iou": float(np.mean(same_channel)),
        "same_disease_spatial_iou": float(np.mean(same_spatial)),
        "cross_disease_channel_iou": float(np.mean(cross_channel)),
        "cross_disease_spatial_iou": float(np.mean(cross_spatial)),
    }


def _save_checkpoint(path: Path, router: R11Router, payload: dict) -> None:
    torch.save(payload, path)


def _forward_image_logits(router, batch, device):
    images = batch["images"].to(device, non_blocking=True)
    images = images.contiguous(memory_format=torch.channels_last)
    logits, base_logits, residual = router(images)
    return logits, base_logits, residual


def train_r11(force: bool = False, smoke: bool = False) -> dict:
    _seed_everything(SEED)
    runtime = load_r5_reference()["runtime"]
    freeze = load_r5_reference()["freeze"]
    if (R11_DIR / "training_status.json").is_file() and not force and not smoke:
        status = read_json(R11_DIR / "training_status.json")
        if status.get("complete"):
            return status
    train_data, valid_data, _ = build_h_datasets()
    train_loader, sampler = _make_loader(
        train_data, shuffle=True, workers=int(runtime["num_workers"]))
    valid_loader, _ = _make_loader(
        valid_data, shuffle=False, workers=int(runtime["num_workers"]))

    device = torch.device("cuda")
    router = build_r11_router(device)
    trainable = router.trainable_parameters()
    total_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=BASE_LR,
                                  weight_decay=WEIGHT_DECAY)
    accumulation = int(runtime["grad_accumulation"])
    max_epochs = 1 if smoke else MAX_EPOCHS
    updates_per_epoch = []
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch + 1)
        updates_per_epoch.append(math.ceil(len(sampler) / accumulation))
    total_updates = sum(updates_per_epoch)
    warmup_updates = sum(updates_per_epoch[:WARMUP_EPOCHS])
    config_payload = {
        "experiment": "R11_DiseaseSpecific_SparseFeatureRouting_512",
        "seed": SEED,
        "image_size": 512,
        "base": "H2_Single_Study_Loss_Weight/best.pth (fully frozen)",
        "base_checkpoint": "H2_Single_Study_Loss_Weight/best.pth",
        "architecture": "frozen H2 + 10 disease-specific channel/spatial Top-K "
                        "selectors + projection/pooling + scalar residual heads",
        "stage3_shape": [384, 32, 32],
        "channel_topk": 192,
        "spatial_topk": 512,
        "single_weight": 1.25,
        "loss": "H2 normalized weighted Study BCE (w_s=1.25 for single-image)",
        "optimizer": "AdamW",
        "learning_rate": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP,
        "warmup_epochs": WARMUP_EPOCHS,
        "max_epochs": MAX_EPOCHS,
        "batch_image_budget": int(runtime["image_budget"]),
        "grad_accumulation": accumulation,
        "bf16_backbone": True,
        "trainable_parameters": total_trainable,
        "branch_parameters": count_branch_parameters(router),
        "test_access": False,
        "early_stopping": EARLY,
    }
    write_json(R11_DIR / "config.json", config_payload)
    write_json(R11_DIR / "branch_parameters.json", count_branch_parameters(router))

    history = []
    per_epoch_class = []
    routing_diag = []
    gradient_diag = []
    best_auroc = -1.0
    best_epoch = 0
    no_improve = 0
    global_step = 0
    train_bce_history = []
    valid_bce_history = []
    start_epoch = 1
    last_path = R11_DIR / "last_checkpoint.pth"
    if last_path.is_file() and not force and not smoke:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        start_epoch = int(payload.get("epoch", 0)) + 1
        history = payload.get("history", [])
        per_epoch_class = payload.get("per_epoch_class_metrics", [])
        routing_diag = payload.get("routing_diagnostics", [])
        gradient_diag = payload.get("gradient_diagnostics", [])
        best_auroc = float(payload.get("best_macro_auroc", -1.0))
        best_epoch = int(payload.get("best_epoch", 0))
        train_bce_history = [float(row["train_bce"]) for row in history]
        valid_bce_history = [float(row["valid_bce"]) for row in history]
        global_step = sum(updates_per_epoch[:start_epoch - 1])
        if "router_state" in payload:
            router.load_state_dict(payload["router_state"], strict=True)
        else:
            router.base.load_state_dict(payload["base_state"], strict=True)
            router.branches.load_state_dict(payload["branch_state"], strict=True)
        print(f"[R11] resuming after epoch {start_epoch - 1}", flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        sampler.set_epoch(epoch)
        router.train()
        router.base.eval()
        iterator = iter(train_loader)
        started = time.perf_counter()
        bce_sum = 0.0
        eligible_studies = 0
        updates = 0
        selector_norms = {"channel": [], "spatial": []}
        residual_values = []
        data_wait = 0.0
        while True:
            window = []
            for _ in range(accumulation):
                fetch_start = time.perf_counter()
                try:
                    window.append(next(iterator))
                except StopIteration:
                    break
                data_wait += time.perf_counter() - fetch_start
            if not window:
                break
            lr = _scheduler_lr(global_step, total_updates, warmup_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, raw_bce, residual = _weighted_window_loss(
                router, window, device)
            loss.backward()
            grad_norms = {}
            for name, module in (("channel", "channel_selector"),
                                 ("spatial", "spatial_selector")):
                total = 0.0
                for branch in router.branches:
                    for parameter in getattr(branch, module).parameters():
                        if parameter.grad is not None:
                            total += float(parameter.grad.pow(2).sum())
                grad_norms[name] = math.sqrt(total)
            selector_norms["channel"].append(grad_norms["channel"])
            selector_norms["spatial"].append(grad_norms["spatial"])
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP,
                                            error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            updates += 1
            bce_sum += raw_bce * sum(
                int(batch["known"].any(dim=1).sum()) for batch in window)
            eligible_studies += sum(
                int(batch["known"].any(dim=1).sum()) for batch in window)
            residual_values.append(residual.abs().mean().item())
            if smoke and updates >= 4:
                break
        train_seconds = time.perf_counter() - started
        train_bce = bce_sum / max(eligible_studies, 1)
        validation = _evaluate(router, valid_loader, device)
        score = float(validation["macro_auroc"])
        improved = score > best_auroc + EARLY["epsilon"]
        if improved:
            best_auroc = score
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        train_bce_history.append(train_bce)
        valid_bce_history.append(float(validation["overall_bce"]))
        row = {
            "epoch": epoch,
            "train_bce": train_bce,
            "objective_loss": train_bce,
            "valid_bce": validation["overall_bce"],
            "macro_auroc": score,
            "macro_auprc": validation["macro_auprc"],
            "lr": lr,
            "updates": updates,
            "best_epoch": best_epoch,
            "best_macro_auroc": best_auroc,
            "no_improve": no_improve,
            "train_seconds": train_seconds,
            "data_wait_seconds": data_wait,
            "residual_abs_mean": validation["residual_stats"]["residual_abs_mean"],
            "residual_std": validation["residual_stats"]["residual_std"],
            "residual_to_base_ratio": validation["residual_stats"]["residual_to_base_ratio"],
        }
        stop_reason = None
        train_down = (
            len(train_bce_history) >= 4
            and all(train_bce_history[-i] < train_bce_history[-i - 1]
                    for i in range(1, 4)))
        valid_up = (
            len(valid_bce_history) >= 4
            and all(valid_bce_history[-i] > valid_bce_history[-i - 1]
                    for i in range(1, 4)))
        if (not smoke and train_down and valid_up
                and no_improve >= EARLY["auroc_no_improve"]):
            stop_reason = "three_condition_early_stop"
        row.update({"train_down_3": bool(train_down),
                    "valid_up_3": bool(valid_up),
                    "stop_reason": stop_reason})
        history.append(row)
        write_csv(R11_DIR / "training_history.csv", history)
        for name in LABEL_NAMES:
            item = validation["per_class"][name]
            per_epoch_class.append({
                "epoch": epoch,
                "disease": name,
                "auroc": item["auroc"],
                "auprc": item["auprc"],
                "bce": item["bce"],
                "positive_bce": item["positive_bce"],
                "negative_bce": item["negative_bce"],
                "branch_off_auroc": item["branch_off_auroc"],
                "branch_off_auprc": item["branch_off_auprc"],
            })
        write_csv(R11_DIR / "per_epoch_class_metrics.csv", per_epoch_class)
        mask_stats = _mask_diagnostics(
            validation["channel_masks"], validation["spatial_masks"])
        routing_diag.append({
            "epoch": epoch,
            "channel_retention": mask_stats["channel_retention"],
            "spatial_retention": mask_stats["spatial_retention"],
            "same_disease_channel_iou": mask_stats["same_disease_channel_iou"],
            "same_disease_spatial_iou": mask_stats["same_disease_spatial_iou"],
            "cross_disease_channel_iou": mask_stats["cross_disease_channel_iou"],
            "cross_disease_spatial_iou": mask_stats["cross_disease_spatial_iou"],
            "selector_channel_grad_norm": float(np.mean(selector_norms["channel"]))
            if selector_norms["channel"] else 0.0,
            "selector_spatial_grad_norm": float(np.mean(selector_norms["spatial"]))
            if selector_norms["spatial"] else 0.0,
            "residual_abs_mean": validation["residual_stats"]["residual_abs_mean"],
            "residual_std": validation["residual_stats"]["residual_std"],
        })
        write_csv(R11_DIR / "sparse_routing_diagnostics.csv", routing_diag)
        gradient_diag.append({
            "epoch": epoch,
            "channel_grad_norm": float(np.mean(selector_norms["channel"]))
            if selector_norms["channel"] else 0.0,
            "spatial_grad_norm": float(np.mean(selector_norms["spatial"]))
            if selector_norms["spatial"] else 0.0,
            "train_bce": train_bce,
            "updates": updates,
        })
        write_csv(R11_DIR / "gradient_diagnostics.csv", gradient_diag)
        payload = {
            "experiment": "R11",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_macro_auroc": best_auroc,
            "history": history,
            "per_epoch_class_metrics": per_epoch_class,
            "routing_diagnostics": routing_diag,
            "gradient_diagnostics": gradient_diag,
            "config": config_payload,
            "base_state": router.base.state_dict(),
            "branch_state": router.branches.state_dict(),
        }
        _save_checkpoint(R11_DIR / "last_checkpoint.pth", router, payload)
        if improved or not (R11_DIR / "best_checkpoint.pth").is_file():
            _save_checkpoint(R11_DIR / "best_checkpoint.pth", router, payload)
        print(
            f"[R11] epoch {epoch}/{max_epochs} train_bce={train_bce:.6f} "
            f"valid_bce={validation['overall_bce']:.6f} "
            f"macro_auroc={score:.6f} best={best_epoch} "
            f"stop={stop_reason} residual={validation['residual_stats']['residual_abs_mean']:.6f}",
            flush=True)
        if stop_reason is not None or smoke:
            break
    status = {
        "experiment": "R11",
        "completed_epochs": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "best_macro_auroc": best_auroc,
        "stop_reason": history[-1].get("stop_reason"),
        "complete": True,
        "test_access": False,
    }
    write_json(R11_DIR / "training_status.json", status)
    write_json(R11_DIR / "training_result.json", status)
    shutdown_loader(train_loader)
    shutdown_loader(valid_loader)
    return status


if __name__ == "__main__":
    print(json.dumps(train_r11(), ensure_ascii=True, indent=1, default=str))
