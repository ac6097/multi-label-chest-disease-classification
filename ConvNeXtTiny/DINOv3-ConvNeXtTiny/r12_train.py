"""Train the three R12 target modules independently on the frozen R12-Fixed."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from genopt_common import (  # noqa: E402
    EXPERIMENT_ROOT,
    LABEL_NAMES,
    SEED,
    TRAIN_CSV,
    VALID_CSV,
    build_h_datasets,
    class_bce_breakdown,
    load_r5_reference,
    read_json,
    study_logit_mean,
    write_json,
)
from dino_data import make_loader, shutdown_loader  # noqa: E402
from dino_feature_models import collate_masked_studies  # noqa: E402
from hybrid.common import write_csv  # noqa: E402
from stage_fusion_analysis import study_metrics  # noqa: E402
from r12_model import (  # noqa: E402
    R12Model, R11_DIR, TARGETS, TARGET_INDICES, VIEW_CATEGORIES,
    fixed_logits_from_r11,
)

R12_ROOT = EXPERIMENT_ROOT / "R12_DiseaseSpecific_ModularOptimization"
EXPERIMENTS = {
    "A": "R12A_Cardiomediastinum_Geometry",
    "B": "R12B_Pneumothorax_LocalEdge",
    "C": "R12C_NoFinding_LabelRelation",
}
BASE_LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_EPOCHS = 20
WARMUP_EPOCHS = 2


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_view_map(csv_path: Path) -> dict[str, int]:
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False,
                        na_filter=False)
    result = {}
    for _, row in frame.iterrows():
        value = str(row.get("ViewPosition", "")).strip().upper()
        result[str(row["Dicom_id"])] = VIEW_CATEGORIES.get(value, 3)
    return result


def _make_loader(dataset, shuffle: bool):
    loader, sampler, _ = make_loader(
        dataset, image_budget=8, shuffle=shuffle, seed=SEED, num_workers=4,
        pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    return loader, sampler


def _batch_views(batch, view_map, device):
    ids = []
    for study_ids in batch["image_ids"]:
        ids.extend([str(value) for value in study_ids])
    return torch.tensor([view_map.get(value, 3) for value in ids],
                        dtype=torch.long, device=device)


def _target_loss(model, window, target_index, view_map, device):
    weight_total = 0.0
    eligible_total = 0
    for batch in window:
        counts = batch["image_counts"].to(device)
        known = batch["known"].to(device)
        eligible = known[:, target_index]
        weights = torch.where(counts == 1, 1.25, 1.0)
        weight_total += float(weights[eligible].sum().cpu())
        eligible_total += int(eligible.sum())
    loss_total = torch.zeros((), device=device)
    raw_sum = 0.0
    residual_mean = 0.0
    for batch in window:
        images = batch["images"].to(device, non_blocking=True)
        images = images.contiguous(memory_format=torch.channels_last)
        view_ids = _batch_views(batch, view_map, device)
        logits, _, parts = model(images, view_ids)
        study_logits = study_logit_mean(
            logits, batch["study_index"].to(device), batch["num_studies"],
            batch["image_counts"].to(device))
        labels = batch["labels"].to(device)[:, target_index]
        known = batch["known"].to(device)[:, target_index]
        counts = batch["image_counts"].to(device)
        per_study = F.binary_cross_entropy_with_logits(
            study_logits[:, target_index], labels, reduction="none")
        weights = torch.where(counts == 1, 1.25, 1.0)
        loss = (per_study[known] * weights[known]).sum() / max(weight_total, 1e-9)
        loss_total = loss_total + loss
        raw_sum += float(per_study[known].sum().detach().float().cpu())
    return loss_total, raw_sum / max(eligible_total, 1)


def _evaluate_target(model, loader, target_index, view_map, device) -> dict:
    model.eval()
    study_logits, labels, subjects = [], [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        images = images.contiguous(memory_format=torch.channels_last)
        view_ids = _batch_views(batch, view_map, device)
        with torch.no_grad():
            logits, _, _ = model(images, view_ids)
            study = study_logit_mean(
                logits, batch["study_index"].to(device), batch["num_studies"],
                batch["image_counts"].to(device))
        study_logits.append(study.float().cpu())
        labels.append(batch["labels"])
        subjects.extend([str(value) for value in batch["subject_id"]])
    logits = torch.cat(study_logits, 0).numpy().astype(np.float64)
    labels_all = torch.cat(labels, 0).numpy().astype(np.int64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    metrics = study_metrics(labels_all, probability)
    class_bce = class_bce_breakdown(logits, labels_all)
    rows = []
    for index, name in enumerate(LABEL_NAMES):
        rows.append({
            "disease": name,
            "auroc": metrics["per_class"][name]["auroc"],
            "auprc": metrics["per_class"][name]["auprc"],
            "bce": class_bce[name]["bce"],
            "positive_bce": class_bce[name]["positive_bce"],
            "negative_bce": class_bce[name]["negative_bce"],
        })
    return {
        "macro_auroc": float(metrics["macro_auroc"]),
        "macro_auprc": float(metrics["macro_auprc"]),
        "target_auroc": float(metrics["per_class"][LABEL_NAMES[target_index]]["auroc"]),
        "target_auprc": float(metrics["per_class"][LABEL_NAMES[target_index]]["auprc"]),
        "target_bce": float(class_bce[LABEL_NAMES[target_index]]["bce"]),
        "target_positive_bce": float(class_bce[LABEL_NAMES[target_index]]["positive_bce"]),
        "target_negative_bce": float(class_bce[LABEL_NAMES[target_index]]["negative_bce"]),
        "overall_bce": float(np.mean([class_bce[name]["bce"] for name in LABEL_NAMES])),
        "per_class": rows,
        "subjects": subjects,
    }


def _bias_only(target_index: int, train_logits, train_labels, valid_logits,
               valid_labels) -> dict:
    """Optimal train-only scalar bias; AUROC/AUPRC are invariant to a shift."""
    logits = np.asarray(train_logits, dtype=np.float64)[:, target_index]
    labels = np.asarray(train_labels, dtype=np.int64)[:, target_index]
    lo, hi = -20.0, 20.0
    for _ in range(100):
        middle = 0.5 * (lo + hi)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits + middle, -80, 80)))
        # derivative sign of BCE wrt bias
        gradient = np.mean(probability - labels)
        if gradient > 0:
            hi = middle
        else:
            lo = middle
    bias = 0.5 * (lo + hi)
    valid_probability = 1.0 / (1.0 + np.exp(-np.clip(
        np.asarray(valid_logits, dtype=np.float64)[:, target_index] + bias,
        -80, 80)))
    metrics = study_metrics(np.asarray(valid_labels, dtype=np.int64),
                            np.repeat(valid_probability[:, None], 10, axis=1))
    bce = class_bce_breakdown(np.asarray(valid_logits)[:, target_index][:, None],
                              np.asarray(valid_labels)[:, target_index][:, None])
    return {"bias": float(bias)}


def train_target(key: str, force: bool = False) -> dict:
    name = EXPERIMENTS[key]
    run_dir = R12_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "training_status.json"
    if status_path.is_file() and not force:
        status = read_json(status_path)
        if status.get("complete"):
            return status
    _seed_everything(SEED)
    device = torch.device("cuda")
    train_data, valid_data, _ = build_h_datasets()
    train_loader, sampler = _make_loader(train_data, True)
    valid_loader, _ = _make_loader(valid_data, False)
    view_train = _load_view_map(TRAIN_CSV)
    view_valid = _load_view_map(VALID_CSV)
    model = R12Model(enable_a=(key == "A"), enable_b=(key == "B"),
                     enable_c=(key == "C")).to(device)
    model.train()
    module = {"A": model.module_a, "B": model.module_b, "C": model.module_c}[key]
    parameters = [parameter for parameter in module.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=BASE_LR,
                                  weight_decay=WEIGHT_DECAY)
    target_index = TARGET_INDICES[key]
    updates_per_epoch = [math.ceil(len(sampler) / 4)
                         for _ in range(MAX_EPOCHS)]
    total_updates = sum(updates_per_epoch)
    warmup_updates = sum(updates_per_epoch[:WARMUP_EPOCHS])
    history = []
    epoch_rows = []
    best_auroc = -1.0
    best_epoch = 0
    no_improve = 0
    global_step = 0
    start_epoch = 1
    last_path = run_dir / "last_checkpoint.pth"
    if last_path.is_file() and not force:
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        start_epoch = int(payload.get("epoch", 0)) + 1
        history = payload.get("history", [])
        epoch_rows = payload.get("per_epoch_target_metrics", [])
        best_auroc = float(payload.get("best_target_auroc", -1.0))
        best_epoch = int(payload.get("best_epoch", 0))
        no_improve = int(payload.get("no_improve", 0))
        module.load_state_dict(payload["module_state"], strict=True)
        global_step = sum(updates_per_epoch[:start_epoch - 1])
        print(f"[R12-{key}] resuming after epoch {start_epoch - 1}", flush=True)
    train_bce_history, valid_bce_history = [], []
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        sampler.set_epoch(epoch)
        model.train()
        iterator = iter(train_loader)
        started = time.perf_counter()
        updates = 0
        bce_sum = 0.0
        eligible = 0
        while True:
            window = []
            for _ in range(4):
                try:
                    window.append(next(iterator))
                except StopIteration:
                    break
            if not window:
                break
            if global_step < warmup_updates:
                lr = BASE_LR * (global_step + 1) / max(warmup_updates, 1)
            else:
                progress = (global_step - warmup_updates) / max(
                    total_updates - warmup_updates, 1)
                lr = BASE_LR * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, raw = _target_loss(model, window, target_index, view_train,
                                     device)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            global_step += 1
            updates += 1
            bce_sum += raw * sum(int(batch["known"][:, target_index].sum())
                                 for batch in window)
            eligible += sum(int(batch["known"][:, target_index].sum())
                            for batch in window)
        train_bce = bce_sum / max(eligible, 1)
        validation = _evaluate_target(model, valid_loader, target_index,
                                      view_valid, device)
        improved = validation["target_auroc"] > best_auroc
        if improved:
            best_auroc = validation["target_auroc"]
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        train_bce_history.append(train_bce)
        valid_bce_history.append(validation["target_bce"])
        row = {
            "epoch": epoch,
            "train_target_bce": train_bce,
            "valid_target_bce": validation["target_bce"],
            "target_auroc": validation["target_auroc"],
            "target_auprc": validation["target_auprc"],
            "target_positive_bce": validation["target_positive_bce"],
            "target_negative_bce": validation["target_negative_bce"],
            "macro_auroc": validation["macro_auroc"],
            "macro_auprc": validation["macro_auprc"],
            "lr": lr,
            "updates": updates,
            "best_epoch": best_epoch,
            "best_target_auroc": best_auroc,
            "no_improve": no_improve,
        }
        stop_reason = None
        train_down = (len(train_bce_history) >= 4 and all(
            train_bce_history[-i] < train_bce_history[-i - 1]
            for i in range(1, 4)))
        valid_up = (len(valid_bce_history) >= 4 and all(
            valid_bce_history[-i] > valid_bce_history[-i - 1]
            for i in range(1, 4)))
        if train_down and valid_up and no_improve >= 3:
            stop_reason = "three_condition_early_stop"
        row["stop_reason"] = stop_reason
        history.append(row)
        write_csv(run_dir / "training_history.csv", history)
        for item in validation["per_class"]:
            epoch_rows.append({
                "epoch": epoch,
                "disease": item["disease"],
                "auroc": item["auroc"],
                "auprc": item["auprc"],
                "bce": item["bce"],
                "positive_bce": item["positive_bce"],
                "negative_bce": item["negative_bce"],
            })
        write_csv(run_dir / "per_epoch_target_metrics.csv", epoch_rows)
        payload = {
            "experiment": f"R12-{key}",
            "target": TARGETS[key],
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_target_auroc": best_auroc,
            "no_improve": no_improve,
            "history": history,
            "per_epoch_target_metrics": epoch_rows,
            "module_state": module.state_dict(),
            "fixed_reference": fixed_logits_from_r11(),
            "config": {"learning_rate": BASE_LR, "weight_decay": WEIGHT_DECAY,
                       "max_epochs": MAX_EPOCHS, "seed": SEED,
                       "target": TARGETS[key]},
        }
        torch.save(payload, run_dir / "last_checkpoint.pth")
        if improved or not (run_dir / "best_checkpoint.pth").is_file():
            torch.save(payload, run_dir / "best_checkpoint.pth")
        print(f"[R12-{key}] epoch {epoch} train={train_bce:.6f} "
              f"valid={validation['target_bce']:.6f} "
              f"auroc={validation['target_auroc']:.6f} best={best_epoch} "
              f"stop={stop_reason}", flush=True)
        if stop_reason is not None:
            break
    status = {
        "experiment": f"R12-{key}",
        "target": TARGETS[key],
        "completed_epochs": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "best_target_auroc": best_auroc,
        "stop_reason": history[-1].get("stop_reason"),
        "complete": True,
        "test_access": False,
    }
    write_json(status_path, status)
    shutdown_loader(train_loader)
    shutdown_loader(valid_loader)
    del model
    torch.cuda.empty_cache()
    return status


def run_series() -> dict:
    results = {}
    for key in ("A", "B", "C"):
        results[key] = train_target(key)
    return results


if __name__ == "__main__":
    print(json.dumps(run_series(), ensure_ascii=True, indent=1, default=str))
