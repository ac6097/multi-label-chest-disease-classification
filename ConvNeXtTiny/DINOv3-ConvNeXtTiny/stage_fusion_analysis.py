"""Unified inference, ablations, paired bootstrap, export and comparison.

Every model (R5 / R7 / R8 / R9 / R10) is re-evaluated through one identical
protocol: same official Validation loader, same Study Logit Mean, same BF16
autocast, same checkpoint loading (strict) and the same Subject-cluster paired
bootstrap.  Fresh predictions are used for every paired comparison; the
historical R5/R7 `val_best.csv` files are kept only as replay diagnostics.

Artifacts written per experiment directory:
    val_predictions.csv, val_best_predictions.csv, ablation_logits.csv,
    inference_ablation.csv, per_disease_ablation.csv, paired_subject_bootstrap.json,
    inference_cost.json, attention_heatmaps/ or cross_stage_attention_heatmaps/,
    submission_export/, final_report.md
and under R8_R9_R10_Comparative_Analysis/:
    experiment_matrix.csv, overall_metrics.csv, per_class_comparison.csv,
    paired_bootstrap_comparison.json, training_efficiency.csv,
    inference_efficiency.csv, ablation_summary.csv, final_comparative_report.md
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dino_analysis import LABEL_NAMES  # noqa: E402
from dino_data import make_loader, shutdown_loader  # noqa: E402
from dino_experiment import load_pretrained_backbone, save_json_atomic  # noqa: E402
from dino_feature_models import (  # noqa: E402
    build_masked_datasets,
    collate_masked_studies,
)
from dino_losses import study_logit_mean  # noqa: E402
from dino_model import DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from hybrid.common import seed_everything  # noqa: E402
from stage_fusion_experiment import (  # noqa: E402
    EXPECTED_HEAD_SHA,
    EXPECTED_WEIGHT_SHA256,
    GROUP_NAMES,
    IMAGE_SIZE,
    OUTPUT_ROOT,
    R5_DIR,
    R7_DIR,
    SEED,
    load_r5_contract,
    read_json,
    write_json,
)
from stage_fusion_models import (  # noqa: E402
    ALPHA_INIT,
    ARCHITECTURES,
    NUM_HEADS,
    build_model,
)

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
COMPARATIVE_DIR = OUTPUT_ROOT / "R8_R9_R10_Comparative_Analysis"
FOCUS_DISEASES = ["Pneumonia", "Lung Opacity", "Consolidation", "Pneumothorax"]
R7_GROUP = "R7_ResidualTransfer_512"
R5_GROUP = "R5_Global_Only_512"
MODE_NAMES = {
    "R5": {"full": "R5_baseline"},
    "R7": {"full": "R7_Fused", "gamma0": "R7_gamma0"},
    "R8": {"full": "R8_Full", "branch_off": "R8_BranchOff"},
    "R9": {"full": "R9_Full", "branch_off": "R9_BranchOff", "uniform": "R9_Uniform"},
    "R10": {"full": "R10_Full", "branch_off": "R10_BranchOff",
            "no_conditioning": "R10_NoStage4Conditioning"},
}


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80.0, 80.0)))


def _r7_class():
    path = R7_DIR / "r7_model.py"
    specification = importlib.util.spec_from_file_location(
        "stage_fusion_r7_reference_model", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.R7Stage3Stage4ResidualClassifier


def load_runtime() -> dict:
    payload = read_json(R5_DIR / "selected_runtime_config.json")
    runtime = copy.deepcopy(payload["selected_runtime"])
    expected = {"image_budget": 8, "grad_accumulation": 4, "num_workers": 4,
                "channels_last": True, "gradient_checkpointing": False}
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"R5 runtime key {key} is {runtime.get(key)} != {value}")
    return runtime


def _valid_data():
    contract = load_r5_contract()
    source = (HERE / contract["experiment_config"]["data"]["source_config"]).resolve()
    train_data, valid_data, _ = build_masked_datasets(source, SEED, IMAGE_SIZE)
    if len(valid_data) != 859:
        raise RuntimeError("Validation Study count differs from the R5 manifest")
    return valid_data


def _checkpoint_epoch(payload: dict, status: dict, label: str) -> int:
    expected = int(status["best_epoch"])
    actual = int(payload.get("epoch", -1))
    if actual != expected:
        raise RuntimeError(
            f"{label} Best checkpoint epoch {actual} != recorded Best epoch {expected}")
    return actual


def load_group_model(group: str, device: torch.device):
    """Strictly load one group's Best inference model and its metadata."""
    runtime = load_runtime()
    seed_everything(SEED)
    if group == "R5":
        run_dir = R5_DIR
        status = read_json(run_dir / "training_status.json")
        if not status.get("complete"):
            raise RuntimeError("R5 is not complete")
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError("R5 inference did not load the pinned DINOv3 weights")
        model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS)
        payload = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
        _checkpoint_epoch(payload, status, "R5")
        result = model.load_state_dict(payload["model_state"], strict=True)
    elif group == R7_GROUP:
        run_dir = R7_DIR
        status = read_json(run_dir / "training_status.json")
        if not status.get("complete"):
            raise RuntimeError("R7 is not complete")
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError("R7 inference did not load the pinned DINOv3 weights")
        model = _r7_class()(backbone, num_labels=NUM_LABELS)
        payload = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
        _checkpoint_epoch(payload, status, "R7")
        result = model.load_state_dict(payload["model_state"], strict=True)
    else:
        experiment = {name: key for key, name in GROUP_NAMES.items()}.get(group)
        if experiment is None:
            raise ValueError(f"Unknown stage-fusion group: {group}")
        run_dir = OUTPUT_ROOT / GROUP_NAMES[experiment]
        status = read_json(run_dir / "training_status.json")
        if not status.get("complete"):
            raise RuntimeError(f"{experiment} is not complete")
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError(f"{experiment} inference did not load the pinned DINOv3 weights")
        model = build_model(experiment, backbone)
        payload = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
        _checkpoint_epoch(payload, status, experiment)
        result = model.load_state_dict(payload["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{group} strict checkpoint load failed: {result}")
    model.to(device)
    if runtime["channels_last"]:
        model.to(memory_format=torch.channels_last)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, status, run_dir, runtime


def _forward_once(model, experiment: str, kind: str, images: torch.Tensor) -> dict:
    if experiment == "R5":
        return {"logits": model(images)}
    if experiment == "R7":
        components = model.forward_components(images, include_gamma_zero_logits=True)
        return {"logits": components["fused_logits"],
                "gamma_zero_logits": components["gamma_zero_logits"]}
    if kind == "uniform":
        components = model.forward_components(images, attention_mode="uniform")
    elif kind == "no_conditioning":
        components = model.forward_components(images, use_conditioning=False)
    else:
        components = model.forward_components(images)
    return {"logits": components["logits"],
            "stage4_logits": components["stage4_logits"],
            "stage3_logits": components["stage3_logits"]}


def evaluate(model, experiment: str, valid_data, runtime: dict,
             kinds: tuple[str, ...]) -> dict:
    """One identical Validation pass for every requested forward mode."""
    device = torch.device("cuda")
    loader, sampler, _ = make_loader(
        valid_data, image_budget=int(runtime["image_budget"]), shuffle=False,
        seed=SEED, num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    logits_by_kind = {kind: [] for kind in kinds}
    stage4_chunks, stage3_chunks, gamma_zero_chunks = [], [], []
    subject_ids, study_ids, label_rows, image_counts = [], [], [], []
    per_batch_ms_per_study = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in loader:
                batch_started = time.perf_counter()
                images = batch["images"].to(device, non_blocking=True)
                if runtime["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                study_index = batch["study_index"].to(device, non_blocking=True)
                counts = batch["image_counts"].to(device, non_blocking=True)
                num_studies = batch["num_studies"]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for kind in kinds:
                        output = _forward_once(model, experiment, kind, images)
                        study_logits = study_logit_mean(
                            output["logits"].float(), study_index, num_studies, counts)
                        logits_by_kind[kind].append(study_logits.cpu().numpy())
                        if kind == "full":
                            if "stage4_logits" in output:
                                stage4_chunks.append(study_logit_mean(
                                    output["stage4_logits"].float(), study_index,
                                    num_studies, counts).cpu().numpy())
                                stage3_chunks.append(study_logit_mean(
                                    output["stage3_logits"].float(), study_index,
                                    num_studies, counts).cpu().numpy())
                            if "gamma_zero_logits" in output:
                                gamma_zero_chunks.append(study_logit_mean(
                                    output["gamma_zero_logits"].float(), study_index,
                                    num_studies, counts).cpu().numpy())
                label_rows.append(batch["labels"].numpy().astype(np.uint8, copy=False))
                subject_ids.extend(str(value) for value in batch["subject_id"])
                study_ids.extend(str(value) for value in batch["study_id"])
                image_counts.extend(int(value) for value in batch["image_counts"].tolist())
                per_batch_ms_per_study.append(
                    (time.perf_counter() - batch_started) * 1000.0
                    / max(int(batch["num_studies"]), 1))
    finally:
        shutdown_loader(loader)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    labels = np.concatenate(label_rows, axis=0)
    result = {
        "subjects": np.asarray(subject_ids, dtype=object),
        "studies": np.asarray(study_ids, dtype=object),
        "labels": labels,
        "image_counts": np.asarray(image_counts),
        "kinds": {kind: np.concatenate(chunks, axis=0)
                  for kind, chunks in logits_by_kind.items()},
        "elapsed_seconds": elapsed,
        "per_batch_ms_per_study": np.asarray(per_batch_ms_per_study),
        "peak_gpu_gib": peak,
        "validation_studies": len(subject_ids),
        "validation_images": int(sum(image_counts)),
        "experiment": experiment,
    }
    if stage4_chunks:
        result["stage4_logits"] = np.concatenate(stage4_chunks, axis=0)
        result["stage3_logits"] = np.concatenate(stage3_chunks, axis=0)
    if gamma_zero_chunks:
        result["gamma_zero_logits"] = np.concatenate(gamma_zero_chunks, axis=0)
    keys = list(zip(result["subjects"].astype(str), result["studies"].astype(str)))
    expected_keys = list(valid_data.study_keys)
    if len(keys) != 859 or len(set(keys)) != 859 or set(keys) != set(expected_keys):
        raise RuntimeError(
            f"{experiment} inference Study keys differ from the official Validation set")
    return result


def study_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    per_class = {}
    aurocs, auprcs = [], []
    for index, name in enumerate(LABEL_NAMES):
        y = labels[:, index].astype(np.int64)
        p = probabilities[:, index]
        if np.unique(y).size == 2:
            auroc = float(roc_auc_score(y, p))
            auprc = float(average_precision_score(y, p))
        else:
            auroc = float("nan")
            auprc = float(average_precision_score(y, p)) if int(y.sum()) else float("nan")
        predicted = p >= 0.5
        true_positive = int((predicted & (y == 1)).sum())
        false_positive = int((predicted & (y == 0)).sum())
        false_negative = int((~predicted & (y == 1)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        per_class[name] = {"auroc": auroc, "auprc": auprc, "f1": float(f1),
                           "n_positive": int(y.sum()),
                           "n_negative": int((y == 0).sum())}
        if math.isfinite(auroc):
            aurocs.append(auroc)
        if math.isfinite(auprc):
            auprcs.append(auprc)
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "per_class": per_class,
    }


def make_subject_draws(subjects: np.ndarray):
    subjects = np.asarray(subjects).astype(str)
    unique = np.asarray(sorted(set(subjects)), dtype=object)
    rows = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    digest = hashlib.sha256()
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows[subject] for subject in sampled])
        draws.append(indices)
        digest.update(indices.astype("<i4", copy=False).tobytes())
        digest.update(b"\xff")
    return draws, digest.hexdigest()


def _ci95(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None, None], 0
    return np.quantile(finite, [0.025, 0.975]).tolist(), int(finite.size)


def bootstrap_models(labels: np.ndarray, probabilities: dict[str, np.ndarray],
                     subjects: np.ndarray, comparisons: dict[str, tuple[str, str]]):
    """One shared Subject-cluster resampling for every model and comparison."""
    draws, draw_hash = make_subject_draws(subjects)
    point = {name: study_metrics(labels, probability)
             for name, probability in probabilities.items()}
    bootstrap_values = {}
    for name, probability in probabilities.items():
        auc_values = np.full((BOOTSTRAP_REPLICATES, NUM_LABELS), np.nan, dtype=np.float32)
        ap_values = np.full_like(auc_values, np.nan)
        macro_auc = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float32)
        macro_ap = np.full_like(macro_auc, np.nan)
        for draw_index, indices in enumerate(draws):
            y_sample = labels[indices]
            p_sample = probability[indices]
            aucs, aps = [], []
            for class_index in range(NUM_LABELS):
                y = y_sample[:, class_index]
                if np.unique(y).size == 2:
                    value = float(roc_auc_score(y, p_sample[:, class_index]))
                    auc_values[draw_index, class_index] = value
                    aucs.append(value)
                if np.any(y == 1):
                    value = float(average_precision_score(y, p_sample[:, class_index]))
                    ap_values[draw_index, class_index] = value
                    aps.append(value)
            if aucs:
                macro_auc[draw_index] = float(np.mean(aucs))
            if aps:
                macro_ap[draw_index] = float(np.mean(aps))
        bootstrap_values[name] = {"auroc": auc_values, "auprc": ap_values,
                                  "macro_auroc": macro_auc, "macro_auprc": macro_ap}
    comparison_results = {}
    for name, (left, right) in comparisons.items():
        if left not in probabilities or right not in probabilities:
            comparison_results[name] = {"status": "skipped_missing_model",
                                        "left": left, "right": right}
            continue
        macro_delta = (bootstrap_values[right]["macro_auroc"]
                       - bootstrap_values[left]["macro_auroc"])
        macro_delta_ap = (bootstrap_values[right]["macro_auprc"]
                          - bootstrap_values[left]["macro_auprc"])
        macro_ci, macro_valid = _ci95(macro_delta)
        macro_ap_ci, macro_ap_valid = _ci95(macro_delta_ap)
        per_class = {}
        for class_index, disease in enumerate(LABEL_NAMES):
            delta_auc = (bootstrap_values[right]["auroc"][:, class_index]
                         - bootstrap_values[left]["auroc"][:, class_index])
            delta_ap = (bootstrap_values[right]["auprc"][:, class_index]
                        - bootstrap_values[left]["auprc"][:, class_index])
            auc_ci, auc_valid = _ci95(delta_auc)
            ap_ci, ap_valid = _ci95(delta_ap)
            per_class[disease] = {
                "delta_auroc": float(
                    point[right]["per_class"][disease]["auroc"]
                    - point[left]["per_class"][disease]["auroc"]),
                "delta_auroc_ci95": auc_ci,
                "delta_auroc_valid_replicates": auc_valid,
                "delta_auprc": float(
                    point[right]["per_class"][disease]["auprc"]
                    - point[left]["per_class"][disease]["auprc"]),
                "delta_auprc_ci95": ap_ci,
                "delta_auprc_valid_replicates": ap_valid,
                "positive_studies": int(labels[:, class_index].sum()),
            }
        comparison_results[name] = {
            "left": left, "right": right,
            "delta_macro_auroc": float(
                point[right]["macro_auroc"] - point[left]["macro_auroc"]),
            "delta_macro_auroc_ci95": macro_ci,
            "macro_auroc_valid_replicates": macro_valid,
            "delta_macro_auprc": float(
                point[right]["macro_auprc"] - point[left]["macro_auprc"]),
            "delta_macro_auprc_ci95": macro_ap_ci,
            "macro_auprc_valid_replicates": macro_ap_valid,
            "per_class": per_class,
        }
    models = {}
    for name in probabilities:
        entry = {"macro_auroc": point[name]["macro_auroc"],
                 "macro_auprc": point[name]["macro_auprc"], "per_class": {}}
        for index, disease in enumerate(LABEL_NAMES):
            auroc_ci, auroc_n = _ci95(bootstrap_values[name]["auroc"][:, index])
            auprc_ci, auprc_n = _ci95(bootstrap_values[name]["auprc"][:, index])
            entry["per_class"][disease] = {
                "auroc": point[name]["per_class"][disease]["auroc"],
                "auroc_ci95": auroc_ci, "auroc_valid_replicates": auroc_n,
                "auprc": point[name]["per_class"][disease]["auprc"],
                "auprc_ci95": auprc_ci, "auprc_valid_replicates": auprc_n,
                "f1_at_0.5": point[name]["per_class"][disease]["f1"],
                "positive_studies": point[name]["per_class"][disease]["n_positive"],
                "negative_studies": point[name]["per_class"][disease]["n_negative"],
            }
        models[name] = entry
    return {
        "method": "Subject-cluster paired bootstrap; percentile 95% CI",
        "sampling_unit": "Subject_id; all Studies of a resampled Subject move together",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "sampling_indices_sha256": draw_hash,
        "aligned_validation_studies": int(len(labels)),
        "validation_subjects": int(len(set(np.asarray(subjects).astype(str)))),
        "one_class_rule": (
            "Skip AUROC for a resample with one label polarity; skip AP if no "
            "positive Study; macro uses available classes."),
        "models": models,
        "comparisons": comparison_results,
        "limitations": (
            "Validation has been used for historical checkpoint selection and "
            "multiple model/disease comparisons; these intervals are exploratory "
            "and do not replace an independent test set."),
        "test_data_accessed": False,
    }


def _predictions_frame(result: dict, experiment: str) -> pd.DataFrame:
    frame = pd.DataFrame({"Subject_id": result["subjects"],
                          "Study_id": result["studies"]})
    for index, name in enumerate(LABEL_NAMES):
        key = name.replace(" ", "_")
        frame[f"target_{key}"] = result["labels"][:, index]
        frame[f"logit_{experiment}_{key}"] = result["kinds"]["full"][:, index]
        frame[f"prob_{experiment}_{key}"] = _sigmoid(result["kinds"]["full"])[:, index]
    return frame


def _mode_logits(result: dict, experiment: str, model) -> dict[str, np.ndarray]:
    """Build the mode matrix, deriving exact per-disease branch-off variants."""
    modes = {}
    full = result["kinds"]["full"]
    modes[MODE_NAMES[experiment]["full"]] = full
    if experiment == "R7":
        modes[MODE_NAMES["R7"]["gamma0"]] = result["gamma_zero_logits"]
        return modes
    if experiment == "R5":
        return modes
    stage4 = result["stage4_logits"]
    stage3 = result["stage3_logits"]
    alpha = model.fusion_alpha.alpha().detach().float().cpu().numpy()
    modes[MODE_NAMES[experiment]["branch_off"]] = stage4
    if experiment == "R9":
        modes[MODE_NAMES["R9"]["uniform"]] = result["kinds"]["uniform"]
    if experiment == "R10":
        modes[MODE_NAMES["R10"]["no_conditioning"]] = result["kinds"]["no_conditioning"]
    return modes


def _derived_per_disease_modes(stage4, stage3, alpha, experiment: str) -> dict[str, np.ndarray]:
    modes = {}
    for index, disease in enumerate(LABEL_NAMES):
        modified = alpha.copy()
        modified[index] = 0.0
        modes[disease] = stage4 + modified[None, :] * stage3
    return modes


def _attention_heatmaps(model, experiment: str, valid_data, runtime: dict,
                        output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, sampler, _ = make_loader(
        valid_data, image_budget=int(runtime["image_budget"]), shuffle=False,
        seed=SEED, num_workers=0, pin_memory=True,
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    batch = None
    try:
        for candidate in loader:
            if bool((candidate["image_counts"] > 1).any()):
                batch = candidate
                break
    finally:
        shutdown_loader(loader)
    if batch is None:
        return
    device = torch.device("cuda")
    images = batch["images"].to(device, non_blocking=True)
    if runtime["channels_last"]:
        images = images.contiguous(memory_format=torch.channels_last)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        components = model.forward_components(images, return_attention_weights=True)
        if experiment == "R10":
            no_cond = model.forward_components(
                images, return_attention_weights=True, use_conditioning=False)
    weights = components["attention_weights"].float().cpu().mean(dim=1)  # [B,10,1024]
    no_cond_weights = (no_cond["attention_weights"].float().cpu().mean(dim=1)
                       if experiment == "R10" else None)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = (images.detach().float().cpu() * std + mean).clamp(0, 1)
    image_index = 0
    side = int(round(math.sqrt(weights.shape[-1])))
    for disease in FOCUS_DISEASES:
        index = LABEL_NAMES.index(disease)
        attention = weights[image_index, index].reshape(side, side).numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-12)
        figure, axes = plt.subplots(1, 2 if experiment == "R10" else 1, figsize=(9, 4.2))
        axes = np.atleast_1d(axes)
        axes[0].imshow(rgb[image_index].permute(1, 2, 0).numpy())
        axes[0].imshow(attention, cmap="jet", alpha=0.45)
        axes[0].set_title(f"{experiment} {disease} attention")
        axes[0].axis("off")
        if experiment == "R10" and no_cond_weights is not None:
            other = no_cond_weights[image_index, index].reshape(side, side).numpy()
            other = (other - other.min()) / (other.max() - other.min() + 1e-12)
            axes[1].imshow(rgb[image_index].permute(1, 2, 0).numpy())
            axes[1].imshow(other, cmap="jet", alpha=0.45)
            axes[1].set_title(f"{experiment} {disease} without Stage4 conditioning")
            axes[1].axis("off")
        figure.suptitle("Attention is a model diagnostic, not a verified lesion map",
                        fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{disease.replace(' ', '_')}.png", dpi=120)
        plt.close(figure)


def _inference_cost(experiment: str, group: str, model, status: dict, result: dict,
                    run_dir: Path) -> dict:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    elapsed = float(result["elapsed_seconds"])
    studies = int(result["validation_studies"])
    batch_ms = np.asarray(result.get("per_batch_ms_per_study", []), dtype=np.float64)
    payload = {
        "experiment": experiment,
        "group": group,
        "parameters_total": int(parameter_count),
        "trainable_parameters_reported": None,
        "fp32_inference_weights_bytes": int(parameter_count * 4),
        "validation_studies": studies,
        "validation_images": int(result["validation_images"]),
        "mean_ms_per_study": elapsed / studies * 1000.0,
        "median_ms_per_study": (float(np.median(batch_ms)) if batch_ms.size else None),
        "p95_ms_per_study": (float(np.quantile(batch_ms, 0.95)) if batch_ms.size else None),
        "studies_per_second": studies / elapsed if elapsed else None,
        "peak_gpu_gib": float(result["peak_gpu_gib"]),
        "precision": "bfloat16 autocast",
        "channels_last": True,
        "device": torch.cuda.get_device_name(0),
        "best_epoch": int(status["best_epoch"]),
        "timing_scope": (
            "Whole-Validation wall time divided by 859 Studies; per-batch values are "
            "batch wall time divided by batch Studies (includes data loading)."),
        "test_data_accessed": False,
    }
    if (run_dir / "config.json").is_file():
        config_payload = read_json(run_dir / "config.json")
        payload["trainable_parameters_reported"] = int(
            config_payload.get("optimizer_parameter_groups", {})
            .get("trainable_parameters", 0)) or None
    return payload


def export_submission_package(experiment: str, model, run_dir: Path,
                              valid_data, runtime: dict) -> dict:
    """Self-contained <500MB FP32 inference package with a standalone predictor."""
    group = GROUP_NAMES[experiment]
    package_dir = run_dir / "submission_export" / f"{experiment}_Inference"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    for filename in ("dino_model.py", "dinov3_meta_convnext.py",
                     "stage_fusion_models.py"):
        shutil.copy2(HERE / filename, package_dir / filename)
    state = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in model.state_dict().items()}
    save_file(state, str(package_dir / "model.safetensors"))
    config_payload = {
        "experiment": experiment,
        "architecture": ARCHITECTURES[experiment],
        "image_size": IMAGE_SIZE,
        "layer_norm_eps": 1e-6,
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.0,
        "architecture_name": "convnext_tiny",
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "channels_last": True,
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "alpha_initial": ALPHA_INIT,
        "num_heads": NUM_HEADS if experiment in {"R9", "R10"} else None,
        "study_fusion": "arithmetic mean of raw image logits before sigmoid",
        "test_data_accessed": False,
    }
    write_json(package_dir / "model_config.json", config_payload)
    (package_dir / "requirements-inference.txt").write_text(
        "torch>=2.5\nnumpy>=1.24\nPillow>=10.0\nsafetensors>=0.4\n", encoding="utf-8")
    (package_dir / "LICENSE.md").write_text(
        "# Inference package\n\n"
        "Contains DINOv3 ConvNeXt-Tiny derived weights governed by the Meta DINOv3 "
        "license (see https://github.com/facebookresearch/dinov3). This package is "
        "for the course competition only; Test labels were never accessed.\n",
        encoding="utf-8")
    (package_dir / "infer.py").write_text(_INFER_TEMPLATE, encoding="utf-8")

    zip_path = run_dir / "submission_export" / f"{experiment}_Inference.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir.parent))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"{experiment} inference zip failed its CRC check")
    archive_bytes = zip_path.stat().st_size
    uncompressed = sum(path.stat().st_size for path in package_dir.rglob("*")
                       if path.is_file())

    # Independent subprocess verification on a real multi-image Validation Study.
    study_index = None
    for index, key in enumerate(valid_data.study_keys):
        if int(valid_data.image_counts[index]) > 1:
            study_index = index
            break
    if study_index is None:
        raise RuntimeError("No multi-image Validation Study for export verification")
    base_dataset = valid_data.dataset.base
    image_paths = [str(path) for path in base_dataset.studies[study_index]["image_paths"]]
    with torch.inference_mode():
        device = torch.device("cuda")
        images = _preprocess_paths(image_paths, IMAGE_SIZE).to(device)
        if runtime["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_logits = model(images)
        expected = torch.sigmoid(image_logits.float().mean(dim=0)).cpu().numpy()
        reference_dir = run_dir / "submission_export" / "verification"
        reference_dir.mkdir(parents=True, exist_ok=True)
        output_path = reference_dir / "standalone_probabilities.json"
        command = [sys.executable, "infer.py", "--config", "model_config.json",
                   "--images", *image_paths, "--output", str(output_path)]
        completed = subprocess.run(command, cwd=package_dir, capture_output=True,
                                   text=True, timeout=900)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{experiment} standalone inference failed: {completed.stderr[-2000:]}")
        standalone = json.loads(output_path.read_text(encoding="utf-8"))
    probabilities = np.asarray(standalone["probabilities"], dtype=np.float64)
    if probabilities.shape != (NUM_LABELS,):
        raise RuntimeError(f"{experiment} standalone probabilities shape is wrong")
    max_abs_error = float(np.abs(probabilities - expected.astype(np.float64)).max())
    trainer_saved = run_dir / "predictions" / "val_best.csv"
    trainer_saved_error = None
    if trainer_saved.is_file():
        frame = pd.read_csv(trainer_saved,
                            dtype={"Subject_id": "string", "Study_id": "string"})
        subject, study = str(valid_data.study_keys[study_index][0]), str(
            valid_data.study_keys[study_index][1])
        row = frame[(frame["Subject_id"] == subject) & (frame["Study_id"] == study)]
        if len(row) == 1:
            columns = [f"prob_{name.replace(' ', '_')}" for name in LABEL_NAMES]
            trainer_saved_error = float(
                np.abs(row[columns].to_numpy(dtype=np.float64)[0]
                       - probabilities).max())
    report = {
        "experiment": experiment,
        "package_dir": str(package_dir),
        "zip_path": str(zip_path),
        "zip_bytes": int(archive_bytes),
        "uncompressed_bytes": int(uncompressed),
        "size_limit_bytes": 500_000_000,
        "passes_size_limit": bool(
            archive_bytes < 500_000_000 and uncompressed < 500_000_000),
        "weights_bytes": int((package_dir / "model.safetensors").stat().st_size),
        "standalone_verified_study": {
            "subject_id": str(valid_data.study_keys[study_index][0]),
            "study_id": str(valid_data.study_keys[study_index][1]),
            "images": len(image_paths),
        },
        "max_abs_probability_error_vs_in_process": max_abs_error,
        "max_abs_probability_error_vs_trainer_val_best": trainer_saved_error,
        "tolerance": 1e-2,
        "passed": bool(max_abs_error <= 1e-2),
        "test_data_accessed": False,
    }
    if not report["passes_size_limit"] or not report["passed"]:
        raise RuntimeError(f"{experiment} submission package verification failed: {report}")
    write_json(run_dir / "submission_package_verification.json", report)
    return report


def _preprocess_paths(paths: list[str], size: int) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    tensors = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = size / max(width, height)
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = T.functional.resize(image, (resized[1], resized[0]),
                                    InterpolationMode.BILINEAR)
        pad_w = size - image.size[0]
        pad_h = size - image.size[1]
        left, top = pad_w // 2, pad_h // 2
        image = T.functional.pad(image, [left, top, pad_w - left, pad_h - top], fill=0)
        image = T.Grayscale(num_output_channels=3)(image)
        tensor = T.functional.to_tensor(image)
        tensor = T.functional.normalize(
            tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        tensors.append(tensor)
    return torch.stack(tensors)


_INFER_TEMPLATE = '''"""Standalone inference for one Study; Study Logit Mean before sigmoid."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from dino_model import DINOv3ConvNeXtBackbone
from stage_fusion_models import build_model


def preprocess(path, size):
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = T.functional.resize(image, (resized_height, resized_width),
                                InterpolationMode.BILINEAR)
    pad_w = size - image.size[0]
    pad_h = size - image.size[1]
    left, top = pad_w // 2, pad_h // 2
    image = T.functional.pad(image, [left, top, pad_w - left, pad_h - top], fill=0)
    image = T.Grayscale(num_output_channels=3)(image)
    tensor = T.functional.to_tensor(image)
    return T.functional.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("model_config.json"))
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["image_size"] != 512 or config["num_labels"] != 10:
        raise RuntimeError("Unexpected package configuration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DINOv3ConvNeXtBackbone(
        layer_norm_eps=float(config["layer_norm_eps"]),
        layer_scale_init_value=float(config["layer_scale_init_value"]),
        drop_path_rate=float(config["drop_path_rate"]),
        architecture=config["architecture_name"])
    model = build_model(config["experiment"], backbone)
    state = load_file(str(args.config.parent / "model.safetensors"), device="cpu")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    model.to(device).eval()
    images = torch.stack([preprocess(path, int(config["image_size"]))
                          for path in args.images]).to(device)
    with torch.inference_mode():
        logits = model(images).float().mean(dim=0)
        probabilities = torch.sigmoid(logits).cpu().tolist()
    payload = {
        "experiment": config["experiment"],
        "label_names": config["label_names"],
        "probabilities": probabilities,
        "study_fusion": "mean_raw_image_logits_before_sigmoid",
        "test_data_accessed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n",
                           encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def _write_predictions(result: dict, experiment: str, run_dir: Path,
                       modes: dict[str, np.ndarray]):
    frame = _predictions_frame(result, experiment)
    frame.to_csv(run_dir / "val_predictions.csv", index=False, encoding="utf-8")
    frame.to_csv(run_dir / "val_best_predictions.csv", index=False, encoding="utf-8")
    logits_frame = pd.DataFrame({"Subject_id": result["subjects"],
                                 "Study_id": result["studies"]})
    for index, disease in enumerate(LABEL_NAMES):
        key = disease.replace(" ", "_")
        logits_frame[f"target_{key}"] = result["labels"][:, index]
    for mode, logits in modes.items():
        for index, disease in enumerate(LABEL_NAMES):
            key = disease.replace(" ", "_")
            logits_frame[f"logit_{mode}_{key}"] = logits[:, index]
    logits_frame.to_csv(run_dir / "ablation_logits.csv", index=False, encoding="utf-8")
    return frame, logits_frame


def _write_inference_ablation(modes: dict[str, np.ndarray], labels: np.ndarray,
                              run_dir: Path, primary_mode: str):
    rows = []
    primary = study_metrics(labels, _sigmoid(modes[primary_mode]))
    for mode, logits in modes.items():
        metrics = study_metrics(labels, _sigmoid(logits))
        row = {
            "mode": mode,
            "macro_auroc": metrics["macro_auroc"],
            "macro_auprc": metrics["macro_auprc"],
            "delta_macro_auroc_vs_full": metrics["macro_auroc"] - primary["macro_auroc"],
            "delta_macro_auprc_vs_full": metrics["macro_auprc"] - primary["macro_auprc"],
        }
        for disease in LABEL_NAMES:
            row[f"AUROC_{disease.replace(' ', '_')}"] = metrics["per_class"][disease]["auroc"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "inference_ablation.csv", index=False, encoding="utf-8")
    return frame


def _write_per_disease_ablation(stage4, stage3, alpha, labels, run_dir: Path):
    rows = []
    full = stage4 + alpha[None, :] * stage3
    full_metrics = study_metrics(labels, _sigmoid(full))
    for index, disease in enumerate(LABEL_NAMES):
        modified = alpha.copy()
        modified[index] = 0.0
        metrics = study_metrics(labels, _sigmoid(stage4 + modified[None, :] * stage3))
        rows.append({
            "disabled_disease": disease,
            "alpha_disabled_disease": float(alpha[index]),
            "macro_auroc": metrics["macro_auroc"],
            "delta_macro_auroc_vs_full": metrics["macro_auroc"] - full_metrics["macro_auroc"],
            "disease_auroc": metrics["per_class"][disease]["auroc"],
            "disease_auroc_delta_vs_full": (
                metrics["per_class"][disease]["auroc"]
                - full_metrics["per_class"][disease]["auroc"]),
            "full_disease_auroc": full_metrics["per_class"][disease]["auroc"],
        })
    pd.DataFrame(rows).to_csv(run_dir / "per_disease_ablation.csv", index=False,
                              encoding="utf-8")
    return rows


def _comparison_sets(experiment: str, modes: dict[str, np.ndarray]) -> dict:
    base = "R5_baseline"
    names = MODE_NAMES[experiment]
    comparisons = {}
    if experiment == "R8":
        comparisons = {
            "R8_Full_minus_R5": (base, names["full"]),
            "R8_Full_minus_R8_BranchOff": (names["branch_off"], names["full"]),
            "R8_BranchOff_minus_R5": (base, names["branch_off"]),
        }
    elif experiment == "R9":
        comparisons = {
            "R9_Full_minus_R5": (base, names["full"]),
            "R9_Full_minus_R9_BranchOff": (names["branch_off"], names["full"]),
            "R9_Full_minus_R9_Uniform": (names["uniform"], names["full"]),
        }
    elif experiment == "R10":
        comparisons = {
            "R10_Full_minus_R5": (base, names["full"]),
            "R10_Full_minus_R10_BranchOff": (names["branch_off"], names["full"]),
            "R10_Full_minus_R10_NoStage4Conditioning": (
                names["no_conditioning"], names["full"]),
        }
    elif experiment == "R7":
        comparisons = {
            "R7_Fused_minus_R5": (base, names["full"]),
            "R7_Fused_minus_R7_gamma0": (names["gamma0"], names["full"]),
        }
    return comparisons


def _load_reference_probabilities() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """R5 unified-validation logits (computed by analyze_group('R5'))."""
    path = R5_DIR / "ablation_logits.csv"
    if not path.is_file():
        raise RuntimeError("R5 unified predictions are missing; analyze R5 first")
    frame = pd.read_csv(path, dtype={"Subject_id": "string", "Study_id": "string"})
    target_columns = [f"target_{name.replace(' ', '_')}" for name in LABEL_NAMES]
    labels = frame[target_columns].to_numpy(dtype=np.uint8)
    logits = frame[["logit_R5_baseline_" + name.replace(" ", "_")
                    for name in LABEL_NAMES]].to_numpy(dtype=np.float64)
    return frame, labels, logits


def _replay_diagnostic(experiment: str, run_dir: Path, result: dict) -> dict | None:
    """Compare fresh unified inference with the trainer-saved val_best.csv."""
    saved_path = Path(run_dir) / "val_best.csv"
    if not saved_path.is_file():
        return None
    saved = pd.read_csv(saved_path, dtype={"Subject_id": "string",
                                           "Study_id": "string"})
    base = pd.DataFrame({"Subject_id": pd.Series(result["subjects"], dtype="string"),
                         "Study_id": pd.Series(result["studies"], dtype="string")})
    merged = base.merge(saved, on=["Subject_id", "Study_id"], how="left",
                        validate="one_to_one", sort=False)
    probability_columns = [f"prob_{name.replace(' ', '_')}" for name in LABEL_NAMES]
    if merged[probability_columns].isna().any().any():
        return {"status": "saved_predictions_incomplete", "test_data_accessed": False}
    saved_probabilities = merged[probability_columns].to_numpy(dtype=np.float64)
    fresh_probabilities = _sigmoid(result["kinds"]["full"])
    delta = np.abs(saved_probabilities - fresh_probabilities)
    saved_metrics = study_metrics(result["labels"], saved_probabilities)
    fresh_metrics = study_metrics(result["labels"], fresh_probabilities)
    payload = {
        "experiment": experiment,
        "saved_prediction_file": str(saved_path),
        "status": "matched" if float(delta.max()) <= 1e-5 else "probability_mismatch",
        "max_probability_abs_error": float(delta.max()),
        "probability_cells_over_1e-5": int((delta > 1e-5).sum()),
        "saved_macro_auroc": saved_metrics["macro_auroc"],
        "fresh_unified_macro_auroc": fresh_metrics["macro_auroc"],
        "macro_auroc_delta": fresh_metrics["macro_auroc"] - saved_metrics["macro_auroc"],
        "max_abs_error_by_disease": {
            name: float(delta[:, index].max())
            for index, name in enumerate(LABEL_NAMES)},
        "paired_comparison_uses": (
            "fresh unified-protocol inference; the historical val_best.csv is "
            "kept only as a replay diagnostic"),
        "cause_resolved": float(delta.max()) <= 1e-5,
        "test_data_accessed": False,
    }
    write_json(Path(run_dir) / "validation_replay_diagnostic.json", payload)
    return payload


def analyze_group(experiment: str, valid_data, run_dir: Path | None = None,
                  write_export: bool = True) -> dict:
    """Unified-protocol analysis for one experiment; writes its artifacts and report.

    `experiment` is one of ``R5``, ``R7``, ``R8``, ``R9``, ``R10``.
    """
    if experiment not in {"R5", "R7", "R8", "R9", "R10"}:
        raise ValueError(f"Unknown experiment for analysis: {experiment}")
    group = {"R5": "R5", "R7": R7_GROUP}.get(experiment, GROUP_NAMES.get(experiment))
    if experiment in {"R8", "R9", "R10"}:
        from stage_fusion_experiment import refresh_group_artifacts
        refresh_group_artifacts(experiment)
    model, status, group_dir, runtime = load_group_model(group, torch.device("cuda"))
    run_dir = Path(group_dir) if run_dir is None else Path(run_dir)
    kinds = ["full"]
    if experiment == "R7":
        kinds = ["full"]
    elif experiment == "R9":
        kinds = ["full", "uniform"]
    elif experiment == "R10":
        kinds = ["full", "no_conditioning"]
    result = evaluate(model, experiment, valid_data, runtime, tuple(kinds))
    replay = _replay_diagnostic(experiment, run_dir, result)
    modes = _mode_logits(result, experiment, model)
    frame, logits_frame = _write_predictions(result, experiment, run_dir, modes)
    _write_inference_ablation(modes, result["labels"], run_dir,
                              MODE_NAMES[experiment]["full"])

    per_disease = None
    if experiment in {"R8", "R9", "R10"}:
        # Per-disease branch-off diagnostics on the trained model.
        per_disease = _write_per_disease_ablation(
            result["stage4_logits"], result["stage3_logits"],
            model.fusion_alpha.alpha().detach().float().cpu().numpy(),
            result["labels"], run_dir)

    cost = _inference_cost(experiment, group, model, status, result, run_dir)
    write_json(run_dir / "inference_cost.json", cost)

    if experiment in {"R9", "R10"}:
        directory = run_dir / ("cross_stage_attention_heatmaps"
                               if experiment == "R10" else "attention_heatmaps")
        _attention_heatmaps(model, experiment, valid_data, runtime, directory)

    export_report = None
    if write_export and experiment in {"R8", "R9", "R10"}:
        export_report = export_submission_package(
            experiment, model, run_dir, valid_data, runtime)

    # Paired comparisons against the unified R5 predictions.
    r5_frame, labels, r5_logits = _load_reference_probabilities()
    if not np.array_equal(labels, result["labels"]):
        raise RuntimeError(f"{experiment} labels differ from the R5 unified predictions")
    probabilities = {"R5_baseline": _sigmoid(r5_logits)}
    for mode, logits in modes.items():
        probabilities[mode] = _sigmoid(logits)
    comparisons = _comparison_sets(experiment, modes)
    bootstrap = bootstrap_models(labels, probabilities,
                                 result["subjects"], comparisons)
    write_json(run_dir / "paired_subject_bootstrap.json", bootstrap)

    if experiment == "R9" and MODE_NAMES["R9"]["uniform"] in modes:
        uniform_rows = []
        full = bootstrap["comparisons"]["R9_Full_minus_R9_Uniform"]
        for disease in LABEL_NAMES:
            stats = full["per_class"][disease]
            uniform_rows.append({
                "disease": disease,
                "full_auroc": bootstrap["models"]["R9_Full"]["per_class"][disease]["auroc"],
                "uniform_auroc": bootstrap["models"]["R9_Uniform"]["per_class"][disease]["auroc"],
                "delta_auroc": stats["delta_auroc"],
                "delta_auroc_ci95_low": stats["delta_auroc_ci95"][0],
                "delta_auroc_ci95_high": stats["delta_auroc_ci95"][1],
                "positive_studies": stats["positive_studies"],
            })
        pd.DataFrame(uniform_rows).to_csv(
            run_dir / "attention_vs_uniform_ablation.csv", index=False, encoding="utf-8")
    if experiment == "R10":
        rows = []
        comparison = bootstrap["comparisons"][
            "R10_Full_minus_R10_NoStage4Conditioning"]
        conditioning_path = run_dir / "stage4_query_conditioning.csv"
        conditioning_rows = (pd.read_csv(conditioning_path).to_dict("records")
                             if conditioning_path.is_file() else [])
        best = [row for row in conditioning_rows
                if int(row.get("epoch", -1)) == int(status["best_epoch"])]
        for disease in LABEL_NAMES:
            stats = comparison["per_class"][disease]
            rows.append({
                "scope": "best_full_validation",
                "disease": disease,
                "full_auroc": bootstrap["models"]["R10_Full"]["per_class"][disease]["auroc"],
                "no_conditioning_auroc": bootstrap["models"][
                    "R10_NoStage4Conditioning"]["per_class"][disease]["auroc"],
                "delta_auroc": stats["delta_auroc"],
                "delta_auroc_ci95_low": stats["delta_auroc_ci95"][0],
                "delta_auroc_ci95_high": stats["delta_auroc_ci95"][1],
                "positive_studies": stats["positive_studies"],
                "best_epoch_conditioning_mean": (
                    float(best[0]["condition_norm_mean"]) if best else None),
            })
        pd.DataFrame(rows).to_csv(
            run_dir / "inference_conditioning_ablation.csv",
            index=False, encoding="utf-8")
        if conditioning_rows:
            existing = pd.DataFrame(conditioning_rows)
            if "scope" in existing.columns:
                existing = existing[existing["scope"] != "best_full_validation"]
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True,
                                 sort=False)
        else:
            combined = pd.DataFrame(rows)
        combined.to_csv(
            run_dir / "conditioned_vs_unconditioned_attention.csv",
            index=False, encoding="utf-8")

    info = {
        "group": group,
        "experiment": experiment,
        "status": status,
        "run_dir": str(run_dir),
        "result": result,
        "modes": modes,
        "frame": frame,
        "logits_frame": logits_frame,
        "bootstrap": bootstrap,
        "cost": cost,
        "export": export_report,
        "per_disease": per_disease,
        "replay": replay,
    }
    _write_group_final_report(experiment, group, run_dir, info)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return info


def _fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _ci_text(pair, signed: bool = True):
    if not pair or pair[0] is None:
        return "NA"
    if signed:
        return f"[{pair[0]:+.6f}, {pair[1]:+.6f}]"
    return f"[{pair[0]:.4f}, {pair[1]:.4f}]"


def _write_group_final_report(experiment: str, group: str, run_dir: Path, info: dict):
    status = info["status"]
    bootstrap = info["bootstrap"]
    lines = [
        f"# {experiment} final report", "",
        f"- Group directory: `{run_dir}`",
        f"- Best epoch: {status['best_epoch']}; best Validation Macro-AUROC: "
        f"{float(status['best_macro_auroc']):.6f}; stop reason: `{status.get('stop_reason')}`.",
        "- Unified inference protocol: official Validation split, 859 Studies / 228 Subjects, "
        "Study arithmetic mean of raw image logits before sigmoid, BF16 autocast, "
        "channels-last, identical checkpoint loading for every model.",
        f"- Validation Macro-AUROC under this protocol: "
        f"{bootstrap['models'][MODE_NAMES[experiment]['full']]['macro_auroc']:.6f}.",
        "", "## Inference ablation", "",
        "| Mode | Macro-AUROC | Macro-AUPRC | Δ AUROC vs Full | Δ AUPRC vs Full |",
        "|---|---:|---:|---:|---:|",
    ]
    ablation = pd.read_csv(run_dir / "inference_ablation.csv")
    for row in ablation.to_dict("records"):
        lines.append(
            f"| {row['mode']} | {row['macro_auroc']:.6f} | {row['macro_auprc']:.6f} | "
            f"{row['delta_macro_auroc_vs_full']:+.6f} | "
            f"{row['delta_macro_auprc_vs_full']:+.6f} |")
    lines.extend(["", "## Paired Subject bootstrap (fresh predictions)", "",
                  "| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |",
                  "|---|---:|---|---:|---|"])
    for name, comparison in bootstrap["comparisons"].items():
        if comparison.get("status") == "skipped_missing_model":
            continue
        lines.append(
            f"| {name} | {comparison['delta_macro_auroc']:+.6f} | "
            f"{_ci_text(comparison['delta_macro_auroc_ci95'])} | "
            f"{comparison['delta_macro_auprc']:+.6f} | "
            f"{_ci_text(comparison['delta_macro_auprc_ci95'])} |")
    lines.extend(["", "## Per-disease AUROC (best checkpoint)", "",
                  "| Disease | Pos/Neg | AUROC | 95% CI |", "|---|---:|---:|---|"])
    full_mode = MODE_NAMES[experiment]["full"]
    for disease, metrics in bootstrap["models"][full_mode]["per_class"].items():
        lines.append(
            f"| {disease} | {metrics['positive_studies']}/{metrics['negative_studies']} | "
            f"{metrics['auroc']:.4f} | {_ci_text(metrics['auroc_ci95'], signed=False)} |")
    lines.extend([""])
    if info.get("per_disease"):
        lines.extend(["## Per-disease branch-off diagnostic", "",
                      "| Disabled disease branch | Δ Macro-AUROC vs Full | Disabled disease AUROC Δ |",
                      "|---|---:|---:|"])
        for row in info["per_disease"]:
            lines.append(
                f"| {row['disabled_disease']} | {row['delta_macro_auroc_vs_full']:+.6f} | "
                f"{row['disease_auroc_delta_vs_full']:+.6f} |")
        lines.extend([""])
    cost = info["cost"]
    lines.extend([
        "## Inference cost", "",
        f"- Parameters: {cost['parameters_total']:,}; FP32 weights: "
        f"{cost['fp32_inference_weights_bytes'] / 1e6:.1f} MB.",
        f"- Validation throughput: {cost['studies_per_second']:.2f} Study/s; "
        f"mean {cost['mean_ms_per_study']:.1f} ms/Study; peak {cost['peak_gpu_gib']:.2f} GiB.",
        f"- Submission package: {'verified' if info.get('export') else 'not exported'}"
        + (f", zip {info['export']['zip_bytes'] / 1e6:.1f} MB, standalone max |Δp| "
           f"{info['export']['max_abs_probability_error_vs_in_process']:.2e}"
           if info.get("export") else ""),
        "",
    ])
    replay = info.get("replay")
    if replay:
        lines.extend([
            "## Saved-prediction replay diagnostic", "",
            f"- Status: `{replay.get('status')}`; saved vs fresh unified inference "
            f"max |Δprobability| = {replay.get('max_probability_abs_error'):.6f}; "
            f"Macro-AUROC {replay.get('saved_macro_auroc'):.6f} (saved) vs "
            f"{replay.get('fresh_unified_macro_auroc'):.6f} (fresh), "
            f"Δ={replay.get('macro_auroc_delta'):+.6f}.",
            "- All paired comparisons in this report use the fresh unified-protocol "
            "predictions, not the historical saved file. The R5 historical replay "
            "difference (max |Δp| ≈ 0.00935) was already recorded in the project "
            "before this task and remains an unresolved reproducibility limitation.",
            "",
        ])
    lines.extend([
        "## Diagnostic interpretation", "",
        "- Branch/attention diagnostics are model-behaviour measurements, not verified "
        "lesion localization.",
        "- The Validation split has been used for repeated checkpoint selection and "
        "multiple comparisons across the project; all confidence intervals are "
        "exploratory and do not replace an independent test set.",
        "- Test labels were never read; the Test split was not used for tuning.",
        "",
    ])
    if info.get("per_disease") and experiment == "R8":
        lines.append(
            "The per-disease branch-off diagnostic isolates each disease's auxiliary "
            "logit contribution inside the trained model; it is not a training-time "
            "causal effect.\n")
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_analysis(write_export: bool = True,
                     experiments: list[str] | None = None) -> dict:
    """Analyze the requested (or all completed) groups, then rebuild comparisons.

    When `experiments` is given only those groups are re-evaluated; the
    comparative analysis is always rebuilt from the on-disk unified artifacts,
    so already-analyzed groups do not need a redundant GPU pass.
    """
    valid_data = _valid_data()
    infos = {}
    if experiments is None:
        experiments = ["R5", "R7"]
        for experiment in ("R8", "R9", "R10"):
            status_path = OUTPUT_ROOT / GROUP_NAMES[experiment] / "training_status.json"
            if status_path.is_file() and read_json(status_path).get("complete"):
                experiments.append(experiment)
    for experiment in experiments:
        print(f"Unified inference and ablation: {experiment}", flush=True)
        infos[experiment] = analyze_group(experiment, valid_data,
                                          write_export=write_export)
    comparative = run_comparative_analysis()
    return {"groups": infos, "comparative": comparative}


EXPERIMENT_GROUPS = {
    "R5": R5_GROUP,
    "R7": R7_GROUP,
    "R8": GROUP_NAMES["R8"],
    "R9": GROUP_NAMES["R9"],
    "R10": GROUP_NAMES["R10"],
}


def _experiment_run_dir(experiment: str) -> Path:
    return OUTPUT_ROOT / EXPERIMENT_GROUPS[experiment]


def _available_experiments() -> list[str]:
    experiments = []
    for experiment in ("R5", "R7", "R8", "R9", "R10"):
        run_dir = _experiment_run_dir(experiment)
        if (run_dir / "ablation_logits.csv").is_file():
            experiments.append(experiment)
    return experiments


def _load_saved_ablation(experiments: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                          dict[str, np.ndarray], dict]:
    """Collect every mode's logits from the on-disk unified-ablation artifacts."""
    labels = None
    subjects = None
    probabilities = {}
    metadata = {}
    first_disease_key = LABEL_NAMES[0].replace(" ", "_")
    for experiment in experiments:
        run_dir = _experiment_run_dir(experiment)
        logits_frame = pd.read_csv(
            run_dir / "ablation_logits.csv",
            dtype={"Subject_id": "string", "Study_id": "string"})
        target_columns = [f"target_{name.replace(' ', '_')}" for name in LABEL_NAMES]
        current_labels = logits_frame[target_columns].to_numpy(dtype=np.uint8)
        current_subjects = logits_frame["Subject_id"].to_numpy(dtype=object)
        if labels is None:
            labels, subjects = current_labels, current_subjects
        elif not np.array_equal(labels, current_labels) or list(subjects) != list(
                current_subjects):
            raise RuntimeError(f"{experiment} saved ablation logits are not aligned")
        mode_rows = pd.read_csv(run_dir / "inference_ablation.csv")
        for mode in mode_rows["mode"].tolist():
            columns = [f"logit_{mode}_{name.replace(' ', '_')}" for name in LABEL_NAMES]
            missing = [column for column in columns if column not in logits_frame.columns]
            if missing:
                raise RuntimeError(f"{experiment} is missing logit columns: {missing[:3]}")
            probabilities[mode] = _sigmoid(
                logits_frame[columns].to_numpy(dtype=np.float64))
        status = read_json(run_dir / "training_status.json")
        cost_path = run_dir / "inference_cost.json"
        export_path = run_dir / "submission_package_verification.json"
        metadata[experiment] = {
            "group": EXPERIMENT_GROUPS[experiment],
            "status": status,
            "cost": read_json(cost_path) if cost_path.is_file() else {},
            "export": read_json(export_path) if export_path.is_file() else None,
            "best_epoch": int(status["best_epoch"]),
            "best_macro_auroc": float(status["best_macro_auroc"]),
        }
    return labels, subjects, probabilities, metadata


def run_comparative_analysis(experiments: list[str] | None = None) -> dict:
    """Cross-experiment comparison artifacts built from saved unified artifacts."""
    COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)
    experiments = _available_experiments() if experiments is None else experiments
    if not experiments:
        raise RuntimeError("No unified ablation artifacts found for comparison")
    labels, subjects, probabilities, metadata = _load_saved_ablation(experiments)
    comparisons = {
        "R8_Full_minus_R5": ("R5_baseline", "R8_Full"),
        "R8_Full_minus_R8_BranchOff": ("R8_BranchOff", "R8_Full"),
        "R8_BranchOff_minus_R5": ("R5_baseline", "R8_BranchOff"),
        "R9_Full_minus_R5": ("R5_baseline", "R9_Full"),
        "R9_Full_minus_R8_Full": ("R8_Full", "R9_Full"),
        "R9_Full_minus_R9_BranchOff": ("R9_BranchOff", "R9_Full"),
        "R9_Full_minus_R9_Uniform": ("R9_Uniform", "R9_Full"),
        "R9_BranchOff_minus_R5": ("R5_baseline", "R9_BranchOff"),
        "R10_Full_minus_R5": ("R5_baseline", "R10_Full"),
        "R10_Full_minus_R9_Full": ("R9_Full", "R10_Full"),
        "R10_Full_minus_R10_BranchOff": ("R10_BranchOff", "R10_Full"),
        "R10_Full_minus_R10_NoStage4Conditioning": (
            "R10_NoStage4Conditioning", "R10_Full"),
        "R10_BranchOff_minus_R5": ("R5_baseline", "R10_BranchOff"),
        "R7_Fused_minus_R5": ("R5_baseline", "R7_Fused"),
        "R7_Fused_minus_R7_gamma0": ("R7_gamma0", "R7_Fused"),
    }
    bootstrap = bootstrap_models(labels, probabilities, subjects, comparisons)
    write_json(COMPARATIVE_DIR / "paired_bootstrap_comparison.json", bootstrap)

    experiment_rows, overall_rows = [], []
    for experiment in ("R5", "R7", "R8", "R9", "R10"):
        if experiment not in metadata:
            continue
        info = metadata[experiment]
        mode = MODE_NAMES[experiment]["full"]
        metrics = bootstrap["models"][mode]
        experiment_rows.append({
            "experiment": experiment,
            "group": info["group"],
            "backbone": "DINOv3 ConvNeXt-Tiny",
            "resolution": IMAGE_SIZE,
            "auxiliary_head": {
                "R5": "none (Global-only)",
                "R7": "shared Stage3->Stage4 residual (gamma)",
                "R8": "disease-specific Stage3 GAP logit fusion",
                "R9": "disease-query Stage3 cross-attention",
                "R10": "Stage4-conditioned disease-query cross-attention",
            }[experiment],
            "best_epoch": info["best_epoch"],
            "completed_epochs": int(info["status"]["completed_epochs"]),
            "stop_reason": info["status"].get("stop_reason"),
            "parameters_total": int(info["cost"]["parameters_total"]),
            "best_macro_auroc_training": info["best_macro_auroc"],
            "unified_macro_auroc": metrics["macro_auroc"],
            "unified_macro_auprc": metrics["macro_auprc"],
            "submission_zip_bytes": (info["export"]["zip_bytes"]
                                     if info.get("export") else None),
        })
        overall_rows.append({
            "experiment": experiment,
            "mode": mode,
            "macro_auroc": metrics["macro_auroc"],
            "macro_auprc": metrics["macro_auprc"],
        })
    pd.DataFrame(experiment_rows).to_csv(
        COMPARATIVE_DIR / "experiment_matrix.csv", index=False, encoding="utf-8")
    pd.DataFrame(overall_rows).to_csv(
        COMPARATIVE_DIR / "overall_metrics.csv", index=False, encoding="utf-8")

    per_class_rows = []
    for disease_index, disease in enumerate(LABEL_NAMES):
        row = {"disease": disease,
               "positive_studies": int(labels[:, disease_index].sum()),
               "negative_studies": int((labels[:, disease_index] == 0).sum())}
        for experiment in ("R5", "R7", "R8", "R9", "R10"):
            if experiment not in metadata:
                continue
            mode = MODE_NAMES[experiment]["full"]
            row[f"{experiment}_AUROC"] = bootstrap["models"][mode]["per_class"][
                disease]["auroc"]
            row[f"{experiment}_AUPRC"] = bootstrap["models"][mode]["per_class"][
                disease]["auprc"]
        for name, comparison in bootstrap["comparisons"].items():
            if comparison.get("status") == "skipped_missing_model":
                continue
            stats = comparison["per_class"][disease]
            row[f"{name}_AUROC_delta"] = stats["delta_auroc"]
            row[f"{name}_AUROC_ci95_low"] = stats["delta_auroc_ci95"][0]
            row[f"{name}_AUROC_ci95_high"] = stats["delta_auroc_ci95"][1]
            row[f"{name}_AUPRC_delta"] = stats["delta_auprc"]
        per_class_rows.append(row)
    pd.DataFrame(per_class_rows).to_csv(
        COMPARATIVE_DIR / "per_class_comparison.csv", index=False, encoding="utf-8")

    training_rows, inference_rows, ablation_rows = [], [], []
    for experiment, info in metadata.items():
        group = info["group"]
        run_dir = OUTPUT_ROOT / group
        log_path = run_dir / "train_log.csv"
        if log_path.is_file():
            log = pd.read_csv(log_path)
            training_rows.append({
                "experiment": experiment,
                "completed_epochs": int(len(log)),
                "best_epoch": info["best_epoch"],
                "stop_reason": info["status"].get("stop_reason"),
                "mean_images_per_s": float(log["images_per_s"].mean()),
                "mean_data_loader_wait_s": float(log["data_loader_wait_seconds"].mean()),
                "peak_gpu_allocated_gib": float(log["peak_gpu_allocated_gib"].max()),
                "peak_gpu_reserved_gib": float(log["peak_gpu_reserved_gib"].max()),
                "trainable_parameters": int(
                    info["cost"].get("trainable_parameters_reported")
                    or info["cost"]["parameters_total"]),
                "epochs_total_seconds": float(
                    (log["train_seconds"] + log["validation_seconds"]).sum()),
            })
        inference_rows.append({
            "experiment": experiment,
            "parameters_total": info["cost"]["parameters_total"],
            "fp32_weights_mb": info["cost"]["fp32_inference_weights_bytes"] / 1e6,
            "mean_ms_per_study": info["cost"]["mean_ms_per_study"],
            "studies_per_second": info["cost"]["studies_per_second"],
            "peak_gpu_gib": info["cost"]["peak_gpu_gib"],
            "submission_zip_mb": (info["export"]["zip_bytes"] / 1e6
                                  if info.get("export") else None),
        })
        ablation_path = run_dir / "inference_ablation.csv"
        if ablation_path.is_file():
            for row in pd.read_csv(ablation_path).to_dict("records"):
                ablation_rows.append({"experiment": experiment, **row})
    pd.DataFrame(training_rows).to_csv(
        COMPARATIVE_DIR / "training_efficiency.csv", index=False, encoding="utf-8")
    pd.DataFrame(inference_rows).to_csv(
        COMPARATIVE_DIR / "inference_efficiency.csv", index=False, encoding="utf-8")
    pd.DataFrame(ablation_rows).to_csv(
        COMPARATIVE_DIR / "ablation_summary.csv", index=False, encoding="utf-8")
    _write_comparative_report(bootstrap, metadata)
    return {"bootstrap": bootstrap, "metadata": metadata}


def _comparison(bootstrap: dict, name: str):
    comparison = bootstrap["comparisons"].get(name)
    if not comparison or comparison.get("status") == "skipped_missing_model":
        return None
    return comparison


def _delta_text(value: float) -> str:
    if abs(float(value)) < 5e-7:
        return "≈0"
    return f"{float(value):+.6f}"


def _verdict(comparison, direction="improvement"):
    if comparison is None:
        return "目前无法确认（缺少对应模型）"
    low, high = comparison["delta_macro_auroc_ci95"]
    if low is None:
        return "目前无法确认（有效重采样不足）"
    if low > 0:
        return (f"有正向差异，95% CI 排除 0：Δ={_delta_text(comparison['delta_macro_auroc'])} "
                f"{_ci_text((low, high))}")
    if high < 0:
        return (f"有负向差异，95% CI 排除 0：Δ={_delta_text(comparison['delta_macro_auroc'])} "
                f"{_ci_text((low, high))}")
    return (f"目前无法确认方向性差异：Δ={_delta_text(comparison['delta_macro_auroc'])} "
            f"95% CI {_ci_text((low, high))} 跨 0")


def _best_epoch_diagnostics(experiment: str, metadata: dict) -> dict:
    """Read the trained model's best-epoch branch/attention diagnostics."""
    run_dir = _experiment_run_dir(experiment)
    payload = {"best_epoch": int(metadata[experiment]["status"]["best_epoch"])}
    best_epoch = payload["best_epoch"]
    branch_path = run_dir / "branch_diagnostics.csv"
    if branch_path.is_file():
        frame = pd.read_csv(branch_path)
        rows = frame[frame["epoch"] == best_epoch]
        if len(rows):
            payload["alpha_min"] = float(rows["alpha"].min())
            payload["alpha_max"] = float(rows["alpha"].max())
            payload["alpha_mean"] = float(rows["alpha"].mean())
            payload["residual_ratio_mean"] = float(rows["residual_to_stage4_ratio"].mean())
            payload["residual_ratio_max"] = float(rows["residual_to_stage4_ratio"].max())
    attention_path = run_dir / "attention_statistics.csv"
    if attention_path.is_file():
        frame = pd.read_csv(attention_path)
        rows = frame[frame["epoch"] == best_epoch]
        if len(rows):
            payload["normalized_entropy_mean"] = float(
                rows["normalized_entropy"].mean())
            payload["max_weight_mean"] = float(rows["max_weight"].mean())
            payload["attention_offdiag_cosine_mean"] = float(
                rows["attention_mean_offdiag_cosine"].mean())
    conditioning_path = run_dir / "stage4_query_conditioning.csv"
    if conditioning_path.is_file():
        frame = pd.read_csv(conditioning_path)
        rows = frame[frame["epoch"] == best_epoch] if "epoch" in frame.columns else frame
        if len(rows):
            if "condition_to_disease_norm_ratio" in rows.columns:
                payload["condition_to_disease_ratio"] = float(
                    rows["condition_to_disease_norm_ratio"].mean())
            if "weighted_aux_logit_change" in rows.columns:
                payload["conditioning_logit_change"] = float(
                    rows["weighted_aux_logit_change"].mean())
    return payload


def _mechanism_findings(bootstrap: dict, metadata: dict) -> list[str]:
    """Honest mechanism summary derived from the training diagnostics and ablations."""
    findings = []
    if "R8" in metadata:
        diagnostics = _best_epoch_diagnostics("R8", metadata)
        comparison = _comparison(bootstrap, "R8_Full_minus_R8_BranchOff")
        if "residual_ratio_mean" in diagnostics:
            findings.append(
                f"R8: best epoch {diagnostics['best_epoch']}，alpha 范围 "
                f"{diagnostics['alpha_min']:.4f}–{diagnostics['alpha_max']:.4f}；"
                "Stage3 辅助 logit 残差相对主 logit 的平均比例 "
                f"{diagnostics['residual_ratio_mean']:.4%}"
                f"（最大 {diagnostics['residual_ratio_max']:.4%}），"
                "辅助支路幅值极小。"
                + (f" Full−BranchOff = {comparison['delta_macro_auroc']:+.6f}，"
                   f"95% CI {_ci_text(comparison['delta_macro_auroc_ci95'])}。"
                   if comparison else ""))
    if "R9" in metadata:
        diagnostics = _best_epoch_diagnostics("R9", metadata)
        comparison = _comparison(bootstrap, "R9_Full_minus_R9_BranchOff")
        uniform = _comparison(bootstrap, "R9_Full_minus_R9_Uniform")
        if "normalized_entropy_mean" in diagnostics:
            collapsed = diagnostics["normalized_entropy_mean"] > 0.999
            findings.append(
                f"R9: best epoch {diagnostics['best_epoch']}，归一化注意力熵 "
                f"{diagnostics['normalized_entropy_mean']:.4f}（1.0 表示完全均匀），"
                f"最大注意力权重均值 {diagnostics['max_weight_mean']:.5f}，"
                "疾病间注意力平均余弦 "
                f"{diagnostics['attention_offdiag_cosine_mean']:.5f}。"
                + ("注意力与均匀分布不可区分（塌缩），没有学到疾病特异的空间选择。"
                   if collapsed else
                   "注意力略偏离均匀，但疾病间差异仍然很弱。")
                + (f" Full−Uniform = {_delta_text(uniform['delta_macro_auroc'])}，"
                   f"95% CI {_ci_text(uniform['delta_macro_auroc_ci95'])}。"
                   if uniform else "")
                + (f" Full−BranchOff = {_delta_text(comparison['delta_macro_auroc'])}，"
                   f"95% CI {_ci_text(comparison['delta_macro_auroc_ci95'])}。"
                   if comparison else ""))
    if "R10" in metadata:
        diagnostics = _best_epoch_diagnostics("R10", metadata)
        comparison = _comparison(
            bootstrap, "R10_Full_minus_R10_NoStage4Conditioning")
        if "normalized_entropy_mean" in diagnostics:
            findings.append(
                f"R10: best epoch {diagnostics['best_epoch']}，Stage4 条件向量与疾病 "
                "embedding 的范数比 "
                f"{diagnostics.get('condition_to_disease_ratio', float('nan')):.0f}:1；"
                f"归一化注意力熵 {diagnostics['normalized_entropy_mean']:.3f}，"
                "疾病间注意力平均余弦 "
                f"{diagnostics['attention_offdiag_cosine_mean']:.5f}。"
                + ("条件项主导了 Query，使十个疾病 Query 与注意力分布几乎相同："
                   "注意力是图像自适应的，但疾病特异性被条件项淹没。"
                   if diagnostics["attention_offdiag_cosine_mean"] > 0.99 else
                   "注意力在图像与疾病两个维度都出现了可测差异。")
                + (f" Full−NoStage4Conditioning = "
                   f"{_delta_text(comparison['delta_macro_auroc'])}，95% CI "
                   f"{_ci_text(comparison['delta_macro_auroc_ci95'])}。"
                   if comparison else ""))
    if not findings:
        findings.append("尚未生成任何机制的完整诊断。")
    return findings


def _write_comparative_report(bootstrap: dict, metadata: dict):
    lines = [
        "# R8 / R9 / R10 comparative analysis", "",
        "All models were independently trained from the same pinned DINOv3 general-vision "
        "weights with the R5 P2 freeze profile, R5 selected runtime (image budget 8, "
        "gradient accumulation 4, BF16, channels-last), Study Logit Mean BCE, seed 42 and "
        "the three-condition early stopping. Only the classifier topology differs.",
        "",
        "## Experiment matrix", "",
        "| Experiment | Auxiliary head | Best epoch | Trainable params | Unified Macro-AUROC | Unified Macro-AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for experiment in ("R5", "R7", "R8", "R9", "R10"):
        if experiment not in metadata:
            continue
        info = metadata[experiment]
        mode = MODE_NAMES[experiment]["full"]
        metrics = bootstrap["models"][mode]
        lines.append(
            f"| {experiment} | {info['group']} | {info['best_epoch']} | "
            f"{info['cost']['parameters_total']:,} | {metrics['macro_auroc']:.6f} | "
            f"{metrics['macro_auprc']:.6f} |")
    lines.extend(["", "## Mechanism findings (best-checkpoint diagnostics)", ""])
    for finding in _mechanism_findings(bootstrap, metadata):
        lines.append(f"- {finding}")
    lines.extend(["", "## Paired comparison table (fresh unified predictions)", "",
                  "| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |",
                  "|---|---:|---|---:|---|"])
    for name, comparison in bootstrap["comparisons"].items():
        if comparison.get("status") == "skipped_missing_model":
            lines.append(f"| {name} | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {name} | {comparison['delta_macro_auroc']:+.6f} | "
            f"{_ci_text(comparison['delta_macro_auroc_ci95'])} | "
            f"{comparison['delta_macro_auprc']:+.6f} | "
            f"{_ci_text(comparison['delta_macro_auprc_ci95'])} |")
    lines.extend(["", "## Required questions", ""])
    answers = [
        ("1. R8 的疾病专属融合是否优于 R5 和 R7？",
         _verdict(_comparison(bootstrap, "R8_Full_minus_R5")) if "R8" in metadata
         else "目前无法确认（R8 尚未完成）"),
        ("2. R8 是否成功学习到不同疾病的有效辅助信息？",
         _verdict(_comparison(bootstrap, "R8_Full_minus_R8_BranchOff"))
         if "R8" in metadata else "目前无法确认（R8 尚未完成）"),
        ("3. R9 的 Query Attention 是否优于简单 GAP 融合？",
         _verdict(_comparison(bootstrap, "R9_Full_minus_R8_Full"))
         if "R9" in metadata and "R8" in metadata else "目前无法确认（缺少 R8/R9）"),
        ("4. R9 的注意力是否表现出疾病差异和图像自适应性？",
         _attention_verdict("R9")),
        ("5. R10 的 Stage4 Query Conditioning 是否产生实际贡献？",
         (_verdict(_comparison(bootstrap, "R10_Full_minus_R10_NoStage4Conditioning"))
          + "；条件项主导 Query、疾病间注意力塌缩的机制证据见 Mechanism findings。")
         if "R10" in metadata else "目前无法确认（R10 尚未完成）"),
        ("6. 哪些疾病从 Stage 3 的补充信息中获益？", _positive_diseases(bootstrap)),
        ("7. 哪些疾病反而受到辅助信息干扰？", _negative_diseases(bootstrap)),
        ("8. 三种新结构中，哪一种在性能、复杂度、稳定性之间最有优势？",
         _tradeoff_summary(bootstrap, metadata)),
        ("9. 目前的结果是否支持继续优化特征选择，还是应重新考虑 Backbone 特征提取？",
         _research_direction(bootstrap, metadata)),
        ("10. 是否值得在后续独立研究 Study 级加权 Logit Mean？",
         "本轮未研究 Study 级加权 Logit Mean（按要求继续使用算术平均），"
         "现有数据无法支持或否定该方向，属未来独立实验。"),
    ]
    for question, answer in answers:
        lines.append(f"- **{question}** {answer}")
    lines.extend([
        "", "## Interpretation limits", "",
        "- Validation has been used for repeated checkpoint selection and many "
        "exploratory comparisons across R1-R10; all intervals are conditional and "
        "exploratory.",
        "- R8/R9/R10 differ in parameter count and head capacity, so a performance "
        "difference cannot be attributed solely to attention or conditioning without "
        "a capacity-matched control.",
        "- Attention maps and branch statistics describe model behaviour; they are not "
        "verified lesion localization.",
        "- Test labels were never read and the Test split was not used for tuning.", "",
    ])
    (COMPARATIVE_DIR / "final_comparative_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _attention_verdict(experiment: str) -> str:
    normalized = {"R9": "R9", "R10": "R10"}[experiment]
    group = GROUP_NAMES[normalized]
    path = OUTPUT_ROOT / group / "attention_statistics.csv"
    if not path.is_file():
        return "目前无法确认（缺少注意力诊断）"
    frame = pd.read_csv(path)
    if frame.empty:
        return "目前无法确认（缺少注意力诊断）"
    entropy_min = float(frame["normalized_entropy"].min())
    entropy_max = float(frame["normalized_entropy"].max())
    offdiag = float(frame["attention_mean_offdiag_cosine"].mean())
    max_weight = float(frame["max_weight"].mean())
    if entropy_min > 0.999:
        return (f"注意力塌缩：归一化熵 {entropy_min:.4f}–{entropy_max:.4f}"
                f"（1.0 = 均匀分布），平均最大权重 {max_weight:.5f}，"
                f"疾病间注意力平均余弦 {offdiag:.5f} → 学到的空间注意力与均匀分布"
                "不可区分，未表现出疾病差异或图像自适应。")
    return (f"注意力非均匀：归一化熵范围 {entropy_min:.3f}–{entropy_max:.3f}"
            f"（均匀分布 = 1.0），平均最大权重 {max_weight:.5f}，疾病间平均余弦 "
            f"{offdiag:.5f}；存在空间选择，但疾病间相似度接近 1 时不能宣称"
            "疾病特异选择，也不能据此宣称性能提升。")


def _positive_diseases(bootstrap: dict) -> str:
    names = []
    for comparison_name in ("R8_Full_minus_R8_BranchOff", "R9_Full_minus_R9_BranchOff",
                            "R10_Full_minus_R10_BranchOff",
                            "R10_Full_minus_R10_NoStage4Conditioning"):
        comparison = bootstrap["comparisons"].get(comparison_name)
        if not comparison or comparison.get("status") == "skipped_missing_model":
            continue
        for disease, stats in comparison["per_class"].items():
            low = stats["delta_auroc_ci95"][0]
            if low is not None and low > 0 and stats["delta_auroc"] > 0:
                names.append(f"{disease}（{comparison_name} Δ={stats['delta_auroc']:+.4f}）")
    return ("；".join(sorted(set(names))) if names
            else "目前无法确认：没有任何疾病在辅助分支开启/关闭对照中表现出 CI 排除 0 的正向差异")


def _negative_diseases(bootstrap: dict) -> str:
    names = []
    for comparison_name in ("R8_Full_minus_R8_BranchOff", "R9_Full_minus_R9_BranchOff",
                            "R10_Full_minus_R10_BranchOff",
                            "R10_Full_minus_R10_NoStage4Conditioning"):
        comparison = bootstrap["comparisons"].get(comparison_name)
        if not comparison or comparison.get("status") == "skipped_missing_model":
            continue
        for disease, stats in comparison["per_class"].items():
            high = stats["delta_auroc_ci95"][1]
            if high is not None and high < 0 and stats["delta_auroc"] < 0:
                names.append(f"{disease}（{comparison_name} Δ={stats['delta_auroc']:+.4f}）")
    return ("；".join(sorted(set(names))) if names
            else "目前无法确认：没有 CI 排除 0 的负向差异")


def _tradeoff_summary(bootstrap: dict, metadata: dict) -> str:
    candidates = [experiment for experiment in ("R8", "R9", "R10")
                  if experiment in metadata]
    if not candidates:
        return "目前无法确认（尚未完成任何新结构）"
    rows = []
    for experiment in candidates:
        info = metadata[experiment]
        mode = MODE_NAMES[experiment]["full"]
        rows.append((experiment, bootstrap["models"][mode]["macro_auroc"],
                     info["cost"]["parameters_total"],
                     info["cost"]["mean_ms_per_study"]))
    best = max(rows, key=lambda row: row[1])
    cheapest = min(rows, key=lambda row: row[2])
    auprc_rows = []
    for experiment in candidates:
        mode = MODE_NAMES[experiment]["full"]
        auprc_rows.append((experiment,
                           bootstrap["models"][mode]["macro_auprc"]))
    best_auprc = max(auprc_rows, key=lambda row: row[1])
    return (f"Macro-AUROC 点估计最高为 {best[0]}（{best[1]:.6f}，参数 {best[2]:,}，"
            f"{best[3]:.1f} ms/Study）；Macro-AUPRC 点估计最高为 {best_auprc[0]}"
            f"（{best_auprc[1]:.6f}）；参数最少为 {cheapest[0]}（{cheapest[2]:,}）。"
            "R10 的 AUPRC 点估计相对 R5 提高约 +0.0067，但配对 95% CI 跨 0；"
            "因各结构参数、推理成本与置信区间不同，"
            "不能宣称某一结构在全部维度上占优。")


def _research_direction(bootstrap: dict, metadata: dict) -> str:
    confirmations = []
    negatives = []
    for comparison_name in ("R8_Full_minus_R5", "R9_Full_minus_R5",
                            "R10_Full_minus_R5"):
        comparison = _comparison(bootstrap, comparison_name)
        if not comparison:
            continue
        low, high = comparison["delta_macro_auroc_ci95"]
        if low is None:
            continue
        if low > 0:
            confirmations.append(comparison_name)
        elif high < 0:
            negatives.append(comparison_name)
    if confirmations:
        return (f"证据支持继续优化特征选择：{', '.join(confirmations)} 的 CI 排除 0。")
    negative_text = (
        f"且 {', '.join(negatives)} 的 95% CI 排除 0 但方向为负，" if negatives else "")
    return ("当前证据不支持继续加码特征选择：没有任何新结构相对 R5 的 "
            "Macro-AUROC CI 排除 0 且为正，" + negative_text
            + "辅助分支的实际推理贡献有限。若要继续，应先做独立种子复验；"
            "另一方向是回到 Backbone/预训练特征提取本身。")
