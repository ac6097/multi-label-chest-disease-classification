"""Shared utilities for the G0/G1 generalization diagnostics.

Read-only with respect to existing R3-R10 checkpoints and reports: this module
only loads models/checkpoints and runs deterministic inference.  New artifacts
are written under outputs/experiments/DINOv3_Generalization_Diagnostics/.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from dino_data import EpochTaggedStudyDataset, read_data_config  # noqa: E402
from dino_experiment import load_pretrained_backbone  # noqa: E402
from dino_feature_models import MaskedStudyDataset  # noqa: E402
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from dino_model import DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from dino_feature_models import DINOv3GlobalLocalClassifier  # noqa: E402
from datasets.dataset import StudyDataset  # noqa: E402
from hybrid.common import seed_everything  # noqa: E402
from stage_fusion_analysis import (  # noqa: E402
    _sigmoid,
    bootstrap_models,
    make_subject_draws,
    study_metrics,
)

OUT_ROOT = HERE / "outputs" / "experiments" / "DINOv3_Feature_Optimization"
R3_DIR = OUT_ROOT / "R3_Global_Local"
R5_DIR = OUT_ROOT / "R5_Global_Only_512"
R3_PACKAGE = R3_DIR / "submission_export" / "R3_Inference"
R5_PACKAGE = R5_DIR / "submission_export" / "R5_Inference"
DATA_CONFIG = PROJECT / "configs" / "hybrid_data.yaml"
DIAG_ROOT = HERE / "outputs" / "experiments" / "DINOv3_Generalization_Diagnostics"
G0_DIR = DIAG_ROOT / "G0_Inference_Consistency"
G1_METRICS_DIR = DIAG_ROOT / "G1_R3_R5_Generalization"
G1_DRIFT_DIR = DIAG_ROOT / "G1_Parameter_Feature_Drift"
G1_AGG_DIR = DIAG_ROOT / "G1_Study_Aggregation_Check"
G2_DIR = DIAG_ROOT / "G2_Backbone_LR_Comparison"
FINAL_DIR = DIAG_ROOT / "Final_Decision"
CACHE_DIR = DIAG_ROOT / "cache"

SEED = 42
IMAGE_SIZE = 512
EXPECTED_WEIGHT_SHA256 = (
    "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2")
LABEL_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity",
    "Pleural Effusion", "No Finding",
]
RESEARCH_RUNTIME = {
    "image_budget": 8, "grad_accumulation": 4, "num_workers": 4,
    "pin_memory": True, "prefetch_factor": 2, "channels_last": True,
}


def ensure_directories() -> None:
    for directory in (DIAG_ROOT, G0_DIR, G1_METRICS_DIR, G1_DRIFT_DIR,
                      G1_AGG_DIR, G2_DIR, FINAL_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    from hybrid.common import write_csv as _write_csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_eval_datasets(image_size: int = IMAGE_SIZE):
    """Train/Validation datasets with deterministic (no-augmentation) transforms."""
    config = read_data_config(DATA_CONFIG, image_size=image_size)
    shared = {
        "label_column": config["label_column"],
        "label_semantics": config["label_semantics"],
        "image_id_column": config["image_id_column"],
        "subject_column": config["subject_column"],
        "study_column": config["study_column"],
    }
    train_base = StudyDataset(config["train_csv"], config["image_root"],
                              transform=None, **shared)
    valid_base = StudyDataset(config["val_csv"], config["image_root"],
                              transform=None, **shared)
    if train_base.split != "train" or valid_base.split != "valid":
        raise RuntimeError("Generalization diagnostics may only use official Train/Validation")
    train_masked = MaskedStudyDataset(train_base, image_size, train=False)
    valid_masked = MaskedStudyDataset(valid_base, image_size, train=False)
    train_data = EpochTaggedStudyDataset(train_masked, SEED)
    valid_data = EpochTaggedStudyDataset(valid_masked, SEED)
    return train_data, valid_data, config


def load_r3_model(device: torch.device, channels_last: bool = True,
                  weights: str = "checkpoint", checkpoint: str = "best"):
    """Load R3 (strict) either from a training checkpoint or the package."""
    config = read_json(R3_DIR / "config.json")
    model_config = config["experiment_config"]["model"]
    r3 = model_config["r3"]
    status = read_json(R3_DIR / "training_status.json")
    seed_everything(SEED)
    backbone, verification = load_pretrained_backbone("cpu")
    if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R3 diagnostics did not load the pinned DINOv3 weights")
    model = DINOv3GlobalLocalClassifier(
        backbone,
        topk_ratio=float(r3["topk_ratio"]),
        initial_alpha=float(r3["initial_alpha"]),
        maximum_alpha=float(r3["maximum_alpha"]),
        mask_valid_fraction_threshold=float(r3["mask_valid_fraction_threshold"]),
    )
    if weights == "checkpoint":
        payload = torch.load(R3_DIR / f"{checkpoint}.pth", map_location="cpu",
                             weights_only=False)
        if checkpoint == "best" and int(payload["epoch"]) != int(
                status["best_epoch"]):
            raise RuntimeError("R3 checkpoint epoch differs from its recorded Best epoch")
        state = payload["model_state"]
    elif weights == "package":
        state = load_file(str(R3_PACKAGE / "model.safetensors"), device="cpu")
    else:
        raise ValueError(f"Unknown R3 weight source: {weights}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"R3 strict load failed: {result}")
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, status, verification


def load_r5_model(device: torch.device, channels_last: bool = True,
                  weights: str = "checkpoint", checkpoint: str = "best"):
    status = read_json(R5_DIR / "training_status.json")
    seed_everything(SEED)
    backbone, verification = load_pretrained_backbone("cpu")
    if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R5 diagnostics did not load the pinned DINOv3 weights")
    model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS)
    if weights == "checkpoint":
        payload = torch.load(R5_DIR / f"{checkpoint}.pth", map_location="cpu",
                             weights_only=False)
        if checkpoint == "best" and int(payload["epoch"]) != int(
                status["best_epoch"]):
            raise RuntimeError("R5 checkpoint epoch differs from its recorded Best epoch")
        state = payload["model_state"]
    elif weights == "package":
        state = load_file(str(R5_PACKAGE / "model.safetensors"), device="cpu")
    else:
        raise ValueError(f"Unknown R5 weight source: {weights}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"R5 strict load failed: {result}")
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, status, verification


def _forward_batch(model, group: str, images: torch.Tensor, batch: dict):
    if group == "R3":
        masks = batch["valid_masks"].to(images.device, non_blocking=True)
        return model(images, masks)
    return model(images)


def infer_dataset(model, group: str, dataset, image_budget: int = 8,
                  num_workers: int = 4, channels_last: bool = True,
                  autocast: bool = True, collect_image_logits: bool = False):
    """Deterministic Study-level inference; returns per-Study arrays."""
    from dino_data import make_loader, shutdown_loader
    from dino_feature_models import collate_masked_studies

    device = next(model.parameters()).device
    loader, sampler, _ = make_loader(
        dataset, image_budget=int(image_budget), shuffle=False, seed=SEED,
        num_workers=int(num_workers), pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    study_logits, labels, subjects, studies, image_counts = [], [], [], [], []
    image_logit_chunks, image_study_chunks, image_id_chunks = [], [], []
    study_offset = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch["images"].to(device, non_blocking=True)
                if channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=autocast and device.type == "cuda"):
                    image_logits = _forward_batch(model, group, images, batch)
                study_index = batch["study_index"].to(device, non_blocking=True)
                counts = batch["image_counts"].to(device, non_blocking=True)
                per_study = study_logit_mean(
                    image_logits.float(), study_index, batch["num_studies"], counts)
                study_logits.append(per_study.cpu().numpy())
                labels.append(batch["labels"].numpy())
                subjects.extend(str(value) for value in batch["subject_id"])
                studies.extend(str(value) for value in batch["study_id"])
                image_counts.extend(int(value) for value in counts.tolist())
                if collect_image_logits:
                    image_logit_chunks.append(image_logits.float().cpu().numpy())
                    image_study_chunks.append(
                        (study_index + study_offset).cpu().numpy())
                    for image_group in batch["image_ids"]:
                        image_id_chunks.extend(str(value) for value in image_group)
                study_offset += int(batch["num_studies"])
    finally:
        shutdown_loader(loader)
    elapsed = time.perf_counter() - started
    peak_gpu = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0)
    result = {
        "study_logits": np.concatenate(study_logits, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.uint8),
        "subjects": np.asarray(subjects, dtype=object),
        "studies": np.asarray(studies, dtype=object),
        "image_counts": np.asarray(image_counts, dtype=np.int16),
        "elapsed_seconds": elapsed,
        "peak_gpu_gib": peak_gpu,
        "studies_per_second": len(subjects) / elapsed if elapsed else None,
        "images": int(sum(image_counts)),
        "channels_last": bool(channels_last),
        "autocast": bool(autocast),
        "group": group,
    }
    if collect_image_logits:
        result["image_logits"] = np.concatenate(image_logit_chunks, axis=0).astype(
            np.float32)
        result["image_study_index"] = np.concatenate(image_study_chunks, axis=0)
        result["image_ids"] = np.asarray(image_id_chunks, dtype=object)
    return result


def result_hash(result: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(result["study_logits"]).tobytes())
    for value in result["studies"]:
        digest.update(str(value).encode("utf-8") + b"\x1f")
    return digest.hexdigest()


def probabilities(result: dict) -> np.ndarray:
    return _sigmoid(result["study_logits"])


def metrics(result: dict) -> dict:
    return study_metrics(result["labels"], probabilities(result))


def save_result(path: Path, result: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: result[key] for key in result
              if isinstance(result[key], np.ndarray)}
    meta = {key: result[key] for key in result
            if not isinstance(result[key], np.ndarray)}
    np.savez_compressed(path, **arrays, meta=np.asarray(json.dumps(meta)))
    write_json(path.with_suffix(".json"), meta)


def load_result(path: Path) -> dict:
    path = Path(path)
    payload = np.load(path, allow_pickle=True)
    meta = json.loads(str(payload["meta"]))
    result = {key: payload[key] for key in payload.files if key != "meta"}
    result.update(meta)
    return result


def _rank_inversions(reference: np.ndarray, candidate: np.ndarray) -> int:
    """Number of pairwise order inversions of candidate under the reference order."""
    values = candidate[np.argsort(-reference)]
    upper = np.triu(values[:, None] > values[None, :], k=1)
    return int(upper.sum())


def compare_results(reference: dict, candidate: dict, top_fraction: float = 0.1) -> dict:
    """Study-level and per-class differences between two inference entries."""
    ref_logits = reference["study_logits"].astype(np.float64)
    cand_logits = candidate["study_logits"].astype(np.float64)
    if not (np.array_equal(reference["studies"], candidate["studies"])
            and np.array_equal(reference["subjects"], candidate["subjects"])):
        raise RuntimeError("Inference results are not Study-aligned")
    if not np.array_equal(reference["labels"], candidate["labels"]):
        raise RuntimeError("Inference results have different labels")
    ref_probability = _sigmoid(ref_logits)
    cand_probability = _sigmoid(cand_logits)
    logit_delta = np.abs(ref_logits - cand_logits)
    probability_delta = np.abs(ref_probability - cand_probability)
    per_class = {}
    reference_metrics = study_metrics(reference["labels"], ref_probability)
    candidate_metrics = study_metrics(candidate["labels"], cand_probability)
    for index, disease in enumerate(LABEL_NAMES):
        ref_rank = np.argsort(np.argsort(-ref_probability[:, index]))
        cand_rank = np.argsort(np.argsort(-cand_probability[:, index]))
        top_count = max(1, int(round(len(ref_rank) * top_fraction)))
        top_reference = set(np.argsort(-ref_probability[:, index])[:top_count])
        top_candidate = set(np.argsort(-cand_probability[:, index])[:top_count])
        ref_auc = reference_metrics["per_class"][disease]["auroc"]
        cand_auc = candidate_metrics["per_class"][disease]["auroc"]
        per_class[disease] = {
            "auroc_reference": ref_auc,
            "auroc_candidate": cand_auc,
            "auroc_delta": cand_auc - ref_auc,
            "auprc_reference": reference_metrics["per_class"][disease]["auprc"],
            "auprc_candidate": candidate_metrics["per_class"][disease]["auprc"],
            "auprc_delta": (
                candidate_metrics["per_class"][disease]["auprc"]
                - reference_metrics["per_class"][disease]["auprc"]),
            "max_probability_abs_delta": float(probability_delta[:, index].max()),
            "mean_probability_abs_delta": float(probability_delta[:, index].mean()),
            "max_logit_abs_delta": float(logit_delta[:, index].max()),
            "mean_logit_abs_delta": float(logit_delta[:, index].mean()),
            "n_rank_changed": int(np.sum(ref_rank != cand_rank)),
            "n_pairwise_inversions": _rank_inversions(
                ref_probability[:, index], cand_probability[:, index]),
            "n_top_decile_changed": int(len(top_reference ^ top_candidate)),
        }
    return {
        "max_logit_abs_delta": float(logit_delta.max()),
        "mean_logit_abs_delta": float(logit_delta.mean()),
        "max_probability_abs_delta": float(probability_delta.max()),
        "mean_probability_abs_delta": float(probability_delta.mean()),
        "macro_auroc_reference": reference_metrics["macro_auroc"],
        "macro_auroc_candidate": candidate_metrics["macro_auroc"],
        "macro_auroc_delta": (
            candidate_metrics["macro_auroc"] - reference_metrics["macro_auroc"]),
        "macro_auprc_reference": reference_metrics["macro_auprc"],
        "macro_auprc_candidate": candidate_metrics["macro_auprc"],
        "macro_auprc_delta": (
            candidate_metrics["macro_auprc"] - reference_metrics["macro_auprc"]),
        "per_class": per_class,
        "study_logits_sha256": result_hash(candidate),
    }


def study_bce(result: dict) -> float:
    logits = torch.from_numpy(result["study_logits"].astype(np.float32))
    labels = torch.from_numpy(result["labels"].astype(np.float32))
    known = torch.ones_like(labels, dtype=torch.bool)
    per_study, eligible = study_bce_per_study(logits, labels, known)
    return float(per_study[eligible].mean())


def saved_val_best_frame(group: str):
    import pandas as pd

    path = R3_DIR / "val_best.csv" if group == "R3" else R5_DIR / "val_best.csv"
    return pd.read_csv(path, dtype={"Subject_id": "string", "Study_id": "string"})


def compare_with_saved(result: dict, frame) -> dict:
    """Compare a fresh inference result with the training-time saved val_best.csv."""
    import pandas as pd

    probability_columns = [f"prob_{name.replace(' ', '_')}" for name in LABEL_NAMES]
    base = pd.DataFrame({"Subject_id": pd.Series(result["subjects"], dtype="string"),
                         "Study_id": pd.Series(result["studies"], dtype="string")})
    merged = base.merge(frame, on=["Subject_id", "Study_id"], how="left",
                        validate="one_to_one", sort=False)
    missing = int(merged[probability_columns].isna().any(axis=1).sum())
    saved = merged[probability_columns].to_numpy(dtype=np.float64)
    fresh = probabilities(result).astype(np.float64)
    valid = ~np.isnan(saved).any(axis=1)
    delta = np.abs(saved[valid] - fresh[valid])
    saved_metrics = study_metrics(result["labels"][valid], saved[valid])
    fresh_metrics = study_metrics(result["labels"][valid], fresh[valid])
    return {
        "saved_file": str(R3_DIR / "val_best.csv") if result["group"] == "R3"
        else str(R5_DIR / "val_best.csv"),
        "n_studies": int(valid.sum()),
        "n_missing_in_saved": missing,
        "max_probability_abs_delta": float(delta.max()),
        "mean_probability_abs_delta": float(delta.mean()),
        "cells_over_1e-5": int((delta > 1e-5).sum()),
        "saved_macro_auroc": saved_metrics["macro_auroc"],
        "fresh_macro_auroc": fresh_metrics["macro_auroc"],
        "macro_auroc_delta": fresh_metrics["macro_auroc"] - saved_metrics["macro_auroc"],
        "saved_macro_auprc": saved_metrics["macro_auprc"],
        "fresh_macro_auprc": fresh_metrics["macro_auprc"],
        "per_class_max_delta": {
            name: float(delta[:, index].max()) for index, name in enumerate(LABEL_NAMES)},
        "per_study_max_delta": {
            "subject": str(result["subjects"][int(np.argmax(delta.max(axis=1)))]),
            "study": str(result["studies"][int(np.argmax(delta.max(axis=1)))]),
            "value": float(delta.max()),
        },
    }


def model_state_comparison(group: str) -> dict:
    """Compare training-checkpoint model_state with packaged inference weights."""
    if group == "R3":
        payload = torch.load(R3_DIR / "best.pth", map_location="cpu", weights_only=False)
        state = payload["model_state"]
        package = load_file(str(R3_PACKAGE / "model.safetensors"), device="cpu")
        model_config = read_json(R3_PACKAGE / "model_config.json")["model"]
    else:
        payload = torch.load(R5_DIR / "best.pth", map_location="cpu", weights_only=False)
        state = payload["model_state"]
        package = load_file(str(R5_PACKAGE / "model.safetensors"), device="cpu")
        model_config = read_json(R5_PACKAGE / "model_config.json")["model"]
    key_mismatch = sorted(set(state) ^ set(package))
    max_delta = 0.0
    exact = True
    for name in sorted(set(state) & set(package)):
        left = state[name].detach().float()
        right = package[name].detach().float()
        if tuple(left.shape) != tuple(right.shape):
            raise RuntimeError(f"{group} state shape mismatch for {name}")
        delta = float((left - right).abs().max())
        max_delta = max(max_delta, delta)
        exact = exact and bool(torch.equal(left, right))
    alpha = None
    if "alpha.logit" in state:
        raw = float(state["alpha.logit"].detach().float())
        alpha = 0.2 / (1.0 + math.exp(-raw))
    return {
        "group": group,
        "checkpoint_state_hash": tensor_state_hash(state),
        "package_state_hash": tensor_state_hash(package),
        "state_tensor_count_checkpoint": len(state),
        "state_tensor_count_package": len(package),
        "key_mismatch": key_mismatch,
        "max_abs_tensor_delta": max_delta,
        "bitwise_equal": bool(exact and not key_mismatch),
        "alpha_value_from_checkpoint": alpha,
        "package_model_config_channels_last": model_config.get("channels_last"),
        "package_model_config": model_config,
    }
