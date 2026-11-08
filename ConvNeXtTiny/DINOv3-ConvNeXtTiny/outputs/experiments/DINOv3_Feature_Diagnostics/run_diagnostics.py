from __future__ import annotations

import gc
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[2]
PROJECT_DIR = WORK_DIR.parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WORK_DIR))

from datasets.dataset import StudyDataset
from datasets.transforms import build_transform
from dino_data import EpochTaggedStudyDataset, make_loader, shutdown_loader
from dino_feature_models import (
    DINOv3GlobalLocalClassifier,
    MaskedStudyDataset,
    collate_masked_studies,
)
from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier


LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
SHORT = ["Enlarged CM", "Pneumothorax", "Consolidation", "Pneumonia",
         "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity",
         "Pleural Effusion", "No Finding"]
FOCUS = ["Pneumonia", "Lung Opacity", "Pneumothorax", "Consolidation",
         "Atelectasis", "Enlarged Cardiomediastinum"]
TARGET_COLS = [f"target_{name.replace(' ', '_')}" for name in LABELS]
PROB_COLS = [f"prob_{name.replace(' ', '_')}" for name in LABELS]
DATA_DIR = REPO_DIR / "cxr_first"
VALID_CSV = DATA_DIR / "valid.csv"
R0_DIR = WORK_DIR / "outputs/experiments/DINOv3_P0_P1_P2/p2"
FEATURE_ROOT = WORK_DIR / "outputs/experiments/DINOv3_Feature_Optimization"
RUNS = {
    "R0": R0_DIR,
    "R1": FEATURE_ROOT / "R1_448_GAP",
    "R2": FEATURE_ROOT / "R2_512_GAP",
    "R3": FEATURE_ROOT / "R3_Global_Local",
}
PREDICTIONS = {
    "R0": R0_DIR / "predictions/val_best.csv",
    "R1": RUNS["R1"] / "predictions/val_best.csv",
    "R2": RUNS["R2"] / "predictions/val_best.csv",
    "R3": RUNS["R3"] / "predictions/val_best.csv",
}
CHECKPOINTS = {key: path / "best.pth" for key, path in RUNS.items()}
MODEL_CONFIGS = {
    key: RUNS[key] / "submission_export" / f"{key}_Inference" / "model_config.json"
    for key in ("R2", "R3")
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
EXPECTED_STUDIES = 859
EXPECTED_SUBJECTS = 228


def json_write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def key_tuples(frame: pd.DataFrame):
    return list(zip(frame["Subject_id"].astype(str), frame["Study_id"].astype(str)))


def official_dataset():
    if not VALID_CSV.is_file():
        raise FileNotFoundError(f"Official Validation CSV is missing: {VALID_CSV}")
    base = StudyDataset(
        VALID_CSV, DATA_DIR, transform=None, label_column="Predict_class",
        label_semantics="closed_world", image_id_column="Dicom_id",
        subject_column="Subject_id", study_column="Study_id",
    )
    records = []
    for study in base.studies:
        row = {"Subject_id": study["subject_id"], "Study_id": study["study_id"]}
        row.update({column: float(study["target"][index])
                    for index, column in enumerate(TARGET_COLS)})
        records.append(row)
    labels = np.asarray([[float(study["target"][i]) for i in range(10)]
                         for study in base.studies], dtype=np.int64)
    return base, pd.DataFrame(records), labels


def read_predictions(path: Path, official: pd.DataFrame, official_labels: np.ndarray):
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"Subject_id": "string", "Study_id": "string"})
    required = {"Subject_id", "Study_id", *TARGET_COLS, *PROB_COLS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} missing required columns: {missing_columns}")
    if frame[["Subject_id", "Study_id"]].isna().any().any():
        raise ValueError(f"{path} has missing Study identifiers")
    duplicate = frame.duplicated(["Subject_id", "Study_id"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["Subject_id", "Study_id"]].head(10).to_dict("records")
        raise ValueError(f"{path} has duplicate Study keys: {examples}")
    expected_keys, found_keys = set(key_tuples(official)), set(key_tuples(frame))
    missing = sorted(expected_keys - found_keys)
    extra = sorted(found_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"{path} Study set mismatch: missing={len(missing)} {missing[:10]}, "
            f"extra={len(extra)} {extra[:10]}"
        )
    ordered = official[["Subject_id", "Study_id"]].merge(
        frame, on=["Subject_id", "Study_id"], how="left", validate="one_to_one", sort=False)
    target_values = ordered[TARGET_COLS].to_numpy(dtype=np.int64)
    if not np.array_equal(target_values, official_labels):
        mismatch = np.argwhere(target_values != official_labels)[0]
        raise ValueError(
            f"{path} target mismatch at Study row {int(mismatch[0])}, label {LABELS[int(mismatch[1])]}; "
            "official closed-world Validation labels remain authoritative"
        )
    scores = ordered[PROB_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return ordered, scores


def metric_vectors(labels: np.ndarray, scores: np.ndarray):
    aucs, aps = [], []
    for column in range(labels.shape[1]):
        y = labels[:, column]
        aucs.append(float(roc_auc_score(y, scores[:, column]))
                    if np.unique(y).size == 2 else float("nan"))
        aps.append(float(average_precision_score(y, scores[:, column]))
                    if np.any(y == 1) else float("nan"))
    return np.asarray(aucs), np.asarray(aps)


def macro(values):
    return float(np.nanmean(values))


def make_subject_draws(official: pd.DataFrame):
    subjects = official["Subject_id"].astype(str).to_numpy()
    keys = sorted(set(subjects))
    rows_by_subject = {subject: np.flatnonzero(subjects == subject) for subject in keys}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    digest = __import__("hashlib").sha256()
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([rows_by_subject[subject] for subject in sampled])
        draws.append(indices)
        digest.update(indices.astype("<i4", copy=False).tobytes())
        digest.update(b"\xff")
    return draws, digest.hexdigest()


def paired_bootstrap(labels: np.ndarray, scores_by_name: dict[str, np.ndarray],
                     comparisons: list[tuple[str, str]], draws: list[np.ndarray],
                     draws_sha256: str):
    result = {
        "method": "Subject-cluster paired bootstrap; percentile 95% CI",
        "unit": "Subject_id; all Studies of a sampled Subject are resampled together",
        "seed": BOOTSTRAP_SEED,
        "requested_replicates": BOOTSTRAP_REPLICATES,
        "valid_macro_replicates": BOOTSTRAP_REPLICATES,
        "sampling_indices_sha256": draws_sha256,
        "one_class_resample_rule": (
            "Skip AUROC for a class if that replicate has only one label; macro delta is "
            "the mean of available paired class AUROC deltas. Per-class CI uses only "
            "replicates where both classes are present."
        ),
        "comparisons": {},
    }
    for left, right in comparisons:
        point_left, _ = metric_vectors(labels, scores_by_name[left])
        point_right, _ = metric_vectors(labels, scores_by_name[right])
        point_delta = point_right - point_left
        per_class = [[] for _ in LABELS]
        macro_deltas = []
        valid_macro_count = 0
        for indices in draws:
            y_boot = labels[indices]
            d_auc = []
            for class_index in range(len(LABELS)):
                y = y_boot[:, class_index]
                if np.unique(y).size < 2:
                    continue
                auc_left = roc_auc_score(y, scores_by_name[left][indices, class_index])
                auc_right = roc_auc_score(y, scores_by_name[right][indices, class_index])
                delta = float(auc_right - auc_left)
                per_class[class_index].append(delta)
                d_auc.append(delta)
            if d_auc:
                macro_deltas.append(float(np.mean(d_auc)))
                valid_macro_count += 1
        if valid_macro_count != BOOTSTRAP_REPLICATES:
            raise RuntimeError(
                f"{left}->{right}: only {valid_macro_count} valid macro bootstrap replicates")
        result["comparisons"][f"{right}_minus_{left}"] = {
            "left": left,
            "right": right,
            "macro_auroc_delta": macro(point_right) - macro(point_left),
            "macro_auroc_ci95": np.quantile(macro_deltas, [0.025, 0.975]).tolist(),
            "valid_macro_replicates": valid_macro_count,
            "per_class": {
                name: {
                    "delta": float(point_delta[i]),
                    "ci95": np.quantile(per_class[i], [0.025, 0.975]).tolist()
                    if per_class[i] else [None, None],
                    "valid_replicates": len(per_class[i]),
                }
                for i, name in enumerate(LABELS)
            },
        }
    return result


def build_model(run: str, device: torch.device):
    model_cfg = json.loads(MODEL_CONFIGS[run].read_text(encoding="utf-8"))["model"]
    backbone = DINOv3ConvNeXtBackbone(
        layer_norm_eps=float(model_cfg["layer_norm_eps"]),
        layer_scale_init_value=float(model_cfg["layer_scale_init_value"]),
        drop_path_rate=float(model_cfg["drop_path_rate"]),
        architecture=str(model_cfg["architecture"]),
    )
    if run == "R3":
        cfg = model_cfg["r3"]
        model = DINOv3GlobalLocalClassifier(
            backbone,
            topk_ratio=float(cfg["topk_ratio"]),
            initial_alpha=float(cfg["initial_alpha"]),
            maximum_alpha=float(cfg["maximum_alpha"]),
            mask_valid_fraction_threshold=float(cfg["mask_valid_fraction_threshold"]),
        )
    else:
        model = DINOv3StudyClassifier(backbone, num_labels=10)
    checkpoint = torch.load(CHECKPOINTS[run], map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        raise KeyError(f"{run} best checkpoint has no model_state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    metadata = {
        "checkpoint": str(CHECKPOINTS[run]),
        "checkpoint_bytes": CHECKPOINTS[run].stat().st_size,
        "checkpoint_epoch": int(checkpoint.get("epoch", model_cfg.get("best_epoch", -1))),
        "config_best_epoch": int(model_cfg["best_epoch"]),
        "checkpoint_best_macro_auroc": float(
            checkpoint.get("best_macro_auroc", model_cfg["best_validation_macro_auroc"])),
        "image_size": int(model_cfg["image_size"]),
        "state_tensor_count": len(checkpoint["model_state"]),
    }
    if metadata["checkpoint_epoch"] != metadata["config_best_epoch"]:
        raise RuntimeError(f"{run} checkpoint epoch disagrees with its export metadata: {metadata}")
    alpha = None
    if run == "R3":
        alpha_logit = checkpoint["model_state"]["alpha.logit"].detach().float()
        alpha = float((model.maximum_alpha * torch.sigmoid(alpha_logit)).item())
        metadata["alpha_best_checkpoint"] = alpha
        metadata["topk_ratio"] = model.topk_ratio
        metadata["mask_valid_fraction_threshold"] = model.mask_valid_fraction_threshold
    del checkpoint
    gc.collect()
    model.to(device).eval()
    runtime_path = RUNS[run] / "selected_runtime_config.json"
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime = runtime_payload.get("selected_runtime", runtime_payload)
    channels_last = bool(runtime.get("channels_last", False))
    metadata["training_runtime_image_budget"] = int(runtime.get("image_budget", 8))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    metadata["training_runtime_channels_last"] = channels_last
    metadata["export_metadata_channels_last"] = bool(model_cfg.get("channels_last", False))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata, alpha


def infer_r3(valid_base: StudyDataset, device: torch.device):
    model, metadata, alpha = build_model("R3", device)
    data = EpochTaggedStudyDataset(
        MaskedStudyDataset(valid_base, 512, train=False), BOOTSTRAP_SEED)
    loader, _, _ = make_loader(
        data, image_budget=metadata["training_runtime_image_budget"], shuffle=False, seed=BOOTSTRAP_SEED,
        num_workers=2, pin_memory=device.type == "cuda",
        collate_fn=collate_masked_studies,
    )
    study_rows: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    image_records = []
    source_studies = {
        (study["subject_id"], study["study_id"]): study
        for study in valid_base.studies
    }
    model.eval()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                images = batch["images"].to(device, non_blocking=True)
                if metadata["training_runtime_channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                masks = batch["valid_masks"].to(device, non_blocking=True)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        components = model(images, masks, return_components=True)
                else:
                    components = model(images, masks, return_components=True)
                image_global = components["global_logits"].float().cpu().numpy()
                image_local = components["local_logits"].float().cpu().numpy()
                image_fused = components["logits"].float().cpu().numpy()
                maps = components["response_maps"].float().cpu().numpy()
                if maps.shape[-2:] != (32, 32):
                    raise RuntimeError(f"Expected Stage3 maps 32x32 at 512 input, got {maps.shape}")
                valid_grid = components["valid_grid"].cpu().numpy().astype(bool)
                topk_grid = components["topk_grid"].cpu().numpy().astype(bool)
                selected_invalid = int(np.logical_and(topk_grid,
                                                       ~valid_grid[:, None]).sum())
                if selected_invalid:
                    raise RuntimeError(f"R3 Top-K selected {selected_invalid} invalid grid positions")
                local_counts = np.asarray(components["valid_counts"], dtype=np.int32)
                selected_counts = np.asarray(components["selected_counts"], dtype=np.int32)
                study_index = batch["study_index"].cpu().numpy().astype(int)
                seen_in_study = np.zeros(batch["num_studies"], dtype=np.int32)
                for image_index, study_index_value in enumerate(study_index):
                    sidx = int(study_index_value)
                    subject = str(batch["subject_id"][sidx])
                    study = str(batch["study_id"][sidx])
                    study_key = (subject, study)
                    within = int(seen_in_study[sidx])
                    seen_in_study[sidx] += 1
                    image_id = str(batch["image_ids"][sidx][within])
                    study_source = source_studies[study_key]
                    image_path = study_source["image_paths"][within]
                    record = {
                        "subject_id": subject,
                        "study_id": study,
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "global_logits": image_global[image_index],
                        "local_logits": image_local[image_index],
                        "fused_logits": image_fused[image_index],
                        "response_maps": maps[image_index],
                        "valid_grid": valid_grid[image_index],
                        "topk_grid": topk_grid[image_index],
                        "valid_count": int(local_counts[image_index]),
                        "selected_count": int(selected_counts[image_index]),
                    }
                    image_records.append(record)
                    group = study_rows.setdefault(study_key, {
                        "global_logits": [], "local_logits": [], "fused_logits": []})
                    group["global_logits"].append(image_global[image_index])
                    group["local_logits"].append(image_local[image_index])
                    group["fused_logits"].append(image_fused[image_index])
                if batch_number % 25 == 0:
                    print(f"R3 inference: {batch_number} batches, {len(image_records)} images",
                          flush=True)
    finally:
        shutdown_loader(loader)
    keys, g, l, f = [], [], [], []
    for study in valid_base.studies:
        key = (study["subject_id"], study["study_id"])
        values = study_rows.get(key)
        if values is None:
            raise RuntimeError(f"R3 inference omitted Study {key}")
        keys.append(key)
        g.append(np.mean(values["global_logits"], axis=0))
        l.append(np.mean(values["local_logits"], axis=0))
        f.append(np.mean(values["fused_logits"], axis=0))
    outputs = {
        "keys": keys,
        "global_logits": np.asarray(g, dtype=np.float64),
        "local_logits": np.asarray(l, dtype=np.float64),
        "fused_logits": np.asarray(f, dtype=np.float64),
        "alpha": alpha,
        "image_records": image_records,
        "metadata": metadata,
        "seconds": time.perf_counter() - started,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def infer_r2(valid_base: StudyDataset, device: torch.device):
    model, metadata, _ = build_model("R2", device)
    transform = build_transform(512, train=False)
    base_data = StudyDataset(
        VALID_CSV, DATA_DIR, transform=transform, label_column="Predict_class",
        label_semantics="closed_world", image_id_column="Dicom_id",
        subject_column="Subject_id", study_column="Study_id",
    )
    data = EpochTaggedStudyDataset(base_data, BOOTSTRAP_SEED)
    loader, _, _ = make_loader(
        data, image_budget=metadata["training_runtime_image_budget"], shuffle=False, seed=BOOTSTRAP_SEED,
        num_workers=2, pin_memory=device.type == "cuda",
    )
    study_rows: dict[tuple[str, str], list[np.ndarray]] = {}
    model.eval()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                images = batch["images"].to(device, non_blocking=True)
                if metadata["training_runtime_channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        logits = model(images)
                else:
                    logits = model(images)
                logits = logits.float().cpu().numpy()
                indices = batch["study_index"].cpu().numpy().astype(int)
                for image_index, study_index_value in enumerate(indices):
                    sidx = int(study_index_value)
                    key = (str(batch["subject_id"][sidx]), str(batch["study_id"][sidx]))
                    study_rows.setdefault(key, []).append(logits[image_index])
                if batch_number % 25 == 0:
                    print(f"R2 inference: {batch_number} batches", flush=True)
    finally:
        shutdown_loader(loader)
    rows = []
    for study in valid_base.studies:
        key = (study["subject_id"], study["study_id"])
        if key not in study_rows:
            raise RuntimeError(f"R2 inference omitted Study {key}")
        rows.append(np.mean(study_rows[key], axis=0))
    result = {
        "keys": [(s["subject_id"], s["study_id"]) for s in valid_base.studies],
        "logits": np.asarray(rows, dtype=np.float64),
        "metadata": metadata,
        "seconds": time.perf_counter() - started,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_input_artifacts(output_dir: Path, valid_base: StudyDataset,
                         official: pd.DataFrame, labels: np.ndarray,
                         saved_frames: dict[str, pd.DataFrame], scores: dict[str, np.ndarray]):
    report = {
        "validation_csv": str(VALID_CSV),
        "image_root": str(DATA_DIR),
        "validation_studies": len(valid_base),
        "validation_subjects": int(official["Subject_id"].nunique()),
        "validation_images": int(sum(valid_base.image_counts)),
        "label_order": LABELS,
        "label_semantics": "closed_world (omitted Predict_class IDs encoded negative)",
        "train_accessed": False,
        "test_accessed": False,
        "runs": {},
    }
    for run, frame in saved_frames.items():
        aucs, aps = metric_vectors(labels, scores[run])
        report["runs"][run] = {
            "prediction_file": str(PREDICTIONS[run]),
            "rows": len(frame),
            "unique_study_keys": int(frame[["Subject_id", "Study_id"]].drop_duplicates().shape[0]),
            "target_agreement": True,
            "macro_auroc_from_saved_predictions": macro(aucs),
            "macro_auprc_from_saved_predictions": macro(aps),
        }
    json_write(output_dir / "input_alignment_audit.json", report)
    return report


def validate_checkpoint_reproduction(run: str, saved: np.ndarray, reproduced: np.ndarray,
                                     labels: np.ndarray):
    old_auc, old_ap = metric_vectors(labels, saved)
    new_auc, new_ap = metric_vectors(labels, reproduced)
    delta = reproduced - saved
    return {
        "saved_macro_auroc": macro(old_auc),
        "reproduced_macro_auroc": macro(new_auc),
        "macro_auroc_abs_error": abs(macro(new_auc) - macro(old_auc)),
        "saved_macro_auprc": macro(old_ap),
        "reproduced_macro_auprc": macro(new_ap),
        "max_probability_abs_error": float(np.max(np.abs(delta))),
        "mean_probability_abs_error": float(np.mean(np.abs(delta))),
        "rmse_probability": float(np.sqrt(np.mean(np.square(delta)))),
        "max_per_class_auroc_abs_error": float(np.nanmax(np.abs(new_auc - old_auc))),
    }


def geometry(width: int, height: int, size: int = 512):
    scale = size / max(width, height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - rw) // 2, (size - rh) // 2
    return {"scale": scale, "resized_width": rw, "resized_height": rh,
            "left": left, "top": top, "right": size - rw - left,
            "bottom": size - rh - top}


def response_statistics(outputs, labels, scores_r2, scores_r3):
    records = outputs["image_records"]
    valid_cells = [[] for _ in LABELS]
    selected_cells = [[] for _ in LABELS]
    padding_shares = [[] for _ in LABELS]
    valid_pixel_areas = [[] for _ in LABELS]
    selected_valid_pixel_shares = [[] for _ in LABELS]
    entropies = [[] for _ in LABELS]
    peaks = [[] for _ in LABELS]
    local_maps = [[] for _ in LABELS]
    pair_correlations = {f"{LABELS[i]}|{LABELS[j]}": []
                         for i in range(10) for j in range(i + 1, 10)}
    label_lookup = {key: i for i, key in enumerate(outputs["keys"])}
    local_logits_by_study = outputs["local_logits"]
    for record in records:
        with Image.open(record["image_path"]) as source:
            width, height = source.size
        geom = geometry(width, height)
        valid_pixel = np.zeros((512, 512), dtype=bool)
        l, t = geom["left"], geom["top"]
        valid_pixel[t:t + geom["resized_height"],
                    l:l + geom["resized_width"]] = True
        valid = record["valid_grid"]
        topk = record["topk_grid"]
        # Feature-map cells are 16x16 at 512 input; this is the exact Stage3 grid scale.
        topk_pixels = np.repeat(np.repeat(topk, 16, axis=1), 16, axis=2)
        for class_index in range(10):
            count = int(valid.sum())
            selected = int(topk[class_index].sum())
            valid_cells[class_index].append(count)
            selected_cells[class_index].append(selected)
            selected_pixel_mask = topk_pixels[class_index]
            total_selected_pixels = int(selected_pixel_mask.sum())
            padding_shares[class_index].append(
                float((selected_pixel_mask & ~valid_pixel).sum() / max(total_selected_pixels, 1)))
            valid_pixel_area = int(valid_pixel.sum())
            valid_pixel_areas[class_index].append(valid_pixel_area)
            selected_valid_pixel_shares[class_index].append(
                float((selected_pixel_mask & valid_pixel).sum() / max(valid_pixel_area, 1)))
            values = record["response_maps"][class_index][valid]
            if values.size:
                probs = np.exp(values - np.max(values))
                probs /= max(float(probs.sum()), 1e-30)
                entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-30))))
                entropies[class_index].append(entropy / math.log(max(len(values), 2)))
                peaks[class_index].append(float(np.max(values) - np.mean(values)))
                local_maps[class_index].append(record["response_maps"][class_index].copy())
        for i in range(10):
            for j in range(i + 1, 10):
                values_i = record["response_maps"][i][valid]
                values_j = record["response_maps"][j][valid]
                if values_i.size > 1 and np.std(values_i) > 0 and np.std(values_j) > 0:
                    pair_correlations[f"{LABELS[i]}|{LABELS[j]}"].append(
                        float(np.corrcoef(values_i, values_j)[0, 1]))
    rows = []
    y = labels
    for class_index, name in enumerate(LABELS):
        positives = y[:, class_index] == 1
        negatives = ~positives
        local = local_logits_by_study[:, class_index]
        rows.append({
            "disease": name,
            "positive_studies": int(positives.sum()),
            "negative_studies": int(negatives.sum()),
            "image_count": len(records),
            "mean_valid_stage3_cells": float(np.mean(valid_cells[class_index])),
            "mean_topk_cells": float(np.mean(selected_cells[class_index])),
            "mean_topk_valid_grid_fraction": float(np.mean(
                np.asarray(selected_cells[class_index]) / np.maximum(valid_cells[class_index], 1))),
            "mean_topk_selected_pixel_share_of_valid_area": float(np.mean(
                selected_valid_pixel_shares[class_index])),
            "mean_topk_pixel_share_in_padding": float(np.mean(padding_shares[class_index])),
            "max_topk_pixel_share_in_padding": float(np.max(padding_shares[class_index])),
            "normalized_spatial_entropy_mean": float(np.mean(entropies[class_index])),
            "normalized_spatial_entropy_std": float(np.std(entropies[class_index], ddof=1)),
            "response_max_minus_mean_mean": float(np.mean(peaks[class_index])),
            "response_max_minus_mean_std": float(np.std(peaks[class_index], ddof=1)),
            "local_logit_positive_mean": float(np.mean(local[positives])),
            "local_logit_positive_std": float(np.std(local[positives], ddof=1)),
            "local_logit_negative_mean": float(np.mean(local[negatives])),
            "local_logit_negative_std": float(np.std(local[negatives], ddof=1)),
            "r3_vs_r2_correct_direction_studies": int(np.sum(
                (2 * y[:, class_index] - 1) * (scores_r3[:, class_index] - scores_r2[:, class_index]) > 0)),
            "r3_vs_r2_wrong_direction_studies": int(np.sum(
                (2 * y[:, class_index] - 1) * (scores_r3[:, class_index] - scores_r2[:, class_index]) < 0)),
            "r3_vs_r2_tied_studies": int(np.sum(
                scores_r3[:, class_index] == scores_r2[:, class_index])),
        })
    stats_frame = pd.DataFrame(rows)
    corr_rows = []
    for pair, values in pair_correlations.items():
        left, right = pair.split("|")
        corr_rows.append({
            "disease_left": left,
            "disease_right": right,
            "mean_response_map_correlation": float(np.mean(values)) if values else None,
            "std_response_map_correlation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "valid_images": len(values),
        })
    correlation_frame = pd.DataFrame(corr_rows)
    stats_frame["mean_abs_pairwise_disease_map_corr"] = [
        float(correlation_frame.loc[
            (correlation_frame.disease_left == name) | (correlation_frame.disease_right == name),
            "mean_response_map_correlation"].abs().mean()) for name in LABELS]
    stats_frame["max_abs_pairwise_disease_map_corr"] = [
        float(correlation_frame.loc[
            (correlation_frame.disease_left == name) | (correlation_frame.disease_right == name),
            "mean_response_map_correlation"].abs().max()) for name in LABELS]
    return stats_frame, correlation_frame


def logit_contribution_frame(outputs, scores_a, scores_b, scores_c, labels):
    g, l, f = outputs["global_logits"], outputs["local_logits"], outputs["fused_logits"]
    delta = f - g
    probability_a = scores_a
    probability_b = scores_b
    probability_shift = probability_a - probability_b
    rows = []
    for i, name in enumerate(LABELS):
        y = labels[:, i]
        pos, neg = y == 1, y == 0
        pair_flip_count = 0
        total_pairs = int(pos.sum() * neg.sum())
        # Count positive-negative pair ordering reversals between Global-only and fused scores.
        pos_g, neg_g = g[pos, i], g[neg, i]
        pos_f, neg_f = f[pos, i], f[neg, i]
        for start in range(0, len(pos_g), 64):
            before = pos_g[start:start + 64, None] - neg_g[None, :]
            after = pos_f[start:start + 64, None] - neg_f[None, :]
            pair_flip_count += int(np.sum((before > 0) != (after > 0)))
        auc_a, ap_a = metric_vectors(labels, scores_a)
        auc_b, ap_b = metric_vectors(labels, scores_b)
        auc_c, ap_c = metric_vectors(labels, scores_c)
        rows.append({
            "disease": name,
            "positive_studies": int(pos.sum()),
            "negative_studies": int(neg.sum()),
            "global_logit_mean": float(np.mean(g[:, i])),
            "global_logit_std": float(np.std(g[:, i], ddof=1)),
            "local_logit_mean": float(np.mean(l[:, i])),
            "local_logit_std": float(np.std(l[:, i], ddof=1)),
            "alpha": outputs["alpha"],
            "alpha_local_minus_global_logit_mean": float(np.mean(delta[:, i])),
            "alpha_local_minus_global_logit_std": float(np.std(delta[:, i], ddof=1)),
            "alpha_local_minus_global_logit_abs_mean": float(np.mean(np.abs(delta[:, i]))),
            "alpha_local_minus_global_logit_abs_p90": float(np.quantile(np.abs(delta[:, i]), 0.9)),
            "global_probability_mean": float(np.mean(probability_b[:, i])),
            "fused_probability_mean": float(np.mean(probability_a[:, i])),
            "probability_shift_mean": float(np.mean(probability_shift[:, i])),
            "probability_shift_std": float(np.std(probability_shift[:, i], ddof=1)),
            "probability_shift_abs_mean": float(np.mean(np.abs(probability_shift[:, i]))),
            "probability_shift_abs_max": float(np.max(np.abs(probability_shift[:, i]))),
            "positive_probability_shift_mean": float(np.mean(probability_shift[pos, i])),
            "negative_probability_shift_mean": float(np.mean(probability_shift[neg, i])),
            "positive_probability_increased": int(np.sum(probability_shift[pos, i] > 0)),
            "positive_probability_decreased": int(np.sum(probability_shift[pos, i] < 0)),
            "negative_probability_decreased": int(np.sum(probability_shift[neg, i] < 0)),
            "negative_probability_increased": int(np.sum(probability_shift[neg, i] > 0)),
            "positive_negative_rank_flips": pair_flip_count,
            "positive_negative_pairs": total_pairs,
            "auc_global_local_fused": float(auc_a[i]),
            "auc_global_only": float(auc_b[i]),
            "auc_r2_gap": float(auc_c[i]),
            "ap_global_local_fused": float(ap_a[i]),
            "ap_global_only": float(ap_b[i]),
            "ap_r2_gap": float(ap_c[i]),
        })
    return pd.DataFrame(rows)


def prediction_correlations(scores: dict[str, np.ndarray]):
    rows = []
    for left, right in (("A_R3_fused", "B_R3_global"),
                        ("B_R3_global", "C_R2_gap"),
                        ("A_R3_fused", "C_R2_gap")):
        for i, name in enumerate(LABELS):
            pearson = float(np.corrcoef(scores[left][:, i], scores[right][:, i])[0, 1])
            spear = float(spearmanr(scores[left][:, i], scores[right][:, i]).statistic)
            rows.append({"left": left, "right": right, "disease": name,
                         "pearson_prediction_correlation": pearson,
                         "spearman_prediction_correlation": spear})
    return pd.DataFrame(rows)


def training_comparison():
    rows = []
    for run, run_dir in RUNS.items():
        log = pd.read_csv(run_dir / "train_log.csv")
        status_path = run_dir / "training_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        best_epoch = int(status["best_epoch"])
        through_best = log.loc[log["epoch"] <= best_epoch]
        best_row = log.loc[log["epoch"] == best_epoch].iloc[0]
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        runtime = config.get("selected_runtime")
        if runtime is None:
            runtime = config.get("freeze_config", {}).get("selected_runtime", {})
        if not runtime:
            runtime_path = run_dir / "selected_runtime_config.json"
            if runtime_path.is_file():
                payload = json.loads(runtime_path.read_text(encoding="utf-8"))
                runtime = payload.get("selected_runtime", payload)
        updates = int(through_best["optimizer_updates"].sum())
        rows.append({
            "run": run,
            "image_size": int(json.loads(MODEL_CONFIGS["R3"].read_text(encoding="utf-8"))["model"]["image_size"])
            if run == "R3" else int(config.get("image_size", 320 if run == "R0" else 448 if run == "R1" else 512)),
            "best_epoch": best_epoch,
            "optimizer_updates_through_best_epoch": updates,
            "best_epoch_updates": int(best_row["optimizer_updates"]),
            "image_budget": int(best_row["image_budget"]),
            "gradient_accumulation": int(best_row["grad_accumulation"]),
            "effective_images_per_update": runtime.get("effective_images_per_update"),
            "effective_studies_per_update": runtime.get("effective_studies_per_update"),
            "gradient_checkpointing": runtime.get("gradient_checkpointing"),
            "channels_last": runtime.get("channels_last"),
            "mean_images_per_s_through_best": float(through_best["images_per_s"].mean()),
            "weighted_images_per_s_through_best": float(
                through_best["train_images"].sum() / through_best["train_seconds"].sum()),
            "total_train_images_through_best": int(through_best["train_images"].sum()),
        })
    return pd.DataFrame(rows)


def plot_resolution(metrics: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [2.1, 1]})
    for label in LABELS:
        row = metrics.loc[metrics.disease == label].iloc[0]
        values = [row["R0_auc"], row["R1_auc"], row["R2_auc"]]
        axes[0].plot([320, 448, 512], values, marker="o", linewidth=1.6, alpha=0.82,
                     label=label)
    axes[0].set(title="Per-disease AUROC by input resolution", xlabel="Input resolution",
                ylabel="AUROC", xticks=[320, 448, 512], ylim=(0.35, 1.0))
    axes[0].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, ncol=2, loc="lower right")
    macro_values = [metrics["R0_auc"].mean(), metrics["R1_auc"].mean(), metrics["R2_auc"].mean()]
    bars = axes[1].bar(["R0 · 320", "R1 · 448", "R2 · 512"], macro_values,
                       color=["#4477AA", "#66CCEE", "#228833"])
    axes[1].set(title="Macro-AUROC", ylabel="Macro-AUROC", ylim=(0.65, 0.85))
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, macro_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.6f}",
                     ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "resolution_auroc_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [2.1, 1]})
    delta = metrics["R2_minus_R0_auc"].to_numpy()
    colors = ["#228833" if value >= 0 else "#CC6677" for value in delta]
    order = np.argsort(delta)
    axes[0].barh(np.asarray(SHORT)[order], delta[order], color=np.asarray(colors)[order])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(title="R2 − R0 AUROC point change", xlabel="Δ AUROC")
    axes[0].grid(axis="x", alpha=0.25)
    y = np.arange(len(metrics))
    axes[1].scatter(metrics["positive_studies"], y, label="Positive Studies", color="#AA3377")
    axes[1].scatter(metrics["negative_studies"], y, label="Negative Studies", color="#4477AA")
    axes[1].set_yticks(y, SHORT)
    axes[1].set(title="Validation sample counts", xlabel="Study count")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "resolution_gain_by_disease.png", dpi=180)
    plt.close(fig)


def plot_ablation(metrics: pd.DataFrame, out: Path):
    x = np.arange(len(LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 7))
    for offset, (col, name, color) in enumerate((
            ("A_auc", "A · R3 fused", "#4477AA"),
            ("B_auc", "B · R3 global", "#66CCEE"),
            ("C_auc", "C · R2 GAP", "#228833"))):
        ax.bar(x + (offset - 1) * width, metrics[col], width, label=name, color=color)
    ax.set_xticks(x, SHORT, rotation=35, ha="right")
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("R3 inference ablation and R2 reference")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "r3_per_class_ablation.png", dpi=180)
    plt.close(fig)


def safe_name(value: str):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:80]


def minmax_display_map(values: np.ndarray, mask: np.ndarray):
    selected = values[mask]
    if selected.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(selected, [2, 98])
    result = np.clip((values - low) / max(float(high - low), 1e-8), 0, 1)
    result[~mask] = np.nan
    return result


def draw_response_sample(record, disease_index, disease_name, study_score,
                         outputs, labels, scores_r2, category, output_path,
                         shared_vmin, shared_vmax):
    with Image.open(record["image_path"]) as source:
        original = source.convert("RGB")
        width, height = original.size
        geom = geometry(width, height)
        scaled = original.resize((geom["resized_width"], geom["resized_height"]), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (512, 512), (0, 0, 0))
        canvas.paste(scaled, (geom["left"], geom["top"]))
        original_gray = np.asarray(original.convert("L"))
        canvas_gray = np.asarray(canvas.convert("L"))
    response = record["response_maps"][disease_index]
    valid_grid = record["valid_grid"]
    response_masked = response.copy()
    response_masked[~valid_grid] = np.nan
    response_512 = np.asarray(Image.fromarray(response.astype(np.float32), mode="F").resize(
        (512, 512), Image.Resampling.BILINEAR)).copy()
    valid512 = np.zeros((512, 512), dtype=bool)
    x0, y0 = geom["left"], geom["top"]
    valid512[y0:y0 + geom["resized_height"], x0:x0 + geom["resized_width"]] = True
    response_512[~valid512] = np.nan
    crop = response_512[y0:y0 + geom["resized_height"], x0:x0 + geom["resized_width"]]
    response_original = np.asarray(Image.fromarray(crop.astype(np.float32), mode="F").resize(
        (width, height), Image.Resampling.BILINEAR))
    topk512 = np.repeat(np.repeat(record["topk_grid"][disease_index], 16, axis=0), 16, axis=1)
    topk512 = topk512[:512, :512]
    topk_crop = topk512[y0:y0 + geom["resized_height"], x0:x0 + geom["resized_width"]]
    topk_original = np.asarray(Image.fromarray(topk_crop.astype(np.uint8) * 255).resize(
        (width, height), Image.Resampling.NEAREST)) > 0
    overlay_map = minmax_display_map(response_512, valid512)

    key = (record["subject_id"], record["study_id"])
    pos = next(i for i, existing in enumerate(outputs["keys"]) if existing == key)
    target = int(labels[pos, disease_index])
    p_r3 = float(scores_r3_from_logits(outputs)[pos, disease_index])
    p_r2 = float(scores_r2[pos, disease_index])
    image_g = float(record["global_logits"][disease_index])
    image_l = float(record["local_logits"][disease_index])
    image_f = float(record["fused_logits"][disease_index])
    display = minmax_display_map(record["response_maps"][disease_index], valid_grid)

    fig, axes = plt.subplots(1, 5, figsize=(21, 5.4))
    axes[0].imshow(original_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Original · {width}×{height}")
    axes[1].imshow(canvas_gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"ResizePad · 512×512\nvalid box {x0},{y0} → {geom['resized_width']}×{geom['resized_height']}")
    im = axes[2].imshow(response_masked, cmap="inferno", vmin=shared_vmin, vmax=shared_vmax)
    axes[2].set_title(f"Stage3 response · shared range\n[{shared_vmin:.3f}, {shared_vmax:.3f}]")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(canvas_gray, cmap="gray", vmin=0, vmax=255)
    axes[3].imshow(topk512, cmap="winter", alpha=0.58, vmin=0, vmax=1,
                   interpolation="nearest")
    axes[3].set_title(f"Top-K · {record['selected_count']}/{record['valid_count']} valid cells")
    axes[4].imshow(original_gray, cmap="gray", vmin=0, vmax=255)
    axes[4].imshow(response_original, cmap="inferno", alpha=0.45,
                   vmin=shared_vmin, vmax=shared_vmax)
    axes[4].contour(topk_original.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.9)
    axes[4].set_title("Mapped to original coordinates")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{category} · {disease_name} · Study {record['study_id']} · Image {record['image_id']}\n"
        f"y={target} | Study p(R3)={p_r3:.4f} | Study p(R2)={p_r2:.4f} | "
        f"view logits: global={image_g:.3f}, local={image_l:.3f}, fused={image_f:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def scores_r3_from_logits(outputs):
    return 1.0 / (1.0 + np.exp(-outputs["fused_logits"]))


def create_response_visualizations(outputs, valid_base, labels, scores_r2, out_dir: Path):
    image_dir = out_dir / "r3_response_visualizations"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = outputs["image_records"]
    values = np.concatenate([record["response_maps"].reshape(10, -1) for record in records], axis=1)
    all_valid_values = []
    for record in records:
        for i in range(10):
            all_valid_values.append(record["response_maps"][i][record["valid_grid"]])
    joined = np.concatenate(all_valid_values)
    shared_vmin, shared_vmax = [float(x) for x in np.quantile(joined, [0.01, 0.99])]
    image_by_study = {}
    for record in records:
        image_by_study.setdefault((record["subject_id"], record["study_id"]), []).append(record)
    ordered_keys = outputs["keys"]
    r3 = scores_r3_from_logits(outputs)
    r2_by_key = {key: scores_r2[i] for i, key in enumerate(ordered_keys)}
    selected, manifest = set(), []
    rng = random.Random(BOOTSTRAP_SEED)
    focus_indices = {name: LABELS.index(name) for name in FOCUS}
    for disease_name, disease_index in focus_indices.items():
        candidates = [i for i, key in enumerate(ordered_keys)]
        positives = [i for i in candidates if labels[i, disease_index] == 1]
        negatives = [i for i in candidates if labels[i, disease_index] == 0]
        specifications = []
        if positives:
            specifications.append(("TP_high_score", max(positives, key=lambda i: r3[i, disease_index])))
            specifications.append(("TP_low_score", min(positives, key=lambda i: r3[i, disease_index])))
        if negatives:
            specifications.append(("TN_high_score", max(negatives, key=lambda i: r3[i, disease_index])))
        directions = np.asarray([
            (2 * labels[i, disease_index] - 1) * (r3[i, disease_index] - scores_r2[i, disease_index])
            for i in candidates
        ])
        already = {index for _, index in specifications}
        remaining = [i for i in candidates if i not in already]
        if remaining:
            improvement_order = sorted(
                remaining,
                key=lambda i: (2 * labels[i, disease_index] - 1)
                * (r3[i, disease_index] - scores_r2[i, disease_index]),
                reverse=True,
            )
            specifications.append(("R3_improved_vs_R2", improvement_order[0]))
            already.add(improvement_order[0])
        remaining = [i for i in candidates if i not in already]
        if remaining:
            deterioration_order = sorted(
                remaining,
                key=lambda i: (2 * labels[i, disease_index] - 1)
                * (r3[i, disease_index] - scores_r2[i, disease_index]),
            )
            specifications.append(("R3_deteriorated_vs_R2", deterioration_order[0]))
            already.add(deterioration_order[0])
        already = {index for _, index in specifications}
        random_pool = [i for i in candidates if i not in already]
        if random_pool:
            specifications.append(("fixed_seed_random_control", rng.choice(random_pool)))
        for category, index in specifications:
            key = ordered_keys[index]
            if not image_by_study.get(key):
                continue
            record = sorted(image_by_study[key], key=lambda item: item["image_id"])[0]
            path = image_dir / f"{safe_name(disease_name)}__{safe_name(category)}__S{safe_name(key[1])}__I{safe_name(record['image_id'])}.png"
            draw_response_sample(record, disease_index, disease_name, None, outputs,
                                 labels, scores_r2, category, path, shared_vmin, shared_vmax)
            manifest.append({
                "disease": disease_name,
                "category": category,
                "study_id": key[1],
                "image_id": record["image_id"],
                "subject_id_internal": key[0],
                "true_label": int(labels[index, disease_index]),
                "r3_probability": float(r3[index, disease_index]),
                "r2_probability": float(scores_r2[index, disease_index]),
                "probability_change_toward_truth": float(directions[candidates.index(index)]),
                "visualization": str(path.relative_to(out_dir)),
            })
            selected.add((disease_name, key))
    pd.DataFrame(manifest).drop(columns=["subject_id_internal"]).to_csv(
        out_dir / "r3_response_samples.csv", index=False, encoding="utf-8")

    # A fixed-seed set of studies with multiple views illustrates view-specific responses.
    multiview = []
    for disease_name in FOCUS:
        disease_index = LABELS.index(disease_name)
        candidates = [key for key, views in image_by_study.items()
                      if len(views) >= 2 and labels[ordered_keys.index(key), disease_index] == 1]
        if candidates:
            multiview.append((disease_name, rng.choice(sorted(candidates))))
    if multiview:
        fig, axes = plt.subplots(len(multiview), 4, figsize=(16, 4 * len(multiview)), squeeze=False)
        for row_index, (disease_name, key) in enumerate(multiview):
            disease_index = LABELS.index(disease_name)
            views = sorted(image_by_study[key], key=lambda item: item["image_id"])[:2]
            for view_index, record in enumerate(views):
                with Image.open(record["image_path"]) as source:
                    width, height = source.size
                    original = source.convert("RGB")
                    geom = geometry(width, height)
                    resized = original.resize((geom["resized_width"], geom["resized_height"]), Image.Resampling.BILINEAR)
                    canvas = Image.new("RGB", (512, 512), (0, 0, 0))
                    canvas.paste(resized, (geom["left"], geom["top"]))
                response = record["response_maps"][disease_index]
                masked = response.copy()
                masked[~record["valid_grid"]] = np.nan
                axes[row_index, view_index * 2].imshow(np.asarray(original.convert("L")), cmap="gray")
                axes[row_index, view_index * 2].set_title(f"{disease_name} · {record['image_id']} original")
                axes[row_index, view_index * 2 + 1].imshow(np.asarray(canvas.convert("L")), cmap="gray")
                axes[row_index, view_index * 2 + 1].imshow(
                    np.asarray(Image.fromarray(response.astype(np.float32), mode="F").resize(
                        (512, 512), Image.Resampling.BILINEAR)), cmap="inferno", alpha=0.48,
                    vmin=shared_vmin, vmax=shared_vmax)
                axes[row_index, view_index * 2 + 1].set_title(f"Stage3 response · {record['image_id']}")
            for col in range(4):
                axes[row_index, col].axis("off")
        fig.suptitle("Same Study, separate image views; shared response-map scale")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(image_dir / "multiview_same_study_diagnostics.png", dpi=150)
        plt.close(fig)
    return manifest, {"raw_response_shared_vmin": shared_vmin,
                      "raw_response_shared_vmax": shared_vmax,
                      "selected_visualizations": len(manifest),
                      "multiview_studies": len(multiview)}


def fmt(value, digits=6):
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def make_resolution_metrics(labels, scores_by_run):
    rows = []
    for i, name in enumerate(LABELS):
        row = {"disease": name,
               "positive_studies": int(labels[:, i].sum()),
               "negative_studies": int((labels[:, i] == 0).sum())}
        for run in ("R0", "R1", "R2"):
            aucs, aps = metric_vectors(labels, scores_by_run[run])
            row[f"{run}_auc"] = float(aucs[i])
            row[f"{run}_auprc"] = float(aps[i])
        for right, left in (("R1", "R0"), ("R2", "R1"), ("R2", "R0")):
            row[f"{right}_minus_{left}_auc"] = row[f"{right}_auc"] - row[f"{left}_auc"]
            row[f"{right}_minus_{left}_auprc"] = row[f"{right}_auprc"] - row[f"{left}_auprc"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_resolution_report(out, metrics, bootstrap, training, alignment, rep_r2, rep_r3,
                            aucs_by_run):
    comparison_labels = {
        "R1_minus_R0": "R1 − R0",
        "R2_minus_R1": "R2 − R1",
        "R2_minus_R0": "R2 − R0",
    }
    lines = [
        "# R0/R1/R2 分辨率逐类别诊断", "",
        "## 数据完整性与复现", "",
        f"- 官方 Validation：{alignment['validation_studies']} Studies / {alignment['validation_subjects']} Subjects / {alignment['validation_images']} images。",
        "- 四组预测按 `(Subject_id, Study_id)` 与官方 Validation Study 表做一对一键连接；重复、缺失和多余 Study 均为 0。标签顺序固定为十类官方顺序，closed-world 规则及目标标签均与官方 Validation CSV 一致。",
        "- Validation 使用确定性 ResizePad、灰度三通道和 ImageNet normalization；没有随机增强。Study 分数按每张图的 raw logits 取算术平均后 sigmoid。",
        f"- R2 Best Checkpoint 重推理：Macro-AUROC {rep_r2['reproduced_macro_auroc']:.6f}，保存预测 {rep_r2['saved_macro_auroc']:.6f}，绝对差 {rep_r2['macro_auroc_abs_error']:.8f}。",
        f"- R3 Best Checkpoint 完整融合重推理：Macro-AUROC {rep_r3['reproduced_macro_auroc']:.6f}，保存预测 {rep_r3['saved_macro_auroc']:.6f}，绝对差 {rep_r3['macro_auroc_abs_error']:.8f}。",
        "", "## 逐类 AUROC", "",
        "| Disease | Pos | Neg | R0 · 320 | R1 · 448 | R2 · 512 | R1−R0 | R2−R1 | R2−R0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['disease']} | {int(row['positive_studies'])} | {int(row['negative_studies'])} | "
            f"{row['R0_auc']:.6f} | {row['R1_auc']:.6f} | {row['R2_auc']:.6f} | "
            f"{row['R1_minus_R0_auc']:+.6f} | {row['R2_minus_R1_auc']:+.6f} | "
            f"{row['R2_minus_R0_auc']:+.6f} |"
        )
    lines += ["", "## 逐类 AUPRC", "",
              "| Disease | R0 · 320 | R1 · 448 | R2 · 512 | R1−R0 | R2−R1 | R2−R0 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['disease']} | {row['R0_auprc']:.6f} | {row['R1_auprc']:.6f} | "
            f"{row['R2_auprc']:.6f} | {row['R1_minus_R0_auprc']:+.6f} | "
            f"{row['R2_minus_R1_auprc']:+.6f} | {row['R2_minus_R0_auprc']:+.6f} |"
        )
    lines += ["", "## Subject-cluster paired Bootstrap", "",
              f"- {BOOTSTRAP_REPLICATES} paired resamples; Seed={BOOTSTRAP_SEED}; unit=Subject. Every sampled Subject contributes all its Validation Studies. The same resampling-index sequence was reused for all model pairs.",
              "- Each replicate skips a class AUROC if it has only one label; macro delta is the mean of available paired class deltas. Per-class confidence intervals use valid replicates only. Intervals are unadjusted for multiple comparisons.",
              "", "| Comparison | Measure | Δ point | 95% CI | Valid replicates |",
              "|---|---|---:|---:|---:|"]
    for key, display in comparison_labels.items():
        result = bootstrap["comparisons"][key]
        lo, hi = result["macro_auroc_ci95"]
        lines.append(f"| {display} | Macro-AUROC | {result['macro_auroc_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {result['valid_macro_replicates']} |")
        for disease in LABELS:
            item = result["per_class"][disease]
            lo, hi = item["ci95"]
            lines.append(f"| {display} | {disease} AUROC | {item['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {item['valid_replicates']} |")
    lines += ["", "## 训练预算与吞吐混杂", "",
              "| Run | Resolution | Best epoch | Image budget | Accumulation | Effective images/Studies per update | Updates in best epoch | Cumulative updates through best | Checkpointing | Channels-last | Mean images/s through best |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for _, row in training.iterrows():
        lines.append(
            f"| {row['run']} | {int(row['image_size'])} | {int(row['best_epoch'])} | "
            f"{int(row['image_budget'])} | {int(row['gradient_accumulation'])} | "
            f"{row['effective_images_per_update']} / {row['effective_studies_per_update']} | "
            f"{int(row['best_epoch_updates'])} | {int(row['optimizer_updates_through_best_epoch'])} | "
            f"{row['gradient_checkpointing']} | "
            f"{row['channels_last']} | {row['mean_images_per_s_through_best']:.2f} |"
        )
    lines.append("- 训练日志中的 `optimizer_updates` 为单轮更新数；上表分别列出 Best Epoch 更新数和截至 Best Epoch 的累计更新数。各组均使用 whole-Study batch sampler，不拆分 Study 或重复/丢弃 Study。")
    lines += [
        "", "## 观察与限制", "",
    ]
    gains = metrics.sort_values("R2_minus_R0_auc", ascending=False)
    lines.append("- R2−R0 AUROC 最大的点估计类别：" + ", ".join(
        f"{row.disease} ({row.R2_minus_R0_auc:+.4f})" for row in gains.head(3).itertuples()) + ".")
    positive = int((metrics["R2_minus_R0_auc"] > 0).sum())
    nonnegative = int((metrics["R2_minus_R0_auc"] >= 0).sum())
    lines.append(f"- R2 的单类 AUROC 点估计高于 R0 的类别为 {positive}/10；类别数并不等同于统计支持。请结合上表各类别置信区间判断。")
    macro_series = [macro(aucs_by_run[x]) for x in ("R0", "R1", "R2")]
    if macro_series[1] > macro_series[0] and macro_series[2] > macro_series[1]:
        lines.append("- Macro-AUROC 点估计随 320→448→512 单调上升；是否在 448 后趋平需结合 R2−R1 区间，不能只凭点估计确认平台。")
    lines.append("- R0/R1/R2 均使用 P2 冻结策略（Stem/Stage1/Stage2 冻结，Stage3/Stage4/末端 norm 与 Global 分类头训练）；但 R0 的 image budget、梯度累积和累计 optimizer updates 与 R1/R2 不同，R1 与 R2 的梯度 checkpointing 配置也不同。R0→R1/R2 的差异不能完全归因于分辨率。")
    lines.append("- 验证集 Best Epoch 参与模型选择；置信区间是条件于该选择的探索性区间，没有校正十类与多组比较，也不是独立测试集确认。")
    lines += ["", "## 图表", "",
              "- `resolution_auroc_curves.png`：十类 AUROC 曲线与 Macro-AUROC 对比。",
              "- `resolution_gain_by_disease.png`：R2−R0 AUROC 点差与每类阳性/阴性 Study 数。", ""]
    (out / "resolution_per_class_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_ablation_report(out, metrics, bootstrap, logits_frame, correlations,
                          rep_r2, rep_r3, r3_metadata):
    lines = [
        "# R3 推理消融与 Logit 贡献", "",
        "## Checkpoint 与历史预测复现", "",
        f"- 使用 R3 `best.pth`，Checkpoint Epoch={r3_metadata['checkpoint_epoch']}；alpha 从该 Best Checkpoint 的 `alpha.logit` 计算为 {r3_metadata['alpha_best_checkpoint']:.8f}，Top-K ratio={r3_metadata['topk_ratio']:.3f}，padding 网格阈值={r3_metadata['mask_valid_fraction_threshold']:.2f}。",
        f"- A 原始融合 Macro-AUROC：{rep_r3['reproduced_macro_auroc']:.6f}；历史 `val_best.csv`：{rep_r3['saved_macro_auroc']:.6f}；最大概率差={rep_r3['max_probability_abs_error']:.8f}。",
        f"- C 使用 R2 `best.pth`、512×512 重推理；Macro-AUROC：{rep_r2['reproduced_macro_auroc']:.6f}；历史预测：{rep_r2['saved_macro_auroc']:.6f}；最大概率差={rep_r2['max_probability_abs_error']:.8f}。",
        "- R3 训练运行配置记录 `channels_last=true`，而导出的 `model_config.json` 记录 `false`；本分析按训练期 `selected_runtime_config.json` 恢复内存格式后复现。此导出元数据不一致未在本任务中修改。",
        "- Study logits 先对同一 Study 的全部图像 logits 求算术均值，再 sigmoid。A/B/C 使用同一 859 Study 与相同官方标签。", "",
        "## 三种推理输出", "",
        "| Disease | A AUROC | B AUROC | C AUROC | A−B | B−C | A−C | A AP | B AP | C AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['disease']} | {row['A_auc']:.6f} | {row['B_auc']:.6f} | {row['C_auc']:.6f} | "
            f"{row['A_minus_B_auc']:+.6f} | {row['B_minus_C_auc']:+.6f} | {row['A_minus_C_auc']:+.6f} | "
            f"{row['A_ap']:.6f} | {row['B_ap']:.6f} | {row['C_ap']:.6f} |"
        )
    lines += ["", f"- Macro-AUROC：A={metrics['A_auc'].mean():.6f}，B={metrics['B_auc'].mean():.6f}，C={metrics['C_auc'].mean():.6f}。",
              f"- Macro-AUPRC：A={metrics['A_ap'].mean():.6f}，B={metrics['B_ap'].mean():.6f}，C={metrics['C_ap'].mean():.6f}。",
              "", "## Paired Subject Bootstrap", "",
              f"- {BOOTSTRAP_REPLICATES} 次，Seed={BOOTSTRAP_SEED}；所有比较复用相同的 Subject 抽样索引。CI 未做多重比较校正。",
              "", "| Comparison | Macro-AUROC Δ [95% CI] | Macro-AUPRC Δ [95% CI] |",
              "|---|---:|---:|"]
    for key, label in (("B_minus_A", "A − B"), ("C_minus_B", "B − C"), ("C_minus_A", "A − C")):
        result = bootstrap["comparisons"][key]
        lo, hi = result["macro_auroc_ci95"]
        ap = bootstrap["auprc_bootstrap"]["comparisons"][key]
        ap_lo, ap_hi = ap["macro_auprc_ci95"]
        lines.append(
            f"| {label} | {-result['macro_auroc_delta']:+.6f} "
            f"[{ -hi:+.6f}, { -lo:+.6f}] | "
            f"{-ap['macro_auprc_delta']:+.6f} [{-ap_hi:+.6f}, {-ap_lo:+.6f}] |"
        )
    lines += ["", "### 逐类 AUROC paired CI", "",
              "| Comparison | Disease | Δ AUROC | 95% CI | Valid replicates |",
              "|---|---|---:|---:|---:|"]
    for key, label in (("B_minus_A", "A − B"), ("C_minus_B", "B − C"),
                       ("C_minus_A", "A − C")):
        for disease in LABELS:
            item = bootstrap["comparisons"][key]["per_class"][disease]
            lo, hi = item["ci95"]
            lines.append(
                f"| {label} | {disease} | {-item['delta']:+.6f} | "
                f"[{ -hi:+.6f}, { -lo:+.6f}] | {item['valid_replicates']} |"
            )
    lines += ["", "### 逐类 AUPRC paired CI", "",
              "| Comparison | Disease | Δ AUPRC | 95% CI | Valid replicates |",
              "|---|---|---:|---:|---:|"]
    for key, label in (("B_minus_A", "A − B"), ("C_minus_B", "B − C"),
                       ("C_minus_A", "A − C")):
        for disease in LABELS:
            item = bootstrap["auprc_bootstrap"]["comparisons"][key]["per_class"][disease]
            lo, hi = item["ci95"]
            lines.append(
                f"| {label} | {disease} | {-item['delta']:+.6f} | "
                f"[{ -hi:+.6f}, { -lo:+.6f}] | {item['valid_replicates']} |"
            )
    lines += ["", "## 机制解释", "",
              "- A−B 是同一 R3 Best Checkpoint 下局部分支直接参与最终融合的推理差异。",
              "- B−C 比较 R3 训练后全局路径与独立 R2 GAP；即便 B 高于 C，也只能说结果与‘局部分支训练过程改变全局表征’假设一致，不能据此建立训练期因果关系。",
              "- 若置信区间覆盖 0，验证集证据不足以确认差异。A/B/C 都经过同一验证集 Best Epoch 选择，统计区间存在选择偏差。",
              "", "## Local Logit 的实际贡献", "",
              "`r3_global_vs_local_logits.csv` 中的 `alpha_local_minus_global_logit_*` 是 `alpha × (z_local − z_global)` 的 Study 级分布统计；概率变化、正负样本方向变化和正负配对排序翻转计数也按 Study 计算。",
              "", "## 同 Study 预测相关性", "",
              "| Pair | Mean Pearson | Mean Spearman |",
              "|---|---:|---:|"]
    for left, right in (("A_R3_fused", "B_R3_global"),
                        ("B_R3_global", "C_R2_gap"),
                        ("A_R3_fused", "C_R2_gap")):
        subset = correlations.loc[(correlations.left == left) & (correlations.right == right)]
        lines.append(
            f"| {left} vs {right} | {subset['pearson_prediction_correlation'].mean():.6f} | "
            f"{subset['spearman_prediction_correlation'].mean():.6f} |"
        )
    lines += ["", "逐类 Pearson/Spearman 相关系数见 `r3_global_vs_local_logits.csv`。不同预测高度相关并不代表数值完全等价。", "",
              "## 图表", "", "- `r3_per_class_ablation.png`：A/B/C 十类 AUROC 对照。", ""]
    (out / "r3_inference_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def write_response_report(out, stats, corr, samples, visualization_meta, r3_metadata):
    padding_grid_max = float(stats["max_topk_pixel_share_in_padding"].max())
    entropy_mean = float(stats["normalized_spatial_entropy_mean"].mean())
    corr_mean = float(corr["mean_response_map_correlation"].mean())
    lines = [
        "# R3 Local Disease Response 行为诊断", "",
        "## 提取定义与几何映射", "",
        f"- 使用 R3 Best Checkpoint Epoch {r3_metadata['checkpoint_epoch']}；Stage3 响应图尺寸为 32×32，十个通道由真实 `Conv1x1(Stage3)` 产生。没有用 Grad-CAM 替代。",
        f"- 模型实现先将 512×512 valid mask adaptive-average-pool 到 Stage3 网格，再按阈值 {r3_metadata['mask_valid_fraction_threshold']:.2f} 生成有效单元；Top-K 只在有效网格中选择，ratio={r3_metadata['topk_ratio']:.2f}。",
        "- 对每张图按原始宽高重新计算等比例 ResizePad 的缩放、左右/上下 padding；热图 32×32→512 输入空间后裁除 padding，再双线性映射回原图；Top-K 使用最近邻映射。可视化统一使用同一 raw response 色标范围；叠加图按全局共同响应范围显示。",
        "", "## 全 Validation 定量统计", "",
        "| Disease | Top-K / valid cells | Top-K有效区像素占比 | Padding像素占Top-K | normalized entropy | max−mean | Local logit + mean±SD | Local logit − mean±SD | R3方向正确/错误 Study |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['disease']} | {row['mean_topk_valid_grid_fraction']:.3f} | "
            f"{row['mean_topk_selected_pixel_share_of_valid_area']:.3f} | "
            f"{row['mean_topk_pixel_share_in_padding']:.4f} | "
            f"{row['normalized_spatial_entropy_mean']:.3f} | "
            f"{row['response_max_minus_mean_mean']:.3f} | "
            f"{row['local_logit_positive_mean']:.3f}±{row['local_logit_positive_std']:.3f} | "
            f"{row['local_logit_negative_mean']:.3f}±{row['local_logit_negative_std']:.3f} | "
            f"{int(row['r3_vs_r2_correct_direction_studies'])}/{int(row['r3_vs_r2_wrong_direction_studies'])} |"
        )
    lines += ["", "- 设计层面 Top-K 选择不可能包含 `valid_grid=False` 的特征单元；本次还计算每个 Top-K 16×16 输入像素块实际落入 Padding 像素的比例，最高为 "
              f"{padding_grid_max:.4f}。这项比例对边界处有效网格单元可能非零，不等同于模型选择了完全 Padding 的 Stage3 单元。",
              f"- 十类响应的归一化空间熵平均值为 {entropy_mean:.3f}（在有效 Stage3 单元上以 temperature=1 对响应 logit 做 softmax 后计算 H/log(N)）；疾病响应图两两相关性的跨图像平均值为 {corr_mean:.3f}。详细 45 个类别对相关系数见 `r3_disease_map_correlations.csv`。熵和相关系数只描述响应图形态，不代表定位性能。",
              "- 不存在肺野分割或病灶框标注，因此没有计算肺组织内比例、病灶定位准确率或医学定位结论。", "",
              "## 样本选择与可视化", "",
              f"- 按预先定义的标签方向，从六个重点类别选择真阳性高分、真阳性低分、真阴性高分、R3 相对 R2 朝真实标签方向改善、朝相反方向退化样本，并以 Seed={BOOTSTRAP_SEED} 增加随机对照；共生成 {len(samples)} 张单样本诊断图。",
              f"- 多视角同 Study 样本数：{visualization_meta['multiview_studies']}；配对图位于 `r3_response_visualizations/multiview_same_study_diagnostics.png`。",
              "- 单样本图显示原图、512 ResizePad、原始 Stage3 response、Top-K 位置、映射回原图的响应，并在标题标明标签、R2/R3 概率和该视角 global/local/fused logits。色标共用；overlay 不做逐图归一化。",
              "- Study/Image ID 清单见 `r3_response_samples.csv`。响应图只是模型自身的局部输出，不是经过医学标注验证的真实病灶位置。", "",
              "## 人工抽样图像观察", "",
              "- 抽查的 Atelectasis 高分与 Pneumonia 低分样本中，Stage3 响应多为较宽的区域梯度；Top-K 同时覆盖胸片边缘、下缘及少量文字/设备标记附近位置，没有稳定出现集中于某一处的模式。该观察限于展示样本，不能外推为全体模型行为。",
              "- 同一 Study 多视角图中，响应随视角明显变化；这符合各图像输入的空间变化，但未显示跨视角稳定的局部热点证据。",
              "- 部分图的 Top-K 单元与 padding 边缘相交；全验证集平均相交比例与最大单图比例分别见 CSV/上表。该现象表示边界特征网格覆盖一部分填充像素，不等价于选择了无效网格。", "",
              "## 结论边界", "",
              "- Top-K 的 padding 网格排除由原模型代码实现，本分析没有修改 mask 或池化逻辑。若 padding 像素比例较高，只能说明边界网格与 padding 重叠，需要进一步审查，不直接等同于伪相关。",
              "- 文字、设备、背景及解剖结构观察只能作为定性检查；没有真实区域标注，不能声称模型‘真正关注病灶’。", ""]
    (out / "r3_local_response_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(out, resolution_metrics, resolution_bootstrap,
                       ablation_metrics, ablation_bootstrap, stats,
                       training, r3_metadata, rep_r3):
    resolution = resolution_bootstrap["comparisons"]
    ablation = ablation_bootstrap["comparisons"]
    gain_order = resolution_metrics.sort_values("R2_minus_R0_auc", ascending=False)
    cls = stats.sort_values("mean_abs_pairwise_disease_map_corr", ascending=False)
    r3_macro = {key: float(ablation_metrics[f"{key}_auc"].mean()) for key in ("A", "B", "C")}
    r2r0 = resolution["R2_minus_R0"]
    r2r1 = resolution["R2_minus_R1"]
    r1r0 = resolution["R1_minus_R0"]
    a_b = ablation["B_minus_A"]["macro_auroc_delta"]
    b_c = ablation["C_minus_B"]["macro_auroc_delta"]
    a_c = ablation["C_minus_A"]["macro_auroc_delta"]
    lines = [
        "# DINOv3 特征机制诊断总报告", "",
        "## 结论摘要", "",
        f"- R0→R1→R2 的 Macro-AUROC 点估计为 {resolution_metrics['R0_auc'].mean():.6f} → {resolution_metrics['R1_auc'].mean():.6f} → {resolution_metrics['R2_auc'].mean():.6f}；Subject-cluster paired Bootstrap：R1−R0 {r1r0['macro_auroc_delta']:+.6f} [{r1r0['macro_auroc_ci95'][0]:+.6f}, {r1r0['macro_auroc_ci95'][1]:+.6f}]，R2−R1 {r2r1['macro_auroc_delta']:+.6f} [{r2r1['macro_auroc_ci95'][0]:+.6f}, {r2r1['macro_auroc_ci95'][1]:+.6f}]，R2−R0 {r2r0['macro_auroc_delta']:+.6f} [{r2r0['macro_auroc_ci95'][0]:+.6f}, {r2r0['macro_auroc_ci95'][1]:+.6f}]。",
        f"- R3 消融 Macro-AUROC：A R3 fused={r3_macro['A']:.6f}，B R3 global-only={r3_macro['B']:.6f}，C R2 GAP={r3_macro['C']:.6f}；A−B={-a_b:+.6f}，B−C={-b_c:+.6f}，A−C={-a_c:+.6f}（CI 见下文）。",
        f"- R3 Best Checkpoint 实际 alpha={r3_metadata['alpha_best_checkpoint']:.8f}；A 复现历史 `val_best.csv` Macro-AUROC 差 {rep_r3['macro_auroc_abs_error']:.8f}。",
        f"- A/B/C Macro-AUPRC={ablation_metrics['A_ap'].mean():.6f}/{ablation_metrics['B_ap'].mean():.6f}/{ablation_metrics['C_ap'].mean():.6f}；各比较的 paired CI 见 20.2。",
        "- Bootstrap 区间如包含 0，不足以确认所比较指标有方向性差异。分辨率效应仍受训练预算差异与 Best Epoch 选择影响；B−C 不证明局部分支训练期的因果作用。", "",
        "## 20.1 分辨率带来了什么？", "",
        "R2−R0 AUROC 点估计最高的三个类别：" + ", ".join(
            f"{row.disease} ({row.R2_minus_R0_auc:+.4f})" for row in gain_order.head(3).itertuples()) + ".",
        f"R2−R0 Macro-AUROC 的 95% CI 为 [{r2r0['macro_auroc_ci95'][0]:+.6f}, {r2r0['macro_auroc_ci95'][1]:+.6f}]。448 是否已足够、512 是否仍有稳定增益，应以 R2−R1 的区间和逐类变化共同判断，不从总体点估计外推到所有疾病。",
        "R0 的 image budget / accumulation / best-epoch 累计更新数与 R1/R2 不同；R1/R2 的 checkpointing 也不同。故观察到的收益与高分辨率带来的采样细节保留相一致，但这组历史实验不能完全隔离分辨率本身的因果效应。", "",
        "| Disease | R2−R0 AUROC Δ | 95% CI | Positive / Negative Studies |",
        "|---|---:|---:|---:|"]
    bres = r2r0["per_class"]
    for _, row in resolution_metrics.iterrows():
        item = bres[row["disease"]]
        lines.append(f"| {row['disease']} | {row['R2_minus_R0_auc']:+.6f} | [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}] | {int(row['positive_studies'])} / {int(row['negative_studies'])} |")
    lines += ["", "## 20.2 R3 改善了什么？", "",
              f"A−B（局部分支在推理融合中的直接增量）Macro-AUROC={-a_b:+.6f}，95% CI=[{-ablation['B_minus_A']['macro_auroc_ci95'][1]:+.6f}, {-ablation['B_minus_A']['macro_auroc_ci95'][0]:+.6f}]。",
              f"B−C（R3 训练后的 Global-only 与独立 R2 GAP）Macro-AUROC={-b_c:+.6f}，95% CI=[{-ablation['C_minus_B']['macro_auroc_ci95'][1]:+.6f}, {-ablation['C_minus_B']['macro_auroc_ci95'][0]:+.6f}]。",
              f"A−C 完整 R3 相对 R2 的 Macro-AUROC={-a_c:+.6f}，95% CI=[{-ablation['C_minus_A']['macro_auroc_ci95'][1]:+.6f}, {-ablation['C_minus_A']['macro_auroc_ci95'][0]:+.6f}]。",
              f"Macro-AUPRC：A−B={-ablation_bootstrap['auprc_bootstrap']['comparisons']['B_minus_A']['macro_auprc_delta']:+.6f}，95% CI=[{-ablation_bootstrap['auprc_bootstrap']['comparisons']['B_minus_A']['macro_auprc_ci95'][1]:+.6f}, {-ablation_bootstrap['auprc_bootstrap']['comparisons']['B_minus_A']['macro_auprc_ci95'][0]:+.6f}]；B−C={-ablation_bootstrap['auprc_bootstrap']['comparisons']['C_minus_B']['macro_auprc_delta']:+.6f}，95% CI=[{-ablation_bootstrap['auprc_bootstrap']['comparisons']['C_minus_B']['macro_auprc_ci95'][1]:+.6f}, {-ablation_bootstrap['auprc_bootstrap']['comparisons']['C_minus_B']['macro_auprc_ci95'][0]:+.6f}]；A−C={-ablation_bootstrap['auprc_bootstrap']['comparisons']['C_minus_A']['macro_auprc_delta']:+.6f}，95% CI=[{-ablation_bootstrap['auprc_bootstrap']['comparisons']['C_minus_A']['macro_auprc_ci95'][1]:+.6f}, {-ablation_bootstrap['auprc_bootstrap']['comparisons']['C_minus_A']['macro_auprc_ci95'][0]:+.6f}]。",
              "以上 CI 若跨 0，不能确认有方向性收益。即使 B−C 为正，也只能说与训练期共享表征变化的解释一致，不能因果归因。alpha 较小与否不作为机制结论；需看 per-class `alpha*(z_local-z_global)`、概率位移和正负样本排序翻转数。", "",
              "| Disease | A−B Δ AUROC | B−C Δ AUROC | alpha·(local−global) mean±SD | Mean |Δ probability| | Positive-negative pair flips |",
              "|---|---:|---:|---:|---:|---:|"]
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['disease']} | {row['A_minus_B_auc']:+.6f} | "
            f"{row['B_minus_C_auc']:+.6f} | "
            f"{row['alpha_local_minus_global_logit_mean']:+.4f}±"
            f"{row['alpha_local_minus_global_logit_std']:.4f} | "
            f"{row['probability_shift_abs_mean']:.4f} | "
            f"{int(row['positive_negative_rank_flips'])} |"
        )
    lines += ["", "## 20.3 局部响应是否合理？", "",
              f"Stage3 mask 在模型内部约束 Top-K；padding 像素在所选特征单元覆盖中的最大比例为 {stats['max_topk_pixel_share_in_padding'].max():.4f}。类别响应图的 mean absolute inter-disease correlation 最高类别为 {cls.iloc[0]['disease']} ({cls.iloc[0]['mean_abs_pairwise_disease_map_corr']:.3f})。",
              "抽查的 Atelectasis 高分与 Pneumonia 低分图中，响应分布较宽，Top-K 有时覆盖图像边缘或文字/设备标记附近；多视角 Study 的响应随视角明显变化，未见稳定的跨视角焦点。该结论仅来自展示样本。",
              "这些统计和真实诊断图仅描述模型内部响应结构。没有肺野分割/病灶标注，不能据此宣称定位到肺组织或真实病灶；高/低熵也不等同于好/坏。具体样本需结合 `r3_response_visualizations/` 人工核查。", "",
              "## 20.4 下一步模型路线", "",
              "- 路线 A（普通 GAP）：若局部分支没有可确认的直接增益，或响应检查发现明显边界/伪相关，保留 R2 GAP 作为主模型。",
              "- 路线 B（优化局部响应）：若 A−B 有稳定正向证据，且响应经独立区域标注验证后，再研究 Top-K 或局部池化方式。",
              "- 路线 C（训练期局部辅助）：若 B−C 有稳定增益而 A−B 直接作用较弱，后续可独立训练检验训练期辅助分支。", ""]
    ci_ab = ablation["B_minus_A"]["macro_auroc_ci95"]
    ci_bc = ablation["C_minus_B"]["macro_auroc_ci95"]
    ci_ac = ablation["C_minus_A"]["macro_auroc_ci95"]
    if ci_ab[1] < 0 and ci_ac[1] < 0:
        route = "路线 B：局部融合呈现有统计支持的直接增益，可进一步研究局部池化；仍需独立验证并检查响应图伪相关。"
    elif ci_bc[1] < 0 and ci_ab[0] <= 0 <= ci_ab[1]:
        route = "路线 C：Global-only 相对 R2 有较强证据，而局部直接融合无明确增量；可将局部模块作为训练期辅助候选，后续用独立训练检验。"
    elif (stats["max_topk_pixel_share_in_padding"].max() > 0.25
          or ci_ab[0] > 0):
        route = "路线 A：当前局部融合未显示正向直接证据或存在较明显边界/负向影响，暂保留普通 GAP，并先排查局部响应行为。"
    else:
        route = "现有验证证据不足以在 A/B/C 三条路线中做确认性选择。保留 R2 GAP 作为稳健参照；下一轮需预先定义假设并用独立种子/独立测试证据验证。"
    lines.append(f"- {route}")
    lines.append("- 本次没有训练、优化器更新、模型权重修改、学习率搜索或 Test 数据访问。")
    lines += ["", "## 交付文件", "",
              "逐类分辨率、R3 推理消融、Local Response 统计和图表分别见本目录相应文件。", ""]
    (out / "feature_diagnostics_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    out = HERE
    vis_dir = out / "r3_response_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint re-inference in this diagnostic run")
    device = torch.device("cuda")
    print(f"Diagnostic output: {out}", flush=True)
    print(f"Validation-only source: {VALID_CSV}", flush=True)

    valid_base, official, labels = official_dataset()
    keys = key_tuples(official)
    subjects = official["Subject_id"].astype(str).nunique()
    image_count = int(sum(valid_base.image_counts))
    if len(valid_base) != EXPECTED_STUDIES or subjects != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Validation count mismatch: {len(valid_base)} Studies, {subjects} Subjects")
    missing_images = [str(path) for study in valid_base.studies for path in study["image_paths"]
                      if not path.is_file()]
    if missing_images:
        json_write(out / "input_alignment_error.json", {
            "missing_validation_images": missing_images[:100],
            "missing_validation_image_count": len(missing_images),
            "test_accessed": False,
        })
        raise FileNotFoundError(f"{len(missing_images)} Validation image files are missing")

    saved_frames, saved_scores = {}, {}
    for run in ("R0", "R1", "R2", "R3"):
        saved_frames[run], saved_scores[run] = read_predictions(
            PREDICTIONS[run], official, labels)
    if len(set(keys)) != EXPECTED_STUDIES:
        raise RuntimeError("Official Validation has duplicate Study keys")
    alignment = {
        "validation_studies": len(valid_base),
        "validation_subjects": int(subjects),
        "validation_images": image_count,
    }
    save_input_artifacts(out, valid_base, official, labels, saved_frames, saved_scores)
    print(f"Validated {len(valid_base)} Studies / {subjects} Subjects / {image_count} images; "
          "all saved predictions aligned by Study key.", flush=True)

    r3 = infer_r3(valid_base, device)
    r3_scores = 1.0 / (1.0 + np.exp(-r3["fused_logits"]))
    r3_global_scores = 1.0 / (1.0 + np.exp(-r3["global_logits"]))
    r3_reproduction = validate_checkpoint_reproduction("R3", saved_scores["R3"], r3_scores, labels)
    print(f"R3 best-checkpoint replay Macro-AUROC={r3_reproduction['reproduced_macro_auroc']:.6f}; "
          f"saved={r3_reproduction['saved_macro_auroc']:.6f}; "
          f"max |Δp|={r3_reproduction['max_probability_abs_error']:.6f}", flush=True)
    if (r3_reproduction["macro_auroc_abs_error"] > 1e-5
            or r3_reproduction["max_probability_abs_error"] > 0.02):
        json_write(out / "r3_reproduction_failure.json", r3_reproduction)
        raise RuntimeError("R3 Best Checkpoint did not reproduce val_best.csv within tolerance; "
                           "stopping before mechanism analysis as requested")
    json_write(out / "r3_checkpoint_reproduction.json", {
        "result": r3_reproduction,
        "checkpoint_runtime": r3["metadata"],
        "passed": True,
    })
    (out / "r3_reproduction_failure.json").unlink(missing_ok=True)

    r2 = infer_r2(valid_base, device)
    r2_scores = 1.0 / (1.0 + np.exp(-r2["logits"]))
    r2_reproduction = validate_checkpoint_reproduction("R2", saved_scores["R2"], r2_scores, labels)
    print(f"R2 best-checkpoint replay Macro-AUROC={r2_reproduction['reproduced_macro_auroc']:.6f}; "
          f"saved={r2_reproduction['saved_macro_auroc']:.6f}; "
          f"max |Δp|={r2_reproduction['max_probability_abs_error']:.6f}", flush=True)
    if (r2_reproduction["macro_auroc_abs_error"] > 1e-5
            or r2_reproduction["max_probability_abs_error"] > 0.02):
        json_write(out / "r2_reproduction_failure.json", r2_reproduction)
        raise RuntimeError("R2 Best Checkpoint did not reproduce val_best.csv within tolerance; "
                           "stopping before inference ablation")
    json_write(out / "r2_checkpoint_reproduction.json", {
        "result": r2_reproduction,
        "checkpoint_runtime": r2["metadata"],
        "passed": True,
    })
    (out / "r2_reproduction_failure.json").unlink(missing_ok=True)

    # From here on the direct checkpoint predictions are the analysis inputs for A/B/C.
    scores = {
        "R0": saved_scores["R0"],
        "R1": saved_scores["R1"],
        "R2": r2_scores,
        "R3": r3_scores,
        "A_R3_fused": r3_scores,
        "B_R3_global": r3_global_scores,
        "C_R2_gap": r2_scores,
    }
    for run in ("R0", "R1", "R2", "R3"):
        aucs, aps = metric_vectors(labels, scores[run])
        print(f"{run}: Macro-AUROC={macro(aucs):.6f}, Macro-AUPRC={macro(aps):.6f}", flush=True)

    draws, draws_sha = make_subject_draws(official)
    res_boot = paired_bootstrap(
        labels, {k: scores[k] for k in ("R0", "R1", "R2")},
        [("R0", "R1"), ("R1", "R2"), ("R0", "R2")], draws, draws_sha)
    json_write(out / "resolution_paired_bootstrap.json", res_boot)
    resolution_metrics = make_resolution_metrics(labels, scores)
    training = training_comparison()
    resolution_metrics.to_csv(out / "resolution_per_class_metrics.csv", index=False,
                              encoding="utf-8")
    plot_resolution(resolution_metrics, out)
    write_resolution_report(out, resolution_metrics, res_boot, training, alignment,
                            r2_reproduction, r3_reproduction,
                            {key: metric_vectors(labels, scores[key])[0]
                             for key in ("R0", "R1", "R2")})

    ab_boot = paired_bootstrap(
        labels,
        {"A": scores["A_R3_fused"],
         "B": scores["B_R3_global"],
         "C": scores["C_R2_gap"]},
        [("A", "B"), ("B", "C"), ("A", "C")], draws, draws_sha)
    # Add point and interval changes for Macro-AUPRC without implying AUC bootstrap is PR bootstrap.
    ab_boot["macro_auprc_point_estimates"] = {
        name: macro(metric_vectors(labels, scores[name])[1])
        for name in ("A_R3_fused", "B_R3_global", "C_R2_gap")
    }
    json_write(out / "r3_inference_bootstrap.json", ab_boot)
    ab_rows = []
    for i, name in enumerate(LABELS):
        auc_a, ap_a = metric_vectors(labels, scores["A_R3_fused"])
        auc_b, ap_b = metric_vectors(labels, scores["B_R3_global"])
        auc_c, ap_c = metric_vectors(labels, scores["C_R2_gap"])
        ab_rows.append({
            "disease": name,
            "positive_studies": int(labels[:, i].sum()),
            "negative_studies": int((labels[:, i] == 0).sum()),
            "A_auc": float(auc_a[i]), "B_auc": float(auc_b[i]), "C_auc": float(auc_c[i]),
            "A_ap": float(ap_a[i]), "B_ap": float(ap_b[i]), "C_ap": float(ap_c[i]),
            "A_minus_B_auc": float(auc_a[i] - auc_b[i]),
            "B_minus_C_auc": float(auc_b[i] - auc_c[i]),
            "A_minus_C_auc": float(auc_a[i] - auc_c[i]),
            "A_minus_B_ap": float(ap_a[i] - ap_b[i]),
            "B_minus_C_ap": float(ap_b[i] - ap_c[i]),
            "A_minus_C_ap": float(ap_a[i] - ap_c[i]),
        })
    ab_metrics = pd.DataFrame(ab_rows)
    ab_metrics.to_csv(out / "r3_inference_ablation.csv", index=False, encoding="utf-8")
    correlations = prediction_correlations({
        "A_R3_fused": scores["A_R3_fused"],
        "B_R3_global": scores["B_R3_global"],
        "C_R2_gap": scores["C_R2_gap"],
    })
    correlation_wide = correlations.pivot(
        index="disease", columns=["left", "right"],
        values=["pearson_prediction_correlation", "spearman_prediction_correlation"])
    correlation_wide.columns = ["_".join(x) for x in correlation_wide.columns]
    correlation_wide = correlation_wide.reset_index()
    logits_frame = logit_contribution_frame(
        r3, scores["A_R3_fused"], scores["B_R3_global"], scores["C_R2_gap"], labels)
    joined_logits = logits_frame.merge(correlation_wide, on="disease", how="left",
                                       validate="one_to_one")
    joined_logits = joined_logits.merge(
        ab_metrics[["disease", "A_auc", "B_auc", "C_auc", "A_ap", "B_ap", "C_ap",
                    "A_minus_B_auc", "B_minus_C_auc", "A_minus_C_auc",
                    "A_minus_B_ap", "B_minus_C_ap", "A_minus_C_ap"]],
        on="disease", how="left", validate="one_to_one")
    joined_logits.to_csv(out / "r3_global_vs_local_logits.csv", index=False,
                         encoding="utf-8")
    plot_ablation(ab_metrics, out)
    write_ablation_report(out, ab_metrics, ab_boot, joined_logits, correlations,
                          r2_reproduction, r3_reproduction, r3["metadata"])

    response_stats, response_corr = response_statistics(
        r3, labels, scores["C_R2_gap"], scores["A_R3_fused"])
    response_stats.to_csv(out / "r3_response_statistics.csv", index=False,
                          encoding="utf-8")
    response_corr.to_csv(out / "r3_disease_map_correlations.csv", index=False,
                         encoding="utf-8")
    local_global = joined_logits.merge(
        response_stats[["disease", "normalized_spatial_entropy_mean",
                        "response_max_minus_mean_mean", "mean_topk_pixel_share_in_padding",
                        "mean_abs_pairwise_disease_map_corr"]],
        on="disease", validate="one_to_one")
    local_global.to_csv(out / "r3_local_global_disease_comparison.csv", index=False,
                        encoding="utf-8")

    manifest, visualization_meta = create_response_visualizations(
        r3, valid_base, labels, scores["C_R2_gap"], out)
    write_response_report(out, response_stats, response_corr, manifest,
                          visualization_meta, r3["metadata"])
    write_final_report(out, resolution_metrics, res_boot, ab_metrics, ab_boot,
                       local_global, training, r3["metadata"], r3_reproduction)

    json_write(out / "diagnostic_run_metadata.json", {
        "validation_only": True,
        "train_accessed": False,
        "test_accessed": False,
        "training_or_optimizer_step": False,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "validation_studies": len(valid_base),
        "validation_subjects": subjects,
        "validation_images": image_count,
        "r2_checkpoint_reproduction": r2_reproduction,
        "r3_checkpoint_reproduction": r3_reproduction,
        "r2_inference_seconds": r2["seconds"],
        "r3_inference_and_response_seconds": r3["seconds"],
        "r2_checkpoint": r2["metadata"],
        "r3_checkpoint": r3["metadata"],
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "sampling_indices_sha256": draws_sha},
        "response_visualizations": visualization_meta,
    })
    print("Diagnostics complete. Train/Test access: none; training/optimizer updates: none.",
          flush=True)


if __name__ == "__main__":
    main()
