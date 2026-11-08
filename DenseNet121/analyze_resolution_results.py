"""Compare the original 320 baseline, matched 320 continuation, and 448 continuation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from datasets import StudyDataset, build_transform
from datasets.transforms import ResizePad
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


CLASS_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]
HISTORY_PATTERN = re.compile(
    r"epoch=(\d+)/(\d+) train_loss=([^\s]+) val_loss=([^\s]+) macro_auroc=([^\s]+).*?"
    r"peak_gpu_gb=([^\s]+) train_s=([^\s]+) val_s=([^\s]+)")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def metric_summary(targets, probabilities):
    class_rows = []
    aucs = []
    for index, name in enumerate(CLASS_NAMES):
        target = targets[:, index].astype(int)
        score = probabilities[:, index].astype(float)
        positives = int(target.sum())
        negatives = int(target.size - positives)
        auc = float(roc_auc_score(target, score)) if positives and negatives else float("nan")
        ap = float(average_precision_score(target, score)) if positives else float("nan")
        if np.isfinite(auc):
            aucs.append(auc)
        class_rows.append({
            "class_id": index + 1,
            "class_name": name,
            "positive_studies": positives,
            "negative_studies": negatives,
            "auroc": auc,
            "auprc": ap,
        })
    return (float(np.mean(aucs)) if aucs else float("nan")), class_rows


def normalize_predictions(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Subject_id": "string", "Study_id": "string"})
    if frame.duplicated(["Subject_id", "Study_id"]).any():
        raise ValueError(f"{path} contains duplicate Study rows")
    rows = pd.DataFrame({
        "Subject_id": frame["Subject_id"].astype(str),
        "Study_id": frame["Study_id"].astype(str),
    })
    for index, name in enumerate(CLASS_NAMES, start=1):
        target_column = f"true_{name}"
        score_column = f"prob_{name}"
        if target_column in frame:
            rows[f"target_{index}"] = pd.to_numeric(frame[target_column], errors="raise")
        elif f"y_{index}" in frame:
            rows[f"target_{index}"] = pd.to_numeric(frame[f"y_{index}"], errors="raise")
        else:
            raise ValueError(f"{path} is missing Study labels for {name}")
        if score_column in frame:
            rows[f"prob_{index}"] = pd.to_numeric(frame[score_column], errors="raise")
        elif f"logit_mean_{index}" in frame:
            rows[f"prob_{index}"] = sigmoid(pd.to_numeric(
                frame[f"logit_mean_{index}"], errors="raise").to_numpy())
        else:
            raise ValueError(f"{path} is missing probabilities for {name}")
    if rows.filter(regex=r"^target_").isna().any().any():
        raise ValueError(f"{path} has missing Study labels")
    if not np.isfinite(rows.filter(regex=r"^prob_").to_numpy(dtype=float)).all():
        raise ValueError(f"{path} has non-finite probabilities")
    rows["method"] = method
    return rows


def extract_arrays(frame):
    targets = frame[[f"target_{index}" for index in range(1, 11)]].to_numpy(dtype=int)
    probabilities = frame[[f"prob_{index}" for index in range(1, 11)]].to_numpy(dtype=float)
    return targets, probabilities


def parse_history(path: Path, method: str):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HISTORY_PATTERN.search(line)
        if not match:
            continue
        epoch, max_epochs, train_loss, val_loss, macro_auc, peak_memory, train_s, val_s = match.groups()
        rows.append({
            "method": method, "epoch": int(epoch), "max_epochs": int(max_epochs),
            "train_loss": float(train_loss), "validation_loss": float(val_loss),
            "macro_auroc": float(macro_auc), "peak_allocated_gib": float(peak_memory),
            "train_seconds": float(train_s), "validation_seconds": float(val_s),
        })
    if not rows:
        raise ValueError(f"No epoch metrics found in {path}")
    return rows


def subject_bootstrap(frames, repetitions: int, seed: int):
    keys = frames["epoch6_base"][["Subject_id", "Study_id"]].copy()
    subject_ids = keys["Subject_id"].to_numpy()
    unique_subjects = np.unique(subject_ids)
    subject_indices = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    results = {"epoch6_base": [], "standard_bce_320": [], "standard_bce_448": []}
    for _ in range(repetitions):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in sampled])
        for method, frame in frames.items():
            targets, probabilities = extract_arrays(frame.iloc[indices])
            metric, _ = metric_summary(targets, probabilities)
            results[method].append(metric)
    comparisons = []
    for left, right in (
        ("standard_bce_448", "standard_bce_320"),
        ("standard_bce_448", "epoch6_base"),
        ("standard_bce_320", "epoch6_base"),
    ):
        delta = np.asarray(results[left]) - np.asarray(results[right])
        delta = delta[np.isfinite(delta)]
        comparisons.append({
            "comparison": f"{left}_minus_{right}",
            "bootstrap_repetitions_requested": repetitions,
            "bootstrap_repetitions_finite": int(delta.size),
            "mean_delta": float(np.mean(delta)),
            "ci95_low": float(np.percentile(delta, 2.5)),
            "ci95_high": float(np.percentile(delta, 97.5)),
            "ci95_crosses_zero": bool(np.percentile(delta, 2.5) <= 0 <= np.percentile(delta, 97.5)),
        })
    return comparisons


def create_model(checkpoint: Path, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(state["config"]["model_config"])
    model_config["pretrained"] = False
    model = ChestXrayClassifier(**{key: value for key, value in model_config.items()
                                   if key != "name"})
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    amp_enabled = device.type == "cuda" and bool(state["config"].get("amp", True))
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float32
    return model, state, amp_enabled, amp_dtype


def prepare_image(image_path: Path, image_size: int):
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    tensor = build_transform(image_size)(original)
    return original, tensor


def make_gradcam(model, image_path: Path, class_index: int, image_size: int,
                 device, amp_enabled, amp_dtype):
    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = model.backbone.denseblock4.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    original, tensor = prepare_image(image_path, image_size)
    tensor = tensor.unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(tensor)
    score = float(logits[0, class_index].detach().cpu())
    logits[0, class_index].backward()
    handle.remove()
    activation = captured["activation"].detach()
    gradient = captured["gradient"].detach()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.float().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    display = ResizePad(image_size)(original).convert("L").convert("RGB")
    source_array = np.asarray(display).astype(np.float32) / 255.0
    heat = np.zeros_like(source_array)
    heat[:, :, 0] = np.clip(cam * 2.0, 0.0, 1.0)
    heat[:, :, 1] = np.clip(1.5 - np.abs(2.0 * cam - 1.0) * 1.5, 0.0, 1.0)
    heat[:, :, 2] = np.clip(1.0 - cam * 1.5, 0.0, 1.0)
    overlay = np.clip(source_array * 0.58 + heat * 0.42, 0.0, 1.0)
    return source_array, overlay, score


def select_gradcam_cases(comparison, image_frame):
    selected = []
    used = set()
    for class_id in (4, 8):
        target = comparison[f"target_{class_id}"].to_numpy(dtype=int)
        delta = (comparison[f"prob_{class_id}"] - comparison[f"prob_{class_id}_320"]).to_numpy()
        candidates = (
            ("positive_gain_448_vs_320", np.flatnonzero(target == 1), True, delta),
            ("positive_drop_448_vs_320", np.flatnonzero(target == 1), False, delta),
            ("highest_score_negative_448", np.flatnonzero(target == 0), True,
             comparison[f"prob_{class_id}"].to_numpy()),
        )
        for kind, indices, choose_high, values in candidates:
            ordered = sorted(indices, key=lambda index: values[index], reverse=choose_high)
            chosen = next((index for index in ordered if (
                comparison.iloc[index]["Subject_id"], comparison.iloc[index]["Study_id"]) not in used), None)
            if chosen is None:
                continue
            row = comparison.iloc[chosen]
            key = (row["Subject_id"], row["Study_id"])
            used.add(key)
            images = image_frame[
                (image_frame["Subject_id"].astype(str) == str(row["Subject_id"])) &
                (image_frame["Study_id"].astype(str) == str(row["Study_id"]))
            ]
            if images.empty:
                continue
            selected.append({
                "class_id": class_id, "class_name": CLASS_NAMES[class_id - 1],
                "case_kind": kind, "Subject_id": row["Subject_id"], "Study_id": row["Study_id"],
                "Dicom_id": str(images.iloc[0]["Dicom_id"]),
                "target": int(row[f"target_{class_id}"]),
                "probability_epoch6_320": float(row[f"prob_{class_id}_base"]),
                "probability_continuation_320": float(row[f"prob_{class_id}_320"]),
                "probability_continuation_448": float(row[f"prob_{class_id}"]),
                "delta_448_minus_320": float(row[f"prob_{class_id}"] - row[f"prob_{class_id}_320"]),
                "n_images": int(len(images)),
            })
    return selected


def export_gradcam(cases, checkpoint_paths, image_root: Path, output_path: Path, device):
    if not cases:
        raise RuntimeError("No validation cases were selected for Grad-CAM comparison")
    models = {}
    for name, checkpoint in checkpoint_paths.items():
        models[name] = create_model(checkpoint, device)
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, max(4, 3.5 * len(cases))), squeeze=False)
    for row_index, case in enumerate(cases):
        image_path = (image_root / "valid" / case["Subject_id"] /
                      case["Study_id"] / f"{case['Dicom_id']}.jpg")
        original_source = None
        for column, (method, image_size) in enumerate(
                (("epoch6_base", 320), ("standard_bce_320", 320), ("standard_bce_448", 448))):
            model, _state, amp_enabled, amp_dtype = models[method]
            source, overlay, image_logit = make_gradcam(
                model, image_path, case["class_id"] - 1, image_size,
                device, amp_enabled, amp_dtype)
            if original_source is None:
                original_source = source
            axis = axes[row_index, column + 1]
            axis.imshow(overlay)
            axis.set_title(f"{method} Grad-CAM\nimage logit={image_logit:.3f}")
        axes[row_index, 0].imshow(original_source, cmap="gray")
        axes[row_index, 0].set_title(
            f"{case['class_name']} / {case['case_kind']}\n"
            f"Study target={case['target']}; 320 p={case['probability_continuation_320']:.3f}; "
            f"448 p={case['probability_continuation_448']:.3f}")
        for axis in axes[row_index]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    model = None
    models.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def latency_for_checkpoint(name, checkpoint: Path, dataset: StudyDataset,
                           device, warmup_studies: int, max_studies: int):
    model, state, amp_enabled, amp_dtype = create_model(checkpoint, device)
    image_size = int(state["config"]["data"]["image_size"])
    transform = build_transform(image_size)
    total = min(len(dataset), warmup_studies + max_studies)
    measured = {
        "single_decode_convert_ms": [], "single_transform_ms": [], "single_h2d_ms": [],
        "single_forward_ms": [], "single_e2e_ms": [],
        "study_preprocess_ms": [], "study_h2d_ms": [], "study_forward_fusion_ms": [],
        "study_e2e_ms": [], "study_images": [],
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for study_index in range(total):
            study = dataset.studies[study_index]
            warmup = study_index < warmup_studies
            for image_path in study["image_paths"]:
                image_started = time.perf_counter()
                decode_started = time.perf_counter()
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                decode_elapsed = time.perf_counter() - decode_started
                transform_started = time.perf_counter()
                tensor = transform(image)
                transform_elapsed = time.perf_counter() - transform_started
                if not warmup:
                    measured["single_decode_convert_ms"].append(decode_elapsed * 1000)
                    measured["single_transform_ms"].append(transform_elapsed * 1000)

                input_cpu = tensor.unsqueeze(0)
                if device.type == "cuda":
                    input_cpu = input_cpu.pin_memory()
                    transfer_start = torch.cuda.Event(enable_timing=True)
                    transfer_end = torch.cuda.Event(enable_timing=True)
                    transfer_start.record()
                    input_device = input_cpu.to(device, non_blocking=True)
                    transfer_end.record()
                    torch.cuda.synchronize(device)
                    transfer_ms = float(transfer_start.elapsed_time(transfer_end))
                    forward_start = torch.cuda.Event(enable_timing=True)
                    forward_end = torch.cuda.Event(enable_timing=True)
                    forward_start.record()
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        _ = model(input_device)
                    forward_end.record()
                    torch.cuda.synchronize(device)
                    forward_ms = float(forward_start.elapsed_time(forward_end))
                else:
                    transfer_started = time.perf_counter()
                    input_device = input_cpu.to(device)
                    transfer_ms = (time.perf_counter() - transfer_started) * 1000
                    forward_started = time.perf_counter()
                    _ = model(input_device)
                    forward_ms = (time.perf_counter() - forward_started) * 1000
                image_e2e = (time.perf_counter() - image_started) * 1000
                if not warmup:
                    measured["single_h2d_ms"].append(transfer_ms)
                    measured["single_forward_ms"].append(forward_ms)
                    measured["single_e2e_ms"].append(image_e2e)

            study_started = time.perf_counter()
            preprocess_started = time.perf_counter()
            cpu_tensors = []
            for image_path in study["image_paths"]:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                cpu_tensors.append(transform(image))
            cpu_images = torch.stack(cpu_tensors)
            if device.type == "cuda":
                cpu_images = cpu_images.pin_memory()
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
            if device.type == "cuda":
                study_index_tensor = torch.zeros(len(cpu_tensors), dtype=torch.long).pin_memory()
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record()
                images = cpu_images.to(device, non_blocking=True)
                study_index_device = study_index_tensor.to(device, non_blocking=True)
                h2d_end.record()
                torch.cuda.synchronize(device)
                h2d_ms = float(h2d_start.elapsed_time(h2d_end))
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    _ = model(images, study_index_device, 1)
                forward_end.record()
                torch.cuda.synchronize(device)
                forward_ms = float(forward_start.elapsed_time(forward_end))
            else:
                h2d_started = time.perf_counter()
                images = cpu_images.to(device)
                study_index_device = torch.zeros(len(cpu_tensors), dtype=torch.long, device=device)
                h2d_ms = (time.perf_counter() - h2d_started) * 1000
                forward_started = time.perf_counter()
                _ = model(images, study_index_device, 1)
                forward_ms = (time.perf_counter() - forward_started) * 1000
            study_e2e_ms = (time.perf_counter() - study_started) * 1000
            if not warmup:
                measured["study_preprocess_ms"].append(preprocess_ms)
                measured["study_h2d_ms"].append(h2d_ms)
                measured["study_forward_fusion_ms"].append(forward_ms)
                measured["study_e2e_ms"].append(study_e2e_ms)
                measured["study_images"].append(len(cpu_tensors))

    peak_allocated_gib = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                          if device.type == "cuda" else None)
    peak_reserved_gib = (torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                         if device.type == "cuda" else None)
    report = {
        "method": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "image_size": image_size,
        "measured_studies": len(measured["study_e2e_ms"]),
        "measured_images": len(measured["single_e2e_ms"]),
        "study_image_count_mean": float(np.mean(measured["study_images"])),
        "single_image": {
            "decode_convert": percentile_summary(measured["single_decode_convert_ms"]),
            "resize_augment_normalize": percentile_summary(measured["single_transform_ms"]),
            "host_to_device": percentile_summary(measured["single_h2d_ms"]),
            "forward": percentile_summary(measured["single_forward_ms"]),
            "end_to_end_direct": percentile_summary(measured["single_e2e_ms"]),
        },
        "study": {
            "decode_preprocess": percentile_summary(measured["study_preprocess_ms"]),
            "host_to_device": percentile_summary(measured["study_h2d_ms"]),
            "forward_and_fusion": percentile_summary(measured["study_forward_fusion_ms"]),
            "end_to_end": percentile_summary(measured["study_e2e_ms"]),
        },
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "timing_scope": "Local RTX 4060 Laptop; validation Studies; synchronous component timings; T4 not measured.",
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", default=str(PROJECT_DIR / "outputs/predictions/val_best.csv"))
    parser.add_argument("--320-predictions", dest="predictions_320", default=str(
        PROJECT_DIR / "outputs/experiments/paired_standard_bce/outputs/predictions/val_best.csv"))
    parser.add_argument("--448-predictions", dest="predictions_448", default=str(
        PROJECT_DIR / "outputs/experiments/densenet_448_e6/outputs/predictions/val_best.csv"))
    parser.add_argument("--base-checkpoint", default=str(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--320-checkpoint", dest="checkpoint_320", default=str(
        PROJECT_DIR / "outputs/experiments/paired_standard_bce/checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--448-checkpoint", dest="checkpoint_448", default=str(
        PROJECT_DIR / "outputs/experiments/densenet_448_e6/checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--data-config", default=str(PROJECT_DIR / "configs/data.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs/experiments/densenet_448_analysis"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-studies", type=int, default=10)
    parser.add_argument("--latency-studies", type=int, default=100)
    args = parser.parse_args()

    output_dir = resolve_config_path(args.output_dir)
    summary_path = output_dir / "comparison_summary.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing analysis artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "epoch6_base": resolve_config_path(args.base_predictions),
        "standard_bce_320": resolve_config_path(args.predictions_320),
        "standard_bce_448": resolve_config_path(args.predictions_448),
    }
    frames = {name: normalize_predictions(path, name) for name, path in sources.items()}
    base = frames["epoch6_base"].sort_values(["Subject_id", "Study_id"]).reset_index(drop=True)
    expected_keys = list(zip(base["Subject_id"], base["Study_id"]))
    for name, frame in frames.items():
        frame.sort_values(["Subject_id", "Study_id"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if list(zip(frame["Subject_id"], frame["Study_id"])) != expected_keys:
            raise ValueError(f"{name} predictions do not cover the same validation Studies")
        if not np.array_equal(
                frame[[f"target_{i}" for i in range(1, 11)]].to_numpy(),
                base[[f"target_{i}" for i in range(1, 11)]].to_numpy()):
            raise ValueError(f"{name} labels differ from the baseline validation labels")

    data_path = resolve_config_path(args.data_config)
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    validation = pd.read_csv(data_cfg["val_csv"], dtype={
        "Dicom_id": "string", "Subject_id": "string", "Study_id": "string"})
    image_counts = validation.groupby(["Subject_id", "Study_id"], sort=False).size()
    key_index = pd.MultiIndex.from_frame(base[["Subject_id", "Study_id"]])
    base["n_images"] = image_counts.reindex(key_index).to_numpy(dtype=int)

    summary_rows, class_rows, group_rows = [], [], []
    for method, frame in frames.items():
        targets, probabilities = extract_arrays(frame)
        macro_auc, per_class = metric_summary(targets, probabilities)
        summary_rows.append({"method": method, "n_studies": len(frame), "macro_auroc": macro_auc})
        for item in per_class:
            class_rows.append({"method": method, **item})
        for group_name, mask in (
            ("single_image", base["n_images"].to_numpy() == 1),
            ("multi_image", base["n_images"].to_numpy() > 1),
        ):
            group_auc, group_classes = metric_summary(targets[mask], probabilities[mask])
            group_rows.append({
                "method": method, "group": group_name, "n_studies": int(mask.sum()),
                "macro_auroc": group_auc,
            })
            for item in group_classes:
                group_rows.append({
                    "method": method, "group": group_name, "n_studies": int(mask.sum()), **item,
                })

    bootstrap = subject_bootstrap(frames, args.bootstrap_repetitions, args.seed)
    comparison = base[["Subject_id", "Study_id", "n_images"]].copy()
    for method, frame in frames.items():
        comparison[f"probability_source_{method}"] = method
        for class_id in range(1, 11):
            suffix = {"epoch6_base": "_base", "standard_bce_320": "_320",
                      "standard_bce_448": ""}[method]
            comparison[f"prob_{class_id}{suffix}"] = frame[f"prob_{class_id}"].to_numpy()
    for class_id in range(1, 11):
        comparison[f"target_{class_id}"] = base[f"target_{class_id}"].to_numpy()
    cases = select_gradcam_cases(comparison, validation)
    pd.DataFrame(summary_rows).to_csv(output_dir / "macro_auroc_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "single_multi_image_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(output_dir / "subject_bootstrap.csv", index=False)
    pd.DataFrame(cases).to_csv(output_dir / "gradcam_case_selection.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    history_rows = []
    history_specs = (
        ("standard_bce_320", PROJECT_DIR / "outputs/experiments/paired_standard_bce/outputs/logs/train.log"),
        ("standard_bce_448", PROJECT_DIR / "outputs/experiments/densenet_448_e6/outputs/logs/train.log"),
    )
    for method, log_path in history_specs:
        history_rows.extend(parse_history(log_path, method))
    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    for method, subset in history_frame.groupby("method", sort=False):
        subset = subset.sort_values("epoch")
        axes[0].plot(subset["epoch"], subset["train_loss"], marker="o", label=f"{method} train")
        axes[0].plot(subset["epoch"], subset["validation_loss"], marker="s", label=f"{method} validation")
        axes[1].plot(subset["epoch"], subset["macro_auroc"], marker="o", label=method)
    axes[0].set(title="Training and validation BCE", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation Study Macro-AUROC", xlabel="Epoch", ylabel="Macro-AUROC")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    class_frame = pd.DataFrame(class_rows)
    pivot = class_frame.pivot(index="class_name", columns="method", values="auroc").reindex(CLASS_NAMES)
    axis = pivot.plot(kind="bar", figsize=(14, 5))
    axis.set(title="Per-class Study AUROC", xlabel="", ylabel="AUROC")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "class_auroc.png", dpi=160)
    plt.close(axis.figure)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested local GPU latency analysis")
    device = torch.device("cuda")
    checkpoint_paths = {
        "epoch6_base": resolve_config_path(args.base_checkpoint),
        "standard_bce_320": resolve_config_path(args.checkpoint_320),
        "standard_bce_448": resolve_config_path(args.checkpoint_448),
    }
    export_gradcam(cases, checkpoint_paths, Path(data_cfg["image_root"]),
                   output_dir / "gradcam_320_vs_448.png", device)
    latency_dataset = StudyDataset(
        data_cfg["val_csv"], data_cfg["image_root"], transform=None, label_column=None,
        image_id_column=data_cfg["image_id_column"],
        subject_column=data_cfg["subject_column"], study_column=data_cfg["study_column"])
    latency_rows = []
    for method, checkpoint in checkpoint_paths.items():
        latency_rows.append(latency_for_checkpoint(
            method, checkpoint, latency_dataset, device,
            max(args.warmup_studies, 0), max(args.latency_studies, 1)))
    latency_report = {
        "device": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "results": latency_rows,
        "scope": "Validation data measured on the local device; this is not a T4 measurement.",
    }
    (output_dir / "latency_results.json").write_text(
        json.dumps(latency_report, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "validation_studies": len(base),
        "validation_images": int(base["n_images"].sum()),
        "summary": summary_rows,
        "checkpoint_sha256": {
            method: sha256(checkpoint) for method, checkpoint in checkpoint_paths.items()
        },
        "subject_bootstrap": bootstrap,
        "latency_device": latency_report["device"],
        "latency": latency_rows,
        "gradcam_cases": len(cases),
        "output_dir": str(output_dir),
        "artifacts": [
            "macro_auroc_summary.csv", "class_metrics.csv", "single_multi_image_metrics.csv",
            "subject_bootstrap.csv", "training_history.csv", "training_curves.png",
            "class_auroc.png", "gradcam_case_selection.csv", "gradcam_320_vs_448.png",
            "latency_results.json",
        ],
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
