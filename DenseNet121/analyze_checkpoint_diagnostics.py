"""Evaluate saved DenseNet checkpoints and cache image logits for Study fusion analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             roc_auc_score)
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from datasets.transforms import ResizePad
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


CLASS_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]
FUSION_NAMES = ["logit_mean", "probability_mean", "probability_max"]
WEAK_CLASS_IDS = [2, 3, 4, 8]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80, 80)))


def f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if not thresholds.size:
        return 0.5, float(f1_score(targets, probabilities >= 0.5, zero_division=0))
    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def metric_summary(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[dict]]:
    per_class = []
    aucs = []
    for class_index, name in enumerate(CLASS_NAMES):
        y = targets[:, class_index].astype(int)
        p = probabilities[:, class_index]
        positives = int(y.sum())
        negatives = int(y.size - positives)
        auc = float(roc_auc_score(y, p)) if positives and negatives else float("nan")
        ap = float(average_precision_score(y, p)) if positives else float("nan")
        threshold, best_f1 = f1_threshold(y, p) if positives and negatives else (0.5, float("nan"))
        f1_at_05 = float(f1_score(y, p >= 0.5, zero_division=0))
        predicted_positive = p >= 0.5
        true_positive_at_05 = int(np.sum((y == 1) & predicted_positive))
        false_positive_at_05 = int(np.sum((y == 0) & predicted_positive))
        false_negative_at_05 = int(np.sum((y == 1) & ~predicted_positive))
        if np.isfinite(auc):
            aucs.append(auc)
        per_class.append({
            "class_id": class_index + 1,
            "class_name": name,
            "n_positive": positives,
            "n_negative": negatives,
            "auroc": auc,
            "auprc": ap,
            "f1_at_0_5": f1_at_05,
            "true_positive_at_0_5": true_positive_at_05,
            "false_positive_at_0_5": false_positive_at_05,
            "false_negative_at_0_5": false_negative_at_05,
            "precision_at_0_5": (true_positive_at_05 / max(true_positive_at_05 + false_positive_at_05, 1)),
            "recall_at_0_5": (true_positive_at_05 / max(true_positive_at_05 + false_negative_at_05, 1)),
            "f1_best_validation_threshold": best_f1,
            "threshold_for_best_validation_f1": threshold,
            "predicted_positive_at_0_5": int((p >= 0.5).sum()),
            "positive_probability_median": float(np.median(p[y == 1])) if positives else float("nan"),
            "negative_probability_median": float(np.median(p[y == 0])) if negatives else float("nan"),
            "positive_probability_p10": float(np.quantile(p[y == 1], 0.1)) if positives else float("nan"),
            "positive_probability_p90": float(np.quantile(p[y == 1], 0.9)) if positives else float("nan"),
            "negative_probability_p90": float(np.quantile(p[y == 0], 0.9)) if negatives else float("nan"),
        })
    return (float(np.mean(aucs)) if aucs else float("nan"), per_class)


def build_model(checkpoint_path: Path, device: torch.device, expected_epoch: int):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(state.get("epoch", 0))
    if epoch != expected_epoch:
        raise ValueError(f"{checkpoint_path} stores epoch {epoch}; expected epoch {expected_epoch}")
    config = state.get("config") or {}
    model_config = dict(config.get("model_config") or {})
    if not model_config:
        raise ValueError(f"{checkpoint_path} is missing the saved model_config")
    model_config["pretrained"] = False
    model = ChestXrayClassifier(**{key: value for key, value in model_config.items() if key != "name"})
    incompatible = model.load_state_dict(state["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state load returned incompatible keys: {incompatible}")
    model.to(device).eval()
    return model, state


def image_metadata_map(csv_path: Path) -> dict[str, dict]:
    frame = pd.read_csv(csv_path, dtype={"Dicom_id": "string", "Subject_id": "string", "Study_id": "string"},
                        keep_default_na=False)
    return frame.set_index("Dicom_id").to_dict(orient="index")


def aggregate_studies(dataset: StudyDataset, image_logits: np.ndarray,
                      image_metadata: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows, study_rows = [], []
    cursor = 0
    for item in dataset.studies:
        count = len(item["image_ids"])
        values = image_logits[cursor:cursor + count]
        if values.shape != (count, 10):
            raise ValueError(f"Bad image-logit block for {(item['subject_id'], item['study_id'])}")
        probs = sigmoid(values)
        fused = {
            "logit_mean": sigmoid(values.mean(axis=0)),
            "probability_mean": probs.mean(axis=0),
            "probability_max": probs.max(axis=0),
        }
        view_values = sorted({str(image_metadata.get(image_id, {}).get("ViewPosition", "")).strip()
                              for image_id in item["image_ids"]} - {""})
        view_group = "mixed" if len(view_values) > 1 else (view_values[0] if view_values else "unknown")
        sizes = []
        for image_index, image_id in enumerate(item["image_ids"]):
            meta = image_metadata.get(image_id, {})
            row = {
                "Subject_id": item["subject_id"], "Study_id": item["study_id"], "Dicom_id": image_id,
                "ViewPosition": meta.get("ViewPosition", ""), "Rows": meta.get("Rows", ""),
                "Columns": meta.get("Columns", ""),
            }
            for class_index, class_id in enumerate(range(1, 11)):
                row[f"logit_{class_id}"] = float(values[image_index, class_index])
            image_rows.append(row)
            try:
                sizes.append((int(meta["Rows"]), int(meta["Columns"])))
            except (KeyError, TypeError, ValueError):
                pass
        short_sides = [min(rows, columns) for rows, columns in sizes]
        long_sides = [max(rows, columns) for rows, columns in sizes]
        study_row = {
            "Subject_id": item["subject_id"], "Study_id": item["study_id"],
            "n_images": count, "view_positions": ";".join(view_values), "view_group": view_group,
            "median_short_side": float(np.median(short_sides)) if short_sides else np.nan,
            "median_long_side": float(np.median(long_sides)) if long_sides else np.nan,
        }
        for class_index, class_id in enumerate(range(1, 11)):
            study_row[f"y_{class_id}"] = float(item["target"][class_index])
            study_row[f"mean_logit_{class_id}"] = float(values[:, class_index].mean())
            for fusion in FUSION_NAMES:
                study_row[f"{fusion}_{class_id}"] = float(fused[fusion][class_index])
        study_rows.append(study_row)
        cursor += count
    if cursor != len(image_logits):
        raise ValueError(f"Consumed {cursor} of {len(image_logits)} image logits")
    return pd.DataFrame(image_rows), pd.DataFrame(study_rows)


def group_metric_rows(checkpoint: str, fusion: str, group_type: str, groups: pd.Series,
                      targets: np.ndarray, probabilities: np.ndarray) -> tuple[list[dict], list[dict]]:
    summary_rows, class_rows = [], []
    for value in sorted(groups.dropna().unique(), key=lambda current: str(current)):
        mask = (groups == value).to_numpy()
        if int(mask.sum()) < 2:
            continue
        macro_auc, per_class = metric_summary(targets[mask], probabilities[mask])
        summary_rows.append({"checkpoint": checkpoint, "fusion": fusion, "group_type": group_type,
                             "group": value, "n_studies": int(mask.sum()), "macro_auroc": macro_auc})
        for item in per_class:
            class_rows.append({"checkpoint": checkpoint, "fusion": fusion, "group_type": group_type,
                               "group": value, "n_studies": int(mask.sum()), **item})
    return summary_rows, class_rows


def choose_error_cases(studies: pd.DataFrame, output_dir: Path) -> dict[int, float]:
    case_rows = []
    thresholds = {}
    for class_id in WEAK_CLASS_IDS:
        name = CLASS_NAMES[class_id - 1]
        target = studies[f"y_{class_id}"].to_numpy().astype(int)
        probability = studies[f"logit_mean_{class_id}"].to_numpy()
        threshold, _ = f1_threshold(target, probability)
        thresholds[class_id] = threshold
        for kind, mask, order in (
            ("high_score_negative", target == 0, "descending"),
            ("low_score_positive", target == 1, "ascending"),
            ("high_score_positive", target == 1, "descending"),
        ):
            indices = np.flatnonzero(mask)
            indices = sorted(indices, key=lambda index: probability[index], reverse=(order == "descending"))[:5]
            for rank, index in enumerate(indices, start=1):
                row = studies.iloc[index]
                positive = bool(target[index])
                predicted_at_f1_threshold = probability[index] >= threshold
                case_rows.append({
                    "class_id": class_id, "class_name": name, "case_kind": kind, "rank": rank,
                    "Subject_id": row["Subject_id"], "Study_id": row["Study_id"],
                    "target": int(target[index]), "probability": float(probability[index]),
                    "f1_threshold_on_validation": threshold,
                    "error_at_validation_f1_threshold": ("false_positive" if predicted_at_f1_threshold and not positive
                                                         else "false_negative" if not predicted_at_f1_threshold and positive
                                                         else "correct"),
                    "n_images": int(row["n_images"]), "view_positions": row["view_positions"],
                    "median_short_side": row["median_short_side"],
                    "cooccurring_positive_classes": ";".join(
                        CLASS_NAMES[other_id - 1] for other_id in range(1, 11)
                        if other_id != class_id and row[f"y_{other_id}"] == 1.0),
                })
    pd.DataFrame(case_rows).to_csv(output_dir / "weak_class_error_cases.csv", index=False)
    return thresholds


def cooccurrence_rows(studies: pd.DataFrame) -> list[dict]:
    result = []
    for target_id in WEAK_CLASS_IDS:
        target_positive = studies[f"y_{target_id}"].to_numpy() == 1
        for other_id, other_name in enumerate(CLASS_NAMES, start=1):
            if other_id == target_id:
                continue
            other_positive = studies[f"y_{other_id}"].to_numpy() == 1
            n_pos = int(target_positive.sum())
            n_neg = int((~target_positive).sum())
            result.append({
                "target_class": CLASS_NAMES[target_id - 1], "cooccurring_class": other_name,
                "target_positive_studies": n_pos,
                "target_positive_cooccurring_studies": int((target_positive & other_positive).sum()),
                "target_positive_cooccurrence_rate": float((target_positive & other_positive).sum() / n_pos),
                "target_negative_cooccurring_studies": int((~target_positive & other_positive).sum()),
                "target_negative_cooccurrence_rate": float((~target_positive & other_positive).sum() / n_neg),
            })
    return result


def make_gradcam(model: ChestXrayClassifier, image_path: Path, class_index: int,
                 device: torch.device, image_size: int = 320) -> tuple[np.ndarray, np.ndarray, float]:
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = model.backbone.denseblock4.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    tensor = build_transform(image_size)(original).unsqueeze(0).to(device)
    amp_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        logits = model(tensor)
    score = float(logits[0, class_index].detach().cpu())
    logits[0, class_index].backward()
    handle.remove()
    activation = captured["activation"].detach()
    gradient = captured["gradient"].detach()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(image_size, image_size), mode="bilinear",
                                          align_corners=False)[0, 0]
    cam = cam.float().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    display = ResizePad(image_size)(original).convert("L").convert("RGB")
    image_array = np.asarray(display).astype(np.float32) / 255.0
    heat = np.zeros_like(image_array)
    heat[:, :, 0] = np.clip(cam * 2.0, 0.0, 1.0)
    heat[:, :, 1] = np.clip(1.5 - np.abs(2.0 * cam - 1.0) * 1.5, 0.0, 1.0)
    heat[:, :, 2] = np.clip(1.0 - cam * 1.5, 0.0, 1.0)
    overlay = np.clip(image_array * 0.58 + heat * 0.42, 0.0, 1.0)
    return image_array, overlay, score


def export_gradcam(model: ChestXrayClassifier, studies: pd.DataFrame, image_rows: pd.DataFrame,
                   image_root: Path, output_path: Path, thresholds: dict[int, float],
                   device: torch.device, image_size: int = 320) -> list[dict]:
    selections = []
    for class_id in [4, 8]:
        target = studies[f"y_{class_id}"].to_numpy().astype(int)
        score = studies[f"logit_mean_{class_id}"].to_numpy()
        threshold = thresholds[class_id]
        for case_kind, mask, choose_high in (
            ("correct_positive", (target == 1) & (score >= threshold), True),
            ("false_positive", (target == 0) & (score >= threshold), True),
            ("false_negative", (target == 1) & (score < threshold), False),
        ):
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            selected_index = max(indices, key=lambda index: score[index]) if choose_high else min(
                indices, key=lambda index: score[index])
            row = studies.iloc[selected_index]
            study_images = image_rows[
                (image_rows["Subject_id"] == row["Subject_id"]) &
                (image_rows["Study_id"] == row["Study_id"])
            ]
            logit_column = f"logit_{class_id}"
            image = study_images.loc[study_images[logit_column].astype(float).idxmax()]
            selections.append({
                "class_id": class_id, "class_name": CLASS_NAMES[class_id - 1], "case_kind": case_kind,
                "Subject_id": row["Subject_id"], "Study_id": row["Study_id"],
                "Dicom_id": image["Dicom_id"], "study_probability": float(score[selected_index]),
                "threshold_for_f1_selected_on_validation": threshold,
                "n_images": int(row["n_images"]), "view_positions": row["view_positions"],
                "rows": int(image["Rows"]), "columns": int(image["Columns"]),
            })
    if not selections:
        return selections

    dataset_root = image_root / "valid"
    fig, axes = plt.subplots(len(selections), 2, figsize=(10, 4.5 * len(selections)), squeeze=False)
    for row_index, selection in enumerate(selections):
        path = dataset_root / selection["Subject_id"] / selection["Study_id"] / f"{selection['Dicom_id']}.jpg"
        original, overlay, image_logit = make_gradcam(model, path, selection["class_id"] - 1,
                                                      device, image_size)
        axes[row_index, 0].imshow(original, cmap="gray")
        axes[row_index, 0].set_title(
            f"{selection['class_name']} / {selection['case_kind']}\n"
            f"study p={selection['study_probability']:.3f}; f1 threshold={selection['threshold_for_f1_selected_on_validation']:.3f}; "
            f"image logit={image_logit:.3f}")
        axes[row_index, 1].imshow(overlay)
        axes[row_index, 1].set_title(f"Grad-CAM — {selection['Dicom_id']}; {selection['rows']}×{selection['columns']}")
        for axis in axes[row_index]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return selections


def plot_auc_comparison(epoch_metrics: pd.DataFrame, fusion_class_metrics: pd.DataFrame,
                        output_dir: Path) -> None:
    epoch6 = epoch_metrics[(epoch_metrics["checkpoint"] == "epoch6") &
                           (epoch_metrics["fusion"] == "logit_mean") &
                           (epoch_metrics["group_type"] == "all")].sort_values("class_id")
    epoch14 = epoch_metrics[(epoch_metrics["checkpoint"] == "epoch14") &
                            (epoch_metrics["fusion"] == "logit_mean") &
                            (epoch_metrics["group_type"] == "all")].sort_values("class_id")
    positions = np.arange(len(CLASS_NAMES))
    fig, axis = plt.subplots(figsize=(13, 5.5))
    axis.bar(positions - 0.2, epoch6["auroc"], width=0.4, label="Epoch 6 best")
    axis.bar(positions + 0.2, epoch14["auroc"], width=0.4, label="Epoch 14 last")
    axis.set_xticks(positions, [name.replace(" ", "\n") for name in CLASS_NAMES], fontsize=8)
    axis.set_ylim(0.5, 1.0)
    axis.set_ylabel("Study AUROC")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "densenet_epoch6_vs_epoch14_auc.png", dpi=160)
    plt.close(fig)

    fusion = fusion_class_metrics[(fusion_class_metrics["checkpoint"] == "epoch6") &
                                  (fusion_class_metrics["group_type"] == "all")]
    pivot = fusion.pivot(index="class_name", columns="fusion", values="auroc").reindex(CLASS_NAMES)
    fig, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(pivot[FUSION_NAMES].to_numpy(), vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xticks(np.arange(len(FUSION_NAMES)), [name.replace("_", " ") for name in FUSION_NAMES])
    for y in range(len(CLASS_NAMES)):
        for x in range(len(FUSION_NAMES)):
            axis.text(x, y, f"{pivot.iloc[y, x]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(image, ax=axis, label="Study AUROC")
    fig.tight_layout()
    fig.savefig(output_dir / "densenet_epoch6_fusion_auc_by_class.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-checkpoint", default=str(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--last-checkpoint", default=str(PROJECT_DIR / "checkpoint/last/last.pth"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs/experiments/densenet_epoch6_diagnostics"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=400)
    parser.add_argument("--reuse-existing-logits", action="store_true",
                        help="Reuse image-logit CSVs from a completed fresh inference after validating keys and values")
    args = parser.parse_args()

    best_path = resolve_config_path(args.best_checkpoint).resolve()
    last_path = resolve_config_path(args.last_checkpoint).resolve()
    output_dir = resolve_config_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = (PROJECT_DIR.parent / "docs" / "figures").resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve_config_path(PROJECT_DIR / "configs/data.yaml")
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    validation_csv = Path(data_cfg["val_csv"])
    dataset = StudyDataset(
        validation_csv, data_cfg["image_root"], build_transform(int(data_cfg["image_size"])),
        label_column=data_cfg["label_column"], label_semantics=data_cfg["label_semantics"],
        image_id_column=data_cfg["image_id_column"], subject_column=data_cfg["subject_column"],
        study_column=data_cfg["study_column"],
    )
    labels, known = dataset.label_tensors()
    if not bool(known.all()):
        raise ValueError("Validation labels are incomplete; closed-world evaluation cannot proceed")
    image_meta = image_metadata_map(validation_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_specs = [("epoch6", best_path, 6), ("epoch14", last_path, 14)]
    manifest = {
        "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
        "validation_csv": str(validation_csv), "validation_studies": len(dataset),
        "validation_images": int(sum(dataset.image_counts)), "image_size": int(data_cfg["image_size"]),
        "class_names": CLASS_NAMES, "checkpoints": {},
    }
    if device.type == "cuda":
        manifest["gpu_name"] = torch.cuda.get_device_name(device)

    all_summary, all_class_metrics, all_group_summary, all_group_classes = [], [], [], []
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]] = {}
    for checkpoint_name, checkpoint_path, expected_epoch in checkpoint_specs:
        model, state = build_model(checkpoint_path, device, expected_epoch)
        checkpoint_cfg = state["config"]
        amp_enabled = bool(checkpoint_cfg.get("amp")) and device.type == "cuda"
        amp_dtype_name = checkpoint_cfg.get("amp_dtype", "float16")
        amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
        if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            amp_dtype = torch.float16
            amp_dtype_name = "float16 (BF16 unsupported; matches trainer fallback)"
        image_budget = int(checkpoint_cfg["image_budget"])
        sampler = StudyBatchSampler(dataset, image_budget, shuffle=False)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=int(data_cfg["num_workers"]),
                            pin_memory=device.type == "cuda", collate_fn=collate_studies)
        image_cache_path = output_dir / f"{checkpoint_name}_image_logits.csv"
        if args.reuse_existing_logits and image_cache_path.exists():
            cache_meta_path = output_dir / f"{checkpoint_name}_image_logits.meta.json"
            if cache_meta_path.exists():
                cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                expected_amp = bool(amp_enabled)
                expected_amp_dtype = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
                if (cache_meta.get("checkpoint_sha256") != sha256(checkpoint_path) or
                        cache_meta.get("amp_enabled") != expected_amp or
                        cache_meta.get("amp_dtype") != expected_amp_dtype):
                    raise ValueError(f"Cached logits provenance does not match checkpoint precision: {image_cache_path}")
            cached_images = pd.read_csv(image_cache_path, dtype={
                "Subject_id": "string", "Study_id": "string", "Dicom_id": "string",
            }, keep_default_na=False)
            expected_image_keys = [
                (item["subject_id"], item["study_id"], image_id)
                for item in dataset.studies for image_id in item["image_ids"]
            ]
            cached_image_keys = list(zip(cached_images["Subject_id"].astype(str),
                                         cached_images["Study_id"].astype(str),
                                         cached_images["Dicom_id"].astype(str)))
            logit_columns = [f"logit_{class_id}" for class_id in range(1, 11)]
            if cached_image_keys != expected_image_keys or cached_images[logit_columns].isna().any().any():
                raise ValueError(f"Cached image logits do not match the validation image order: {image_cache_path}")
            image_logits = cached_images[logit_columns].to_numpy(dtype=float)
        else:
            image_logits_batches = []
            with torch.inference_mode():
                for batch in loader:
                    images = batch["images"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        logits = model(images)
                    image_logits_batches.append(logits.float().cpu().numpy())
            image_logits = np.concatenate(image_logits_batches, axis=0)
        if image_logits.shape != (sum(dataset.image_counts), 10) or not np.isfinite(image_logits).all():
            raise ValueError(f"Invalid image logits shape or non-finite values for {checkpoint_name}: "
                             f"{image_logits.shape}")
        image_rows, study_rows = aggregate_studies(dataset, image_logits, image_meta)
        image_rows.to_csv(image_cache_path, index=False)
        (output_dir / f"{checkpoint_name}_image_logits.meta.json").write_text(json.dumps({
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
            "stored_epoch": expected_epoch, "strict_model_state_load": True,
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": "bfloat16" if amp_dtype == torch.bfloat16 else "float16",
            "image_rows": len(image_rows), "image_logits_sha256": sha256(image_cache_path),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        study_rows.to_csv(output_dir / f"{checkpoint_name}_study_predictions.csv", index=False)

        targets = labels.numpy()
        fusion_scores = {}
        for fusion in FUSION_NAMES:
            probabilities = study_rows[[f"{fusion}_{class_id}" for class_id in range(1, 11)]].to_numpy(float)
            fusion_scores[fusion] = probabilities
            macro_auc, per_class = metric_summary(targets, probabilities)
            all_summary.append({"checkpoint": checkpoint_name, "epoch": expected_epoch,
                                "fusion": fusion, "group_type": "all", "group": "all",
                                "n_studies": len(dataset), "macro_auroc": macro_auc,
                                "macro_f1_at_0_5": float(np.mean([item["f1_at_0_5"] for item in per_class]))})
            for item in per_class:
                all_class_metrics.append({"checkpoint": checkpoint_name, "epoch": expected_epoch,
                                          "fusion": fusion, "group_type": "all", "group": "all", **item})
            for group_type, groups in (
                ("image_count", study_rows["n_images"].astype(str)),
                ("single_vs_multi", study_rows["n_images"].map(
                    lambda count: "single_image" if count == 1 else "multi_image")),
                ("view_group", study_rows["view_group"]),
            ):
                summary_rows, class_rows = group_metric_rows(checkpoint_name, fusion, group_type,
                                                             groups, targets, probabilities)
                all_group_summary.extend(summary_rows)
                all_group_classes.extend(class_rows)
        cache[checkpoint_name] = (image_rows, study_rows, fusion_scores)
        manifest["checkpoints"][checkpoint_name] = {
            "path": str(checkpoint_path), "sha256": sha256(checkpoint_path),
            "stored_epoch": int(state["epoch"]), "best_metric": state.get("best_metric"),
            "strict_model_state_load": True, "model_config": checkpoint_cfg["model_config"],
            "training_config": {key: checkpoint_cfg.get(key) for key in (
                "loss", "image_budget", "grad_accumulation", "learning_rate_backbone",
                "learning_rate_head", "weight_decay", "warmup_epochs", "patience",
                "freeze_backbone_epochs", "amp", "amp_dtype", "seed")},
            "weight_counts": checkpoint_cfg.get("weight_counts"),
            "state_dict_tensors": len(state["model"]),
        }
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_summary)
    class_metrics_df = pd.DataFrame(all_class_metrics)
    group_summary_df = pd.DataFrame(all_group_summary)
    group_classes_df = pd.DataFrame(all_group_classes)
    summary_df.to_csv(output_dir / "fusion_summary.csv", index=False)
    class_metrics_df.to_csv(output_dir / "class_metrics.csv", index=False)
    group_summary_df.to_csv(output_dir / "group_summary.csv", index=False)
    group_classes_df.to_csv(output_dir / "group_class_metrics.csv", index=False)

    # Cross-check the fresh Epoch 6 aggregation against the prediction CSV written by training.
    stored_csv = PROJECT_DIR / "outputs/predictions/val_best.csv"
    if stored_csv.exists():
        stored = pd.read_csv(stored_csv, dtype={"Subject_id": "string", "Study_id": "string"})
        fresh = cache["epoch6"][1].copy()
        for frame in (stored, fresh):
            frame["Subject_id"] = frame["Subject_id"].astype(str)
            frame["Study_id"] = frame["Study_id"].astype(str)
        stored = stored.set_index(["Subject_id", "Study_id"])
        fresh = fresh.set_index(["Subject_id", "Study_id"])
        if (len(stored) != len(fresh) or stored.index.has_duplicates or fresh.index.has_duplicates or
                set(stored.index.to_list()) != set(fresh.index.to_list())):
            raise ValueError("Epoch 6 prediction CSV keys differ from the freshly evaluated validation set")
        differences = []
        for class_id, class_name in enumerate(CLASS_NAMES, start=1):
            differences.append(max(abs(float(stored.loc[key, f"prob_{class_name}"]) -
                                       float(fresh.loc[key, f"logit_mean_{class_id}"]))
                                   for key in fresh.index.to_list()))
        manifest["epoch6_cached_prediction_max_abs_difference"] = float(max(differences))

    epoch6_images, epoch6_studies, epoch6_scores = cache["epoch6"]
    weak_thresholds = choose_error_cases(epoch6_studies, output_dir)
    pd.DataFrame(cooccurrence_rows(epoch6_studies)).to_csv(
        output_dir / "weak_class_cooccurrence.csv", index=False)
    epoch6_studies[["Subject_id", "Study_id", "n_images", "view_positions", "median_short_side"] +
                   [f"y_{class_id}" for class_id in range(1, 11)] +
                   [f"logit_mean_{class_id}" for class_id in range(1, 11)]].to_csv(
        output_dir / "epoch6_weak_class_analysis.csv", index=False)

    # Subject-cluster bootstrap keeps multiple Studies from one Subject in the same resample.
    rng = np.random.default_rng(42)
    subject_ids = epoch6_studies["Subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subject_ids)
    subject_indices = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    deltas = {"probability_mean": [], "probability_max": []}
    rare_class_aucs = []
    for _ in range(max(args.bootstrap_repetitions, 0)):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in sampled_subjects])
        y = labels.numpy()[indices]
        base_auc, _ = metric_summary(y, epoch6_scores["logit_mean"][indices])
        if 0 < int(y[:, 0].sum()) < len(y):
            rare_class_aucs.append(float(roc_auc_score(y[:, 0], epoch6_scores["logit_mean"][indices, 0])))
        for fusion in deltas:
            other_auc, _ = metric_summary(y, epoch6_scores[fusion][indices])
            deltas[fusion].append(other_auc - base_auc)
    bootstrap_rows = []
    for fusion, values in deltas.items():
        if values:
            bootstrap_rows.append({"fusion_delta_vs_logit_mean": fusion,
                                   "mean_delta": float(np.mean(values)),
                                   "ci95_low": float(np.quantile(values, 0.025)),
                                   "ci95_high": float(np.quantile(values, 0.975)),
                                   "repetitions": len(values), "cluster": "Subject_id"})
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "fusion_subject_bootstrap.csv", index=False)
    if rare_class_aucs:
        pd.DataFrame([{
            "class_name": CLASS_NAMES[0], "point_estimate_auroc": float(
                class_metrics_df[(class_metrics_df["checkpoint"] == "epoch6") &
                                 (class_metrics_df["fusion"] == "logit_mean") &
                                 (class_metrics_df["group_type"] == "all") &
                                 (class_metrics_df["class_id"] == 1)]["auroc"].iloc[0]),
            "ci95_low": float(np.quantile(rare_class_aucs, 0.025)),
            "ci95_high": float(np.quantile(rare_class_aucs, 0.975)),
            "repetitions": len(rare_class_aucs), "cluster": "Subject_id",
        }]).to_csv(output_dir / "enlarged_cardiomediastinum_subject_bootstrap.csv", index=False)

    best_model, _best_state = build_model(best_path, device, 6)
    selected_cam_cases = export_gradcam(
        best_model, epoch6_studies, epoch6_images, Path(data_cfg["image_root"]),
        figure_dir / "densenet_weak_class_gradcam.png", weak_thresholds, device,
        int(data_cfg["image_size"]))
    pd.DataFrame(selected_cam_cases).to_csv(output_dir / "gradcam_case_selection.csv", index=False)
    del best_model, _best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    plot_auc_comparison(class_metrics_df, class_metrics_df, figure_dir)
    manifest["weak_class_f1_thresholds_selected_on_validation"] = weak_thresholds
    manifest["gradcam_cases"] = selected_cam_cases
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "device": str(device), "validation_studies": len(dataset),
        "validation_images": int(sum(dataset.image_counts)),
        "strict_load_epochs": [manifest["checkpoints"][name]["stored_epoch"] for name, _, _ in checkpoint_specs],
        "fusion_summary": summary_df.to_dict(orient="records"),
        "output_dir": str(output_dir), "figures": str(figure_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
