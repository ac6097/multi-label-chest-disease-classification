from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[3]
PROJECT_DIR = WORK_DIR.parent
REPO_DIR = PROJECT_DIR.parent
DATA_DIR = REPO_DIR / "cxr_first"
OPT_DIR = WORK_DIR / "outputs/experiments/DINOv3_Feature_Optimization"
R5_DIR = OPT_DIR / "R5_Global_Only_512"
R6_DIR = OPT_DIR / "R6_Full_Finetune_512"
CACHE_DIR = HERE / "cached_features"
PROBE_CACHE_DIR = HERE / "probe_cache"
CHART_DIR = HERE / "charts"
DISEASE_DIR = HERE / "per_disease_reports"
IMAGE_SIZE = 512
SEED = 42
BOOTSTRAP_REPLICATES = 2000
C_VALUES = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
EXTRACTOR_VERSION = "r5r6-stage-gap-study-mean-v1"
PROBE_VERSION = "subject-group-l2-logistic-v1"
LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
SHORT = [
    "Enlarged CM", "Pneumothorax", "Consolidation", "Pneumonia", "Edema",
    "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]
TARGET_COLS = [f"target_{name.replace(' ', '_')}" for name in LABELS]
STAGES = ("stage1", "stage2", "stage3", "stage4")
DISPLAY_STAGE = {name: f"Stage {index}" for index, name in enumerate(STAGES, 1)}
EXPECTED_DIMS = {"stage1": 96, "stage2": 192, "stage3": 384,
                 "stage4": 768, "stage4_norm": 768}

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WORK_DIR))
from datasets.dataset import StudyDataset, collate_studies  # noqa: E402
from datasets.transforms import build_transform  # noqa: E402
from dino_data import EpochTaggedStudyDataset, make_loader, shutdown_loader  # noqa: E402
from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier  # noqa: E402


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe_float(value):
    return float(value) if np.isfinite(value) else None


def metadata_of(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"].item()))


def save_npz_atomic(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def official_datasets():
    if not (DATA_DIR / "train.csv").is_file() or not (DATA_DIR / "valid.csv").is_file():
        raise FileNotFoundError(f"Official Train/Validation files missing under {DATA_DIR}")
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Official image root missing: {DATA_DIR}")
    transform = build_transform(IMAGE_SIZE, train=False)
    train = StudyDataset(
        DATA_DIR / "train.csv", DATA_DIR, transform=transform,
        label_column="Predict_class", label_semantics="closed_world",
        image_id_column="Dicom_id", subject_column="Subject_id", study_column="Study_id")
    valid = StudyDataset(
        DATA_DIR / "valid.csv", DATA_DIR, transform=transform,
        label_column="Predict_class", label_semantics="closed_world",
        image_id_column="Dicom_id", subject_column="Subject_id", study_column="Study_id")
    train_subjects = {str(item["subject_id"]) for item in train.studies}
    valid_subjects = {str(item["subject_id"]) for item in valid.studies}
    overlap = sorted(train_subjects & valid_subjects)
    if overlap:
        raise RuntimeError(f"Train/Validation Subject leakage: {len(overlap)} subjects")
    if len(train) != 18513 or int(sum(train.image_counts)) != 30344:
        raise RuntimeError("Train Study/image counts differ from the R5 manifest")
    if len(valid) != 859 or int(sum(valid.image_counts)) != 1422 or len(valid_subjects) != 228:
        raise RuntimeError("Validation Study/image/Subject counts differ from the official manifest")
    for dataset in (train, valid):
        if dataset.split not in {"train", "valid"}:
            raise RuntimeError(f"Unexpected official split: {dataset.split}")
        if len(set(dataset.study_keys)) != len(dataset.study_keys):
            raise RuntimeError(f"Duplicate Study keys in {dataset.csv_file}")
        _, known = dataset.label_tensors()
        if not bool(known.all()):
            raise RuntimeError(f"Unknown label cells found in closed-world split {dataset.split}")
    return {"train": train, "valid": valid}, {
        "train_studies": len(train), "train_images": int(sum(train.image_counts)),
        "train_subjects": len(train_subjects), "validation_studies": len(valid),
        "validation_images": int(sum(valid.image_counts)),
        "validation_subjects": len(valid_subjects), "subject_overlap": 0,
        "label_order": LABELS, "label_semantics": "closed_world",
        "preprocessing": (
            "ResizePad(512), Grayscale(3), ToTensor, ImageNet Normalize; "
            "deterministic evaluation transform for both Train feature extraction and Validation"
        ),
        "train_csv_sha256": file_sha256(DATA_DIR / "train.csv"),
        "validation_csv_sha256": file_sha256(DATA_DIR / "valid.csv"),
        "test_data_accessed": False,
    }


def checkpoint_details(model_name: str):
    run_dir = R5_DIR if model_name == "R5" else R6_DIR
    checkpoint_path = run_dir / "best.pth"
    config_path = run_dir / "config.json"
    status_path = run_dir / "training_status.json"
    for path in (checkpoint_path, config_path, status_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required {model_name} Best artifact missing: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    verification_path = run_dir / "model_verification.json"
    if not status.get("complete"):
        raise RuntimeError(f"{model_name} training is not complete; refusing feature extraction")
    best_epoch = int(status["best_epoch"])
    if int(status["completed_epochs"]) < best_epoch:
        raise RuntimeError(f"{model_name} status Best Epoch exceeds completed epochs")
    return {
        "run_dir": run_dir, "checkpoint": checkpoint_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": best_epoch,
        "best_macro_auroc": float(status["best_macro_auroc"]),
        "config": config, "status": status,
        "model_verification": (json.loads(verification_path.read_text(encoding="utf-8"))
                               if verification_path.is_file() else {}),
    }


def load_best_model(model_info, device):
    checkpoint = torch.load(model_info["checkpoint"], map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state")
    if state is None:
        raise RuntimeError(f"{model_info['checkpoint']} has no model_state")
    if "epoch" in checkpoint and int(checkpoint["epoch"]) != model_info["best_epoch"]:
        raise RuntimeError(
            f"Checkpoint epoch {checkpoint['epoch']} differs from recorded Best {model_info['best_epoch']}")
    model = DINOv3StudyClassifier(DINOv3ConvNeXtBackbone())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict Best checkpoint load failed: {incompatible}")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def features_for_split(model_name, model, model_info, dataset, split, device):
    hooks = {}
    handles = []
    stages = model.backbone.model.stages
    for index, stage in enumerate(stages):
        name = f"stage{index + 1}"
        handles.append(stage[-1].register_forward_hook(
            lambda _module, _inputs, output, feature_name=name: hooks.__setitem__(feature_name, output)))
    tagged_dataset = EpochTaggedStudyDataset(dataset, SEED)
    loader, sampler, _ = make_loader(
        tagged_dataset, image_budget=8, shuffle=False, seed=SEED, num_workers=4,
        pin_memory=True, prefetch_factor=2, collate_fn=collate_studies)
    sampler.set_epoch(0)
    chunks = {key: [] for key in (*STAGES, "stage4_norm")}
    subjects, studies, counts, labels = [], [], [], []
    expected_shapes = {
        "stage1": (96, 128, 128), "stage2": (192, 64, 64),
        "stage3": (384, 32, 32), "stage4": (768, 16, 16),
    }
    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                images = batch["images"].to(device, non_blocking=True)
                images = images.contiguous(memory_format=torch.channels_last)
                pooled_norm, _stage4 = model.backbone.forward_features(images)
                for stage in STAGES:
                    fmap = hooks[stage]
                    if tuple(fmap.shape[1:]) != expected_shapes[stage]:
                        raise RuntimeError(
                            f"{model_name} {stage} shape {tuple(fmap.shape[1:])} != "
                            f"{expected_shapes[stage]}")
                    chunks[stage].append(fmap.float().mean(dim=(-2, -1)).cpu().numpy())
                chunks["stage4_norm"].append(pooled_norm.float().cpu().numpy())
                study_index = batch["study_index"].to(device, non_blocking=True)
                image_counts = batch["image_counts"].to(device, non_blocking=True)
                for stage in (*STAGES, "stage4_norm"):
                    image_vectors = torch.as_tensor(chunks[stage][-1], device=device)
                    study_vectors = torch.zeros(
                        int(batch["num_studies"]), image_vectors.shape[-1],
                        dtype=torch.float32, device=device)
                    study_vectors.index_add_(0, study_index, image_vectors)
                    study_vectors /= image_counts.to(device).unsqueeze(1)
                    chunks[stage][-1] = study_vectors.cpu().numpy()
                subjects.extend(str(value) for value in batch["subject_id"])
                studies.extend(str(value) for value in batch["study_id"])
                counts.extend(int(value) for value in batch["image_counts"].tolist())
                labels.append(batch["labels"].numpy().astype(np.uint8, copy=False))
                if batch_number % 500 == 0:
                    print(f"{model_name}/{split}: extracted {len(studies)} Studies", flush=True)
    finally:
        shutdown_loader(loader)
        for handle in handles:
            handle.remove()

    result = {
        "subject_ids": np.asarray(subjects, dtype="U64"),
        "study_ids": np.asarray(studies, dtype="U64"),
        "image_counts": np.asarray(counts, dtype=np.int16),
        "labels": np.concatenate(labels, axis=0),
        "metadata": None,
    }
    if len(set(zip(result["subject_ids"], result["study_ids"]))) != len(studies):
        raise RuntimeError(f"{model_name}/{split} extraction has duplicate Study IDs")
    if len(studies) != len(dataset):
        raise RuntimeError(f"{model_name}/{split} extracted {len(studies)} of {len(dataset)} Studies")
    for stage in (*STAGES, "stage4_norm"):
        result[stage] = np.concatenate(chunks[stage], axis=0).astype(np.float32, copy=False)
        if result[stage].shape != (len(dataset), EXPECTED_DIMS[stage]):
            raise RuntimeError(
                f"{model_name}/{split}/{stage} has unexpected Study feature shape {result[stage].shape}")
        if not np.isfinite(result[stage]).all():
            raise FloatingPointError(f"Non-finite {model_name}/{split}/{stage} feature values")
    if result["labels"].shape != (len(dataset), 10) or not np.isfinite(result["labels"]).all():
        raise RuntimeError(f"Unexpected {model_name}/{split} labels")
    expected_keys = list(dataset.study_keys)
    extracted_keys = list(zip(result["subject_ids"], result["study_ids"]))
    if set(expected_keys) != set(extracted_keys):
        raise RuntimeError(f"{model_name}/{split} extracted Study key set differs from official CSV")
    return result


def cache_metadata(model_name, split, stage, model_info, feature_dim):
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "backbone": model_name,
        "split": split,
        "stage": stage,
        "feature_dimension": int(feature_dim),
        "checkpoint": str(model_info["checkpoint"]),
        "checkpoint_sha256": model_info["checkpoint_sha256"],
        "best_epoch": int(model_info["best_epoch"]),
        "input_size": IMAGE_SIZE,
        "preprocessing": "deterministic ResizePad(512)+Grayscale3+ToTensor+ImageNetNormalize",
        "study_aggregation": "arithmetic mean of per-image GAP vectors within Study",
        "test_data_accessed": False,
    }


def save_feature_caches(model_name, split, model_info, extracted):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for stage in (*STAGES, "stage4_norm"):
        path = CACHE_DIR / f"{model_name.lower()}_{stage}_{split}.npz"
        meta = cache_metadata(model_name, split, stage, model_info, EXPECTED_DIMS[stage])
        arrays = {
            "Subject_id": extracted["subject_ids"],
            "Study_id": extracted["study_ids"],
            "Image_Count": extracted["image_counts"],
            "labels": extracted["labels"],
            "features": extracted[stage],
            "metadata": np.asarray(json.dumps(meta, ensure_ascii=False)),
        }
        if path.exists():
            existing_meta = metadata_of(path)
            if existing_meta != meta:
                raise RuntimeError(f"Existing cache metadata does not match current source: {path}")
            with np.load(path, allow_pickle=False) as existing:
                for key, value in arrays.items():
                    if key == "metadata":
                        continue
                    if existing[key].shape != value.shape or not np.array_equal(existing[key], value):
                        raise RuntimeError(f"Existing feature cache data mismatch: {path}:{key}")
        else:
            save_npz_atomic(path, **arrays)


def load_feature_set(model_name, split):
    result = {}
    common = None
    for stage in (*STAGES, "stage4_norm"):
        path = CACHE_DIR / f"{model_name.lower()}_{stage}_{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Required extracted Study feature cache missing: {path}")
        metadata = metadata_of(path)
        if metadata.get("extractor_version") != EXTRACTOR_VERSION:
            raise RuntimeError(f"Feature cache extractor version mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            features = archive["features"].astype(np.float32, copy=False)
            subjects = archive["Subject_id"].astype(str)
            studies = archive["Study_id"].astype(str)
            image_counts = archive["Image_Count"].copy()
            labels = archive["labels"].astype(np.uint8, copy=False)
        if features.ndim != 2 or features.shape[1] != EXPECTED_DIMS[stage]:
            raise RuntimeError(f"Bad feature dimension in {path}: {features.shape}")
        if not np.isfinite(features).all():
            raise FloatingPointError(f"Non-finite values in {path}")
        keys = list(zip(subjects, studies))
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"Duplicate Study IDs in {path}")
        if common is None:
            common = {"Subject_id": subjects, "Study_id": studies,
                      "Image_Count": image_counts, "labels": labels}
        else:
            for key in ("Subject_id", "Study_id", "Image_Count", "labels"):
                if not np.array_equal(common[key], {
                    "Subject_id": subjects, "Study_id": studies,
                    "Image_Count": image_counts, "labels": labels}[key]):
                    raise RuntimeError(f"Study metadata differs across stage caches: {path}")
        result[stage] = features
    result.update(common)
    return result


def feature_variants():
    variants = {}
    for stage in STAGES:
        variants[stage] = {"kind": "single", "keys": [stage], "stage": DISPLAY_STAGE[stage]}
        variants[f"pca64_{stage}"] = {
            "kind": "pca64", "keys": [stage], "stage": f"{DISPLAY_STAGE[stage]} PCA-64"}
    variants["stage4_norm"] = {
        "kind": "single", "keys": ["stage4_norm"], "stage": "Stage 4 Norm(GAP)"}
    variants["joint_s3_s4"] = {
        "kind": "joint", "keys": ["stage3", "stage4"], "stage": "Stage 3 + Stage 4"}
    variants["joint_s2_s3"] = {
        "kind": "joint", "keys": ["stage2", "stage3"], "stage": "Stage 2 + Stage 3"}
    return variants


def transform_features(train_features, eval_features, variant):
    keys = variant["keys"]
    if variant["kind"] == "single":
        scaler = StandardScaler(copy=True)
        train_x = scaler.fit_transform(train_features[keys[0]])
        eval_x = scaler.transform(eval_features[keys[0]])
        return train_x, eval_x
    if variant["kind"] == "joint":
        train_parts, eval_parts = [], []
        for key in keys:
            scaler = StandardScaler(copy=True)
            train_parts.append(scaler.fit_transform(train_features[key]))
            eval_parts.append(scaler.transform(eval_features[key]))
        return np.concatenate(train_parts, axis=1), np.concatenate(eval_parts, axis=1)
    if variant["kind"] == "pca64":
        scaler = StandardScaler(copy=True)
        scaled_train = scaler.fit_transform(train_features[keys[0]])
        scaled_eval = scaler.transform(eval_features[keys[0]])
        pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED)
        return pca.fit_transform(scaled_train), pca.transform(scaled_eval)
    raise ValueError(f"Unknown feature transform kind: {variant['kind']}")


def valid_group_splits(features, labels, groups, disease):
    for n_splits in (3, 2):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=SEED)
        splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
        valid = all(np.unique(labels[val_index]).size == 2
                    and np.unique(labels[train_index]).size == 2
                    for train_index, val_index in splits)
        if valid:
            return splits, n_splits
    positive_groups = len(set(groups[labels == 1]))
    raise RuntimeError(
        f"Cannot construct subject-grouped folds with both labels for {disease}; "
        f"positive studies={int(labels.sum())}, positive Subjects={positive_groups}")


def fit_probe_variant(model_name, variant_name, variant, train_set, valid_set,
                      checkpoint_sha):
    PROBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name.lower()}_{variant_name}"
    prediction_path = PROBE_CACHE_DIR / f"{stem}.npz"
    cv_path = PROBE_CACHE_DIR / f"{stem}_cv.csv"
    expected_metadata = {
        "probe_version": PROBE_VERSION, "backbone": model_name,
        "variant": variant_name, "checkpoint_sha256": checkpoint_sha,
        "c_values": list(C_VALUES), "seed": SEED,
    }
    if prediction_path.is_file() and cv_path.is_file():
        with np.load(prediction_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata != expected_metadata:
                raise RuntimeError(f"Probe cache metadata mismatch: {prediction_path}")
            result = {key: archive[key].copy() for key in (
                "validation_probabilities", "oof_probabilities", "selected_C",
                "oof_threshold", "converged", "feature_dimension")}
        return result, pd.read_csv(cv_path)

    y_train_all = train_set["labels"].astype(np.uint8)
    y_valid_all = valid_set["labels"].astype(np.uint8)
    groups = train_set["Subject_id"].astype(str)
    n_train, n_labels = y_train_all.shape
    oof_matrix = np.full((n_train, n_labels), np.nan, dtype=np.float32)
    validation_probabilities = np.zeros((len(y_valid_all), n_labels), dtype=np.float32)
    selected_c = np.zeros(n_labels, dtype=np.float64)
    thresholds = np.zeros(n_labels, dtype=np.float64)
    converged = np.zeros(n_labels, dtype=np.uint8)
    cv_rows = []
    feature_dim = None
    with threadpool_limits(limits=2):
        for disease_index, disease in enumerate(LABELS):
            y = y_train_all[:, disease_index].astype(np.int64)
            splits, fold_count = valid_group_splits(
                train_set[variant["keys"][0]], y, groups, disease)
            oof_by_c = {float(value): np.full(n_train, np.nan, dtype=np.float32)
                        for value in C_VALUES}
            fold_auc_by_c = {float(value): [] for value in C_VALUES}
            convergence_by_c = {float(value): [] for value in C_VALUES}
            for fold_index, (fit_index, hold_index) in enumerate(splits, start=1):
                fit_features = {key: value[fit_index] for key, value in train_set.items()
                                if key in (*STAGES, "stage4_norm")}
                hold_features = {key: value[hold_index] for key, value in train_set.items()
                                 if key in (*STAGES, "stage4_norm")}
                fit_x, hold_x = transform_features(fit_features, hold_features, variant)
                if feature_dim is None:
                    feature_dim = int(fit_x.shape[1])
                if fit_x.shape[1] != feature_dim:
                    raise RuntimeError("Probe feature dimension changed between folds")
                y_fit, y_hold = y[fit_index], y[hold_index]
                classifier = None
                for c_value in C_VALUES:
                    if classifier is None:
                        classifier = LogisticRegression(
                            C=float(c_value), penalty="l2", solver="lbfgs",
                            max_iter=1000, tol=1e-4, random_state=SEED,
                            warm_start=True)
                    else:
                        classifier.set_params(C=float(c_value))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        classifier.fit(fit_x, y_fit)
                    prediction = classifier.predict_proba(hold_x)[:, 1].astype(np.float32)
                    oof_by_c[float(c_value)][hold_index] = prediction
                    fold_auc = float(roc_auc_score(y_hold, prediction))
                    fold_ap = float(average_precision_score(y_hold, prediction))
                    is_converged = not any(issubclass(w.category, ConvergenceWarning)
                                           for w in caught)
                    if int(classifier.n_iter_.max()) >= 1000:
                        is_converged = False
                    fold_auc_by_c[float(c_value)].append(fold_auc)
                    convergence_by_c[float(c_value)].append(is_converged)
                    cv_rows.append({
                        "Backbone": model_name, "Variant": variant_name,
                        "Disease": disease, "C": float(c_value),
                        "Fold": fold_index, "Fold_Count": fold_count,
                        "Train_Studies": int(len(fit_index)), "Validation_Studies": int(len(hold_index)),
                        "Train_Positive_Studies": int(y_fit.sum()),
                        "Train_Negative_Studies": int((y_fit == 0).sum()),
                        "Fold_Positive_Studies": int(y_hold.sum()),
                        "Fold_Negative_Studies": int((y_hold == 0).sum()),
                        "AUROC": fold_auc, "AUPRC": fold_ap,
                        "Converged": is_converged,
                        "Iterations": int(classifier.n_iter_.max()),
                    })

            mean_auc = {c: float(np.mean(values))
                        for c, values in fold_auc_by_c.items()}
            chosen_c = max(C_VALUES, key=lambda value: (mean_auc[float(value)], -float(value)))
            selected_c[disease_index] = float(chosen_c)
            chosen_oof = oof_by_c[float(chosen_c)]
            if not np.isfinite(chosen_oof).all():
                raise RuntimeError(f"OOF predictions are incomplete for {model_name}/{variant_name}/{disease}")
            oof_matrix[:, disease_index] = chosen_oof
            converged[disease_index] = int(all(convergence_by_c[float(chosen_c)]))
            final_train_x, final_valid_x = transform_features(
                {key: train_set[key] for key in (*STAGES, "stage4_norm")},
                {key: valid_set[key] for key in (*STAGES, "stage4_norm")}, variant)
            final_model = LogisticRegression(
                C=float(chosen_c), penalty="l2", solver="lbfgs",
                max_iter=1000, tol=1e-4, random_state=SEED)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                final_model.fit(final_train_x, y)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught) or int(final_model.n_iter_.max()) >= 1000:
                converged[disease_index] = 0
            validation_probabilities[:, disease_index] = final_model.predict_proba(final_valid_x)[:, 1]
            precision, recall, threshold_values = precision_recall_curve(y, chosen_oof)
            if len(threshold_values):
                f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
                    precision[:-1] + recall[:-1], 1e-15)
                best_f1 = np.flatnonzero(f1_values == f1_values.max())
                thresholds[disease_index] = float(threshold_values[best_f1[-1]])
            else:
                thresholds[disease_index] = 0.5
            print(
                f"Probe {model_name}/{variant_name}/{disease}: C={chosen_c:g}, "
                f"OOF AUROC={roc_auc_score(y, chosen_oof):.4f}, "
                f"selected-folds={fold_count}, convergence={bool(converged[disease_index])}",
                flush=True)

    result = {
        "validation_probabilities": validation_probabilities,
        "oof_probabilities": oof_matrix,
        "selected_C": selected_c,
        "oof_threshold": thresholds,
        "converged": converged,
        "feature_dimension": np.asarray([feature_dim], dtype=np.int32),
    }
    save_npz_atomic(prediction_path, **result,
                    metadata=np.asarray(json.dumps(expected_metadata, ensure_ascii=False)))
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)
    return result, pd.DataFrame(cv_rows)


def find_best_f1_threshold(y, oof_prediction):
    precision, recall, thresholds = precision_recall_curve(y, oof_prediction)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(f1_values == f1_values.max())
    return float(thresholds[best[-1]])


def threshold_metrics(y, probabilities, threshold):
    prediction = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction.astype(np.uint8), labels=[0, 1]).ravel()
    precision = float(precision_score(y, prediction, zero_division=0))
    recall = float(recall_score(y, prediction, zero_division=0))
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    f1 = float(f1_score(y, prediction, zero_division=0))
    balanced = float((recall + specificity) / 2) if np.isfinite(specificity) else float("nan")
    return {
        "Precision": precision, "Recall_Sensitivity": recall,
        "Specificity": specificity, "F1": f1,
        "Balanced_Accuracy": balanced,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Accuracy": float(accuracy_score(y, prediction)),
    }


def point_metrics(labels, probabilities):
    auroc, auprc = [], []
    for index in range(labels.shape[1]):
        y = labels[:, index]
        p = probabilities[:, index]
        auroc.append(float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"))
        auprc.append(float(average_precision_score(y, p)) if np.any(y == 1) else float("nan"))
    return np.asarray(auroc), np.asarray(auprc)


def subject_bootstrap_draws(subject_ids):
    subjects = np.asarray(subject_ids).astype(str)
    unique = np.asarray(sorted(set(subjects)), dtype=object)
    rows_by_subject = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(SEED)
    draws = []
    digest = hashlib.sha256()
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows_by_subject[subject] for subject in sampled])
        draws.append(indices)
        digest.update(indices.astype("<i4", copy=False).tobytes())
        digest.update(b"\xff")
    return draws, digest.hexdigest()


def bootstrap_all(labels, probabilities_by_model, draws):
    point = {}
    boot = {}
    for model_key, probabilities in probabilities_by_model.items():
        auc_point, ap_point = point_metrics(labels, probabilities)
        auc_samples = np.full((BOOTSTRAP_REPLICATES, labels.shape[1]), np.nan, dtype=np.float32)
        ap_samples = np.full_like(auc_samples, np.nan)
        macro_auc = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float32)
        macro_ap = np.full_like(macro_auc, np.nan)
        for replicate, indices in enumerate(draws):
            y_sample = labels[indices]
            p_sample = probabilities[indices]
            aucs, aps = [], []
            for class_index in range(labels.shape[1]):
                y = y_sample[:, class_index]
                if np.unique(y).size == 2:
                    auc_value = roc_auc_score(y, p_sample[:, class_index])
                    auc_samples[replicate, class_index] = auc_value
                    aucs.append(auc_value)
                if np.any(y == 1):
                    ap_value = average_precision_score(y, p_sample[:, class_index])
                    ap_samples[replicate, class_index] = ap_value
                    aps.append(ap_value)
            if aucs:
                macro_auc[replicate] = float(np.mean(aucs))
            if aps:
                macro_ap[replicate] = float(np.mean(aps))
            if (replicate + 1) % 250 == 0:
                print(f"Bootstrap {model_key}: {replicate + 1}/{BOOTSTRAP_REPLICATES}", flush=True)
        point[model_key] = {"auroc": auc_point, "auprc": ap_point}
        boot[model_key] = {
            "auroc": auc_samples, "auprc": ap_samples,
            "macro_auroc": macro_auc, "macro_auprc": macro_ap,
        }
    return point, boot


def ci(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None], 0
    return np.quantile(values, [0.025, 0.975]).tolist(), int(len(values))


def model_metric_record(model_key, labels, point, boot):
    auc_point = point[model_key]["auroc"]
    ap_point = point[model_key]["auprc"]
    auc_macro_ci, auc_macro_n = ci(boot[model_key]["macro_auroc"])
    ap_macro_ci, ap_macro_n = ci(boot[model_key]["macro_auprc"])
    record = {
        "Macro_AUROC": float(np.nanmean(auc_point)),
        "Macro_AUROC_CI95": auc_macro_ci,
        "Macro_AUROC_Valid_Replicates": auc_macro_n,
        "Macro_AUPRC": float(np.nanmean(ap_point)),
        "Macro_AUPRC_CI95": ap_macro_ci,
        "Macro_AUPRC_Valid_Replicates": ap_macro_n,
        "per_class": {},
    }
    for index, disease in enumerate(LABELS):
        auc_ci, auc_n = ci(boot[model_key]["auroc"][:, index])
        ap_ci, ap_n = ci(boot[model_key]["auprc"][:, index])
        record["per_class"][disease] = {
            "AUROC": json_safe_float(auc_point[index]), "AUROC_CI95": auc_ci,
            "AUROC_Valid_Replicates": auc_n,
            "AUPRC": json_safe_float(ap_point[index]), "AUPRC_CI95": ap_ci,
            "AUPRC_Valid_Replicates": ap_n,
            "Positive_Studies": int(labels[:, index].sum()),
            "Negative_Studies": int((labels[:, index] == 0).sum()),
        }
    return record


def compare_bootstrap(left_key, right_key, class_index, metric, point, boot):
    metric_key = "auroc" if metric == "AUROC" else "auprc"
    if class_index is None:
        left_point = float(np.nanmean(point[left_key][metric_key]))
        right_point = float(np.nanmean(point[right_key][metric_key]))
        samples = (boot[right_key]["macro_auroc"] - boot[left_key]["macro_auroc"]
                   if metric == "AUROC" else
                   boot[right_key]["macro_auprc"] - boot[left_key]["macro_auprc"])
    else:
        left_point = float(point[left_key][metric_key][class_index])
        right_point = float(point[right_key][metric_key][class_index])
        samples = (boot[right_key][metric_key][:, class_index]
                   - boot[left_key][metric_key][:, class_index])
    ci_values, valid = ci(samples)
    return {
        "left": left_key, "right": right_key, "metric": metric,
        "delta": right_point - left_point,
        "ci95": ci_values, "valid_replicates": valid,
    }


def metric_rows(model_specs, predictions, train_features, valid_features,
                train_labels, valid_labels, probe_results, bootstrap_records):
    rows = []
    for key, spec in model_specs.items():
        model_name, variant_name = key.split("|", 1)
        result = probe_results[key]
        probabilities = predictions[key]
        model_ci = bootstrap_records[key]
        selected = result["selected_C"]
        thresholds = result["oof_threshold"]
        oof = result["oof_probabilities"]
        train_subjects = train_features[model_name]["train"]["Subject_id"]
        feature_dimension = int(result["feature_dimension"][0])
        for class_index, disease in enumerate(LABELS):
            y = valid_labels[:, class_index].astype(np.uint8)
            p = probabilities[:, class_index].astype(np.float64)
            oof_y = train_labels[:, class_index].astype(np.uint8)
            oof_p = oof[:, class_index].astype(np.float64)
            auc_value = float(roc_auc_score(y, p))
            ap_value = float(average_precision_score(y, p))
            auc_ci = model_ci["per_class"][disease]["AUROC_CI95"]
            ap_ci = model_ci["per_class"][disease]["AUPRC_CI95"]
            base = {
                "Backbone": model_name, "Feature_Variant": variant_name,
                "Stage": spec["stage"], "Probe": spec["kind"], "Disease": disease,
                "Selected_C": float(selected[class_index]),
                "Train_OOF_AUROC": float(roc_auc_score(oof_y, oof_p)),
                "Train_OOF_AUPRC": float(average_precision_score(oof_y, oof_p)),
                "Validation_AUROC": auc_value,
                "Validation_AUROC_CI95_Low": auc_ci[0],
                "Validation_AUROC_CI95_High": auc_ci[1],
                "Validation_AUPRC": ap_value,
                "Validation_AUPRC_CI95_Low": ap_ci[0],
                "Validation_AUPRC_CI95_High": ap_ci[1],
                "Positive_Studies": int(y.sum()), "Negative_Studies": int((y == 0).sum()),
                "Subject_Count": int(len(set(valid_features["R5"]["valid"]["Subject_id"]))),
                "Train_Subject_Count": int(len(set(train_subjects))),
                "Feature_Dimension": feature_dimension,
                "Selected_C_Converged": bool(result["converged"][class_index]),
                "Probability_Mean_Positive": float(p[y == 1].mean()),
                "Probability_Mean_Negative": float(p[y == 0].mean()),
                "Probability_Std": float(p.std()),
                "Brier_Score": float(brier_score_loss(y, p)),
                "Log_Loss": float(log_loss(y, p, labels=[0, 1])),
            }
            protocols = (
                ("fixed_0.5", 0.5),
                ("train_oof_best_f1", float(thresholds[class_index])),
            )
            for protocol, threshold in protocols:
                row = dict(base)
                row["Threshold_Protocol"] = protocol
                row["Threshold"] = threshold
                row.update(threshold_metrics(y, p, threshold))
                rows.append(row)
    return pd.DataFrame(rows)


def prediction_frame(model_specs, predictions, valid_set):
    base = pd.DataFrame({
        "Subject_id": valid_set["Subject_id"],
        "Study_id": valid_set["Study_id"],
        "Image_Count": valid_set["Image_Count"],
    })
    for index, disease in enumerate(LABELS):
        base[TARGET_COLS[index]] = valid_set["labels"][:, index]
    frames = []
    for key in model_specs:
        model_name, variant_name = key.split("|", 1)
        frame = base.copy()
        frame.insert(0, "Backbone", model_name)
        frame.insert(1, "Feature_Variant", variant_name)
        for index, disease in enumerate(LABELS):
            frame[f"prob_{disease.replace(' ', '_')}"] = predictions[key][:, index]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def feature_integrity_report(path, dataset_metadata, features, model_info):
    lines = [
        "# Feature Cache Integrity Report", "",
        f"- Train: {dataset_metadata['train_studies']:,} Studies / {dataset_metadata['train_images']:,} images / {dataset_metadata['train_subjects']:,} Subjects.",
        f"- Validation: {dataset_metadata['validation_studies']:,} Studies / {dataset_metadata['validation_images']:,} images / {dataset_metadata['validation_subjects']:,} Subjects.",
        "- Subject overlap between Train and Validation: 0.",
        "- Each Study key `(Subject_id, Study_id)` appears once per split and is aligned across R5/R6 and all Stage caches.",
        "- Labels: ten official classes, closed-world encoding, order matches the official class ID order.",
        "- Feature extraction uses each checkpoint in `eval()` and `torch.inference_mode()`; all backbone parameters have `requires_grad=False`.",
        "- No Test CSV, Test image path, or Test label was opened.",
        "", "## Source Best Checkpoints", "",
        "| Backbone | Best Epoch | Macro-AUROC in training status | Checkpoint SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for model_name in ("R5", "R6"):
        info = model_info[model_name]
        lines.append(
            f"| {model_name} | {info['best_epoch']} | {info['best_macro_auroc']:.6f} | `{info['checkpoint_sha256']}` |")
    lines += ["", "## Cached Study feature dimensions", "",
              "| Split | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 4 Norm(GAP) |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("train", "valid"):
        sizes = [features["R5"][split][stage].shape for stage in (*STAGES, "stage4_norm")]
        lines.append(f"| {split} | " + " | ".join(f"{shape[0]}×{shape[1]}" for shape in sizes) + " |")
    lines += ["", "Stage 1–4 features are GAP vectors computed at each actual Stage output before the next Downsample. Stage 4 Norm(GAP) is the same final normalized pooled vector used by the Global Head. Each Study vector is the arithmetic mean of its per-image vectors. Feature caches include Subject_id, Study_id, Image_Count, labels, source checkpoint SHA-256, stage and extraction metadata.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_ci_map(metric_frame, model_name, metric, threshold_protocol="fixed_0.5"):
    frame = metric_frame[(metric_frame["Backbone"] == model_name)
                         & (metric_frame["Threshold_Protocol"] == threshold_protocol)]
    suffix = "AUROC" if metric == "AUROC" else "AUPRC"
    return {
        row.Disease: {
            "point": float(getattr(row, f"Validation_{suffix}")),
            "low": float(getattr(row, f"Validation_{suffix}_CI95_Low")),
            "high": float(getattr(row, f"Validation_{suffix}_CI95_High")),
        }
        for row in frame.itertuples(index=False)
    }


def make_comparison_outputs(metric_frame, specs, point, boot, valid_labels,
                            draw_hash, valid_set):
    comparisons = {}
    transition_rows = []
    r6r5_rows = []
    joint_rows = []
    pca_rows = []
    best_rows = []
    def add_pair(name, left, right, disease_index):
        record_auc = compare_bootstrap(left, right, disease_index, "AUROC", point, boot)
        record_ap = compare_bootstrap(left, right, disease_index, "AUPRC", point, boot)
        comparisons[name] = {"AUROC": record_auc, "AUPRC": record_ap}
        return record_auc, record_ap

    for model_name in ("R5", "R6"):
        for contrast, left_stage, right_stage in (
            ("Stage2-Stage1", "stage1", "stage2"),
            ("Stage3-Stage2", "stage2", "stage3"),
            ("Stage4-Stage3", "stage3", "stage4"),
            ("Stage4-Stage1", "stage1", "stage4"),
        ):
            left, right = f"{model_name}|{left_stage}", f"{model_name}|{right_stage}"
            for class_index, disease in enumerate(LABELS):
                name = f"{model_name}|{contrast}|{disease}"
                auc_delta, ap_delta = add_pair(name, left, right, class_index)
                transition_rows.append({
                    "Backbone": model_name, "Contrast": contrast, "Disease": disease,
                    "Delta_AUROC": auc_delta["delta"],
                    "AUROC_CI95_Low": auc_delta["ci95"][0],
                    "AUROC_CI95_High": auc_delta["ci95"][1],
                    "AUROC_Valid_Replicates": auc_delta["valid_replicates"],
                    "Delta_AUPRC": ap_delta["delta"],
                    "AUPRC_CI95_Low": ap_delta["ci95"][0],
                    "AUPRC_CI95_High": ap_delta["ci95"][1],
                    "AUPRC_Valid_Replicates": ap_delta["valid_replicates"],
                })
            auc_macro, ap_macro = add_pair(f"{model_name}|{contrast}|Macro", left, right, None)
            transition_rows.append({
                "Backbone": model_name, "Contrast": contrast, "Disease": "Macro",
                "Delta_AUROC": auc_macro["delta"],
                "AUROC_CI95_Low": auc_macro["ci95"][0], "AUROC_CI95_High": auc_macro["ci95"][1],
                "AUROC_Valid_Replicates": auc_macro["valid_replicates"],
                "Delta_AUPRC": ap_macro["delta"],
                "AUPRC_CI95_Low": ap_macro["ci95"][0], "AUPRC_CI95_High": ap_macro["ci95"][1],
                "AUPRC_Valid_Replicates": ap_macro["valid_replicates"],
            })

        for feature_name, comparator_pairs in (
            ("joint_s3_s4", (("Stage3+4−Stage3", "stage3"),
                             ("Stage3+4−Stage4", "stage4"))),
            ("joint_s2_s3", (("Stage2+3−Stage2", "stage2"),
                             ("Stage2+3−Stage3", "stage3"))),
        ):
            joint = f"{model_name}|{feature_name}"
            for contrast, base_stage in comparator_pairs:
                baseline = f"{model_name}|{base_stage}"
                for class_index, disease in enumerate(LABELS):
                    name = f"{model_name}|{contrast}|{disease}"
                    auc_delta, ap_delta = add_pair(name, baseline, joint, class_index)
                    joint_rows.append({
                        "Backbone": model_name, "Contrast": contrast, "Disease": disease,
                        "Delta_AUROC": auc_delta["delta"],
                        "AUROC_CI95_Low": auc_delta["ci95"][0],
                        "AUROC_CI95_High": auc_delta["ci95"][1],
                        "Delta_AUPRC": ap_delta["delta"],
                        "AUPRC_CI95_Low": ap_delta["ci95"][0],
                        "AUPRC_CI95_High": ap_delta["ci95"][1],
                    })
                auc_macro, ap_macro = add_pair(f"{model_name}|{contrast}|Macro", baseline, joint, None)
                joint_rows.append({
                    "Backbone": model_name, "Contrast": contrast, "Disease": "Macro",
                    "Delta_AUROC": auc_macro["delta"], "AUROC_CI95_Low": auc_macro["ci95"][0],
                    "AUROC_CI95_High": auc_macro["ci95"][1],
                    "Delta_AUPRC": ap_macro["delta"], "AUPRC_CI95_Low": ap_macro["ci95"][0],
                    "AUPRC_CI95_High": ap_macro["ci95"][1],
                })

        for stage in STAGES:
            pca_key, raw_key = f"{model_name}|pca64_{stage}", f"{model_name}|{stage}"
            for class_index, disease in enumerate(LABELS):
                name = f"{model_name}|PCA64-{DISPLAY_STAGE[stage]}|{disease}"
                auc_delta, ap_delta = add_pair(name, raw_key, pca_key, class_index)
                pca_rows.append({
                    "Backbone": model_name, "Stage": DISPLAY_STAGE[stage], "Disease": disease,
                    "Raw_AUROC": point[raw_key]["auroc"][class_index],
                    "PCA64_AUROC": point[pca_key]["auroc"][class_index],
                    "Delta_AUROC": auc_delta["delta"],
                    "AUROC_CI95_Low": auc_delta["ci95"][0], "AUROC_CI95_High": auc_delta["ci95"][1],
                    "Raw_AUPRC": point[raw_key]["auprc"][class_index],
                    "PCA64_AUPRC": point[pca_key]["auprc"][class_index],
                    "Delta_AUPRC": ap_delta["delta"],
                    "AUPRC_CI95_Low": ap_delta["ci95"][0], "AUPRC_CI95_High": ap_delta["ci95"][1],
                })

    for class_index, disease in enumerate(LABELS):
        for compared_model in ("R5", "R6"):
            values = [point[f"{compared_model}|{stage}"]["auroc"][class_index]
                      for stage in STAGES]
            order = np.argsort(values)[::-1]
            first, second = int(order[0]), int(order[1])
            first_stage, second_stage = STAGES[first], STAGES[second]
            delta, _ = add_pair(
                f"{compared_model}|best-vs-second|{disease}",
                f"{compared_model}|{second_stage}", f"{compared_model}|{first_stage}", class_index)
            best_rows.append({
                "Backbone": compared_model, "Disease": disease,
                "Best_Stage": DISPLAY_STAGE[first_stage],
                "Second_Stage": DISPLAY_STAGE[second_stage],
                "Best_AUROC": float(values[first]),
                "Second_AUROC": float(values[second]),
                "Best_minus_Second_AUROC": delta["delta"],
                "CI95_Low": delta["ci95"][0], "CI95_High": delta["ci95"][1],
                "Stage_Advantage": (
                    "Clear in this paired interval" if delta["ci95"][0] is not None
                    and delta["ci95"][0] > 0 else "No clear stage advantage"),
            })

    for stage in STAGES:
        left, right = f"R5|{stage}", f"R6|{stage}"
        for class_index, disease in enumerate(LABELS):
            name = f"R6-R5|{stage}|{disease}"
            auc_delta, ap_delta = add_pair(name, left, right, class_index)
            r6r5_rows.append({
                "Stage": DISPLAY_STAGE[stage], "Disease": disease,
                "Delta_AUROC": auc_delta["delta"], "AUROC_CI95_Low": auc_delta["ci95"][0],
                "AUROC_CI95_High": auc_delta["ci95"][1],
                "Delta_AUPRC": ap_delta["delta"], "AUPRC_CI95_Low": ap_delta["ci95"][0],
                "AUPRC_CI95_High": ap_delta["ci95"][1],
            })
        auc_macro, ap_macro = add_pair(f"R6-R5|{stage}|Macro", left, right, None)
        r6r5_rows.append({
            "Stage": DISPLAY_STAGE[stage], "Disease": "Macro",
            "Delta_AUROC": auc_macro["delta"], "AUROC_CI95_Low": auc_macro["ci95"][0],
            "AUROC_CI95_High": auc_macro["ci95"][1],
            "Delta_AUPRC": ap_macro["delta"], "AUPRC_CI95_Low": ap_macro["ci95"][0],
            "AUPRC_CI95_High": ap_macro["ci95"][1],
        })

    json_payload = {
        "method": "Subject-cluster paired bootstrap; percentile 95% CI",
        "sampling_unit": "Subject_id; all Study rows belonging to each sampled Subject are included together",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "sampling_indices_sha256": draw_hash,
        "aligned_validation_studies": int(len(valid_labels)),
        "validation_subjects": int(len(set(valid_set["Subject_id"]))),
        "one_class_rule": (
            "AUC is skipped when a bootstrap sample has one class; AP is skipped when it has no positives. "
            "Macro metrics average the available classes for that replicate. Every comparison uses identical draws."),
        "metrics_by_model": {},
        "comparisons": comparisons,
        "multiple_comparison_note": (
            "The many Stage × Disease × Probe comparisons increase false-positive risk. "
            "Validation has also been used for historical Best-checkpoint selection; intervals are exploratory."),
    }
    for key in point:
        json_payload["metrics_by_model"][key] = model_metric_record(
            key, valid_labels, point, boot)
    (HERE / "stage_full_metrics_by_disease.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(transition_rows).to_csv(HERE / "stage_transition_deltas.csv", index=False)
    pd.DataFrame(r6r5_rows).to_csv(HERE / "r6_minus_r5_stage_deltas.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(HERE / "stage_joint_probe_comparison.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(HERE / "pca64_probe_comparison.csv", index=False)
    pd.DataFrame(best_rows).to_csv(HERE / "best_stage_paired_comparisons.csv", index=False)
    write_json(HERE / "stage_paired_bootstrap.json", json_payload)
    return json_payload, pd.DataFrame(transition_rows), pd.DataFrame(r6r5_rows), \
        pd.DataFrame(joint_rows), pd.DataFrame(pca_rows), pd.DataFrame(best_rows)


def matrix_from_metrics(metric_frame, backbone, variant_prefix, metric, column="Validation_AUROC"):
    frame = metric_frame[(metric_frame["Backbone"] == backbone)
                         & (metric_frame["Threshold_Protocol"] == "fixed_0.5")
                         & metric_frame["Feature_Variant"].isin(STAGES)]
    result = frame.pivot(index="Disease", columns="Feature_Variant", values=column)
    return result.reindex(index=LABELS, columns=STAGES)


def write_stage_matrices(metric_frame, transition_frame, r6r5_frame):
    for backbone in ("R5", "R6"):
        for metric, column in (
            ("auroc", "Validation_AUROC"), ("auprc", "Validation_AUPRC"),
            ("f1", "F1"), ("recall", "Recall_Sensitivity"),
            ("specificity", "Specificity"), ("balanced_accuracy", "Balanced_Accuracy"),
        ):
            matrix = matrix_from_metrics(metric_frame, backbone, None, metric, column)
            if metric == "auroc":
                best = matrix.idxmax(axis=1)
                matrix["Best Stage"] = best
            matrix.to_csv(HERE / f"stage_{metric}_matrix_{backbone.lower()}.csv")
    auroc_r6r5 = r6r5_frame[r6r5_frame["Disease"] != "Macro"].pivot(
        index="Disease", columns="Stage", values="Delta_AUROC").reindex(
            index=LABELS, columns=[DISPLAY_STAGE[x] for x in STAGES])
    auprc_r6r5 = r6r5_frame[r6r5_frame["Disease"] != "Macro"].pivot(
        index="Disease", columns="Stage", values="Delta_AUPRC").reindex(
            index=LABELS, columns=[DISPLAY_STAGE[x] for x in STAGES])
    auroc_r6r5.to_csv(HERE / "r6_minus_r5_auroc_matrix.csv")
    auprc_r6r5.to_csv(HERE / "r6_minus_r5_auprc_matrix.csv")


def heatmap(matrix, title, path, cmap="viridis", center=None, cbar_label=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = matrix.to_numpy(dtype=float)
    if center is not None:
        limit = max(float(np.nanmax(np.abs(data))), 1e-6)
        image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), [name.replace(" ", "\n", 2) for name in matrix.index])
    ax.set_title(title)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Disease")
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isfinite(data[row, col]):
                ax.text(col, row, f"{data[row, col]:+.3f}" if center is not None
                        else f"{data[row, col]:.3f}", ha="center", va="center",
                        color="white" if (center is None and data[row, col] < np.nanmean(data))
                        or (center is not None and abs(data[row, col]) < limit * 0.45)
                        else "black", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    if cbar_label:
        colorbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_charts(metric_frame, transition_frame, r6r5_frame, joint_frame,
                  train_log_r5, train_log_r6):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for backbone in ("R5", "R6"):
        for metric, column in (("AUROC", "Validation_AUROC"),
                               ("AUPRC", "Validation_AUPRC")):
            matrix = matrix_from_metrics(metric_frame, backbone, None, metric.lower(), column)
            heatmap(matrix, f"{backbone} {metric}: Disease × Stage",
                    CHART_DIR / f"{backbone.lower()}_{metric.lower()}_heatmap.png",
                    cbar_label=metric)
    for metric, column in (("AUROC", "Delta_AUROC"), ("AUPRC", "Delta_AUPRC")):
        frame = r6r5_frame[r6r5_frame["Disease"] != "Macro"]
        matrix = frame.pivot(index="Disease", columns="Stage", values=column).reindex(
            index=LABELS, columns=[DISPLAY_STAGE[x] for x in STAGES])
        heatmap(matrix, f"R6−R5 {metric} Change by Stage",
                CHART_DIR / f"r6_minus_r5_{metric.lower()}_heatmap.png",
                cmap="coolwarm", center=0, cbar_label=f"Δ{metric}")

    stage_labels = [DISPLAY_STAGE[stage] for stage in STAGES]
    for metric, column in (("AUROC", "Validation_AUROC"), ("AUPRC", "Validation_AUPRC")):
        fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
        for disease_index, disease in enumerate(LABELS):
            ax = axes.flat[disease_index]
            for backbone, style in (("R5", "o-"), ("R6", "s--")):
                frame = metric_frame[
                    (metric_frame["Backbone"] == backbone)
                    & (metric_frame["Disease"] == disease)
                    & (metric_frame["Threshold_Protocol"] == "fixed_0.5")
                    & metric_frame["Feature_Variant"].isin(STAGES)
                ].set_index("Feature_Variant").reindex(STAGES)
                ax.plot(stage_labels, frame[column], style, label=backbone, linewidth=1.3)
            ax.set_title(disease, fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Per-disease {metric} across Stages")
        fig.tight_layout()
        fig.savefig(CHART_DIR / f"per_disease_stage_{metric.lower()}_curves.png", dpi=180)
        plt.close(fig)

    transition = transition_frame[(transition_frame["Disease"] != "Macro")
                                  & transition_frame["Contrast"].isin(
                                      ["Stage2-Stage1", "Stage3-Stage2", "Stage4-Stage3"])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    contrasts = ["Stage2-Stage1", "Stage3-Stage2", "Stage4-Stage3"]
    for ax, contrast in zip(axes, contrasts):
        frame = transition[transition["Contrast"] == contrast]
        x = np.arange(len(LABELS))
        for offset, backbone, color in ((-0.18, "R5", "#3478bf"), (0.18, "R6", "#d97924")):
            subset = frame[frame["Backbone"] == backbone].set_index("Disease").reindex(LABELS)
            ax.bar(x + offset, subset["Delta_AUROC"], width=0.34, label=backbone, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(contrast)
        ax.set_xticks(x, SHORT, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Paired point estimates for adjacent Stage AUROC changes")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "stage_transition_auroc_deltas.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    for disease_index, disease in enumerate(LABELS):
        ax = axes.flat[disease_index]
        for backbone, style in (("R5", "o-"), ("R6", "s--")):
            stage3 = metric_frame[(metric_frame["Backbone"] == backbone)
                                  & (metric_frame["Disease"] == disease)
                                  & (metric_frame["Feature_Variant"] == "stage3")
                                  & (metric_frame["Threshold_Protocol"] == "fixed_0.5")]["Validation_AUROC"].iloc[0]
            stage4 = metric_frame[(metric_frame["Backbone"] == backbone)
                                  & (metric_frame["Disease"] == disease)
                                  & (metric_frame["Feature_Variant"] == "stage4")
                                  & (metric_frame["Threshold_Protocol"] == "fixed_0.5")]["Validation_AUROC"].iloc[0]
            joint = metric_frame[(metric_frame["Backbone"] == backbone)
                                 & (metric_frame["Disease"] == disease)
                                 & (metric_frame["Feature_Variant"] == "joint_s3_s4")
                                 & (metric_frame["Threshold_Protocol"] == "fixed_0.5")]["Validation_AUROC"].iloc[0]
            ax.plot(["Stage 3", "Stage 4", "Stage 3+4"], [stage3, stage4, joint],
                    style, label=backbone, linewidth=1.2)
        ax.set_title(disease, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Standalone Stage 3/4 and Joint Stage 3+4 AUROC")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "stage3_stage4_joint_auroc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, log in (("R5", train_log_r5), ("R6", train_log_r6)):
        log = log.sort_values("epoch")
        ax.plot(log["epoch"], log["train_bce"], marker="o", label=f"{model_name} Train")
        ax.plot(log["epoch"], log["valid_loss"], marker="s", linestyle="--", label=f"{model_name} Validation")
    ax.set(xlabel="Epoch", ylabel="Study BCE", title="R5/R6 Training and Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "r5_r6_train_valid_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, log in (("R5", train_log_r5), ("R6", train_log_r6)):
        log = log.sort_values("epoch")
        ax.plot(log["epoch"], log["macro_auroc"], marker="o", label=model_name)
    ax.set(xlabel="Epoch", ylabel="Macro-AUROC", title="R5/R6 Validation Macro-AUROC")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "r5_r6_macro_auroc.png", dpi=180)
    plt.close(fig)


def fmt_ci(point, low, high, digits=4):
    if low is None or high is None or not np.isfinite(point):
        return "NA"
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def metric_lookup(frame, backbone, variant, disease):
    row = frame[(frame["Backbone"] == backbone)
                & (frame["Feature_Variant"] == variant)
                & (frame["Disease"] == disease)
                & (frame["Threshold_Protocol"] == "fixed_0.5")]
    if len(row) != 1:
        raise RuntimeError(f"Expected one metric row for {backbone}/{variant}/{disease}, got {len(row)}")
    return row.iloc[0]


def comparison_lookup(frame, backbone, contrast, disease):
    row = frame[(frame["Backbone"] == backbone)
                & (frame["Contrast"] == contrast)
                & (frame["Disease"] == disease)]
    if len(row) != 1:
        raise RuntimeError(f"Expected one comparison for {backbone}/{contrast}/{disease}, got {len(row)}")
    return row.iloc[0]


def create_per_disease_reports(metric_frame, transition_frame, r6r5_frame,
                               joint_frame, pca_frame, best_frame,
                               dataset_meta):
    DISEASE_DIR.mkdir(parents=True, exist_ok=True)
    for disease_index, disease in enumerate(LABELS):
        sample = metric_lookup(metric_frame, "R5", "stage1", disease)
        rows = [
            f"# {disease}", "",
            f"- Validation positive / negative Studies: {int(sample.Positive_Studies)} / {int(sample.Negative_Studies)}; Subjects: {int(sample.Subject_Count)}.",
            "", "## Single-stage probe", "",
            "| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model_name in ("R5", "R6"):
            for stage in STAGES:
                row = metric_lookup(metric_frame, model_name, stage, disease)
                rows.append(
                    f"| {model_name} | {DISPLAY_STAGE[stage]} | "
                    f"{fmt_ci(row.Validation_AUROC, row.Validation_AUROC_CI95_Low, row.Validation_AUROC_CI95_High)} | "
                    f"{fmt_ci(row.Validation_AUPRC, row.Validation_AUPRC_CI95_Low, row.Validation_AUPRC_CI95_High)} | "
                    f"{row.Train_OOF_AUROC:.4f} | {row.Train_OOF_AUPRC:.4f} | {row.Selected_C:g} | "
                    f"{row.Feature_Dimension} | {row.F1:.4f} | {row.Balanced_Accuracy:.4f} |")
        rows += ["", "## Stage transitions and R6−R5", "", "| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |", "|---|---:|---:|"]
        for model_name in ("R5", "R6"):
            for contrast in ("Stage2-Stage1", "Stage3-Stage2", "Stage4-Stage3", "Stage4-Stage1"):
                row = comparison_lookup(transition_frame, model_name, contrast, disease)
                rows.append(
                    f"| {model_name} {contrast} | "
                    f"{fmt_ci(row.Delta_AUROC, row.AUROC_CI95_Low, row.AUROC_CI95_High, 4)} | "
                    f"{fmt_ci(row.Delta_AUPRC, row.AUPRC_CI95_Low, row.AUPRC_CI95_High, 4)} |")
        for stage in STAGES:
            row = r6r5_frame[(r6r5_frame["Stage"] == DISPLAY_STAGE[stage])
                             & (r6r5_frame["Disease"] == disease)].iloc[0]
            rows.append(
                f"| R6−R5 {DISPLAY_STAGE[stage]} | "
                f"{fmt_ci(row.Delta_AUROC, row.AUROC_CI95_Low, row.AUROC_CI95_High, 4)} | "
                f"{fmt_ci(row.Delta_AUPRC, row.AUPRC_CI95_Low, row.AUPRC_CI95_High, 4)} |")
        rows += ["", "## Best Stage, joint probes, and PCA-64", "", "| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |", "|---|---|---:|---|"]
        for model_name in ("R5", "R6"):
            row = best_frame[(best_frame["Backbone"] == model_name)
                             & (best_frame["Disease"] == disease)].iloc[0]
            rows.append(
                f"| {model_name} | {row.Best_Stage} | "
                f"{fmt_ci(row.Best_minus_Second_AUROC, row.CI95_Low, row.CI95_High, 4)} | {row.Stage_Advantage} |")
        for model_name in ("R5", "R6"):
            for contrast in ("Stage3+4−Stage3", "Stage3+4−Stage4", "Stage2+3−Stage2", "Stage2+3−Stage3"):
                row = comparison_lookup(joint_frame, model_name, contrast, disease)
                rows.append(
                    f"| {model_name} {contrast} | "
                    f"{fmt_ci(row.Delta_AUROC, row.AUROC_CI95_Low, row.AUROC_CI95_High, 4)} | "
                    f"{fmt_ci(row.Delta_AUPRC, row.AUPRC_CI95_Low, row.AUPRC_CI95_High, 4)} |")
        rows += ["", "## Disease-specific evidence and next direction", ""]
        for model_name in ("R5", "R6"):
            best = best_frame[(best_frame["Backbone"] == model_name)
                              & (best_frame["Disease"] == disease)].iloc[0]
            stage34 = comparison_lookup(joint_frame, model_name, "Stage3+4−Stage4", disease)
            stage_transition = comparison_lookup(transition_frame, model_name,
                                                 "Stage4-Stage3", disease)
            if stage34.AUROC_CI95_Low is not None and stage34.AUROC_CI95_Low > 0:
                direction = "Stage3+Stage4 provides positive paired evidence beyond Stage4 under this probe; prioritize a controlled Stage3→4 feature-transfer study."
            elif stage34.Delta_AUROC > 0 and stage34.AUROC_CI95_Low is not None and stage34.AUROC_CI95_Low <= 0:
                direction = "Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed."
            elif best.Best_Stage == "Stage 4" and stage_transition.AUROC_CI95_Low is not None and stage_transition.AUROC_CI95_High < 0:
                direction = "Stage4 is the top single-stage point estimate and Stage4−Stage3 is negative with an interval below zero; verify aggregation and head utilization before modifying early-stage structure."
            else:
                direction = "Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up."
            rows.append(
                f"- {model_name}: point-best {best.Best_Stage}; Stage3+4−Stage4 AUROC "
                f"{stage34.Delta_AUROC:+.4f} [{stage34.AUROC_CI95_Low:+.4f}, "
                f"{stage34.AUROC_CI95_High:+.4f}]. {direction}")
        rows.append("")
        rows += ["", "PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.", "", "| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |", "|---|---|---:|---:|"]
        for model_name in ("R5", "R6"):
            for stage in STAGES:
                row = pca_frame[(pca_frame["Backbone"] == model_name)
                                & (pca_frame["Stage"] == DISPLAY_STAGE[stage])
                                & (pca_frame["Disease"] == disease)].iloc[0]
                rows.append(
                    f"| {model_name} | {DISPLAY_STAGE[stage]} | "
                    f"{fmt_ci(row.Delta_AUROC, row.AUROC_CI95_Low, row.AUROC_CI95_High, 4)} | "
                    f"{fmt_ci(row.Delta_AUPRC, row.AUPRC_CI95_Low, row.AUPRC_CI95_High, 4)} |")
        rows += ["", "## Interpretation limits", "", "These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.", ""]
        filename = disease.replace(" ", "_").replace("/", "_") + ".md"
        (DISEASE_DIR / filename).write_text("\n".join(rows), encoding="utf-8")


def write_final_report(path, model_info, metrics, transitions, deltas, joint,
                       pca, best, bootstrap, r5_log, r6_log):
    r5_status = model_info["R5"]["status"]
    r6_status = model_info["R6"]["status"]
    def train_summary(name, status, log, info):
        best_epoch = int(status["best_epoch"])
        best_row = log.loc[log["epoch"].astype(int) == best_epoch].iloc[0]
        through_best = int(log.loc[log["epoch"].astype(int) <= best_epoch,
                                   "optimizer_updates"].astype(int).sum())
        verification = info.get("model_verification", {})
        return {
            "name": name, "best_epoch": best_epoch,
            "best_auc": float(status["best_macro_auroc"]),
            "train_bce": float(best_row["train_bce"]),
            "valid_bce": float(best_row["valid_loss"]),
            "updates": through_best,
            "images_per_s": float(log.loc[log["epoch"].astype(int) <= best_epoch,
                                          "images_per_s"].mean()),
            "peak_reserved": float(log["peak_gpu_reserved_gib"].max()),
            "trainable": int(verification.get("trainable_parameters", 0)),
        }
    training_comparison = [
        train_summary("R5", r5_status, r5_log, model_info["R5"]),
        train_summary("R6", r6_status, r6_log, model_info["R6"]),
    ]
    metric = {(row.Backbone, row.Feature_Variant, row.Disease): row
              for row in metrics[metrics["Threshold_Protocol"] == "fixed_0.5"].itertuples(index=False)}
    lines = [
        "# R5 / R6 四阶段疾病特征可分性诊断", "",
        "## 摘要", "",
        f"- R5 Best：Epoch {r5_status['best_epoch']}，Macro-AUROC {r5_status['best_macro_auroc']:.6f}；R6 Best：Epoch {r6_status['best_epoch']}，Macro-AUROC {r6_status['best_macro_auroc']:.6f}。",
        f"- 正式模型 R6−R5 Macro-AUROC：{bootstrap['r6_vs_r5_macro_auroc']['delta']:+.6f}，95% CI [{bootstrap['r6_vs_r5_macro_auroc']['ci95'][0]:+.6f}, {bootstrap['r6_vs_r5_macro_auroc']['ci95'][1]:+.6f}]；Macro-AUPRC 差值 {bootstrap['r6_vs_r5_macro_auprc']['delta']:+.6f}，95% CI [{bootstrap['r6_vs_r5_macro_auprc']['ci95'][0]:+.6f}, {bootstrap['r6_vs_r5_macro_auprc']['ci95'][1]:+.6f}]。",
        "- 下表展示冻结 Study 特征经 Subject 分组 Train-CV 选择 C 的 L2 Logistic Regression Probe；不是 R5/R6 正式模型分数。", "",
        "## 1. 正式 R6 对比 R5", "",
        "完整逐类监督模型指标与训练状态见 `R6_Full_Finetune_512/r5_vs_r6_comparison.md`。全量解冻后是否改善必须结合配对 Subject Bootstrap 区间；Validation 已用于 Best checkpoint 选择，统计结果是探索性证据。", "",
        "| Run | Best Epoch | Macro-AUROC | Train BCE at Best | Valid BCE at Best | Optimizer Updates through Best | Mean images/s through Best | Peak reserved GiB | Trainable Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in training_comparison:
        lines.append(
            f"| {summary['name']} | {summary['best_epoch']} | {summary['best_auc']:.6f} | "
            f"{summary['train_bce']:.6f} | {summary['valid_bce']:.6f} | {summary['updates']:,} | "
            f"{summary['images_per_s']:.2f} | {summary['peak_reserved']:.2f} | {summary['trainable']:,} |")
    lines += [
        "",
        "## 2. R5 / R6 Stage × Disease AUROC Probe 矩阵", "",
        "AUROC 与 AUPRC 矩阵及对应 Bootstrap CI 位于 `stage_auroc_matrix_*.csv`、`stage_auprc_matrix_*.csv` 和 `stage_paired_bootstrap.json`。", "",
        "| Disease | R5 Best Stage (AUROC) | R6 Best Stage (AUROC) | R5 Stage3→4 Δ | R6 Stage3→4 Δ | R5 Stage3+4 vs Stage4 | R6 Stage3+4 vs Stage4 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for disease in LABELS:
        best_r5 = best[(best["Backbone"] == "R5") & (best["Disease"] == disease)].iloc[0]
        best_r6 = best[(best["Backbone"] == "R6") & (best["Disease"] == disease)].iloc[0]
        t5 = comparison_lookup(transitions, "R5", "Stage4-Stage3", disease)
        t6 = comparison_lookup(transitions, "R6", "Stage4-Stage3", disease)
        j5 = comparison_lookup(joint, "R5", "Stage3+4−Stage4", disease)
        j6 = comparison_lookup(joint, "R6", "Stage3+4−Stage4", disease)
        lines.append(
            f"| {disease} | {best_r5.Best_Stage} ({best_r5.Best_AUROC:.3f}) | "
            f"{best_r6.Best_Stage} ({best_r6.Best_AUROC:.3f}) | "
            f"{fmt_ci(t5.Delta_AUROC, t5.AUROC_CI95_Low, t5.AUROC_CI95_High)} | "
            f"{fmt_ci(t6.Delta_AUROC, t6.AUROC_CI95_Low, t6.AUROC_CI95_High)} | "
            f"{fmt_ci(j5.Delta_AUROC, j5.AUROC_CI95_Low, j5.AUROC_CI95_High)} | "
            f"{fmt_ci(j6.Delta_AUROC, j6.AUROC_CI95_Low, j6.AUROC_CI95_High)} |")
    lines += ["", "## 3. 弱类别与对照类别", ""]
    for disease in ("Pneumonia", "Lung Opacity", "Consolidation", "Pneumothorax"):
        lines.append(f"### {disease}")
        for model_name in ("R5", "R6"):
            scores = [metric[(model_name, stage, disease)].Validation_AUROC for stage in STAGES]
            ap = [metric[(model_name, stage, disease)].Validation_AUPRC for stage in STAGES]
            transition = comparison_lookup(transitions, model_name, "Stage4-Stage3", disease)
            joint_vs4 = comparison_lookup(joint, model_name, "Stage3+4−Stage4", disease)
            rows = [
                f"- {model_name}: AUROC Stage1→4 `{', '.join(f'{v:.3f}' for v in scores)}`；AUPRC `{', '.join(f'{v:.3f}' for v in ap)}`。",
                f"- Stage4−Stage3 AUROC Δ {transition.Delta_AUROC:+.4f} [{transition.AUROC_CI95_Low:+.4f}, {transition.AUROC_CI95_High:+.4f}]；Stage3+4−Stage4 Δ {joint_vs4.Delta_AUROC:+.4f} [{joint_vs4.AUROC_CI95_Low:+.4f}, {joint_vs4.AUROC_CI95_High:+.4f}].",
            ]
            lines.extend(rows)
        lines.append("")

    lines += [
        "## 4. 全量微调改变了哪些阶段？", "",
        "逐疾病、逐 Stage 的 R6−R5 AUROC/AUPRC 及 paired CI 见 `r6_minus_r5_stage_deltas.csv` 与对应热力图。R6 与 R5 特征空间不同；Probe 变化描述可分性差异，不将通道数值对应解释为相同医学语义。", "",
        "## 5. Stage 互补性与特征维度敏感性", "",
        "`stage_joint_probe_comparison.csv` 对 Stage3+4 与两者单独 Probe，以及 Stage2+3 与 Stage2/3 单独 Probe 进行同 Study 配对比较。只有配对 CI 不跨零时，才将验证结果描述为当前 Probe 协议下的可检测补充价值；这不证明 Stage4 不可逆地丢失病理信息。`pca64_probe_comparison.csv` 比较各 Stage 原维度和统一 PCA-64 后的 Probe。", "",
        "## 6. 训练曲线", "",
        f"- R5 实际训练 {len(r5_log)} Epoch，Best Epoch {int(r5_log.loc[r5_log.macro_auroc.idxmax(), 'epoch'])}；Train BCE {float(r5_log.iloc[-1].train_bce):.4f}、Validation BCE {float(r5_log.iloc[-1].valid_loss):.4f}。",
        f"- R6 实际训练 {len(r6_log)} Epoch，Best Epoch {int(r6_log.loc[r6_log.macro_auroc.idxmax(), 'epoch'])}；Train BCE {float(r6_log.iloc[-1].train_bce):.4f}、Validation BCE {float(r6_log.iloc[-1].valid_loss):.4f}。",
        "训练曲线和停止规则见 `charts/` 及各自训练目录。", "",
        "## 7. 下一轮研究判断", "",
        "- 四个单 Stage Probe 的 Macro-AUROC：R5 Stage1/2/3/4=`0.7116/0.7215/0.7818/0.7880`；R6=`0.7189/0.7354/0.7789/0.7842`。当前证据不支持把主要瓶颈概括为 Stage3 特征获取不足；判别性主要在 Stage3/4 已形成。",
        "- Stage3+Stage4 相对单独 Stage4 的 Macro-AUROC 增量：R5 `+0.0011`，95% CI `[-0.0058,+0.0067]`；R6 `+0.0035`，95% CI `[-0.0005,+0.0073]`。宏平均没有清楚的联合收益。",
        "- 存在类别级互补信号：R6 Consolidation 的 Stage3+4−Stage4 为 `+0.0268`，95% CI `[+0.0094,+0.0477]`，且 Stage3+4−Stage3 也为正；R5 Pneumonia 的 Stage3+4−Stage4 为 `+0.0269`，95% CI `[+0.0042,+0.0532]`，但相对 Stage3 的区间跨零。因为比较数量多且 Validation 已参与历史选模，这些结果是需要独立复验的候选假设。",
        "- 因此若设计下一轮结构验证，优先测试一个轻量 Stage3→Stage4/Global GAP 特征融合，预先指定 Consolidation 为主要类别、Pneumonia 为次要类别，同时保留普通 R5 GAP 对照；本轮不实现也不启动新训练。",
        "- R6 正式 Macro-AUROC 点估计低于 R5 `0.0047`，配对 95% CI `[-0.0146,+0.0029]` 跨零，且 R6 可训练参数更多。当前保留 R5 Global-only 作为后续结构改进基线，R6 保留为全量微调对照；这不构成 R6 明确劣于 R5 的统计证据。",
        "", "## 8. 统计与方法限制", "",
        f"- Validation：{dataset_metadata_text(model_info)}。Bootstrap 以 Subject 为单位，Seed=42，{BOOTSTRAP_REPLICATES} 次；同一 Study 的多张图在特征提取阶段先做 image-level GAP，再按 Study 平均。",
        "- 正则化强度只在 Train 内部 Subject 分组 3-fold CV（若折内类别缺失则降为 2-fold）选择；Scaler、PCA 只在当前 CV 训练折拟合。Train OOF 概率选择补充 F1 阈值；正式 Validation 未用于阈值或 C 选择。",
        "- 对重复疾病、Stage 与 Probe 比较存在多重比较风险；Validation 还曾用于历史 checkpoint 选择。Bootstrap CI 是探索性不确定性描述，不替代独立测试集确认。",
        "- Probe 测量线性可分性与当前预处理/聚合下的特征可用性，不证明因果机制、真实病灶定位或临床有效性。未使用 Test。", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dataset_metadata_text(model_info):
    return "859 Studies / 228 Subjects；标签为官方 closed-world 十类。R5 与 R6 使用相同的 Validation Study keys 和标签顺序"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Stage feature extraction requires the configured CUDA GPU")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    DISEASE_DIR.mkdir(parents=True, exist_ok=True)
    datasets, dataset_meta = official_datasets()
    model_info = {name: checkpoint_details(name) for name in ("R5", "R6")}
    torch.cuda.init()
    device = torch.device("cuda")

    all_features = {name: {} for name in ("R5", "R6")}
    extraction_manifest = {
        "input_size": IMAGE_SIZE,
        "label_order": LABELS,
        "feature_stages": {key: {"dimension": value} for key, value in EXPECTED_DIMS.items()},
        "aggregation": "GAP each image feature map, then arithmetic mean of image vectors within Study",
        "training_feature_transform": "deterministic evaluation transform; no random augmentation",
        "normalization": "ImageNet mean/std after ResizePad and Grayscale3",
        "dataset": dataset_meta,
        "checkpoints": {},
        "feature_cache_files": [],
        "test_data_accessed": False,
    }
    for model_name in ("R5", "R6"):
        info = model_info[model_name]
        extraction_manifest["checkpoints"][model_name] = {
            "path": str(info["checkpoint"]),
            "sha256": info["checkpoint_sha256"],
            "best_epoch": info["best_epoch"],
            "best_macro_auroc": info["best_macro_auroc"],
        }
        model = load_best_model(info, device)
        for split in ("train", "valid"):
            start_cached = all(
                (CACHE_DIR / f"{model_name.lower()}_{stage}_{split}.npz").is_file()
                for stage in (*STAGES, "stage4_norm"))
            if not start_cached:
                extracted = features_for_split(
                    model_name, model, info, datasets[split], split, device)
                save_feature_caches(model_name, split, info, extracted)
            all_features[model_name][split] = load_feature_set(model_name, split)
            for stage in (*STAGES, "stage4_norm"):
                path = CACHE_DIR / f"{model_name.lower()}_{stage}_{split}.npz"
                extraction_manifest["feature_cache_files"].append({
                    "path": str(path), "sha256": file_sha256(path),
                    "metadata": metadata_of(path),
                })
        del model
        torch.cuda.empty_cache()

    for split in ("train", "valid"):
        r5, r6 = all_features["R5"][split], all_features["R6"][split]
        for key in ("Subject_id", "Study_id", "Image_Count", "labels"):
            if not np.array_equal(r5[key], r6[key]):
                raise RuntimeError(f"R5/R6 {split} feature cache misalignment for {key}")
    if not set(all_features["R5"]["train"]["Subject_id"]).isdisjoint(
            set(all_features["R5"]["valid"]["Subject_id"])):
        raise RuntimeError("Feature cache Subject leakage check failed")
    feature_config_path = HERE / "feature_extraction_config.json"
    write_json(feature_config_path, extraction_manifest)
    feature_integrity_report(HERE / "feature_integrity_report.md", dataset_meta,
                             all_features, model_info)

    variants = feature_variants()
    model_specs, probe_results, cv_frames, prediction_matrices = {}, {}, [], {}
    train_labels = all_features["R5"]["train"]["labels"].astype(np.uint8)
    valid_labels = all_features["R5"]["valid"]["labels"].astype(np.uint8)
    for model_name in ("R5", "R6"):
        train_set = all_features[model_name]["train"]
        valid_set = all_features[model_name]["valid"]
        if not np.array_equal(train_set["labels"], train_labels):
            raise RuntimeError(f"{model_name} train labels are misaligned")
        if not np.array_equal(valid_set["labels"], valid_labels):
            raise RuntimeError(f"{model_name} validation labels are misaligned")
        for variant_name, variant in variants.items():
            model_key = f"{model_name}|{variant_name}"
            print(f"Starting CV probe {model_key}", flush=True)
            result, cv_frame = fit_probe_variant(
                model_name, variant_name, variant, train_set, valid_set,
                model_info[model_name]["checkpoint_sha256"])
            model_specs[model_key] = variant
            probe_results[model_key] = result
            prediction_matrices[model_key] = result["validation_probabilities"]
            cv_frames.append(cv_frame)
    cv_all = pd.concat(cv_frames, ignore_index=True)
    cv_all.to_csv(HERE / "linear_probe_cv_results.csv", index=False)

    draws, draw_hash = subject_bootstrap_draws(all_features["R5"]["valid"]["Subject_id"])
    point, boot = bootstrap_all(valid_labels, prediction_matrices, draws)
    metric_rows_frame = metric_rows(
        model_specs, prediction_matrices, all_features, all_features,
        train_labels, valid_labels, probe_results,
        {key: model_metric_record(key, valid_labels, point, boot) for key in point})
    metric_rows_frame.to_csv(HERE / "linear_probe_all_metrics.csv", index=False)
    metric_rows_frame.to_csv(HERE / "stage_full_metrics_by_disease.csv", index=False)
    prediction_frame(model_specs, prediction_matrices, all_features["R5"]["valid"]).to_csv(
        HERE / "validation_probe_predictions.csv", index=False)

    bootstrap_records = {key: model_metric_record(key, valid_labels, point, boot)
                         for key in point}
    boot_payload, transitions, deltas, joint, pca, best = make_comparison_outputs(
        metric_rows_frame, model_specs, point, boot, valid_labels, draw_hash,
        all_features["R5"]["valid"])
    write_stage_matrices(metric_rows_frame, transitions, deltas)
    r5_log = pd.read_csv(R5_DIR / "train_log.csv")
    r6_log = pd.read_csv(R6_DIR / "train_log.csv")
    create_charts(metric_rows_frame, transitions, deltas, joint, r5_log, r6_log)
    create_per_disease_reports(metric_rows_frame, transitions, deltas, joint,
                               pca, best, dataset_meta)

    # Formal-model R5/R6 metrics already use the exact Study-aligned prediction outputs.
    r5_pred = pd.read_csv(R5_DIR / "val_best.csv", dtype={"Subject_id": "string", "Study_id": "string"})
    r6_pred = pd.read_csv(R6_DIR / "val_best.csv", dtype={"Subject_id": "string", "Study_id": "string"})
    expected_key_set = set(zip(
        all_features["R5"]["valid"]["Subject_id"],
        all_features["R5"]["valid"]["Study_id"],
    ))
    for name, frame in (("R5", r5_pred), ("R6", r6_pred)):
        if frame.duplicated(["Subject_id", "Study_id"]).any():
            raise RuntimeError(f"{name} formal Best predictions contain duplicate Study keys")
        observed = set(zip(frame["Subject_id"].astype(str), frame["Study_id"].astype(str)))
        if observed != expected_key_set:
            raise RuntimeError(f"{name} formal Best predictions do not match the official Validation Study set")
    valid_order = pd.DataFrame({
        "Subject_id": all_features["R5"]["valid"]["Subject_id"],
        "Study_id": all_features["R5"]["valid"]["Study_id"],
    })
    merged = valid_order.merge(
        r5_pred, on=["Subject_id", "Study_id"], how="left", validate="one_to_one", sort=False)
    merged = merged.merge(
        r6_pred, on=["Subject_id", "Study_id"], how="left", validate="one_to_one",
        sort=False, suffixes=("_r5", "_r6"))
    if len(merged) != 859:
        raise RuntimeError("Formal R5/R6 Validation predictions are not aligned for stage report")
    prob_cols = [f"prob_{name.replace(' ', '_')}" for name in LABELS]
    formal_scores = {
        "R5": merged[[f"{column}_r5" for column in prob_cols]].to_numpy(dtype=float),
        "R6": merged[[f"{column}_r6" for column in prob_cols]].to_numpy(dtype=float),
    }
    formal_labels = merged[[f"{column}_r5" for column in TARGET_COLS]].to_numpy(dtype=np.uint8)
    formal_labels_r6 = merged[[f"{column}_r6" for column in TARGET_COLS]].to_numpy(dtype=np.uint8)
    if (not np.array_equal(formal_labels, formal_labels_r6)
            or not np.array_equal(formal_labels, all_features["R5"]["valid"]["labels"])):
        raise RuntimeError("Formal R5/R6 Best prediction labels differ from the official feature-cache labels")
    formal_point, formal_boot = bootstrap_all(formal_labels, formal_scores,
                                              draws)
    formal_r5_auc = formal_point["R5"]["auroc"]
    formal_r6_auc = formal_point["R6"]["auroc"]
    formal_r5_ap = formal_point["R5"]["auprc"]
    formal_r6_ap = formal_point["R6"]["auprc"]
    r5_r6_auc_delta = formal_boot["R6"]["macro_auroc"] - formal_boot["R5"]["macro_auroc"]
    r5_r6_ap_delta = formal_boot["R6"]["macro_auprc"] - formal_boot["R5"]["macro_auprc"]
    auc_ci, auc_n = ci(r5_r6_auc_delta)
    ap_ci, ap_n = ci(r5_r6_ap_delta)
    boot_payload["r6_vs_r5_macro_auroc"] = {
        "r5": float(np.nanmean(formal_r5_auc)), "r6": float(np.nanmean(formal_r6_auc)),
        "delta": float(np.nanmean(formal_r6_auc - formal_r5_auc)),
        "ci95": auc_ci, "valid_replicates": auc_n,
    }
    boot_payload["r6_vs_r5_macro_auprc"] = {
        "r5": float(np.nanmean(formal_r5_ap)), "r6": float(np.nanmean(formal_r6_ap)),
        "delta": float(np.nanmean(formal_r6_ap - formal_r5_ap)),
        "ci95": ap_ci, "valid_replicates": ap_n,
    }
    write_json(HERE / "stage_paired_bootstrap.json", boot_payload)
    write_final_report(HERE / "stage_feature_diagnostics_final_report.md", model_info,
                       metric_rows_frame, transitions, deltas, joint, pca, best,
                       boot_payload, r5_log, r6_log)
    print(json.dumps({
        "status": "complete", "feature_dir": str(HERE),
        "feature_models": len(model_specs), "cv_rows": len(cv_all),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "R5_probe_stage4_norm": bootstrap_records.get("R5|stage4_norm", {}).get("Macro_AUROC"),
        "R6_probe_stage4_norm": bootstrap_records.get("R6|stage4_norm", {}).get("Macro_AUROC"),
        "formal_R5_macro_auroc": float(np.nanmean(formal_r5_auc)),
        "formal_R6_macro_auroc": float(np.nanmean(formal_r6_auc)),
        "formal_R6_minus_R5_ci95": auc_ci,
        "test_data_accessed": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
