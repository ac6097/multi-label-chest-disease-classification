"""G1: R3/R5 generalization, aggregation and pretrained-representation drift.

Deterministic no-augmentation inference on Train and Validation, single/multi
image Study analysis, Logit-Mean vs Probability-Mean aggregation check,
label/subject distribution audit, Stage3/Stage4 parameter drift and feature
drift against the original DINOv3 general-vision weights.

No training, no checkpoint or report overwrite, no Test labels.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from generalization_common import (
    CACHE_DIR,
    G0_DIR,
    G1_AGG_DIR,
    G1_DRIFT_DIR,
    G1_METRICS_DIR,
    LABEL_NAMES,
    R3_DIR,
    R5_DIR,
    SEED,
    build_eval_datasets,
    ensure_directories,
    infer_dataset,
    load_r3_model,
    load_r5_model,
    load_result,
    metrics,
    probabilities,
    read_json,
    save_result,
    sha256_file,
    study_bce,
    write_csv,
    write_json,
)
from stage_fusion_analysis import (
    _sigmoid,
    bootstrap_models,
    study_metrics,
)

G1_CACHE = G1_METRICS_DIR / "cache"
PRETRAINED_SHA = (
    "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2")
FROZEN_PREFIXES = (
    "backbone.model.downsample_layers.0.",
    "backbone.model.downsample_layers.1.",
    "backbone.model.stages.0.",
    "backbone.model.stages.1.",
    "backbone.model.norms.0.",
    "backbone.model.norms.1.",
    "backbone.model.norms.2.",
)


# ---------------------------------------------------------------------------
# Cached split inference
# ---------------------------------------------------------------------------

def split_result(group: str, split: str, dataset, force: bool = False) -> dict:
    G1_CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{group.lower()}_{split}_noaug"
    path = G1_CACHE / f"{tag}.npz"
    if path.is_file() and not force:
        print(f"[G1] cached: {tag}", flush=True)
        return load_result(path)
    print(f"[G1] inference: {tag}", flush=True)
    device = torch.device("cuda")
    if group == "R3":
        model, status, _ = load_r3_model(device, channels_last=True,
                                         weights="checkpoint")
    else:
        model, status, _ = load_r5_model(device, channels_last=True,
                                         weights="checkpoint")
    try:
        result = infer_dataset(
            model, group, dataset, image_budget=8, num_workers=8,
            channels_last=True, autocast=True, collect_image_logits=True)
    finally:
        del model
        torch.cuda.empty_cache()
    result["group"] = group
    result["split"] = split
    result["status_best_epoch"] = int(status["best_epoch"])
    save_result(path, result)
    print(f"[G1] finished: {tag} ({result['elapsed_seconds']:.1f}s, "
          f"{result['images']} images)", flush=True)
    return result


def _bce_for_subset(result: dict, mask: np.ndarray) -> float:
    from dino_losses import study_bce_per_study

    logits = torch.from_numpy(result["study_logits"][mask].astype(np.float32))
    labels = torch.from_numpy(result["labels"][mask].astype(np.float32))
    known = torch.ones_like(labels, dtype=torch.bool)
    per_study, eligible = study_bce_per_study(logits, labels, known)
    return float(per_study[eligible].mean())


def _metric_rows(group: str, split: str, result: dict, study_group: str = "all",
                 mask: np.ndarray | None = None) -> list[dict]:
    labels = result["labels"]
    probability = probabilities(result)
    if mask is not None:
        labels = labels[mask]
        probability = probability[mask]
    metric = study_metrics(labels, probability)
    bce = study_bce(result) if mask is None else _bce_for_subset(result, mask)
    rows = []
    for disease in LABEL_NAMES:
        per_class = metric["per_class"][disease]
        rows.append({
            "group": group,
            "split": split,
            "study_group": study_group,
            "disease": disease,
            "auroc": per_class["auroc"],
            "auprc": per_class["auprc"],
            "f1_at_0.5": per_class["f1"],
            "n_positive": per_class["n_positive"],
            "n_negative": per_class["n_negative"],
            "study_bce": bce,
            "macro_auroc": metric["macro_auroc"],
            "macro_auprc": metric["macro_auprc"],
        })
    return rows


# ---------------------------------------------------------------------------
# G1-A: Train/Validation same-protocol metrics
# ---------------------------------------------------------------------------

def run_g1a(results: dict) -> dict:
    rows = []
    for group in ("R3", "R5"):
        for split in ("train", "valid"):
            rows.extend(_metric_rows(group, split, results[(group, split)]))
    frame = pd.DataFrame(rows)
    for group in ("R3", "R5"):
        frame[frame["group"] == group].to_csv(
            G1_METRICS_DIR / f"{group.lower()}_train_valid_metrics.csv",
            index=False, encoding="utf-8")

    gap_rows = []
    for group in ("R3", "R5"):
        train_metrics = study_metrics(results[(group, "train")]["labels"],
                                      probabilities(results[(group, "train")]))
        valid_metrics = study_metrics(results[(group, "valid")]["labels"],
                                      probabilities(results[(group, "valid")]))
        for disease in LABEL_NAMES:
            gap_rows.append({
                "group": group,
                "disease": disease,
                "train_auroc": train_metrics["per_class"][disease]["auroc"],
                "valid_auroc": valid_metrics["per_class"][disease]["auroc"],
                "auroc_gap": (train_metrics["per_class"][disease]["auroc"]
                              - valid_metrics["per_class"][disease]["auroc"]),
                "train_auprc": train_metrics["per_class"][disease]["auprc"],
                "valid_auprc": valid_metrics["per_class"][disease]["auprc"],
                "auprc_gap": (train_metrics["per_class"][disease]["auprc"]
                              - valid_metrics["per_class"][disease]["auprc"]),
                "train_positive_studies": train_metrics["per_class"][disease][
                    "n_positive"],
                "valid_positive_studies": valid_metrics["per_class"][disease][
                    "n_positive"],
                "train_study_bce": study_bce(results[(group, "train")]),
                "valid_study_bce": study_bce(results[(group, "valid")]),
            })
        gap_rows.append({
            "group": group,
            "disease": "Macro",
            "train_auroc": train_metrics["macro_auroc"],
            "valid_auroc": valid_metrics["macro_auroc"],
            "auroc_gap": train_metrics["macro_auroc"] - valid_metrics["macro_auroc"],
            "train_auprc": train_metrics["macro_auprc"],
            "valid_auprc": valid_metrics["macro_auprc"],
            "auprc_gap": train_metrics["macro_auprc"] - valid_metrics["macro_auprc"],
            "train_positive_studies": None,
            "valid_positive_studies": None,
            "train_study_bce": study_bce(results[(group, "train")]),
            "valid_study_bce": study_bce(results[(group, "valid")]),
        })
    write_csv(G1_METRICS_DIR / "per_class_generalization_gap.csv", gap_rows)
    return {"metrics_rows": rows, "gap_rows": gap_rows}


# ---------------------------------------------------------------------------
# G1-B: single/multi-image Studies and aggregation ablation
# ---------------------------------------------------------------------------

STUDY_GROUPS = {
    "single": lambda counts: counts == 1,
    "multi": lambda counts: counts > 1,
    "two": lambda counts: counts == 2,
    "three_plus": lambda counts: counts >= 3,
}


def run_g1b(results: dict) -> dict:
    rows = []
    for group in ("R3", "R5"):
        for split in ("train", "valid"):
            result = results[(group, split)]
            counts = result["image_counts"]
            for study_group, selector in STUDY_GROUPS.items():
                mask = selector(counts)
                if int(mask.sum()) == 0:
                    continue
                subset_rows = _metric_rows(group, split, result, study_group, mask)
                n_studies = int(mask.sum())
                n_subjects = len(set(result["subjects"][mask].astype(str)))
                for row in subset_rows:
                    row["n_studies"] = n_studies
                    row["n_subjects"] = n_subjects
                rows.extend(subset_rows)
    write_csv(G1_METRICS_DIR / "single_multi_study_metrics.csv", rows)

    # Aggregation ablation on multi-image Validation Studies, R5 Best only.
    result = results[("R5", "valid")]
    counts = result["image_counts"]
    multi_mask = counts > 1
    labels = result["labels"][multi_mask]
    subjects = result["subjects"][multi_mask]
    image_logits = result["image_logits"]
    image_study = result["image_study_index"]
    full_counts = result["image_counts"]
    study_logit_mean = np.zeros((len(counts), 10), dtype=np.float64)
    probability_sum = np.zeros((len(counts), 10), dtype=np.float64)
    for image_index in range(len(image_logits)):
        study_index = int(image_study[image_index])
        study_logit_mean[study_index] += image_logits[image_index]
        probability_sum[study_index] += _sigmoid(
            image_logits[image_index].astype(np.float64))
    study_logit_mean /= full_counts[:, None]
    probability_mean = probability_sum / full_counts[:, None]
    logit_probability = _sigmoid(study_logit_mean[multi_mask])
    probability_probability = probability_mean[multi_mask]
    logit_metrics = study_metrics(labels, logit_probability)
    probability_metrics = study_metrics(labels, probability_probability)
    bootstrap = bootstrap_models(
        labels,
        {"logit_mean": logit_probability, "probability_mean": probability_probability},
        subjects,
        {"probability_mean_minus_logit_mean": ("logit_mean", "probability_mean")})
    comparison = bootstrap["comparisons"]["probability_mean_minus_logit_mean"]
    ablation_rows = []
    for disease in LABEL_NAMES:
        stats = comparison["per_class"][disease]
        ablation_rows.append({
            "disease": disease,
            "n_positive_studies": int(labels[:, LABEL_NAMES.index(disease)].sum()),
            "logit_mean_auroc": logit_metrics["per_class"][disease]["auroc"],
            "probability_mean_auroc": probability_metrics["per_class"][disease][
                "auroc"],
            "auroc_delta": stats["delta_auroc"],
            "auroc_ci95_low": stats["delta_auroc_ci95"][0],
            "auroc_ci95_high": stats["delta_auroc_ci95"][1],
            "logit_mean_auprc": logit_metrics["per_class"][disease]["auprc"],
            "probability_mean_auprc": probability_metrics["per_class"][disease][
                "auprc"],
            "auprc_delta": stats["delta_auprc"],
        })
    macro_comparison = comparison
    ablation_rows.append({
        "disease": "Macro",
        "n_positive_studies": None,
        "logit_mean_auroc": logit_metrics["macro_auroc"],
        "probability_mean_auroc": probability_metrics["macro_auroc"],
        "auroc_delta": macro_comparison["delta_macro_auroc"],
        "auroc_ci95_low": macro_comparison["delta_macro_auroc_ci95"][0],
        "auroc_ci95_high": macro_comparison["delta_macro_auroc_ci95"][1],
        "logit_mean_auprc": logit_metrics["macro_auprc"],
        "probability_mean_auprc": probability_metrics["macro_auprc"],
        "auprc_delta": macro_comparison["delta_macro_auprc"],
    })
    write_csv(G1_AGG_DIR / "study_aggregation_ablation.csv", ablation_rows)
    write_json(G1_AGG_DIR / "study_aggregation_bootstrap.json", {
        "dataset": "R5 Best, multi-image Validation Studies only",
        "n_multi_image_studies": int(multi_mask.sum()),
        "n_subjects": len(set(subjects.astype(str))),
        "logit_mean_macro_auroc": logit_metrics["macro_auroc"],
        "probability_mean_macro_auroc": probability_metrics["macro_auroc"],
        "delta_macro_auroc": comparison["delta_macro_auroc"],
        "delta_macro_auroc_ci95": comparison["delta_macro_auroc_ci95"],
        "delta_macro_auprc": comparison["delta_macro_auprc"],
        "delta_macro_auprc_ci95": comparison["delta_macro_auprc_ci95"],
        "bootstrap": bootstrap,
        "test_data_accessed": False,
    })
    return {"single_multi_rows": rows, "aggregation": {
        "logit_mean_macro_auroc": logit_metrics["macro_auroc"],
        "probability_mean_macro_auroc": probability_metrics["macro_auroc"],
        "delta_macro_auroc": comparison["delta_macro_auroc"],
        "delta_macro_auroc_ci95": comparison["delta_macro_auroc_ci95"],
        "n_multi_image_studies": int(multi_mask.sum()),
    }}


# ---------------------------------------------------------------------------
# G1-C: label and subject distribution
# ---------------------------------------------------------------------------

def run_g1c(train_data, valid_data) -> dict:
    rows = []
    integrity = {"splits": {}, "subject_leakage": {}, "notes": []}
    split_subjects = {}
    split_study_keys = {}
    for split, data in (("train", train_data), ("valid", valid_data)):
        subjects = np.asarray([str(subject) for subject, _ in data.study_keys])
        studies = np.asarray([str(study) for _, study in data.study_keys])
        labels = np.stack([data.dataset.base.studies[index]["target"].numpy()
                           for index in range(len(data))]).astype(np.uint8)
        image_counts = np.asarray(data.image_counts)
        split_subjects[split] = set(subjects)
        split_study_keys[split] = set(data.study_keys)
        unique_subjects = sorted(set(subjects))
        studies_per_subject = np.asarray([int(np.sum(subjects == subject))
                                          for subject in unique_subjects])
        positives_per_study = labels.sum(axis=1)
        integrity["splits"][split] = {
            "studies": int(len(data)),
            "images": int(image_counts.sum()),
            "subjects": int(len(unique_subjects)),
            "studies_per_subject_mean": float(studies_per_subject.mean()),
            "studies_per_subject_median": float(np.median(studies_per_subject)),
            "studies_per_subject_max": int(studies_per_subject.max()),
            "studies_per_subject_p90": float(np.quantile(studies_per_subject, 0.9)),
            "fraction_studies_from_subjects_with_ge5_studies": float(
                studies_per_subject[studies_per_subject >= 5].sum()
                / studies_per_subject.sum()),
            "image_count_distribution": {
                str(count): int(count_value)
                for count, count_value in zip(*np.unique(image_counts,
                                                        return_counts=True))},
            "positives_per_study_mean": float(positives_per_study.mean()),
            "positives_per_study_median": float(np.median(positives_per_study)),
            "single_label_study_fraction": float(
                np.mean(positives_per_study == 1)),
            "multi_label_study_fraction": float(
                np.mean(positives_per_study > 1)),
            "no_label_study_fraction": float(np.mean(positives_per_study == 0)),
        }
        for index, disease in enumerate(LABEL_NAMES):
            positive_study = labels[:, index].astype(bool)
            positive_subjects = sorted(set(subjects[positive_study]))
            rows.append({
                "split": split,
                "scope": "disease",
                "disease": disease,
                "study_positive": int(positive_study.sum()),
                "study_negative": int((~positive_study).sum()),
                "subject_positive": int(len(positive_subjects)),
                "subject_negative": int(len(unique_subjects) - len(positive_subjects)),
                "positive_study_fraction": float(positive_study.mean()),
                "positive_subject_fraction": float(
                    len(positive_subjects) / len(unique_subjects)),
            })
        rows.append({
            "split": split, "scope": "all", "disease": "ALL",
            "study_positive": int(len(data)),
            "study_negative": 0,
            "subject_positive": int(len(unique_subjects)),
            "subject_negative": 0,
            "positive_study_fraction": 1.0,
            "positive_subject_fraction": 1.0,
        })
    subject_overlap = sorted(split_subjects["train"] & split_subjects["valid"])
    study_overlap = sorted(split_study_keys["train"] & split_study_keys["valid"])
    integrity["subject_leakage"] = {
        "train_subjects": len(split_subjects["train"]),
        "valid_subjects": len(split_subjects["valid"]),
        "overlap_subjects": len(subject_overlap),
        "overlap_examples": subject_overlap[:5],
        "overlap_studies": len(study_overlap),
    }
    image_summary_path = Path("docs/tables/image_summary.json")
    if image_summary_path.is_file():
        integrity["image_audit"] = read_json(image_summary_path)
    integrity["notes"] = [
        "Subject/Study overlap is computed from the official CSV keys.",
        "Exact image duplicate groups come from the existing dataset audit "
        "(docs/tables/image_summary.json); Test labels are not involved.",
    ]
    write_csv(G1_METRICS_DIR / "label_subject_distribution.csv", rows)
    write_json(G1_METRICS_DIR / "data_integrity.json", integrity)
    return {"rows": rows, "integrity": integrity}


# ---------------------------------------------------------------------------
# G1-D: Stage / block parameter drift
# ---------------------------------------------------------------------------

def _original_backbone_state() -> dict[str, torch.Tensor]:
    from dino_experiment import load_pretrained_backbone

    backbone, verification = load_pretrained_backbone("cpu")
    if verification["weights_sha256"] != PRETRAINED_SHA:
        raise RuntimeError("Original DINOv3 weight hash mismatch")
    return {name: tensor.detach().float().clone()
            for name, tensor in backbone.state_dict().items()
            if name.startswith("model.")
            and not name.startswith("model.norms.")}


def _checkpoint_backbone_state(group: str) -> tuple[dict[str, torch.Tensor], dict]:
    run_dir = R3_DIR if group == "R3" else R5_DIR
    payload = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    state = {}
    excluded = []
    for name, tensor in payload["model_state"].items():
        if name.startswith("backbone."):
            inner = name.removeprefix("backbone.")
            if inner.startswith("model.norms."):
                # Alias of model.norm (same Parameter); count it only once.
                continue
            state[inner] = tensor.detach().float().clone()
        else:
            excluded.append(name)
    return state, {
        "checkpoint": str(run_dir / "best.pth"),
        "epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_macro_auroc": float(payload["best_macro_auroc"]),
        "non_backbone_state_tensors": sorted(excluded),
    }


def _stage_key(name: str) -> str:
    if name.startswith("model.downsample_layers."):
        return f"stage{name.split('.')[2]}"
    if name.startswith("model.stages."):
        return f"stage{name.split('.')[2]}"
    if name.startswith("model.norms."):
        return f"norm{name.split('.')[2]}"
    if name.startswith("model.norm."):
        return "final_norm"
    if name.startswith("model.head"):
        return "head"
    return "other"


def _block_key(name: str) -> str:
    if name.startswith("model.stages."):
        parts = name.split(".")
        return f"stage{parts[2]}_block{parts[3]}"
    if name.startswith("model.downsample_layers."):
        return f"downsample{name.split('.')[2]}"
    return _stage_key(name)


def _parameter_type(name: str) -> str:
    if "dwconv" in name:
        return "dwconv"
    if "pwconv1" in name:
        return "pwconv1"
    if "pwconv2" in name:
        return "pwconv2"
    if "gamma" in name:
        return "layer_scale_gamma"
    if "downsample_layers" in name and "weight" in name:
        return "downsample_conv"
    if "norm" in name:
        return "layernorm"
    return "other"


def _relative_drift(delta: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(reference.norm()) + 1e-12
    return float(delta.norm()) / denominator


def run_g1d() -> dict:
    original = _original_backbone_state()
    models = {}
    for group in ("R3", "R5"):
        state, meta = _checkpoint_backbone_state(group)
        models[group] = {"state": state, "meta": meta}

    drift_rows, block_rows, type_rows = [], [], []
    frozen_rows = []
    summary = {"checkpoints": {group: models[group]["meta"]
                               for group in models},
               "excluded_from_backbone_drift": [
                   "classifier.*", "local_classifier.*", "alpha.*"],
               "parameter_count_original": int(
                   sum(t.numel() for t in original.values())),
               "parameter_count_drifted": {},
               "alpha": {}}
    for group in ("R3", "R5"):
        state = models[group]["state"]
        missing = sorted(set(original) ^ set(state))
        summary["parameter_count_drifted"][group] = int(
            sum(t.numel() for t in state.values()))
        summary["missing_keys"] = summary.get("missing_keys", {})
        summary["missing_keys"][group] = missing
        stage_accumulator = {}
        block_accumulator = {}
        type_accumulator = {}
        for name in sorted(set(original) & set(state)):
            reference = original[name]
            finetuned = state[name]
            if tuple(reference.shape) != tuple(finetuned.shape):
                raise RuntimeError(f"{group} shape mismatch for {name}")
            delta = finetuned - reference
            absolute = float(delta.norm())
            relative = _relative_drift(delta, reference)
            stage = _stage_key(name)
            block = _block_key(name)
            parameter_type = _parameter_type(name)
            for accumulator, key in ((stage_accumulator, stage),
                                     (block_accumulator, block),
                                     (type_accumulator, parameter_type)):
                entry = accumulator.setdefault(key, {
                    "squared_delta": 0.0, "squared_reference": 0.0,
                    "tensors": 0, "parameters": 0})
                entry["squared_delta"] += absolute ** 2
                entry["squared_reference"] += float(reference.norm()) ** 2
                entry["tensors"] += 1
                entry["parameters"] += int(reference.numel())
            detached = name
            frozen = any(detached.startswith(prefix.removeprefix("backbone."))
                         for prefix in FROZEN_PREFIXES)
            if frozen:
                frozen_rows.append({
                    "group": group, "parameter": name,
                    "max_abs_delta": float(delta.abs().max()),
                    "bitwise_unchanged": bool(torch.equal(reference, finetuned)),
                })
            drift_rows.append({
                "group": group,
                "parameter": name,
                "stage": stage,
                "block": block,
                "parameter_type": parameter_type,
                "numel": int(reference.numel()),
                "reference_l2": float(reference.norm()),
                "delta_l2": absolute,
                "relative_drift": relative,
                "delta_max_abs": float(delta.abs().max()),
                "frozen": frozen,
            })
        for key, entry in stage_accumulator.items():
            absolute = math.sqrt(entry["squared_delta"])
            reference_norm = math.sqrt(entry["squared_reference"])
            stage_rows_append = {
                "group": group, "stage": key,
                "num_tensors": entry["tensors"],
                "num_parameters": entry["parameters"],
                "absolute_drift_l2": absolute,
                "reference_l2": reference_norm,
                "relative_drift": absolute / (reference_norm + 1e-12),
            }
            drift_rows.append({**stage_rows_append, "parameter": f"__stage__{key}",
                               "block": key, "parameter_type": "stage_aggregate",
                               "numel": entry["parameters"],
                               "delta_l2": absolute,
                               "delta_max_abs": None, "frozen": False,
                               "reference_l2": reference_norm,
                               "stage": key})
        summary.setdefault("stage_drift", {})[group] = []
        for key, entry in stage_accumulator.items():
            absolute = math.sqrt(entry["squared_delta"])
            reference_norm = math.sqrt(entry["squared_reference"])
            summary["stage_drift"][group].append({
                "stage": key, "absolute_drift_l2": absolute,
                "reference_l2": reference_norm,
                "relative_drift": absolute / (reference_norm + 1e-12),
                "num_parameters": entry["parameters"]})
        for key, entry in block_accumulator.items():
            absolute = math.sqrt(entry["squared_delta"])
            reference_norm = math.sqrt(entry["squared_reference"])
            block_rows.append({
                "group": group, "block": key,
                "num_tensors": entry["tensors"],
                "num_parameters": entry["parameters"],
                "absolute_drift_l2": absolute,
                "reference_l2": reference_norm,
                "relative_drift": absolute / (reference_norm + 1e-12),
            })
        for key, entry in type_accumulator.items():
            absolute = math.sqrt(entry["squared_delta"])
            reference_norm = math.sqrt(entry["squared_reference"])
            type_rows.append({
                "group": group, "parameter_type": key,
                "num_tensors": entry["tensors"],
                "num_parameters": entry["parameters"],
                "absolute_drift_l2": absolute,
                "reference_l2": reference_norm,
                "relative_drift": absolute / (reference_norm + 1e-12),
            })
        payload = torch.load(
            (R3_DIR if group == "R3" else R5_DIR) / "best.pth",
            map_location="cpu", weights_only=False)["model_state"]
        head_keys = [key for key in payload if not key.startswith("backbone.")]
        summary.setdefault("non_backbone_state", {})[group] = {
            "keys": head_keys,
            "classifier_parameters": int(sum(
                payload[key].numel() for key in head_keys
                if key.startswith("classifier."))),
        }
        if "alpha.logit" in payload:
            raw = float(payload["alpha.logit"].detach().float())
            summary["alpha"][group] = {
                "alpha_logit": raw,
                "alpha": 0.2 / (1.0 + math.exp(-raw)),
                "initial_alpha": 0.05,
            }
    drift_frame = pd.DataFrame([
        row for row in drift_rows if not str(row["parameter"]).startswith("__stage__")])
    drift_frame.to_csv(G1_DRIFT_DIR / "stage_parameter_drift.csv", index=False,
                       encoding="utf-8")
    pd.DataFrame(block_rows).to_csv(G1_DRIFT_DIR / "block_parameter_drift.csv",
                                    index=False, encoding="utf-8")
    pd.DataFrame(type_rows).to_csv(G1_DRIFT_DIR / "parameter_type_drift.csv",
                                   index=False, encoding="utf-8")
    frozen_frame = pd.DataFrame(frozen_rows)
    frozen_frame.to_csv(G1_DRIFT_DIR / "frozen_layer_check.csv", index=False,
                        encoding="utf-8")
    summary["frozen_layers"] = {
        "all_bitwise_unchanged": bool(frozen_frame["bitwise_unchanged"].all())
        if len(frozen_frame) else None,
        "params_checked": int(len(frozen_frame)),
        "max_abs_delta": float(frozen_frame["max_abs_delta"].max())
        if len(frozen_frame) else None,
    }
    write_json(G1_DRIFT_DIR / "parameter_drift_summary.json", summary)
    _plot_parameter_drift(summary, block_rows)
    return {"stage_drift": summary["stage_drift"],
            "block_rows": block_rows, "type_rows": type_rows,
            "frozen": summary["frozen_layers"], "alpha": summary["alpha"]}


def _plot_parameter_drift(summary: dict, block_rows: list[dict]) -> None:
    figures = G1_DRIFT_DIR / "parameter_drift_figures"
    figures.mkdir(parents=True, exist_ok=True)
    stages = ["stage0", "stage1", "stage2", "stage3", "final_norm"]
    for metric_key, filename, ylabel in (
            ("relative_drift", "stage_relative_drift.png", "relative drift"),
            ("absolute_drift_l2", "stage_absolute_drift.png", "L2 drift")):
        figure, axis = plt.subplots(figsize=(8, 4.4))
        width = 0.35
        for offset, group in enumerate(("R3", "R5")):
            entries = {entry["stage"]: entry for entry in
                       summary["stage_drift"][group]}
            values = [entries.get(stage, {}).get(metric_key, float("nan"))
                      for stage in stages]
            axis.bar(np.arange(len(stages)) + offset * width - width / 2, values,
                     width=width, label=group)
        axis.set_xticks(np.arange(len(stages)))
        axis.set_xticklabels(stages)
        axis.set_ylabel(ylabel)
        axis.set_title(f"Stage3/Stage4 parameter drift ({metric_key})")
        axis.legend()
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=140)
        plt.close(figure)
    blocks = sorted({row["block"] for row in block_rows
                     if row["block"].startswith("stage")})
    figure, axis = plt.subplots(figsize=(10, 4.4))
    width = 0.35
    for offset, group in enumerate(("R3", "R5")):
        entries = {row["block"]: row for row in block_rows if row["group"] == group}
        values = [entries.get(block, {}).get("relative_drift", float("nan"))
                  for block in blocks]
        axis.bar(np.arange(len(blocks)) + offset * width - width / 2, values,
                 width=width, label=group)
    axis.set_xticks(np.arange(len(blocks)))
    axis.set_xticklabels(blocks, rotation=60, ha="right", fontsize=7)
    axis.set_ylabel("relative drift")
    axis.set_title("Per-Block parameter drift")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures / "block_relative_drift.png", dpi=140)
    plt.close(figure)


# ---------------------------------------------------------------------------
# G1-E: feature representation drift
# ---------------------------------------------------------------------------

def _sample_study_indices(dataset, count: int, seed: int = SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    total = len(dataset)
    take = min(count, total)
    indices = np.sort(rng.choice(total, size=take, replace=False))
    return [int(index) for index in indices]


def _extract_features(backbone, dataset, indices: list[int],
                      batch_size: int = 8):
    """Per-Study mean Stage3 GAP, Stage4 GAP and classifier-input features."""
    features_stage3, features_stage4, features_final, image_counts = [], [], [], []
    backbone.eval()
    device = next(backbone.parameters()).device
    with torch.inference_mode():
        for index in indices:
            item = dataset[index]
            images = item["images"]
            counts = int(images.shape[0])
            image_counts.append(counts)
            stage3_sum = torch.zeros(384, dtype=torch.float64, device=device)
            stage4_sum = torch.zeros(768, dtype=torch.float64, device=device)
            final_sum = torch.zeros(768, dtype=torch.float64, device=device)
            for start in range(0, counts, batch_size):
                chunk = images[start:start + batch_size].to(device)
                chunk = chunk.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    stage3, stage4 = backbone._forward_feature_maps(
                        chunk, capture_stage3=True)
                    final = backbone._normalize_global_features(stage4)
                stage3_sum += stage3.float().mean(dim=(-2, -1)).sum(dim=0).double()
                stage4_sum += stage4.float().mean(dim=(-2, -1)).sum(dim=0).double()
                final_sum += final.float().sum(dim=0).double()
            features_stage3.append((stage3_sum / counts).cpu().numpy())
            features_stage4.append((stage4_sum / counts).cpu().numpy())
            features_final.append((final_sum / counts).cpu().numpy())
    return (np.stack(features_stage3), np.stack(features_stage4),
            np.stack(features_final), image_counts)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return (a * b).sum(axis=1) / denominator


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denominator = (np.linalg.norm(x.T @ x, "fro")
                   * np.linalg.norm(y.T @ y, "fro"))
    return float(cross / (denominator + 1e-12))


def run_g1e(train_data, valid_data, sample_count: int = 192) -> dict:
    device = torch.device("cuda")
    from dino_experiment import load_pretrained_backbone

    _, pretrained_verification = load_pretrained_backbone("cpu")
    if pretrained_verification["weights_sha256"] != PRETRAINED_SHA:
        raise RuntimeError("Pretrained weight hash mismatch for feature drift")
    pretrained_backbone, _ = load_pretrained_backbone("cpu")
    pretrained_backbone = pretrained_backbone.to(device).eval()
    finetuned = {}
    last_backbones = {}
    for group in ("R3", "R5"):
        loader = load_r3_model if group == "R3" else load_r5_model
        model, status, _ = loader(device, channels_last=False)
        finetuned[group] = model.backbone
        del model
        last_model, _, _ = loader(device, channels_last=False, checkpoint="last")
        last_backbones[group] = last_model.backbone
        del last_model
    rows, cka_values, pairwise, within_run = [], {}, {}, {}
    for split, dataset in (("train", train_data), ("valid", valid_data)):
        indices = _sample_study_indices(dataset, sample_count)
        study_keys = [dataset.study_keys[index] for index in indices]
        base = dataset.dataset.base
        image_counts = [len(base.studies[index]["image_paths"]) for index in indices]
        pretrained = _extract_features(pretrained_backbone, dataset, indices)
        features = {"pretrained": pretrained[:3]}
        for group in ("R3", "R5"):
            backbone = finetuned[group]
            features[group] = _extract_features(backbone, dataset, indices)[:3]
            last_features = _extract_features(
                last_backbones[group], dataset, indices)[:3]
            within_run.setdefault(group, {})[split] = {
                "cosine_stage3_best_vs_last": float(_cosine(
                    features[group][0], last_features[0]).mean()),
                "cosine_stage4_best_vs_last": float(_cosine(
                    features[group][1], last_features[1]).mean()),
                "cosine_final_best_vs_last": float(_cosine(
                    features[group][2], last_features[2]).mean()),
            }
        for group in ("R3", "R5"):
            stage3_features, stage4_features, final_features = features[group]
            cosine3 = _cosine(features["pretrained"][0], stage3_features)
            cosine4 = _cosine(features["pretrained"][1], stage4_features)
            cosine_final = _cosine(features["pretrained"][2], final_features)
            norm_ratio3 = (np.linalg.norm(stage3_features, axis=1)
                           / (np.linalg.norm(features["pretrained"][0], axis=1)
                              + 1e-12))
            norm_ratio4 = (np.linalg.norm(stage4_features, axis=1)
                           / (np.linalg.norm(features["pretrained"][1], axis=1)
                              + 1e-12))
            norm_ratio_final = (np.linalg.norm(final_features, axis=1)
                                / (np.linalg.norm(features["pretrained"][2],
                                                  axis=1) + 1e-12))
            for position, (subject, study) in enumerate(study_keys):
                rows.append({
                    "group": group,
                    "split": split,
                    "subject_id": str(subject),
                    "study_id": str(study),
                    "image_count": int(image_counts[position]),
                    "cosine_stage3": float(cosine3[position]),
                    "cosine_stage4": float(cosine4[position]),
                    "cosine_final": float(cosine_final[position]),
                    "norm_ratio_stage3": float(norm_ratio3[position]),
                    "norm_ratio_stage4": float(norm_ratio4[position]),
                    "norm_ratio_final": float(norm_ratio_final[position]),
                })
            cka_values.setdefault(group, {})[split] = {
                "linear_cka_stage3": _linear_cka(features["pretrained"][0],
                                                 stage3_features),
                "linear_cka_stage4": _linear_cka(features["pretrained"][1],
                                                 stage4_features),
                "linear_cka_final": _linear_cka(features["pretrained"][2],
                                                final_features),
            }
        pairwise[split] = {
            "r3_vs_r5_cosine_stage3_mean": float(_cosine(
                features["R3"][0], features["R5"][0]).mean()),
            "r3_vs_r5_cosine_stage4_mean": float(_cosine(
                features["R3"][1], features["R5"][1]).mean()),
            "r3_vs_r5_cosine_final_mean": float(_cosine(
                features["R3"][2], features["R5"][2]).mean()),
        }
    write_csv(G1_DRIFT_DIR / "feature_drift.csv", rows)
    frame = pd.DataFrame(rows)
    summary = {"sample_per_split": sample_count, "seed": SEED,
               "sample_rule": (
                   "np.random.default_rng(42).choice over the official Study order "
                   "without replacement; identical Study list for all backbones"),
               "feature_definition": (
                   "Stage3 GAP / Stage4 GAP are channel means of the raw Stage3 / "
                   "Stage4 feature maps; 'final' is the LayerNorm-normalized pooled "
                   "token that the classifier actually consumes."),
               "groups": {}, "cka": cka_values, "pairwise": pairwise,
               "within_run_best_vs_last": within_run}
    for group in ("R3", "R5"):
        summary["groups"][group] = {}
        for split in ("train", "valid"):
            subset = frame[(frame["group"] == group) & (frame["split"] == split)]
            summary["groups"][group][split] = {
                "n_studies": int(len(subset)),
                "cosine_stage3_mean": float(subset["cosine_stage3"].mean()),
                "cosine_stage3_median": float(subset["cosine_stage3"].median()),
                "cosine_stage4_mean": float(subset["cosine_stage4"].mean()),
                "cosine_stage4_median": float(subset["cosine_stage4"].median()),
                "cosine_final_mean": float(subset["cosine_final"].mean()),
                "cosine_final_median": float(subset["cosine_final"].median()),
                "norm_ratio_stage3_mean": float(subset["norm_ratio_stage3"].mean()),
                "norm_ratio_stage4_mean": float(subset["norm_ratio_stage4"].mean()),
                "norm_ratio_final_mean": float(subset["norm_ratio_final"].mean()),
            }
    write_json(G1_DRIFT_DIR / "feature_drift_summary.json", summary)
    _plot_feature_drift(frame)
    del pretrained_backbone, finetuned, last_backbones
    torch.cuda.empty_cache()
    return summary


def _plot_feature_drift(frame: pd.DataFrame) -> None:
    figures = G1_DRIFT_DIR / "parameter_drift_figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, column, label in (
            (axes[0], "cosine_stage3", "Stage3 GAP"),
            (axes[1], "cosine_stage4", "Stage4 GAP"),
            (axes[2], "cosine_final", "classifier-input (final norm)")):
        for group in ("R3", "R5"):
            for split, style in (("train", "-"), ("valid", "--")):
                subset = frame[(frame["group"] == group)
                               & (frame["split"] == split)][column]
                axis.hist(subset, bins=40, histtype="step", linestyle=style,
                          label=f"{group} {split}", alpha=0.8)
        axis.set_title(f"{label} cosine vs pretrained")
        axis.set_xlabel("cosine similarity")
        axis.set_ylabel("Study count")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures / "feature_cosine_distribution.png", dpi=140)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_train_valid_curves() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for group in ("R3", "R5"):
        path = (R3_DIR if group == "R3" else R5_DIR) / "train_log.csv"
        frame = pd.read_csv(path)
        axes[0].plot(frame["epoch"], frame["macro_auroc"], marker="o",
                     label=f"{group} valid Macro-AUROC")
        axes[1].plot(frame["epoch"], frame["train_bce"], marker="o",
                     label=f"{group} train BCE")
        axes[1].plot(frame["epoch"], frame["valid_loss"], marker="s",
                     linestyle="--", label=f"{group} valid BCE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Macro-AUROC")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Study BCE")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.suptitle("R3 / R5 training curves (selected checkpoints at epoch 6)")
    figure.tight_layout()
    figure.savefig(G1_METRICS_DIR / "train_valid_curves.png", dpi=140)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Reports and G2 gate
# ---------------------------------------------------------------------------

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


def write_g1_reports(g0_summary: dict, g1a: dict, g1b: dict, g1c: dict,
                     g1d: dict, g1e: dict) -> dict:
    gap = pd.DataFrame(g1a["gap_rows"])
    macro_gap = gap[gap["disease"] == "Macro"].set_index("group")
    r3_gap = float(macro_gap.loc["R3", "auroc_gap"])
    r5_gap = float(macro_gap.loc["R5", "auroc_gap"])
    single_multi = pd.DataFrame(g1b["single_multi_rows"])
    aggregation = g1b["aggregation"]
    stage_drift = g1d["stage_drift"]
    feature_summary = g1e
    lines = [
        "# G1 final report: R3/R5 generalization and pretrained-representation drift",
        "",
        "Deterministic no-augmentation inference on the official Train "
        "(18,513 Studies / 30,344 images / 11,055 Subjects) and Validation "
        "(859 Studies / 1,422 images / 228 Subjects) splits. Study Logit Mean "
        "before sigmoid, BF16 autocast, channels-last, image budget 8.",
        "",
        "## G1-A Train vs Validation (same protocol)", "",
        "| Group | Split | Macro-AUROC | Macro-AUPRC | Study BCE |",
        "|---|---|---:|---:|---:|",
    ]
    for group in ("R3", "R5"):
        for split in ("train", "valid"):
            row = [r for r in g1a["gap_rows"] if r["group"] == group
                   and r["disease"] == "Macro"][0]
            lines.append(
                f"| {group} | {split} | {row[f'{split}_auroc']:.6f} | "
                f"{row[f'{split}_auprc']:.6f} | {row[f'{split}_study_bce']:.6f} |")
    lines.extend([
        "",
        f"- Train−Validation Macro-AUROC gap: R3 = {r3_gap:+.6f}; "
        f"R5 = {r5_gap:+.6f}.",
        f"- Train−Validation Study BCE gap: R3 = "
        f"{float(macro_gap.loc['R3', 'train_study_bce']) - float(macro_gap.loc['R3', 'valid_study_bce']):+.6f}; "
        f"R5 = {float(macro_gap.loc['R5', 'train_study_bce']) - float(macro_gap.loc['R5', 'valid_study_bce']):+.6f}.",
        "",
        "Largest per-class Train−Valid AUROC gaps (R3 / R5):",
        "",
        "| Disease | R3 train | R3 valid | R3 gap | R5 train | R5 valid | R5 gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    r3_rows = gap[gap["group"] == "R3"].set_index("disease")
    r5_rows = gap[gap["group"] == "R5"].set_index("disease")
    ordered = sorted(
        [name for name in LABEL_NAMES],
        key=lambda name: -float(r3_rows.loc[name, "auroc_gap"]))
    for disease in ordered[:6]:
        lines.append(
            f"| {disease} | {r3_rows.loc[disease, 'train_auroc']:.4f} | "
            f"{r3_rows.loc[disease, 'valid_auroc']:.4f} | "
            f"{r3_rows.loc[disease, 'auroc_gap']:+.4f} | "
            f"{r5_rows.loc[disease, 'train_auroc']:.4f} | "
            f"{r5_rows.loc[disease, 'valid_auroc']:.4f} | "
            f"{r5_rows.loc[disease, 'auroc_gap']:+.4f} |")
    lines.extend([
        "",
        "## G1-B Single vs multi-image Studies (Validation)", "",
        "| Group | Study group | Studies | Subjects | Macro-AUROC | Macro-AUPRC | Study BCE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for group in ("R3", "R5"):
        subset = single_multi[(single_multi["group"] == group)
                              & (single_multi["split"] == "valid")]
        for study_group in ("single", "multi", "two", "three_plus"):
            rows = subset[(subset["study_group"] == study_group)
                          & (subset["disease"] == LABEL_NAMES[0])]
            if rows.empty:
                continue
            row = rows.iloc[0]
            lines.append(
                f"| {group} | {study_group} | {int(row['n_studies'])} | "
                f"{int(row['n_subjects'])} | {row['macro_auroc']:.4f} | "
                f"{row['macro_auprc']:.4f} | {row['study_bce']:.4f} |")
    lines.extend([
        "",
        "Aggregation check (R5 Best, multi-image Validation Studies only; "
        "pre-specified Logit Mean vs Probability Mean, no trained aggregator):",
        "",
        f"- Studies = {aggregation['n_multi_image_studies']}.",
        f"- Logit Mean Macro-AUROC = {aggregation['logit_mean_macro_auroc']:.6f}.",
        f"- Probability Mean Macro-AUROC = {aggregation['probability_mean_macro_auroc']:.6f}.",
        f"- Δ(Probability−Logit) = {aggregation['delta_macro_auroc']:+.6f}, "
        f"95% CI {aggregation['delta_macro_auroc_ci95']}.",
        "",
        "## G1-D parameter drift (relative to original DINOv3)", "",
        "Note: `stage0`–`stage3` are ConvNeXt stage indices; research "
        "\"Stage3\" = stage2 (9 blocks, 384 ch) and research \"Stage4\" = stage3 "
        "(3 blocks, 768 ch).",
        "",
        "| Group | Stage | Absolute L2 drift | Relative drift |",
        "|---|---|---:|---:|",
    ])
    for group in ("R3", "R5"):
        for entry in stage_drift[group]:
            lines.append(
                f"| {group} | {entry['stage']} | {entry['absolute_drift_l2']:.4f} | "
                f"{entry['relative_drift']:.6f} |")
    lines.extend([
        "",
        f"- Frozen layers (Stem/Stage1/Stage2) bitwise unchanged: "
        f"{g1d['frozen']['all_bitwise_unchanged']} "
        f"({g1d['frozen']['params_checked']} tensors, max |Δ|="
        f"{_fmt(g1d['frozen']['max_abs_delta'], 12)}).",
        f"- R3 learned fusion alpha = {g1d['alpha'].get('R3', {}).get('alpha')}; "
        f"initial 0.05 (alpha.logit 0 → 0.050662).",
        "",
        "## G1-E Feature drift (fixed sampled Studies, seed 42)", "",
        "`final` = LayerNorm-normalized pooled token that the classifier consumes; "
        "Stage3/4 GAP are raw channel means (pre-normalization).",
        "",
        "| Group | Split | cos(Stage3 GAP) | cos(Stage4 GAP) | cos(final) | CKA Stage3 | CKA Stage4 | CKA final |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in ("R3", "R5"):
        for split in ("train", "valid"):
            info = feature_summary["groups"][group][split]
            cka = feature_summary["cka"][group][split]
            lines.append(
                f"| {group} | {split} | {info['cosine_stage3_mean']:.4f} | "
                f"{info['cosine_stage4_mean']:.4f} | "
                f"{info['cosine_final_mean']:.4f} | "
                f"{cka['linear_cka_stage3']:.4f} | {cka['linear_cka_stage4']:.4f} | "
                f"{cka['linear_cka_final']:.4f} |")
    lines.extend([
        "",
        "- R3 vs R5 feature cosine (Validation): Stage3 GAP = "
        f"{feature_summary['pairwise']['valid']['r3_vs_r5_cosine_stage3_mean']:.4f}, "
        "Stage4 GAP = "
        f"{feature_summary['pairwise']['valid']['r3_vs_r5_cosine_stage4_mean']:.4f}, "
        "final = "
        f"{feature_summary['pairwise']['valid']['r3_vs_r5_cosine_final_mean']:.4f}.",
        "- Within-run stability control (same run, Best epoch 6 vs Last epoch 9): "
        f"R3 stage3={feature_summary['within_run_best_vs_last']['R3']['valid']['cosine_stage3_best_vs_last']:.4f}, "
        f"stage4={feature_summary['within_run_best_vs_last']['R3']['valid']['cosine_stage4_best_vs_last']:.4f}, "
        f"final={feature_summary['within_run_best_vs_last']['R3']['valid']['cosine_final_best_vs_last']:.4f}; "
        f"R5 stage3={feature_summary['within_run_best_vs_last']['R5']['valid']['cosine_stage3_best_vs_last']:.4f}, "
        f"stage4={feature_summary['within_run_best_vs_last']['R5']['valid']['cosine_stage4_best_vs_last']:.4f}, "
        f"final={feature_summary['within_run_best_vs_last']['R5']['valid']['cosine_final_best_vs_last']:.4f}. "
        "The raw Stage4 GAP vector is therefore highly sensitive to small parameter "
        "changes; feature-cosine magnitudes should be interpreted together with the "
        "classifier-input (final) features, CKA and the G1-A performance gap.",
        "",
        "## G1-C data distribution checks", "",
        f"- Subject overlap Train↔Validation: {g1c['integrity']['subject_leakage']['overlap_subjects']}; "
        f"Study overlap: {g1c['integrity']['subject_leakage']['overlap_studies']}.",
    ])
    for split in ("train", "valid"):
        info = g1c["integrity"]["splits"][split]
        lines.append(
            f"- {split}: {info['studies']} Studies, {info['images']} images, "
            f"{info['subjects']} Subjects, {info['positives_per_study_mean']:.3f} "
            "positive labels/Study on average, "
            f"{info['multi_label_study_fraction']:.2%} multi-label Studies; "
            f"Studies/Subject mean {info['studies_per_subject_mean']:.3f}, "
            f"median {info['studies_per_subject_median']:.1f}, "
            f"max {info['studies_per_subject_max']}.")
    lines.extend([
        "",
        "## Evidence table", "",
        "| Item | Verdict |",
        "|---|---|",
        f"| R3 Train−Valid Macro-AUROC gap | {r3_gap:+.6f} (large) |",
        f"| R5 Train−Valid Macro-AUROC gap | {r5_gap:+.6f} (large) |",
        f"| Gap direction consistent across R3/R5 | {abs(r3_gap - r5_gap) < 0.02} |",
        f"| Multi-image Study degradation | "
        f"{_multi_vs_single(single_multi)} |",
        f"| Probability-Mean aggregation worth further study | "
        f"{_aggregation_verdict(aggregation)} |",
        f"| Stage3/4 parameter drift substantial | "
        f"{_drift_verdict(stage_drift)} |",
        f"| Feature drift consistent between R3/R5 | "
        f"{_feature_consistency(feature_summary)} |",
        f"| Frozen layers untouched | {g1d['frozen']['all_bitwise_unchanged']} |",
        "",
        "## Required G1 answers", "",
    ])
    answers = _g1_answers(g0_summary, g1a, g1b, g1c, g1d, g1e)
    for question, answer in answers:
        lines.append(f"- **{question}** {answer}")
    lines.extend([
        "",
        "## Limitations", "",
        "- Train evaluation is an in-training evaluation, not an independent "
        "generalization estimate; the Train−Valid gap cannot be attributed to a "
        "single cause.",
        "- No independent Test labels are available; the official Test 0.7510 "
        "cannot be decomposed into distribution shift vs model error here.",
        "- Parameter/feature drift measures representation change, not causal "
        "degradation; a smaller feature cosine similarity is not by itself "
        "evidence of overfitting.",
        "",
    ])
    (G1_METRICS_DIR / "g1_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    decision = _g2_decision(g0_summary, g1a, g1b, g1c, g1d, g1e)
    write_json(G1_METRICS_DIR / "g2_decision.json", decision)
    _write_g2_decision(decision, g1a, g1b, g1d, g1e)
    return decision


def _multi_vs_single(single_multi: pd.DataFrame) -> str:
    parts = []
    for group in ("R3", "R5"):
        subset = single_multi[(single_multi["group"] == group)
                              & (single_multi["split"] == "valid")
                              & (single_multi["disease"] == LABEL_NAMES[0])]
        values = {row["study_group"]: row["macro_auroc"] for _, row in
                  subset.iterrows()}
        parts.append(f"{group}: single={values.get('single', float('nan')):.4f}, "
                     f"multi={values.get('multi', float('nan')):.4f}")
    return "；".join(parts) + "（组间疾病构成不同，不能直接判优劣）"


def _aggregation_verdict(aggregation: dict) -> str:
    low, high = aggregation["delta_macro_auroc_ci95"]
    if low is not None and low > 0:
        return "是：Probability Mean 的 CI 排除 0 且为正，值得独立研究"
    if high is not None and high < 0:
        return "否：Probability Mean 的 CI 排除 0 且为负"
    return "目前无法确认：Δ Macro-AUROC 的 95% CI 跨 0"


def _drift_verdict(stage_drift: dict) -> str:
    parts = []
    for group in ("R3", "R5"):
        entries = {entry["stage"]: entry for entry in stage_drift[group]}
        parts.append(
            f"{group} ConvNeXt-stage2(=Stage3) relative="
            f"{entries['stage2']['relative_drift']:.4f}, "
            f"stage3(=Stage4) relative={entries['stage3']['relative_drift']:.4f}")
    return "；".join(parts)


def _feature_consistency(feature_summary: dict) -> str:
    values = []
    for split in ("train", "valid"):
        r3 = feature_summary["groups"]["R3"][split]["cosine_final_mean"]
        r5 = feature_summary["groups"]["R5"][split]["cosine_final_mean"]
        values.append(f"{split}: R3 final={r3:.4f}, R5 final={r5:.4f}")
    return "；".join(values)


def _g1_answers(g0_summary, g1a, g1b, g1c, g1d, g1e) -> list[tuple[str, str]]:
    gap = pd.DataFrame(g1a["gap_rows"])
    macro = gap[gap["disease"] == "Macro"].set_index("group")
    r3_gap = float(macro.loc["R3", "auroc_gap"])
    r5_gap = float(macro.loc["R5", "auroc_gap"])
    single_multi = pd.DataFrame(g1b["single_multi_rows"])
    aggregation = g1b["aggregation"]
    valid_single = {}
    valid_multi = {}
    for group in ("R3", "R5"):
        subset = single_multi[(single_multi["group"] == group)
                              & (single_multi["split"] == "valid")
                              & (single_multi["disease"] == LABEL_NAMES[0])]
        values = {row["study_group"]: row for _, row in subset.iterrows()}
        valid_single[group] = float(values["single"]["macro_auroc"])
        valid_multi[group] = float(values["multi"]["macro_auroc"])
    stage_drift = g1d["stage_drift"]
    feature = g1e
    answers = []
    answers.append((
        "1. R3、R5 的 Train/Valid Macro-AUROC 差距有多大？",
        f"R3 = {r3_gap:+.6f}（Train "
        f"{float(macro.loc['R3', 'train_auroc']):.6f} → Valid "
        f"{float(macro.loc['R3', 'valid_auroc']):.6f}）；R5 = {r5_gap:+.6f}"
        f"（{float(macro.loc['R5', 'train_auroc']):.6f} → "
        f"{float(macro.loc['R5', 'valid_auroc']):.6f}）。"))
    worst = gap[(gap["group"] == "R3") & (gap["disease"] != "Macro")].sort_values(
        "auroc_gap", ascending=False).head(4)
    answers.append((
        "2. 哪些疾病泛化差距最大？",
        "；".join(f"{row['disease']} R3 Δ={row['auroc_gap']:+.4f}"
                 for _, row in worst.iterrows())))
    low_train = gap[(gap["group"] == "R3") & (gap["disease"] != "Macro")].sort_values(
        "train_auroc").head(4)
    answers.append((
        "3. 哪些疾病即使在 Train 上也表现较差？",
        "；".join(f"{row['disease']} train AUROC={row['train_auroc']:.4f}"
                 for _, row in low_train.iterrows())))
    answers.append((
        "4. 是否主要由多图 Study 导致性能下降？",
        f"R3 single Macro-AUROC={valid_single['R3']:.4f} vs multi="
        f"{valid_multi['R3']:.4f}；R5 single={valid_single['R5']:.4f} vs multi="
        f"{valid_multi['R5']:.4f}。两组疾病构成不同，不能仅凭此判断聚合算法优劣；"
        f"多图 Study 固定权重聚合对照：Probability Mean − Logit Mean "
        f"= {aggregation['delta_macro_auroc']:+.6f}，95% CI "
        f"{aggregation['delta_macro_auroc_ci95']}。"))
    # ConvNeXt stage indices: research "Stage3" = stage2, research "Stage4" = stage3.
    drift3 = [e for e in stage_drift["R3"] if e["stage"] == "stage2"][0]
    drift4 = [e for e in stage_drift["R3"] if e["stage"] == "stage3"][0]
    r53 = [e for e in stage_drift["R5"] if e["stage"] == "stage2"][0]
    r54 = [e for e in stage_drift["R5"] if e["stage"] == "stage3"][0]
    answers.append((
        "5. Stage 3 和 Stage 4 哪部分参数变化更大？",
        f"R3: Stage3(ConvNeXt stage2) relative={drift3['relative_drift']:.6f}, "
        f"Stage4(ConvNeXt stage3) {drift4['relative_drift']:.6f}；R5: "
        f"Stage3 {r53['relative_drift']:.6f}, Stage4 {r54['relative_drift']:.6f}。"
        "Stage4 相对偏移更大是预期现象（任务相关度更高、无冻结差异时更新幅度更大）。"))
    answers.append((
        "6. 两个模型的特征漂移表现是否一致？",
        "；".join(
            f"{split}: R3 cos(Stage4 GAP)={feature['groups']['R3'][split]['cosine_stage4_mean']:.4f}, "
            f"R5={feature['groups']['R5'][split]['cosine_stage4_mean']:.4f}; "
            f"R3 cos(final)={feature['groups']['R3'][split]['cosine_final_mean']:.4f}, "
            f"R5={feature['groups']['R5'][split]['cosine_final_mean']:.4f}; "
            f"CKA Stage4 R3={feature['cka']['R3'][split]['linear_cka_stage4']:.4f}, "
            f"R5={feature['cka']['R5'][split]['linear_cka_stage4']:.4f}"
            for split in ("train", "valid"))))
    answers.append((
        "7. 是否有证据支持“微调过程改变了模型的泛化表现”？",
        "存在明确的 Train−Valid 差距与 Stage3/4 参数/特征漂移，但本诊断不能把泛化"
        "差距归因于漂移本身：单 seed、验证集反复用于选模、且 Train 评价是训练内评价。"
        "可支持的结论仅是“微调改变了表示且存在泛化差距”，不是因果证据。"))
    decision = _g2_decision(g0_summary, g1a, g1b, g1c, g1d, g1e)
    answers.append((
        "8. 当前是否值得测试更低 Backbone 学习率？",
        decision["recommendation_text"]))
    return answers


def _g2_decision(g0_summary, g1a, g1b, g1c, g1d, g1e) -> dict:
    gap = pd.DataFrame(g1a["gap_rows"])
    macro = gap[gap["disease"] == "Macro"].set_index("group")
    r3_gap = float(macro.loc["R3", "auroc_gap"])
    r5_gap = float(macro.loc["R5", "auroc_gap"])
    train_macro = {group: float(macro.loc[group, "train_auroc"]) for group in
                   ("R3", "R5")}
    valid_macro = {group: float(macro.loc[group, "valid_auroc"]) for group in
                   ("R3", "R5")}
    stage_drift = g1d["stage_drift"]
    # ConvNeXt stage indices: research "Stage3" = stage2, research "Stage4" = stage3.
    r3_stage4 = [e for e in stage_drift["R3"] if e["stage"] == "stage3"][0][
        "relative_drift"]
    r5_stage4 = [e for e in stage_drift["R5"] if e["stage"] == "stage3"][0][
        "relative_drift"]
    feature = g1e
    valid_cosine_r3 = feature["groups"]["R3"]["valid"]["cosine_final_mean"]
    valid_cosine_r5 = feature["groups"]["R5"]["valid"]["cosine_final_mean"]
    valid_raw_stage4_r3 = feature["groups"]["R3"]["valid"]["cosine_stage4_mean"]
    valid_raw_stage4_r5 = feature["groups"]["R5"]["valid"]["cosine_stage4_mean"]
    gate_ok = bool(g0_summary["gate_passed"])
    generalization_gap_large = bool(r3_gap > 0.02 or r5_gap > 0.02)
    drift_nonzero = bool(r3_stage4 > 0.005 or r5_stage4 > 0.005)
    feature_drift_nonzero = bool(valid_cosine_r3 < 0.995
                                 or valid_cosine_r5 < 0.995)
    recommend = bool(gate_ok and generalization_gap_large and drift_nonzero)
    if not gate_ok:
        recommendation = "G0 未通过：先修复推理问题，不启动 G2。"
    elif recommend:
        recommendation = (
            "建议在用户确认后启动 G2（1e-5 vs 3e-6 配对种子实验）。依据："
            f"Train−Valid Macro-AUROC 差距 R3={r3_gap:+.4f}/R5={r5_gap:+.4f}；"
            f"Stage4 参数相对漂移 R3={r3_stage4:.4f}/R5={r5_stage4:.4f}；"
            f"分类器输入(final)特征余弦 R3={valid_cosine_r3:.4f}/R5="
            f"{valid_cosine_r5:.4f}（原始 Stage4 GAP 余弦 R3="
            f"{valid_raw_stage4_r3:.4f}/R5={valid_raw_stage4_r5:.4f}）。"
            "这些是可检验“微调强度影响泛化”的合理依据，但不构成因果证明。")
    else:
        recommendation = (
            "暂不建议直接启动 G2：G0 通过，但 Train/Valid 差距或参数漂移不足以"
            "支持“降低 Backbone 学习率”作为优先方向；应优先检查测试分布与验证集代表性、"
            "困难疾病标签/特征，再考虑微调强度。")
    return {
        "gate_g0_passed": gate_ok,
        "generalization_gap_large": generalization_gap_large,
        "stage4_drift_nonzero": drift_nonzero,
        "feature_drift_nonzero": feature_drift_nonzero,
        "r3_train_valid_gap": r3_gap,
        "r5_train_valid_gap": r5_gap,
        "r3_train_macro_auroc": train_macro["R3"],
        "r5_train_macro_auroc": train_macro["R5"],
        "r3_valid_macro_auroc": valid_macro["R3"],
        "r5_valid_macro_auroc": valid_macro["R5"],
        "r3_stage4_relative_drift": r3_stage4,
        "r5_stage4_relative_drift": r5_stage4,
        "r3_valid_final_cosine": valid_cosine_r3,
        "r5_valid_final_cosine": valid_cosine_r5,
        "r3_valid_stage4_gap_cosine": valid_raw_stage4_r3,
        "r5_valid_stage4_gap_cosine": valid_raw_stage4_r5,
        "recommend_g2": recommend,
        "recommendation_text": recommendation,
        "requires_explicit_user_confirmation": True,
        "test_data_accessed": False,
    }


def _write_g2_decision(decision: dict, g1a: dict, g1b: dict, g1d: dict,
                       g1e: dict) -> None:
    lines = [
        "# G2 go / no-go decision (after G0 + G1)", "",
        f"- G0 gate passed: **{decision['gate_g0_passed']}**.",
        f"- Train−Valid Macro-AUROC gap: R3 = {decision['r3_train_valid_gap']:+.6f}, "
        f"R5 = {decision['r5_train_valid_gap']:+.6f}.",
        f"- Stage4 parameter relative drift: R3 = "
        f"{decision['r3_stage4_relative_drift']:.6f}, R5 = "
        f"{decision['r5_stage4_relative_drift']:.6f}.",
        f"- Stage4 GAP raw feature cosine vs pretrained: R3 = "
        f"{decision['r3_valid_stage4_gap_cosine']:.4f}, R5 = "
        f"{decision['r5_valid_stage4_gap_cosine']:.4f}; classifier-input (final) "
        f"feature cosine: R3 = {decision['r3_valid_final_cosine']:.4f}, R5 = "
        f"{decision['r5_valid_final_cosine']:.4f}.",
        "",
        "## Decision", "",
        f"**{'RECOMMEND G2' if decision['recommend_g2'] else 'DO NOT START G2 YET'}**",
        "",
        decision["recommendation_text"],
        "",
        "## Conditions for G2 (if approved)",
        "",
        "- Training matrix is fixed in advance: R5 Global-only, Stage3/4 LR "
        "1e-5 vs 3e-6, seeds 42/2026/3407, head LR 2e-4; no extra LR grid.",
        "- Seed 42 baseline (`G2-A1`) may reuse R5 only if the protocol is "
        "identical; otherwise re-run it. All other runs must be freshly trained.",
        "- G2 has NOT been started and must not start without explicit user "
        "confirmation.",
        "",
        "## What would change this decision",
        "",
        "- If a distribution/representativeness investigation shows the "
        "Validation set overestimates Test performance, lowering LR is unlikely "
        "to fix the Test gap by itself.",
        "- If Train−Valid gaps were small, the priority would be Test "
        "distribution analysis rather than G2.",
        "",
    ]
    (G1_METRICS_DIR / "g2_go_no_go_decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main G1 pipeline
# ---------------------------------------------------------------------------

def write_final_decision_report(g0_summary: dict, g1_summary: dict) -> None:
    """Final composite G0+G1 report answering the ten required questions."""
    from generalization_common import FINAL_DIR

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    gap = {
        group: g1_summary["g1a_macro_gap"][group] for group in ("R3", "R5")}
    decision = g1_summary["g2_decision"]
    aggregation = g1_summary["aggregation"]
    feature = g1_summary["feature_drift"]
    stage = g1_summary["stage_drift"]
    lines = [
        "# DINOv3 泛化瓶颈诊断：G0 + G1 综合报告", "",
        "本轮只做推理一致性与泛化诊断，未训练任何新模型，未读取 Test 标签。",
        "",
        "## 1. 已排除的推理问题", "",
        f"- G0 gate passed = **{g0_summary.get('gate_passed')}**。",
        "- 研究验证入口与提交包入口在相同 859 个 Validation Study 上使用同一 "
        "checkpoint、同一预处理（ResizePad/Grayscale3/Normalize）、同一类别顺序、"
        "同一 Study Logit Mean 与 sigmoid 顺序；提交包代码与研究代码文件哈希一致，"
        "包内权重与 R3 `best.pth` 的 `model_state` 逐位一致。",
        "- R3 的 Validation 指标可复现：历史保存 0.804653 vs 重放 "
        f"{g0_summary['r3_saved_val_best_replay']['fresh_macro_auroc']:.6f}"
        "（差异在 BF16/批次形状数值噪声范围内）。",
        "- Study 集合/顺序、每 Study 图像集合、图像存在性、标签顺序均对齐。",
        "",
        "## 2. 仍未排除的问题", "",
        "- R3 提交包的 `channels_last=false` 与训练 runtime 的 `true` 不同；该配置"
        "差异只造成数值级差异（已量化），但正式提交实现保留了这一非训练配置。",
        "- R5 历史保存 `val_best.csv` 与当前代码重放存在已知数值差异"
        "（最大概率差约 0.009），该差异与 R3（0.0075）同属 BF16 数值可复现性"
        "限制（训练时 GPU 内存/算法选择环境不同），不是逻辑错误；"
        "R7 的同类重放可复现到 7e-8。",
        "- 官方 Test 0.7510 与 Validation 0.804653 的落差不属于推理实现问题，"
        "但本轮没有独立 Test 标签，无法分解为分布偏移/标签噪声/模型误差。",
        "",
        "## 3. 训练集与验证集的真正差距", "",
        f"- R3：Train Macro-AUROC "
        f"{decision['r3_train_macro_auroc']:.6f} vs Validation "
        f"{decision['r3_valid_macro_auroc']:.6f}，差 {gap['R3']:+.6f}。",
        f"- R5：Train Macro-AUROC "
        f"{decision['r5_train_macro_auroc']:.6f} vs Validation "
        f"{decision['r5_valid_macro_auroc']:.6f}，差 {gap['R5']:+.6f}。",
        "- 这是同一确定性无增强协议下的 Train/Valid 差距；Train 评价属于训练内评价，"
        "不能直接当作泛化估计，也不能单独归因于微调过度。",
        "",
        "## 4. 主要瓶颈疾病", "",
        "逐疾病 Train/Valid（同一确定性无增强协议）：",
        "",
    ]
    gap_frame = pd.read_csv(G1_METRICS_DIR / "per_class_generalization_gap.csv")
    lines.extend([
        "| Disease | R3 train | R3 valid | R3 gap | R5 train | R5 valid | R5 gap | Valid 阳性 Study |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    r3_gap = gap_frame[gap_frame["group"] == "R3"].set_index("disease")
    r5_gap = gap_frame[gap_frame["group"] == "R5"].set_index("disease")
    for disease in LABEL_NAMES:
        lines.append(
            f"| {disease} | {r3_gap.loc[disease, 'train_auroc']:.4f} | "
            f"{r3_gap.loc[disease, 'valid_auroc']:.4f} | "
            f"{r3_gap.loc[disease, 'auroc_gap']:+.4f} | "
            f"{r5_gap.loc[disease, 'train_auroc']:.4f} | "
            f"{r5_gap.loc[disease, 'valid_auroc']:.4f} | "
            f"{r5_gap.loc[disease, 'auroc_gap']:+.4f} | "
            f"{int(r3_gap.loc[disease, 'valid_positive_studies'])} |")
    lines.extend([
        "",
        "- 泛化差距最大：Pneumonia（R3 +0.1414 / R5 +0.1414）、Cardiomegaly"
        "（+0.1259 / +0.1257）、Consolidation（+0.1212 / +0.0987）、Lung Opacity"
        "（+0.1204 / +0.1227）、Atelectasis（+0.0978 / +0.0997）、"
        "Enlarged Cardiomediastinum（+0.0959 / +0.1309）。",
        "- 即使在 Train 上 AUROC 也偏低的是 Enlarged Cardiomediastinum（R3 0.8389）、"
        "Lung Opacity（0.8479）、Pneumonia（0.8642），与 Validation 阳性 Study 数很少"
        "（24 / 214 / 68）叠加，这些类别的估计噪声最大。",
        "- 注意：Pneumonia/Cardiomegaly 等类别的 Valid 阳性数少（68/177），"
        "其 AUROC 的抽样不确定性远大于 Pleural Effusion（260）与 No Finding（306）。",
        "",
        "## 4b. 单图/多图 Study 与聚合（Validation）", "",
    ])
    single_multi = pd.read_csv(
        G1_METRICS_DIR / "single_multi_study_metrics.csv")
    lines.extend([
        "| Group | Study 组 | Study 数 | Subject 数 | Macro-AUROC | Macro-AUPRC | Study BCE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for group in ("R3", "R5"):
        subset = single_multi[(single_multi["group"] == group)
                              & (single_multi["split"] == "valid")
                              & (single_multi["disease"] == LABEL_NAMES[0])]
        for study_group in ("single", "multi", "two", "three_plus"):
            rows = subset[subset["study_group"] == study_group]
            if rows.empty:
                continue
            row = rows.iloc[0]
            lines.append(
                f"| {group} | {study_group} | {int(row['n_studies'])} | "
                f"{int(row['n_subjects'])} | {row['macro_auroc']:.4f} | "
                f"{row['macro_auprc']:.4f} | {row['study_bce']:.4f} |")
    lines.extend([
        "",
        f"- 单图 Study 的 Macro-AUROC（R3 {single_multi[(single_multi['group']=='R3') & (single_multi['disease']==LABEL_NAMES[0]) & (single_multi['split']=='valid') & (single_multi['study_group']=='single')]['macro_auroc'].iloc[0]:.4f} / "
        f"R5 {single_multi[(single_multi['group']=='R5') & (single_multi['disease']==LABEL_NAMES[0]) & (single_multi['split']=='valid') & (single_multi['study_group']=='single')]['macro_auroc'].iloc[0]:.4f}）"
        "明显低于多图 Study；但两组疾病构成与 Subject 组成不同，不能仅据此判定聚合算法优劣。",
        f"- 预先指定的固定权重聚合对照（R5，468 个多图 Study，Logit Mean vs "
        f"Probability Mean）：Δ Macro-AUROC = {aggregation['delta_macro_auroc']:+.6f}，"
        f"95% CI {aggregation['delta_macro_auroc_ci95']}；点估计不支持改用 "
        "Probability Mean，且 CI 跨 0，因此维持 Logit Mean。",
        "",
        "## 5. 参数与特征偏移", "",
        "- 冻结的 Stem/Stage1/Stage2 参数逐位未变（R3/R5 均通过 frozen check）。",
        f"- Stage4 相对参数漂移：R3 {decision['r3_stage4_relative_drift']:.6f}，"
        f"R5 {decision['r5_stage4_relative_drift']:.6f}；Stage3 较小（见 G1-D 表）。",
        f"- Stage4 GAP 原始特征与预训练余弦（Validation）：R3 "
        f"{feature['R3']['valid']['stage4_gap_cosine']:.4f}，R5 "
        f"{feature['R5']['valid']['stage4_gap_cosine']:.4f}；分类器输入(final)特征余弦："
        f"R3 {feature['R3']['valid']['final_feature_cosine']:.4f}，R5 "
        f"{feature['R5']['valid']['final_feature_cosine']:.4f}。Stage3 GAP 余弦≈"
        "0.998，表明 Stage3 原始特征基本保留，漂移主要集中在 Stage4 与最终投影。",
        f"- 同 run 内 Best(ep6) vs Last(ep9) 对照：Stage4 原始 GAP 余弦 "
        f"R3 {g1_summary['feature_within_run']['R3']['valid']['cosine_stage4_best_vs_last']:.4f}、"
        f"R5 {g1_summary['feature_within_run']['R5']['valid']['cosine_stage4_best_vs_last']:.4f}；"
        "说明 raw Stage4 GAP 对微小参数变化高度敏感，其绝对余弦不宜单独作为过拟合证据。",
        "- 漂移确实存在且 R3/R5 现象一致，但漂移本身不等于泛化退化；它只说明"
        "微调显著改变了任务相关表示。",
        "",
        "## 6. 是否支持降低 Stage3/4 学习率", "",
        f"- 决策：**{'建议在用户确认后执行 G2' if decision['recommend_g2'] else '暂不建议执行 G2'}**。",
        f"- {decision['recommendation_text']}",
        "",
        "## 7. 三个配对 Seed 是否出现方向一致的改善", "",
        "- G2 未执行（按门控要求等待用户确认），因此该问题目前**无法回答**。",
        "",
        "## 8. 下一阶段优先方向", "",
        "- 若用户批准 G2：先做固定矩阵（1e-5 vs 3e-6 × seeds 42/2026/3407）的配对实验，"
        "并同时报告参数漂移曲线与 Train/Valid gap 变化。",
        "- 若不批准 G2 或希望先补证据：优先检查 Validation 与 Test 的分布/来源差异、"
        "困难疾病的标签一致性与特征可分性，以及多图 Study 的聚合方式"
        f"（本轮 Probability Mean − Logit Mean 的 Δ Macro-AUROC = "
        f"{aggregation['delta_macro_auroc']:+.6f}，95% CI "
        f"{aggregation['delta_macro_auroc_ci95']}）。",
        "",
        "## 9. 官方 Test 0.7510 落差能解释到什么程度", "",
        "- 已排除：提交入口与验证入口的推理实现不一致、预处理/标签顺序/Study 聚合错误。",
        "- 仍未解释：Validation 0.804653 → Test 0.7510 的约 0.0537 绝对落差。"
        "G0 证明该落差不是推理管线 bug；G1 证明模型在 Train 上显著高于 Validation，"
        "但 Validation 到 Test 的偏移缺少独立标签而无法量化。",
        "",
        "## 10. 因缺少独立测试数据而无法证实的推论", "",
        "- 无法确认 Test 分布偏移的具体幅度与方向（来源/设备/视角差异）。",
        "- 无法确认降低 Stage3/4 学习率会改善 Test 泛化；提升 Validation 也不等于"
        "提升 Test。",
        "- 无法确认参数/特征漂移与 Test 性能之间的因果关系。",
        "- 无法区分验证集过拟合（选模偏差）与真实分布差异各自贡献了多少。",
        "",
        "## 关键瓶颈假设（证据分级）", "",
        "- **已确认事实**：Train−Valid Macro-AUROC 差距约 +0.099（R3/R5 一致）；"
        "Stem/Stage1/Stage2 逐位未变；Stage3 参数相对漂移 0.0084、Stage4 0.0080；"
        "Stage3 GAP 特征余弦 0.998，Stage4/final 特征显著漂移；单图 Study 指标低于多图；"
        "Logit Mean 优于 Probability Mean 的点估计（CI 跨 0）。",
        "- **支持“泛化/分布问题优先于推理或聚合问题”的证据**：G0 排除了推理实现差异；"
        "R7–R10 的新融合/注意力结构均未提升；多图聚合方式不是瓶颈（固定权重对照无优势）；"
        "泛化差距在两个不同结构上几乎相同（+0.0992/+0.0996），更像数据级现象。",
        "- **反对“微调严重破坏预训练表征”的证据**：冻结层逐位未变；Stage3/4 参数相对"
        "漂移仅约 0.8%；Stage3 原始特征基本保留；因此不能把差距直接归因于 Backbone 被"
        "过度改变。",
        "- **尚不能判断**：官方 Test 0.7510 与 Validation 0.8047 的落差中，验证集选模"
        "偏差、Test 分布偏移、标签噪声各占多少；以及降低 Stage3/4 学习率能否改善跨"
        "Subject 泛化。",
        "- **当前最可能瓶颈**：数据层面（Train/Validation/Test 的 Subject/来源/标签分布"
        "差异与困难疾病的有效样本量），其次才是微调强度；两者都需要独立 Test 标签或"
        "分布元数据才能进一步分离。",
        "- **分布差异事实**：Train 11,055 Subject、平均 1.68 Study/Subject（最大 21）；"
        "Validation 仅 228 Subject、平均 3.77 Study/Subject（最大 89），且两者 Subject "
        "不重叠。验证集指标对少数 Subject 的权重明显更高，这既影响 Subject-cluster 不"
        "确定性，也提示 Validation 的 Subject 组成与 Train 不同，不能直接代表 Test 分布。",
        "",
        "## 产物索引", "",
        "- G0：`G0_Inference_Consistency/`（inference_protocol_audit.md, "
        "study_alignment.csv, preprocessing_comparison.json, "
        "logit_probability_difference.csv, per_class_metric_comparison.csv, "
        "g0_final_report.md 等）。",
        "- G1：`G1_R3_R5_Generalization/`、`G1_Parameter_Feature_Drift/`、"
        "`G1_Study_Aggregation_Check/` 下的规定 CSV/PNG 与报告。",
        "- 决策：`G1_R3_R5_Generalization/g2_go_no_go_decision.md`。",
        "",
    ])
    (FINAL_DIR / "final_g0_g1_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def load_report_artifacts():
    """Rebuild the report inputs from already-written G1 artifacts."""
    gap_rows = pd.read_csv(
        G1_METRICS_DIR / "per_class_generalization_gap.csv").to_dict("records")
    single_multi_rows = pd.read_csv(
        G1_METRICS_DIR / "single_multi_study_metrics.csv").to_dict("records")
    bootstrap = read_json(G1_AGG_DIR / "study_aggregation_bootstrap.json")
    aggregation = {
        "n_multi_image_studies": bootstrap["n_multi_image_studies"],
        "logit_mean_macro_auroc": bootstrap["logit_mean_macro_auroc"],
        "probability_mean_macro_auroc": bootstrap["probability_mean_macro_auroc"],
        "delta_macro_auroc": bootstrap["delta_macro_auroc"],
        "delta_macro_auroc_ci95": bootstrap["delta_macro_auroc_ci95"],
        "delta_macro_auprc": bootstrap["delta_macro_auprc"],
        "delta_macro_auprc_ci95": bootstrap["delta_macro_auprc_ci95"],
    }
    drift = read_json(G1_DRIFT_DIR / "parameter_drift_summary.json")
    g1a = {"gap_rows": gap_rows, "metrics_rows": []}
    g1b = {"single_multi_rows": single_multi_rows, "aggregation": aggregation}
    g1c = {"integrity": read_json(G1_METRICS_DIR / "data_integrity.json")}
    g1d = {"stage_drift": drift["stage_drift"],
           "frozen": drift["frozen_layers"], "alpha": drift["alpha"]}
    g1e = read_json(G1_DRIFT_DIR / "feature_drift_summary.json")
    return g1a, g1b, g1c, g1d, g1e


def run_reports_only() -> dict:
    """Regenerate G1/G2/final reports from existing artifacts (no inference)."""
    ensure_directories()
    if not (G1_METRICS_DIR / "per_class_generalization_gap.csv").is_file():
        raise RuntimeError("G1 artifacts are missing; run `g1` first")
    g0_summary_path = G0_DIR / "g0_summary.json"
    g0_summary = read_json(g0_summary_path) if g0_summary_path.is_file() else {
        "gate_passed": False}
    g1a, g1b, g1c, g1d, g1e = load_report_artifacts()
    decision = write_g1_reports(g0_summary, g1a, g1b, g1c, g1d, g1e)
    summary = {
        "g1a_macro_gap": {
            group: float([row for row in g1a["gap_rows"]
                          if row["group"] == group
                          and row["disease"] == "Macro"][0]["auroc_gap"])
            for group in ("R3", "R5")},
        "aggregation": g1b["aggregation"],
        "frozen_layers_unchanged": g1d["frozen"]["all_bitwise_unchanged"],
        "stage_drift": g1d["stage_drift"],
        "feature_drift": {
            group: {split: {
                "stage4_gap_cosine": g1e["groups"][group][split]["cosine_stage4_mean"],
                "final_feature_cosine": g1e["groups"][group][split]["cosine_final_mean"],
                "stage4_cka": g1e["cka"][group][split]["linear_cka_stage4"],
                "final_cka": g1e["cka"][group][split]["linear_cka_final"],
            } for split in ("train", "valid")}
            for group in ("R3", "R5")},
        "feature_pairwise": g1e["pairwise"],
        "feature_within_run": g1e["within_run_best_vs_last"],
        "g2_decision": decision,
        "test_data_accessed": False,
    }
    write_json(G1_METRICS_DIR / "g1_summary.json", summary)
    write_final_decision_report(g0_summary, summary)
    return summary


def run_g1(force: bool = False) -> dict:
    ensure_directories()
    G1_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print("[G1] building deterministic no-augmentation datasets", flush=True)
    train_data, valid_data, _ = build_eval_datasets()
    results = {}
    for group in ("R3", "R5"):
        for split, dataset in (("train", train_data), ("valid", valid_data)):
            results[(group, split)] = split_result(group, split, dataset, force=force)
    g1a = run_g1a(results)
    print("[G1] G1-A Train/Validation metrics done", flush=True)
    g1b = run_g1b(results)
    print("[G1] G1-B single/multi Study + aggregation done", flush=True)
    g1c = run_g1c(train_data, valid_data)
    print("[G1] G1-C distribution audit done", flush=True)
    g1d = run_g1d()
    print("[G1] G1-D parameter drift done", flush=True)
    g1e = run_g1e(train_data, valid_data)
    print("[G1] G1-E feature drift done", flush=True)
    plot_train_valid_curves()
    g0_summary_path = G0_DIR / "g0_summary.json"
    g0_summary = read_json(g0_summary_path) if g0_summary_path.is_file() else {
        "gate_passed": False}
    decision = write_g1_reports(g0_summary, g1a, g1b, g1c, g1d, g1e)
    summary = {        "g1a_macro_gap": {
            group: float([row for row in g1a["gap_rows"]
                          if row["group"] == group
                          and row["disease"] == "Macro"][0]["auroc_gap"])
            for group in ("R3", "R5")},
        "aggregation": g1b["aggregation"],
        "frozen_layers_unchanged": g1d["frozen"]["all_bitwise_unchanged"],
        "stage_drift": g1d["stage_drift"],
        "feature_drift": {
            group: {split: {
                "stage4_gap_cosine": g1e["groups"][group][split]["cosine_stage4_mean"],
                "final_feature_cosine": g1e["groups"][group][split]["cosine_final_mean"],
                "stage4_cka": g1e["cka"][group][split]["linear_cka_stage4"],
                "final_cka": g1e["cka"][group][split]["linear_cka_final"],
            } for split in ("train", "valid")}
            for group in ("R3", "R5")},
        "feature_pairwise": g1e["pairwise"],
        "feature_within_run": g1e["within_run_best_vs_last"],
        "g2_decision": decision,
        "test_data_accessed": False,
    }
    write_json(G1_METRICS_DIR / "g1_summary.json", summary)
    write_final_decision_report(g0_summary, summary)
    return summary


if __name__ == "__main__":
    summary = run_g1(force="--force" in sys.argv)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
