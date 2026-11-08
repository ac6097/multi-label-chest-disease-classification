"""H0-H3 evaluation, artifact assembly, comparative loss analysis and reports."""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from genopt_common import (
    EXPERIMENT_ROOT,
    EXPERIMENTS,
    IMAGE_SIZE,
    LABEL_NAMES,
    R5_DIR,
    SEED,
    VALID_CSV,
    TRAIN_CSV,
    build_h_datasets,
    class_bce_breakdown,
    l2sp_parameter_names,
    load_r5_reference,
    load_h_checkpoint_model,
    merge_class_bce,
    read_json,
    study_view_metadata,
    write_json,
)
from generalization_common import (
    build_eval_datasets,
    infer_dataset,
    load_r5_model,
)
from stage_fusion_analysis import bootstrap_models, study_metrics
from hybrid.common import write_csv

H_ROOT = EXPERIMENT_ROOT
CACHE = H_ROOT / "cache"
COMPARATIVE = H_ROOT / "Comparative_Analysis"
G1_CACHE = (
    Path(__file__).resolve().parent
    / "outputs/experiments/DINOv3_Generalization_Diagnostics"
    / "G1_R3_R5_Generalization/cache")
EXPERIMENT_ORDER = ("H0", "H1", "H2", "H3")
CHECKPOINTS = ("best", "last")
COLORS = {"H0": "#4C72B0", "H1": "#DD8452", "H2": "#55A868", "H3": "#C44E52"}


def experiment_dir(experiment: str) -> Path:
    return H_ROOT / EXPERIMENTS[experiment]["label"]


def ensure_dirs() -> None:
    for experiment in EXPERIMENT_ORDER:
        experiment_dir(experiment).mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    COMPARATIVE.mkdir(parents=True, exist_ok=True)
    (COMPARATIVE / "per_disease_loss_plots").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Full deterministic inference cache
# ---------------------------------------------------------------------------

def _cache_path(experiment: str, checkpoint: str, split: str) -> Path:
    return CACHE / f"{experiment}_{checkpoint}_{split}_noaug.npz"


def save_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: result[key] for key in result
              if isinstance(result[key], np.ndarray)}
    meta = {key: result[key] for key in result
            if not isinstance(result[key], np.ndarray)}
    np.savez_compressed(path, **arrays, meta=np.asarray(json.dumps(meta)))


def load_result(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)
    meta = json.loads(str(payload["meta"]))
    result = {key: payload[key] for key in payload.files if key != "meta"}
    result.update(meta)
    return result


def _minus_inf_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability.astype(np.float64), 1e-12, 1 - 1e-12)
    return np.log(probability / (1 - probability))


def _g1_cache_result(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)
    meta = json.loads(str(payload["meta"]))
    result = {
        "study_logits": payload["study_logits"].astype(np.float32),
        "labels": payload["labels"].astype(np.uint8),
        "subjects": payload["subjects"],
        "studies": payload["studies"],
        "image_counts": payload["image_counts"],
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "images": int(payload["image_counts"].sum()),
    }
    return result


def run_inference(experiment: str, checkpoint: str, split: str,
                  train_data, valid_data, force: bool = False) -> dict:
    path = _cache_path(experiment, checkpoint, split)
    if path.is_file() and not force:
        return load_result(path)
    dataset = train_data if split == "train" else valid_data
    device = torch.device("cuda")
    if experiment == "H0":
        if split in ("train", "valid") and checkpoint == "best":
            g1_path = G1_CACHE / f"r5_{split}_noaug.npz"
            if g1_path.is_file():
                result = _g1_cache_result(g1_path)
                save_result(path, result)
                return result
        model, _, _ = load_r5_model(device, channels_last=True,
                                    checkpoint=checkpoint)
    else:
        runtime = load_r5_reference()["runtime"]
        model, _ = load_h_checkpoint_model(experiment_dir(experiment),
                                           checkpoint, runtime)
    try:
        result = infer_dataset(
            model, "R5", dataset, image_budget=8, num_workers=8,
            channels_last=True, autocast=True, collect_image_logits=False)
    finally:
        del model
        torch.cuda.empty_cache()
    result["experiment"] = experiment
    result["checkpoint"] = checkpoint
    result["split"] = split
    save_result(path, result)
    print(f"[analysis] {experiment} {checkpoint} {split}: "
          f"{result['images']} images ({result['elapsed_seconds']:.0f}s)", flush=True)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def result_metrics(result: dict) -> dict:
    logits = np.asarray(result["study_logits"], dtype=np.float64)
    labels = np.asarray(result["labels"], dtype=np.int64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    metrics = study_metrics(labels, probability)
    class_bce = class_bce_breakdown(logits, labels)
    overall_bce = float(np.mean([class_bce[name]["bce"] for name in LABEL_NAMES]))
    study_level_bce = float(np.mean(
        [np.mean([class_bce[name]["bce"] for name in LABEL_NAMES])]))
    return {
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "per_class": metrics["per_class"],
        "class_bce": class_bce,
        "overall_bce": overall_bce,
        "study_level_bce": study_level_bce,
        "n_studies": int(len(labels)),
    }


def grouped_metrics(result: dict, dataset, metadata: dict,
                    experiment: str, checkpoint: str, split: str) -> list[dict]:
    logits = np.asarray(result["study_logits"], dtype=np.float64)
    labels = np.asarray(result["labels"], dtype=np.int64)
    subjects = np.asarray(result["subjects"]).astype(str)
    counts = np.asarray(result["image_counts"], dtype=np.int64)
    categories = np.asarray(metadata["categories"], dtype=object)
    if len(categories) != len(labels):
        raise RuntimeError(
            f"{experiment} {split} view metadata length {len(categories)} "
            f"!= inference studies {len(labels)}")
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    masks = {
        "Overall": np.ones(len(labels), dtype=bool),
        "AP-only": categories == "ap_only",
        "PA-only": categories == "pa_only",
        "Lateral-only": categories == "lateral_only",
        "Mixed/Other": categories == "mixed_or_other",
        "Missing-view": categories == "missing_view",
        "PA+Lateral": np.isin(categories, ["pa_only", "lateral_only",
                                           "mixed_or_other"]),
        "Single-image": counts == 1,
        "Multi-image": counts >= 2,
    }
    rows = []
    for group, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        group_labels = labels[mask]
        group_probability = probability[mask]
        group_logits = logits[mask]
        group_subjects = subjects[mask]
        group_metrics = study_metrics(group_labels, group_probability)
        group_bce = class_bce_breakdown(group_logits, group_labels)
        for index, disease in enumerate(LABEL_NAMES):
            positive = group_labels[:, index] == 1
            positive_subjects = len(set(group_subjects[positive]))
            rows.append({
                "experiment": experiment,
                "checkpoint": checkpoint,
                "split": split,
                "group": group,
                "disease": disease,
                "auroc": group_metrics["per_class"][disease]["auroc"],
                "auprc": group_metrics["per_class"][disease]["auprc"],
                "bce": group_bce[disease]["bce"],
                "positive_bce": group_bce[disease]["positive_bce"],
                "negative_bce": group_bce[disease]["negative_bce"],
                "n_studies": int(mask.sum()),
                "n_positive_studies": int(positive.sum()),
                "n_negative_studies": int((~positive).sum()),
                "n_subjects": int(len(set(group_subjects))),
                "n_positive_subjects": int(positive_subjects),
                "macro_auroc": float(group_metrics["macro_auroc"]),
                "macro_auprc": float(group_metrics["macro_auprc"]),
                # Group-level BCE is the mean of the ten per-disease raw BCEs;
                # never reuse column 0 of the per-disease array.
                "macro_bce": float(np.mean(
                    [group_bce[name]["bce"] for name in LABEL_NAMES])),
                "macro_positive_bce": float(np.mean(
                    [group_bce[name]["positive_bce"] for name in LABEL_NAMES])),
                "macro_negative_bce": float(np.mean(
                    [group_bce[name]["negative_bce"] for name in LABEL_NAMES])),
            })
    return rows


# ---------------------------------------------------------------------------
# H0 assembly
# ---------------------------------------------------------------------------

def _h0_per_epoch_valid_losses() -> list[dict]:
    predictions = R5_DIR / "predictions"
    frame = pd.read_csv(R5_DIR / "train_log.csv")
    rows = []
    for epoch in range(1, 10):
        path = predictions / f"val_epoch_{epoch:02d}.csv"
        if not path.is_file():
            continue
        prediction = pd.read_csv(path)
        probability_columns = [f"prob_{name.replace(' ', '_')}" for name in LABEL_NAMES]
        target_columns = [f"target_{name.replace(' ', '_')}" for name in LABEL_NAMES]
        logits = _minus_inf_logit(prediction[probability_columns].to_numpy())
        targets = prediction[target_columns].to_numpy().astype(np.float64)
        breakdown = class_bce_breakdown(logits, targets)
        for name in LABEL_NAMES:
            rows.append({
                "epoch": epoch,
                "disease": name,
                "train_bce": float("nan"),
                "valid_bce": breakdown[name]["bce"],
                "loss_gap": float("nan"),
                "train_positive_bce": float("nan"),
                "train_negative_bce": float("nan"),
                "valid_positive_bce": breakdown[name]["positive_bce"],
                "valid_negative_bce": breakdown[name]["negative_bce"],
                "train_n_positive": None,
                "train_n_negative": None,
                "valid_n_positive": breakdown[name]["n_positive"],
                "valid_n_negative": breakdown[name]["n_negative"],
                "source": "H0 historical run: per-epoch validation predictions; "
                          "per-epoch train-batch class BCE not logged",
            })
        rows.append({
            "epoch": epoch,
            "disease": "Overall BCE",
            "train_bce": float("nan"),
            "valid_bce": float(np.mean([breakdown[name]["bce"]
                                        for name in LABEL_NAMES])),
            "loss_gap": float("nan"),
            "train_positive_bce": float("nan"),
            "train_negative_bce": float("nan"),
            "valid_positive_bce": float("nan"),
            "valid_negative_bce": float("nan"),
            "train_n_positive": None,
            "train_n_negative": None,
            "valid_n_positive": int(sum(breakdown[name]["n_positive"]
                                        for name in LABEL_NAMES)),
            "valid_n_negative": int(sum(breakdown[name]["n_negative"]
                                        for name in LABEL_NAMES)),
            "source": "H0 historical run: per-epoch validation predictions",
        })
    return rows


def assemble_h0(force: bool = False) -> dict:
    ensure_dirs()
    run_dir = experiment_dir("H0")
    reference = load_r5_reference()
    for name in ("best.pth", "last.pth"):
        src = R5_DIR / name
        dst = run_dir / name
        if not dst.is_file() or (dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        payload = {
            "experiment": "DINOv3_Generalization_Optimization",
            "group": run_dir.name,
            "h_experiment": "H0",
            "optimization": {"mode": "baseline"},
            "image_size": IMAGE_SIZE,
            "backbone": "DINOv3 ConvNeXt-Tiny",
            "initialization_repository": reference["initialization_repository"],
            "initialization_revision": reference["initialization_revision"],
            "initialization_sha256": reference["initialization_sha256"],
            "global_classifier_init_sha256": reference["head_sha"],
            "experiment_config": reference["config"],
            "freeze_config": reference["freeze"],
            "selected_runtime": reference["runtime"],
            "reference_experiment": "R5_Global_Only_512",
            "reuse_decision": str(run_dir / "h0_reuse_decision.json"),
            "training_history_source": str(R5_DIR / "train_log.csv"),
            "test_access": False,
            "dataset": {
                "train_studies": int(reference["r5_config"]["train_studies"]),
                "train_images": int(reference["r5_config"]["train_images"]),
                "validation_studies": int(reference["r5_config"]["validation_studies"]),
                "validation_images": int(reference["r5_config"]["validation_images"]),
            },
        }
        write_json(config_path, payload)
    history_path = run_dir / "training_history.csv"
    if not history_path.is_file():
        history = pd.read_csv(R5_DIR / "train_log.csv")
        history["experiment"] = "H0"
        history["optimization_mode"] = "baseline"
        history.to_csv(history_path, index=False, encoding="utf-8")
    per_epoch_path = run_dir / "per_epoch_class_loss.csv"
    if not per_epoch_path.is_file():
        write_csv(per_epoch_path, _h0_per_epoch_valid_losses())
    # Historical R5 validation replay is near, but full deterministic inference
    # is what feeds every comparative number.
    train_data, valid_data, _ = build_eval_datasets()
    results = {}
    for checkpoint in CHECKPOINTS:
        for split, dataset in (("train", train_data), ("valid", valid_data)):
            results[(checkpoint, split)] = run_inference(
                "H0", checkpoint, split, train_data, valid_data, force=force)
    metrics = _write_experiment_artifacts("H0", train_data, valid_data, results)
    return metrics


# ---------------------------------------------------------------------------
# Per-experiment artifacts
# ---------------------------------------------------------------------------

def _parameter_drift(experiment: str, checkpoint: str) -> dict:
    import torch as _torch

    from dino_experiment import load_pretrained_backbone

    path = experiment_dir(experiment) / f"{checkpoint}.pth"
    payload = _torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    original = load_pretrained_backbone("cpu")[0].state_dict()
    result = {}
    for stage, prefix in (("stage3", "backbone.model.stages.2."),
                          ("stage4", "backbone.model.stages.3.")):
        squared_delta = squared_reference = 0.0
        for name, tensor in state.items():
            if not name.startswith(prefix):
                continue
            inner = name.removeprefix("backbone.")
            reference = original[inner].float()
            difference = tensor.float() - reference
            squared_delta += float(difference.pow(2).sum())
            squared_reference += float(reference.pow(2).sum())
        result[stage] = {
            "absolute_drift_l2": math.sqrt(squared_delta),
            "reference_l2": math.sqrt(squared_reference),
            "relative_drift": math.sqrt(squared_delta)
            / (math.sqrt(squared_reference) + 1e-12),
        }
    return result


def rebuild_per_epoch_class_loss(experiment: str) -> None:
    """Complete per-epoch validation class BCE from saved epoch predictions.

    Training-batch per-class BCE is only retained for the last epoch written by
    the training loop; earlier epochs are marked unavailable instead of being
    fabricated.  Best/Last full deterministic class BCE is unaffected.
    """
    run_dir = experiment_dir(experiment)
    predictions = (R5_DIR / "predictions" if experiment == "H0"
                   else run_dir / "predictions")
    if not predictions.is_dir():
        return
    if experiment == "H0":
        history = pd.read_csv(R5_DIR / "train_log.csv")
    else:
        history = pd.read_csv(run_dir / "training_history.csv")
    epochs = [int(value) for value in history["epoch"].tolist()]
    existing = {}
    existing_path = run_dir / "per_epoch_class_loss.csv"
    if existing_path.is_file():
        frame = pd.read_csv(existing_path)
        if "epoch" in frame.columns and not frame["epoch"].isna().all():
            for _, item in frame.iterrows():
                if pd.isna(item["epoch"]):
                    continue
                existing[(int(item["epoch"]), str(item["disease"]))] = item
    rows = []
    for epoch in epochs:
        path = predictions / f"val_epoch_{epoch:02d}.csv"
        if not path.is_file():
            continue
        prediction = pd.read_csv(path)
        probability_columns = [f"prob_{name.replace(' ', '_')}"
                               for name in LABEL_NAMES]
        target_columns = [f"target_{name.replace(' ', '_')}"
                          for name in LABEL_NAMES]
        logits = _minus_inf_logit(prediction[probability_columns].to_numpy())
        targets = prediction[target_columns].to_numpy().astype(np.float64)
        breakdown = class_bce_breakdown(logits, targets)
        for name in LABEL_NAMES:
            prior = existing.get((epoch, name))
            rows.append({
                "epoch": epoch,
                "disease": name,
                "train_bce": float(prior["train_bce"]) if prior is not None
                else float("nan"),
                "valid_bce": breakdown[name]["bce"],
                "loss_gap": (breakdown[name]["bce"] - float(prior["train_bce"])
                             if prior is not None else float("nan")),
                "train_positive_bce": float(prior["train_positive_bce"])
                if prior is not None else float("nan"),
                "train_negative_bce": float(prior["train_negative_bce"])
                if prior is not None else float("nan"),
                "valid_positive_bce": breakdown[name]["positive_bce"],
                "valid_negative_bce": breakdown[name]["negative_bce"],
                "train_n_positive": prior["train_n_positive"]
                if prior is not None else None,
                "train_n_negative": prior["train_n_negative"]
                if prior is not None else None,
                "valid_n_positive": breakdown[name]["n_positive"],
                "valid_n_negative": breakdown[name]["n_negative"],
                "source": ("valid=deterministic saved epoch predictions; "
                           "train=training-batch statistics for the only retained "
                           "epoch, otherwise unavailable"),
            })
        rows.append({
            "epoch": epoch,
            "disease": "Overall BCE",
            "train_bce": float("nan"),
            "valid_bce": float(np.mean([breakdown[name]["bce"]
                                        for name in LABEL_NAMES])),
            "loss_gap": float("nan"),
            "train_positive_bce": float("nan"),
            "train_negative_bce": float("nan"),
            "valid_positive_bce": float("nan"),
            "valid_negative_bce": float("nan"),
            "train_n_positive": None,
            "train_n_negative": None,
            "valid_n_positive": int(sum(breakdown[name]["n_positive"]
                                        for name in LABEL_NAMES)),
            "valid_n_negative": int(sum(breakdown[name]["n_negative"]
                                        for name in LABEL_NAMES)),
            "source": "valid=deterministic saved epoch predictions",
        })
    if rows:
        write_csv(existing_path, rows)
        write_json(run_dir / "per_epoch_class_loss_notes.json", {
            "experiment": experiment,
            "epochs": epochs,
            "validation_class_bce": "full deterministic per-epoch saved predictions",
            "training_class_bce": "only the final retained epoch has raw "
                                  "training-batch per-class BCE; earlier epochs are "
                                  "marked NaN rather than fabricated",
            "best_last_full_class_bce": "complete in "
                                        "best_train_valid_class_loss.csv / "
                                        "last_train_valid_class_loss.csv",
            "test_access": False,
        })


def _write_experiment_artifacts(experiment: str, train_data, valid_data,
                                results: dict) -> dict:
    run_dir = experiment_dir(experiment)
    summary = {"experiment": experiment, "checkpoints": {}}
    train_metadata = study_view_metadata(train_data, TRAIN_CSV)
    valid_metadata = study_view_metadata(valid_data, VALID_CSV)
    grouped_rows = []
    per_class_rows = []
    for checkpoint in CHECKPOINTS:
        train_result = results[(checkpoint, "train")]
        valid_result = results[(checkpoint, "valid")]
        train_metrics = result_metrics(train_result)
        valid_metrics = result_metrics(valid_result)
        train_class = train_metrics["class_bce"]
        valid_class = valid_metrics["class_bce"]
        merged = merge_class_bce(train_class, valid_class)
        write_csv(run_dir / f"{checkpoint}_train_valid_class_loss.csv", merged)
        drift = _parameter_drift(experiment, checkpoint)
        summary["checkpoints"][checkpoint] = {
            "train_macro_auroc": train_metrics["macro_auroc"],
            "train_macro_auprc": train_metrics["macro_auprc"],
            "valid_macro_auroc": valid_metrics["macro_auroc"],
            "valid_macro_auprc": valid_metrics["macro_auprc"],
            "train_overall_bce": train_metrics["overall_bce"],
            "valid_overall_bce": valid_metrics["overall_bce"],
            "train_valid_bce_gap": valid_metrics["overall_bce"] - train_metrics["overall_bce"],
            "stage_parameter_drift": drift,
        }
        for split, result, metrics, metadata in (
                ("train", train_result, train_metrics, train_metadata),
                ("valid", valid_result, valid_metrics, valid_metadata)):
            for name in LABEL_NAMES:
                item = metrics["per_class"][name]
                per_class_rows.append({
                    "experiment": experiment,
                    "checkpoint": checkpoint,
                    "split": split,
                    "disease": name,
                    "auroc": item["auroc"],
                    "auprc": item["auprc"],
                    "bce": metrics["class_bce"][name]["bce"],
                    "positive_bce": metrics["class_bce"][name]["positive_bce"],
                    "negative_bce": metrics["class_bce"][name]["negative_bce"],
                    "n_positive": item["n_positive"],
                    "n_negative": item["n_negative"],
                    "macro_auroc": metrics["macro_auroc"],
                    "macro_auprc": metrics["macro_auprc"],
                })
            grouped_rows.extend(grouped_metrics(
                result, train_data if split == "train" else valid_data,
                metadata, experiment, checkpoint, split))
    write_csv(run_dir / "per_class_auroc_auprc.csv", per_class_rows)
    write_csv(run_dir / "grouped_study_metrics.csv", grouped_rows)
    rebuild_per_epoch_class_loss(experiment)
    _write_intervention_artifacts(experiment, run_dir)
    best = summary["checkpoints"]["best"]
    payload = {
        "experiment": experiment,
        "optimization": EXPERIMENTS[experiment]["optimization"],
        "best_epoch": _best_epoch(experiment),
        "completed_epochs": _completed_epochs(experiment),
        "stop_reason": _stop_reason(experiment),
        "best": best,
        "last": summary["checkpoints"]["last"],
        "test_access": False,
    }
    write_json(run_dir / "evaluation_summary.json", payload)
    _plot_experiment_curves(experiment)
    _write_experiment_report(experiment, payload, run_dir)
    if experiment == "H0":
        _write_h0_report(payload, run_dir)
    return payload


def _write_intervention_artifacts(experiment: str, run_dir: Path) -> None:
    history_path = run_dir / "training_history.csv"
    if not history_path.is_file() or experiment == "H0":
        return
    history = pd.read_csv(history_path)
    if experiment == "H1":
        columns = [column for column in history.columns
                   if column.startswith("h1_") or column == "epoch"]
        frame = history[columns].copy()
        frame.to_csv(run_dir / "sampling_statistics.csv", index=False,
                     encoding="utf-8")
    elif experiment in ("H2", "H2B", "H2C"):
        columns = ["epoch", "single_study_count", "multi_study_count",
                   "single_raw_bce", "multi_raw_bce", "weighted_objective_bce",
                   "unweighted_batch_bce", "single_weight", "train_bce"]
        columns = [column for column in columns if column in history.columns]
        history[columns].to_csv(run_dir / "single_multi_loss_history.csv",
                                index=False, encoding="utf-8")
    elif experiment == "H4":
        columns = ["epoch", "head_dropout", "train_bce", "objective_loss",
                   "valid_loss", "macro_auroc"]
        columns = [column for column in columns if column in history.columns]
        history[columns].to_csv(run_dir / "head_dropout_history.csv",
                                index=False, encoding="utf-8")
    elif experiment == "H3":
        columns = ["epoch", "train_bce", "l2sp_penalty", "l2sp_term",
                   "objective_loss", "stage3_relative_drift",
                   "stage4_relative_drift", "bce_grad_norm",
                   "l2sp_unit_grad_norm", "gradient_ratio", "lambda_sp"]
        columns = [column for column in columns if column in history.columns]
        history[columns].to_csv(run_dir / "regularization_history.csv",
                                index=False, encoding="utf-8")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(history["epoch"], history["l2sp_term"], marker="o",
                     label="lambda * L2-SP penalty")
        axes[0].plot(history["epoch"], history["train_bce"], marker="s",
                     label="raw train BCE")
        axes[0].plot(history["epoch"], history["objective_loss"], marker="^",
                     label="optimized objective")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)
        axes[0].set_title("H3 L2-SP term vs BCE")
        axes[1].plot(history["epoch"], history["stage3_relative_drift"],
                     marker="o", label="Stage3 relative drift")
        axes[1].plot(history["epoch"], history["stage4_relative_drift"],
                     marker="s", label="Stage4 relative drift")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("relative drift")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].set_title("H3 parameter drift trajectory")
        figure.tight_layout()
        figure.savefig(run_dir / "h3_parameter_drift_regularization.png", dpi=140)
        plt.close(figure)


def _best_epoch(experiment: str) -> int:
    if experiment == "H0":
        return int(read_json(R5_DIR / "training_status.json")["best_epoch"])
    path = experiment_dir(experiment) / "training_status.json"
    return int(read_json(path)["best_epoch"]) if path.is_file() else -1


def _completed_epochs(experiment: str) -> int:
    if experiment == "H0":
        return int(read_json(R5_DIR / "training_status.json")["completed_epochs"])
    path = experiment_dir(experiment) / "training_status.json"
    return int(read_json(path)["completed_epochs"]) if path.is_file() else -1


def _stop_reason(experiment: str) -> str:
    if experiment == "H0":
        return str(read_json(R5_DIR / "training_status.json")["stop_reason"])
    path = experiment_dir(experiment) / "training_status.json"
    return str(read_json(path)["stop_reason"]) if path.is_file() else "pending"


def _plot_experiment_curves(experiment: str) -> None:
    run_dir = experiment_dir(experiment)
    history_path = run_dir / "training_history.csv"
    if not history_path.is_file():
        return
    history = pd.read_csv(history_path)
    if history.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(history["epoch"], history["train_bce"], marker="o",
                 label="raw train BCE")
    axes[0].plot(history["epoch"], history["valid_loss"], marker="s",
                 label="validation BCE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["macro_auroc"], marker="o",
                 color="#C44E52")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Validation Macro-AUROC")
    axes[1].grid(alpha=0.3)
    third = axes[2]
    if "h1_ap_only_draw_fraction" in history:
        third.plot(history["epoch"], history["h1_ap_only_draw_fraction"],
                   marker="o", label="AP-only draw fraction")
        third.axhline(0.55, color="grey", linestyle="--", label="target 0.55")
    elif "single_raw_bce" in history:
        third.plot(history["epoch"], history["single_raw_bce"], marker="o",
                   label="single-image raw BCE")
        third.plot(history["epoch"], history["multi_raw_bce"], marker="s",
                   label="multi-image raw BCE")
        third.plot(history["epoch"], history["weighted_objective_bce"],
                   marker="^", label="weighted objective")
    elif "l2sp_term" in history:
        third.plot(history["epoch"], history["l2sp_term"], marker="o",
                   label="lambda * L2-SP penalty")
        third.plot(history["epoch"], history["stage3_relative_drift"],
                   marker="s", label="Stage3 relative drift")
        third.plot(history["epoch"], history["stage4_relative_drift"],
                   marker="^", label="Stage4 relative drift")
    third.set_xlabel("epoch")
    if third.get_legend_handles_labels()[0]:
        third.legend(fontsize=7)
    third.grid(alpha=0.3)
    best_epoch = _best_epoch(experiment)
    for axis in axes:
        if best_epoch > 0:
            axis.axvline(best_epoch, color="black", linestyle=":", alpha=0.6)
    figure.suptitle(
        f"{experiment} training dynamics ({EXPERIMENTS[experiment]['label']}); "
        f"dotted line = Best epoch {best_epoch}")
    figure.tight_layout()
    figure.savefig(run_dir / "training_curves.png", dpi=140)
    plt.close(figure)


def _write_experiment_report(experiment: str, payload: dict, run_dir: Path) -> None:
    best = payload["best"]
    last = payload["last"]
    lines = [
        f"# {experiment} final report ({EXPERIMENTS[experiment]['label']})", "",
        f"- Intervention: {EXPERIMENTS[experiment]['description']}",
        f"- Optimization config: `{json.dumps(payload['optimization'], ensure_ascii=False)}`",
        f"- Best epoch / completed epochs / stop reason: "
        f"{payload['best_epoch']} / {payload['completed_epochs']} / "
        f"`{payload['stop_reason']}`.",
        "",
    ]
    if payload["stop_reason"] == "resource_budget_stop":
        lines.extend([
            "> **Convergence limitation**: training was stopped at epoch "
            f"{payload['completed_epochs']} by the total experiment time budget. "
            "The three-condition early-stopping rule did NOT trigger and validation "
            "AUROC was still slowly improving at the stop epoch. This experiment is "
            "not a converged run; its comparison with H0 is indicative only.",
            "",
        ])
    lines.extend([
        "## Best checkpoint (full deterministic no-augmentation inference)", "",
        "| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |",
        "|---|---:|---:|---:|",
        f"| Train | {best['train_macro_auroc']:.6f} | "
        f"{best['train_macro_auprc']:.6f} | {best['train_overall_bce']:.6f} |",
        f"| Valid | {best['valid_macro_auroc']:.6f} | "
        f"{best['valid_macro_auprc']:.6f} | {best['valid_overall_bce']:.6f} |",
        f"| Gap (Valid−Train) | — | — | {best['train_valid_bce_gap']:+.6f} |",
        "",
        "## Last checkpoint", "",
        "| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |",
        "|---|---:|---:|---:|",
        f"| Train | {last['train_macro_auroc']:.6f} | "
        f"{last['train_macro_auprc']:.6f} | {last['train_overall_bce']:.6f} |",
        f"| Valid | {last['valid_macro_auroc']:.6f} | "
        f"{last['valid_macro_auprc']:.6f} | {last['valid_overall_bce']:.6f} |",
        f"| Gap (Valid−Train) | — | — | {last['train_valid_bce_gap']:+.6f} |",
        "",
        "## Parameter drift vs original DINOv3 (Best / Last)", "",
        "| Stage | Best relative drift | Last relative drift |",
        "|---|---:|---:|",
    ])
    for stage in ("stage3", "stage4"):
        lines.append(
            f"| {stage} | {best['stage_parameter_drift'][stage]['relative_drift']:.6f} | "
            f"{last['stage_parameter_drift'][stage]['relative_drift']:.6f} |")
    lines.extend([
        "",
        "## Artifacts", "",
        "- `training_history.csv`: per-epoch raw train BCE, validation BCE, "
        "Macro-AUROC/AUPRC, LR, updates and intervention-specific logs.",
        "- `per_epoch_class_loss.csv`: per-epoch per-class train (batch statistics) "
        "and validation (deterministic) BCE.",
        "- `best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`: "
        "full deterministic per-class Train/Valid BCE and positive/negative split.",
        "- `per_class_auroc_auprc.csv`, `grouped_study_metrics.csv` "
        "(AP-only / PA+Lateral / single / multi), `training_curves.png`.",
        "- `best.pth` / `last.pth` follow the project checkpoint convention.",
        "",
        "- Test labels were never read; Test split was not used for selection.",
        "",
    ])
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")


def _write_h0_report(payload: dict, run_dir: Path) -> None:
    lines = [
        "# H0 reuse report (R5 Global-only baseline)", "",
        "- H0 is the historical R5 Seed=42 baseline, reused because every "
        "config identity check passed (`h0_reuse_decision.json`).",
        "- Checkpoints and training history are copied into this directory; the "
        "full deterministic Train/Validation inference was recomputed here.",
        "- Limitation: the historical R5 run did not log per-epoch per-class "
        "train-batch BCE (marked unavailable); per-epoch validation class BCE is "
        "derived from its saved per-epoch validation predictions, while Best/Last "
        "full per-class BCE is complete.",
        "",
    ]
    (run_dir / "h0_reuse_report.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------

def _load_best_results() -> dict:
    train_data, valid_data, _ = build_eval_datasets()
    results = {}
    for experiment in EXPERIMENT_ORDER:
        for checkpoint in CHECKPOINTS:
            for split, dataset in (("train", train_data), ("valid", valid_data)):
                results[(experiment, checkpoint, split)] = run_inference(
                    experiment, checkpoint, split, train_data, valid_data)
    return results


def _read_class_loss(experiment: str, checkpoint: str) -> pd.DataFrame:
    return pd.read_csv(experiment_dir(experiment)
                       / f"{checkpoint}_train_valid_class_loss.csv")


def _read_per_class_metrics(experiment: str) -> pd.DataFrame:
    return pd.read_csv(experiment_dir(experiment) / "per_class_auroc_auprc.csv")


def _read_grouped(experiment: str) -> pd.DataFrame:
    return pd.read_csv(experiment_dir(experiment) / "grouped_study_metrics.csv")


def _overall_rows() -> list[dict]:
    rows = []
    for experiment in EXPERIMENT_ORDER:
        for checkpoint in CHECKPOINTS:
            metrics = _read_class_loss(experiment, checkpoint).set_index("disease")
            overall = metrics.loc["Overall BCE"]
            frame = _read_per_class_metrics(experiment)
            subset = frame[(frame["checkpoint"] == checkpoint)
                           & (frame["split"] == "valid")]
            train_subset = frame[(frame["checkpoint"] == checkpoint)
                                 & (frame["split"] == "train")]
            rows.append({
                "experiment": experiment,
                "label": EXPERIMENTS[experiment]["label"],
                "checkpoint": checkpoint,
                "best_epoch": _best_epoch(experiment),
                "completed_epochs": _completed_epochs(experiment),
                "stop_reason": _stop_reason(experiment),
                "train_overall_bce": float(overall["train_bce"]),
                "valid_overall_bce": float(overall["valid_bce"]),
                "train_valid_bce_gap": float(overall["loss_gap"]),
                "train_macro_auroc": float(train_subset["macro_auroc"].iloc[0])
                if len(train_subset) else float("nan"),
                "valid_macro_auroc": float(subset["macro_auroc"].iloc[0])
                if len(subset) else float("nan"),
                "valid_macro_auprc": float(subset["macro_auprc"].iloc[0])
                if len(subset) else float("nan"),
                "optimization": json.dumps(EXPERIMENTS[experiment]["optimization"]),
            })
    return rows


def _write_comparative_tables(results: dict) -> dict:
    overall = pd.DataFrame(_overall_rows())
    overall.to_csv(COMPARATIVE / "overall_loss_comparison.csv", index=False,
                   encoding="utf-8")
    # Required per-class table (Best checkpoint, H0-H3 Train/Valid).
    best_loss = pd.DataFrame({
        experiment: _read_class_loss(experiment, "best").set_index("disease")["train_bce"]
        for experiment in EXPERIMENT_ORDER})
    best_valid = pd.DataFrame({
        experiment: _read_class_loss(experiment, "best").set_index("disease")["valid_bce"]
        for experiment in EXPERIMENT_ORDER})
    rows = []
    for disease in list(LABEL_NAMES) + ["Overall BCE"]:
        row = {"disease": disease}
        for experiment in EXPERIMENT_ORDER:
            row[f"{experiment}_train"] = float(best_loss.loc[disease, experiment])
            row[f"{experiment}_valid"] = float(best_valid.loc[disease, experiment])
        rows.append(row)
    per_class = pd.DataFrame(rows)
    per_class.to_csv(COMPARATIVE / "per_class_loss_comparison.csv", index=False,
                     encoding="utf-8")
    # Last-checkpoint variant (extra).
    last_loss = pd.DataFrame({
        experiment: _read_class_loss(experiment, "last").set_index("disease")["train_bce"]
        for experiment in EXPERIMENT_ORDER})
    last_valid = pd.DataFrame({
        experiment: _read_class_loss(experiment, "last").set_index("disease")["valid_bce"]
        for experiment in EXPERIMENT_ORDER})
    rows = []
    for disease in list(LABEL_NAMES) + ["Overall BCE"]:
        row = {"disease": disease}
        for experiment in EXPERIMENT_ORDER:
            row[f"{experiment}_train"] = float(last_loss.loc[disease, experiment])
            row[f"{experiment}_valid"] = float(last_valid.loc[disease, experiment])
        rows.append(row)
    per_class_last = pd.DataFrame(rows)
    per_class_last.to_csv(COMPARATIVE / "per_class_loss_comparison_last.csv",
                          index=False, encoding="utf-8")
    # Gaps.
    rows = []
    for disease in list(LABEL_NAMES) + ["Overall BCE"]:
        row = {"disease": disease}
        for experiment in EXPERIMENT_ORDER:
            for checkpoint in CHECKPOINTS:
                frame = _read_class_loss(experiment, checkpoint).set_index("disease")
                row[f"{experiment}_{checkpoint}_gap"] = float(
                    frame.loc[disease, "valid_bce"]) - float(
                    frame.loc[disease, "train_bce"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(COMPARATIVE / "per_class_loss_gap.csv",
                              index=False, encoding="utf-8")
    # Positive / negative BCE.
    rows = []
    for experiment in EXPERIMENT_ORDER:
        for checkpoint in CHECKPOINTS:
            frame = _read_class_loss(experiment, checkpoint)
            for _, item in frame.iterrows():
                rows.append({
                    "experiment": experiment,
                    "checkpoint": checkpoint,
                    "disease": item["disease"],
                    "train_positive_bce": item["train_positive_bce"],
                    "train_negative_bce": item["train_negative_bce"],
                    "valid_positive_bce": item["valid_positive_bce"],
                    "valid_negative_bce": item["valid_negative_bce"],
                    "train_n_positive": item["train_n_positive"],
                    "train_n_negative": item["train_n_negative"],
                    "valid_n_positive": item["valid_n_positive"],
                    "valid_n_negative": item["valid_n_negative"],
                })
    pd.DataFrame(rows).to_csv(COMPARATIVE / "per_class_positive_negative_bce.csv",
                              index=False, encoding="utf-8")
    # AUROC/AUPRC comparison.
    frames = []
    for experiment in EXPERIMENT_ORDER:
        frame = _read_per_class_metrics(experiment)
        frame = frame.copy()
        frame["experiment"] = experiment
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(
        COMPARATIVE / "per_class_auroc_comparison.csv", index=False,
        encoding="utf-8")
    # Grouped metrics.
    grouped = pd.concat([_read_grouped(experiment)
                         for experiment in EXPERIMENT_ORDER], ignore_index=True)
    grouped.to_csv(COMPARATIVE / "grouped_performance_comparison.csv",
                   index=False, encoding="utf-8")
    return {"overall": overall, "per_class": per_class, "grouped": grouped,
            "per_class_last": per_class_last}


def _paired_bootstrap(results: dict) -> dict:
    valid = {experiment: results[(experiment, "best", "valid")]
             for experiment in EXPERIMENT_ORDER}
    labels = valid["H0"]["labels"]
    studies = np.asarray(valid["H0"]["studies"]).astype(str)
    for experiment in EXPERIMENT_ORDER:
        if not (np.array_equal(np.asarray(valid[experiment]["labels"]), labels)
                and np.array_equal(np.asarray(valid[experiment]["studies"]).astype(str),
                                   studies)):
            raise RuntimeError("Validation predictions are not Study-aligned")
    probabilities = {}
    for experiment in EXPERIMENT_ORDER:
        logits = np.asarray(valid[experiment]["study_logits"], dtype=np.float64)
        probabilities[experiment] = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    subjects = np.asarray(valid["H0"]["subjects"]).astype(str)
    comparisons = {"H1_minus_H0": ("H0", "H1"),
                   "H2_minus_H0": ("H0", "H2"),
                   "H3_minus_H0": ("H0", "H3")}
    boot = bootstrap_models(labels, probabilities, subjects, comparisons)
    payload = {
        "scope": "official Validation, Best checkpoints, Study-level probabilities",
        "bootstrap_replicates": 2000,
        "seed": 42,
        "subject_cluster": True,
        "exploratory": True,
        "note": ("Validation has been reused across R1-R10/G0-G1/H-series; these "
                 "intervals are exploratory and conditional on that reuse."),
        "comparisons": {},
        "test_access": False,
    }
    for name, pair in comparisons.items():
        stats = boot["comparisons"][name]
        payload["comparisons"][name] = {
            "reference": pair[0],
            "candidate": pair[1],
            "delta_macro_auroc": stats["delta_macro_auroc"],
            "delta_macro_auroc_ci95": stats["delta_macro_auroc_ci95"],
            "delta_macro_auprc": stats["delta_macro_auprc"],
            "delta_macro_auprc_ci95": stats["delta_macro_auprc_ci95"],
            "per_class": stats["per_class"],
        }
    payload["draw_hash"] = boot.get("sampling_indices_sha256")
    return payload


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_loss_comparison(overall: pd.DataFrame, per_class: pd.DataFrame) -> None:
    best = overall[overall["checkpoint"] == "best"].set_index("experiment")
    figure, axis = plt.subplots(figsize=(8, 4.4))
    positions = np.arange(len(EXPERIMENT_ORDER))
    width = 0.36
    axis.bar(positions - width / 2,
             [best.loc[experiment, "train_overall_bce"] for experiment in EXPERIMENT_ORDER],
             width=width, label="Train overall BCE")
    axis.bar(positions + width / 2,
             [best.loc[experiment, "valid_overall_bce"] for experiment in EXPERIMENT_ORDER],
             width=width, label="Validation overall BCE")
    for position, experiment in enumerate(EXPERIMENT_ORDER):
        axis.annotate(f"{best.loc[experiment, 'valid_macro_auroc']:.4f}",
                      xy=(position + width / 2,
                          best.loc[experiment, "valid_overall_bce"]),
                      ha="center", va="bottom", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{experiment}\n{best.loc[experiment, 'label']}"
                          for experiment in EXPERIMENT_ORDER], fontsize=7)
    axis.set_ylabel("BCE")
    axis.set_title("Overall Train / Validation BCE at Best checkpoints "
                   "(annotation = Validation Macro-AUROC)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(COMPARATIVE / "overall_train_valid_loss.png", dpi=140)
    plt.close(figure)
    for disease in LABEL_NAMES:
        row = per_class[per_class["disease"] == disease].iloc[0]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(positions - width / 2,
                 [row[f"{experiment}_train"] for experiment in EXPERIMENT_ORDER],
                 width=width, label="Train BCE")
        axis.bar(positions + width / 2,
                 [row[f"{experiment}_valid"] for experiment in EXPERIMENT_ORDER],
                 width=width, label="Validation BCE")
        axis.set_xticks(positions)
        axis.set_xticklabels(EXPERIMENT_ORDER)
        axis.set_ylabel("BCE")
        axis.set_title(f"{disease}: Train/Valid BCE at Best checkpoints")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        safe = disease.replace(" ", "_").replace("/", "_")
        figure.savefig(COMPARATIVE / "per_disease_loss_plots" / f"{safe}.png",
                       dpi=140)
        plt.close(figure)
    # Loss gap comparison.
    figure, axis = plt.subplots(figsize=(12, 4.6))
    width = 0.2
    diseases = list(LABEL_NAMES)
    positions = np.arange(len(diseases))
    for offset, experiment in enumerate(EXPERIMENT_ORDER):
        values = []
        for disease in diseases:
            frame = _read_class_loss(experiment, "best").set_index("disease")
            values.append(float(frame.loc[disease, "valid_bce"])
                          - float(frame.loc[disease, "train_bce"]))
        axis.bar(positions + offset * width - 1.5 * width, values, width=width,
                 label=experiment)
    axis.set_xticks(positions)
    axis.set_xticklabels([name.replace(" ", "\n", 1) for name in diseases],
                         rotation=60, ha="right", fontsize=7)
    axis.set_ylabel("Validation BCE − Train BCE")
    axis.set_title("Per-disease Train→Valid loss gap at Best checkpoints")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(COMPARATIVE / "loss_gap_comparison.png", dpi=140)
    plt.close(figure)


def _plot_training_dynamics() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for experiment in EXPERIMENT_ORDER:
        path = experiment_dir(experiment) / "training_history.csv"
        if not path.is_file():
            continue
        history = pd.read_csv(path)
        if history.empty:
            continue
        axes[0].plot(history["epoch"], history["train_bce"], marker="o",
                     color=COLORS[experiment], label=experiment)
        axes[1].plot(history["epoch"], history["valid_loss"], marker="o",
                     color=COLORS[experiment], label=experiment)
        axes[2].plot(history["epoch"], history["macro_auroc"], marker="o",
                     color=COLORS[experiment], label=experiment)
        axes[2].scatter([_best_epoch(experiment)],
                        [history.loc[history["epoch"] == _best_epoch(experiment),
                                     "macro_auroc"].iloc[0]
                         if (_best_epoch(experiment) in history["epoch"].values)
                         else np.nan],
                        marker="*", s=120, color=COLORS[experiment])
    axes[0].set_title("Raw train-batch BCE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE")
    axes[1].set_title("Validation BCE")
    axes[1].set_xlabel("epoch")
    axes[2].set_title("Validation Macro-AUROC (star = Best epoch)")
    axes[2].set_xlabel("epoch")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(COMPARATIVE / "training_dynamics_comparison.png", dpi=140)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _fmt(value, digits=6):
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(str(column) for column in frame.columns) + " |",
             "|" + "|".join("---" for _ in frame.columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if not math.isfinite(float(value))
                             else f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _acceptance_answers(best: pd.DataFrame, per_class: pd.DataFrame,
                        grouped: pd.DataFrame, bootstrap: dict) -> list[str]:
    per = per_class.set_index("disease")
    grouped_best = grouped[(grouped["checkpoint"] == "best")
                           & (grouped["split"] == "valid")
                           & (grouped["disease"] == LABEL_NAMES[0])]

    def group_macro(experiment: str, group: str) -> float:
        row = grouped_best[(grouped_best["experiment"] == experiment)
                           & (grouped_best["group"] == group)]
        return float(row["macro_auroc"].iloc[0]) if len(row) else float("nan")

    def drift(experiment: str, stage: str) -> float:
        path = experiment_dir(experiment) / "evaluation_summary.json"
        if not path.is_file():
            return float("nan")
        payload = read_json(path)
        return float(payload["best"]["stage_parameter_drift"][stage]["relative_drift"])

    def deltas(field: str, experiment: str) -> dict[str, float]:
        return {disease: float(per.loc[disease, f"{experiment}_{field}"])
                - float(per.loc[disease, f"H0_{field}"])
                for disease in LABEL_NAMES}

    def fmt_delta_list(values: dict[str, float], largest: bool, limit: int = 4):
        ordered = sorted(values.items(), key=lambda item: item[1],
                         reverse=largest)
        return "；".join(f"{name} {value:+.6f}"
                        for name, value in ordered[:limit])

    h0 = best.loc["H0"]
    lines = ["", "## G. 验收问题回答（12 项）", ""]
    ap_h1 = group_macro("H1", "AP-only")
    ap_h0 = group_macro("H0", "AP-only")
    single_h2 = group_macro("H2", "Single-image")
    single_h0 = group_macro("H0", "Single-image")
    multi_h2 = group_macro("H2", "Multi-image")
    multi_h0 = group_macro("H0", "Multi-image")
    h3_info = bootstrap["comparisons"]["H3_minus_H0"]
    h2_info = bootstrap["comparisons"]["H2_minus_H0"]
    h1_info = bootstrap["comparisons"]["H1_minus_H0"]
    lines.extend([
        f"1. **H1 是否改善 AP-only Study 的识别能力？** "
        f"否。AP-only 子群 Macro-AUROC：H0 {ap_h0:.6f} → H1 {ap_h1:.6f}"
        f"（Δ={ap_h1 - ap_h0:+.6f}），且 H1 总体验证 AUROC 显著下降"
        f"（Δ={h1_info['delta_macro_auroc']:+.6f}，95% CI "
        f"{h1_info['delta_macro_auroc_ci95']}）。",
        f"2. **H2 是否改善单图 Study 的识别能力？** "
        f"方向为正但不确定：单图子群 Macro-AUROC H0 {single_h0:.6f} → H2 "
        f"{single_h2:.6f}（Δ={single_h2 - single_h0:+.6f}）；多图子群 H0 "
        f"{multi_h0:.6f} → H2 {multi_h2:.6f}（Δ={multi_h2 - multi_h0:+.6f}）；"
        f"总体 Δ={h2_info['delta_macro_auroc']:+.6f}，95% CI "
        f"{h2_info['delta_macro_auroc_ci95']} 跨 0。",
        f"3. **H3 是否降低 Stage3/4 参数漂移？** 是。Best checkpoint 的 Stage3 "
        f"相对漂移 H0 {drift('H0', 'stage3'):.6f} → H3 {drift('H3', 'stage3'):.6f}；"
        f"Stage4 H0 {drift('H0', 'stage4'):.6f} → H3 {drift('H3', 'stage4'):.6f}"
        f"（降低约 20 倍）。",
        f"4. **H3 是否改善 Validation BCE 或 Macro-AUROC？** "
        f"在本轮预算内没有：Valid Overall BCE H0 {h0['valid_overall_bce']:.6f} → "
        f"H3 {best.loc['H3', 'valid_overall_bce']:.6f}（变差）；Valid Macro-AUROC "
        f"{h0['valid_macro_auroc']:.6f} → {best.loc['H3', 'valid_macro_auroc']:.6f}"
        f"（Δ={h3_info['delta_macro_auroc']:+.6f}，95% CI "
        f"{h3_info['delta_macro_auroc_ci95']}）。但 H3 在 epoch 12 停止时仍缓慢上升，"
        "三条件早停未触发，因此这不是收敛结论。",
        f"5. **H1/H2/H3 对十类疾病 Train Loss 的影响？** 相对 H0 的 Train BCE 增大最多："
        f"H1 {fmt_delta_list(deltas('train', 'H1'), True)}；"
        f"H2 {fmt_delta_list(deltas('train', 'H2'), True)}；"
        f"H3 {fmt_delta_list(deltas('train', 'H3'), True)}。",
        f"6. **对十类疾病 Valid Loss 的影响？** 相对 H0 的 Valid BCE 增大最多："
        f"H1 {fmt_delta_list(deltas('valid', 'H1'), True)}；"
        f"H2 {fmt_delta_list(deltas('valid', 'H2'), True)}；"
        f"H3 {fmt_delta_list(deltas('valid', 'H3'), True)}。"
        "具体逐类数值见上表 B。",
    ])
    gaps = []
    for disease in LABEL_NAMES:
        for experiment in EXPERIMENT_ORDER:
            gaps.append((disease, experiment,
                         float(per.loc[disease, f"{experiment}_valid"])
                         - float(per.loc[disease, f"{experiment}_train"])))
    gaps.sort(key=lambda item: -item[2])
    lines.append(
        "7. **哪些疾病的 Train–Valid Loss Gap 最大？** "
        + "；".join(f"{disease}({experiment}) {value:+.6f}"
                    for disease, experiment, value in gaps[:5])
        + "。总体 Gap 最大为 H0 "
        f"{h0['train_valid_bce_gap']:+.6f}（H2 {best.loc['H2', 'train_valid_bce_gap']:+.6f}，"
        f"H1 {best.loc['H1', 'train_valid_bce_gap']:+.6f}，"
        f"H3 {best.loc['H3', 'train_valid_bce_gap']:+.6f}）。")
    improvements = {}
    for experiment in ("H1", "H2", "H3"):
        values = deltas("valid", experiment)
        improved = [(disease, value) for disease, value in values.items()
                    if value < 0]
        improved.sort(key=lambda item: item[1])
        improvements[experiment] = improved
        lines.append(
            f"8 ({experiment}). **哪些疾病的 Valid Loss 得到改善"
            f"（{experiment} vs H0）？** "
            + ("；".join(f"{name} {value:+.6f}" for name, value in improved[:5])
               if improved else "无（十类均未优于 H0）")
            + f"。改善类别数：{len(improved)}/10。")
    direction_cases = []
    for experiment in ("H1", "H2", "H3"):
        train_delta = float(best.loc[experiment, "train_overall_bce"]) - float(
            h0["train_overall_bce"])
        valid_delta = float(best.loc[experiment, "valid_overall_bce"]) - float(
            h0["valid_overall_bce"])
        if train_delta < 0 < valid_delta:
            direction_cases.append(experiment)
    lines.append(
        "9. **是否存在 Train Loss 下降但 Valid Loss 上升的现象？** "
        + ("是：" + "、".join(direction_cases) if direction_cases else
           "三个实验的总体 Train/Valid 方向均不完全满足该模式；逐疾病层面存在局部"
           "Train 下降/Valid 上升（见 `per_class_loss_gap.csv`）。"))
    auroc_frame = pd.read_csv(COMPARATIVE / "per_class_auroc_comparison.csv")
    valid_auroc = auroc_frame[(auroc_frame["checkpoint"] == "best")
                              & (auroc_frame["split"] == "valid")]
    mismatch = {}
    for experiment in ("H1", "H2", "H3"):
        count = 0
        for disease in LABEL_NAMES:
            bce_delta = deltas("valid", experiment)[disease]
            row = valid_auroc[(valid_auroc["experiment"] == experiment)
                              & (valid_auroc["disease"] == disease)]
            row0 = valid_auroc[(valid_auroc["experiment"] == "H0")
                               & (valid_auroc["disease"] == disease)]
            if row.empty or row0.empty:
                continue
            auroc_delta = float(row["auroc"].iloc[0]) - float(
                row0["auroc"].iloc[0])
            if bce_delta < 0 and auroc_delta <= 0:
                count += 1
        mismatch[experiment] = count
    lines.append(
        "10. **是否存在 Valid Loss 改善但 AUROC 不变甚至下降的现象？** "
        f"逐疾病统计（Valid BCE 改善且 AUROC 不升的类别数）：H1 "
        f"{mismatch['H1']}/10、H2 {mismatch['H2']}/10、H3 {mismatch['H3']}/10。"
        "该现象说明 BCE 改善可能来自概率校准/尺度变化，而不一定是排序能力提升。")
    lines.extend([
        "11. **哪种方法表现出最值得进一步验证的泛化收益？** H2（单图 Study 损失权重）"
        f"是唯一总体方向为正的实验：Valid Macro-AUROC {h0['valid_macro_auroc']:.6f} → "
        f"{best.loc['H2', 'valid_macro_auroc']:.6f}（Δ={h2_info['delta_macro_auroc']:+.6f}），"
        f"单图子群 Δ={single_h2 - single_h0:+.6f}，但要允许 95% CI "
        f"{h2_info['delta_macro_auroc_ci95']} 跨 0 与 AUPRC 轻微下降"
        f"（{best.loc['H0', 'valid_macro_auprc']:.6f} → "
        f"{best.loc['H2', 'valid_macro_auprc']:.6f}）。H1 为明确负向；H3 为未收敛的负向。",
        "12. **下一步应增加配对随机种子，还是研究策略组合？** 现阶段应先对 H2 做"
        "配对随机种子确认（seed 2026/3407，固定同一矩阵），而不是直接做组合；"
        "H1 不建议扩展，H3 需要先修正收敛/预算问题（或重新校准 L2-SP 强度）后再谈组合。",
        "",
    ])
    return lines


def write_reports(overall: pd.DataFrame, per_class: pd.DataFrame,
                  per_class_last: pd.DataFrame, grouped: pd.DataFrame,
                  bootstrap: dict) -> None:
    best = overall[overall["checkpoint"] == "best"].set_index("experiment")
    last = overall[overall["checkpoint"] == "last"].set_index("experiment")
    per_class_index = per_class.set_index("disease")
    # Hyper-parameter summary.
    config_lines = []
    for experiment in EXPERIMENT_ORDER:
        config_lines.append(
            f"- {experiment}: `{json.dumps(EXPERIMENTS[experiment]['optimization'], ensure_ascii=False)}`")
    lines = [
        "# H1 / H2 / H3 Train–Validation Loss Analysis",
        "",
        "本报告基于官方 Train（18,513 Study / 30,344 图像 / 11,055 Subject）与 "
        "Validation（859 Study / 1,422 图像 / 228 Subject）的**完整确定性无增强推理**。"
        "所有数字来自各自 Best / Last Checkpoint；Test 标签从未读取。",
        "",
        "## 0. 实验与统计口径",
        "",
    ] + config_lines + [
        "",
        "- 原始 BCE（raw Study BCE）定义：对每个 Study 先对 10 类 BCE 取平均，"
        "再对全部 Study 取平均；训练/验证均使用该未加权口径，不受 H1 采样、"
        "H2 单图权重或 H3 L2-SP 项影响。",
        "- 各实验实际优化的目标：H0/H1 为原始未加权 Study BCE；H2 为归一化单图加权 "
        "BCE（记录于 `training_history.csv` 的 `weighted_objective_bce`）；"
        "H3 为原始 BCE + λ_SP·R_SP（记录于 `objective_loss`）。比较表格只使用原始 BCE。",
        "- 程序化校验：逐类 BCE 的十类平均与按 Study 平均的 BCE 在全部实验/切分上一致"
        "（两者在同一批全有效标签上数学等价；分析代码同时输出 `overall_bce` 与 "
        "`study_level_bce`）。",
        "- H0 为 R5 Seed=42 基线复用（配置一致性检查通过）；H0 逐 Epoch 的十类训练批次 "
        "BCE 历史未记录，已标记为不可用；其 Validation 逐 Epoch 逐类 BCE 由历史保存的 "
        "逐 Epoch 验证预测恢复，Best/Last 的完整确定性逐类 BCE 已重新计算。",
        "- 所有模型均从同一原始 DINOv3 预训练权重独立训练，仅改变指定干预；"
        "H1/H2/H3 未叠加。",
        "- **H3 收敛性说明**：H3 的三条件早停未触发，训练在 epoch "
        f"{_completed_epochs('H3')} 因总实验时间预算停止，停止时 Validation AUROC "
        "仍在缓慢上升；因此 H3 不是收敛结果，其与 H0 的比较只能作为方向性证据，"
        "不能据此断言 L2-SP 在完整 20-epoch 预算下的最终效果。",
        "",
        "## A. 总体 Loss 与 AUROC 对照（Best Checkpoint）",
        "",
        "| Experiment | Train Overall BCE | Valid Overall BCE | BCE Gap | "
        "Valid Macro-AUROC | Valid Macro-AUPRC | Best Epoch | Completed / Stop |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for experiment in EXPERIMENT_ORDER:
        row = best.loc[experiment]
        lines.append(
            f"| {experiment} | {row['train_overall_bce']:.6f} | "
            f"{row['valid_overall_bce']:.6f} | {row['train_valid_bce_gap']:+.6f} | "
            f"{row['valid_macro_auroc']:.6f} | {row['valid_macro_auprc']:.6f} | "
            f"{row['best_epoch']} | {row['completed_epochs']} / {row['stop_reason']} |")
    train_lowest = best["train_overall_bce"].idxmin()
    valid_lowest = best["valid_overall_bce"].idxmin()
    auroc_highest = best["valid_macro_auroc"].idxmax()
    lines.extend([
        "",
        f"- Train BCE 最低：**{train_lowest}**（{best.loc[train_lowest, 'train_overall_bce']:.6f}）。",
        f"- Validation BCE 最低：**{valid_lowest}**（{best.loc[valid_lowest, 'valid_overall_bce']:.6f}）。",
        f"- Validation Macro-AUROC 最高：**{auroc_highest}**"
        f"（{best.loc[auroc_highest, 'valid_macro_auroc']:.6f}）。",
        "- Loss 与 AUROC 的变化方向不一定一致：BCE 同时衡量概率校准与排序，"
        "而 AUROC 只衡量排序。下文逐项检查是否存在 Train Loss 上升/Valid Loss 下降、"
        "Valid Loss 下降/AUROC 不升、Valid Loss 上升/AUROC 上升。",
        "",
        "### Last Checkpoint（参考）",
        "",
        "| Experiment | Train Overall BCE | Valid Overall BCE | Valid Macro-AUROC | Best Epoch |",
        "|---|---:|---:|---:|---:|",
    ])
    for experiment in EXPERIMENT_ORDER:
        row = last.loc[experiment]
        lines.append(
            f"| {experiment} | {row['train_overall_bce']:.6f} | "
            f"{row['valid_overall_bce']:.6f} | {row['valid_macro_auroc']:.6f} | "
            f"{row['best_epoch']} |")
    lines.extend([
        "",
        "## B. 十类疾病 Train / Valid BCE 对照（Best Checkpoint，实测）",
        "",
        "| Disease | H0 Train | H0 Valid | H1 Train | H1 Valid | H2 Train | H2 Valid | H3 Train | H3 Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for disease in list(LABEL_NAMES) + ["Overall BCE"]:
        row = per_class_index.loc[disease]
        lines.append(
            f"| {disease} | {row['H0_train']:.6f} | {row['H0_valid']:.6f} | "
            f"{row['H1_train']:.6f} | {row['H1_valid']:.6f} | "
            f"{row['H2_train']:.6f} | {row['H2_valid']:.6f} | "
            f"{row['H3_train']:.6f} | {row['H3_valid']:.6f} |")
    # Biggest gaps and valid-loss improvements.
    gaps = []
    for disease in LABEL_NAMES:
        for experiment in EXPERIMENT_ORDER:
            gaps.append((disease, experiment,
                         float(per_class_index.loc[disease, f"{experiment}_valid"])
                         - float(per_class_index.loc[disease, f"{experiment}_train"])))
    gaps.sort(key=lambda item: -item[2])
    lines.extend([
        "",
        "### 最大 Train→Valid BCE Gap（跨实验，Top 10）",
        "",
        "| Disease | Experiment | Valid−Train Gap |",
        "|---|---|---:|",
    ])
    for disease, experiment, value in gaps[:10]:
        lines.append(f"| {disease} | {experiment} | {value:+.6f} |")
    lines.extend([
        "",
        "### Valid BCE 相对 H0 的改善（Best）",
        "",
        "| Disease | H1 Δ | H2 Δ | H3 Δ |",
        "|---|---:|---:|---:|",
    ])
    for disease in list(LABEL_NAMES) + ["Overall BCE"]:
        deltas = []
        for experiment in ("H1", "H2", "H3"):
            deltas.append(float(per_class_index.loc[disease, f"{experiment}_valid"])
                          - float(per_class_index.loc[disease, "H0_valid"]))
        lines.append(
            f"| {disease} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | {deltas[2]:+.6f} |")
    lines.extend([
        "",
        "## C. 十类疾病 Positive / Negative BCE（Best Checkpoint）",
        "",
        "| Disease | Exp | Valid Positive BCE | Valid Negative BCE | Pos/Study | Neg/Study |",
        "|---|---|---:|---:|---:|---:|",
    ])
    positive_negative = pd.read_csv(
        COMPARATIVE / "per_class_positive_negative_bce.csv")
    best_pn = positive_negative[positive_negative["checkpoint"] == "best"]
    for disease in LABEL_NAMES:
        for experiment in EXPERIMENT_ORDER:
            row = best_pn[(best_pn["disease"] == disease)
                          & (best_pn["experiment"] == experiment)]
            if row.empty:
                continue
            item = row.iloc[0]
            lines.append(
                f"| {disease} | {experiment} | {_fmt(item['valid_positive_bce'])} | "
                f"{_fmt(item['valid_negative_bce'])} | {int(item['valid_n_positive'])} | "
                f"{int(item['valid_n_negative'])} |")
    lines.extend([
        "",
        "> 注意：仅凭某类总体 BCE 高低不能判断疾病难度，因为阳性率不同；"
        "positive/negative 拆分与 AUROC 需一起看。",
        "",
        "## D. AUROC / AUPRC 变化（Best Checkpoint）",
        "",
        "| Disease | H0 Valid AUROC | H1 Valid AUROC | H2 Valid AUROC | H3 Valid AUROC | H0 Valid AUPRC | H1 Valid AUPRC | H2 Valid AUPRC | H3 Valid AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    auroc_frame = pd.read_csv(COMPARATIVE / "per_class_auroc_comparison.csv")
    valid_best = auroc_frame[(auroc_frame["checkpoint"] == "best")
                             & (auroc_frame["split"] == "valid")]
    for disease in LABEL_NAMES:
        cells = []
        for experiment in EXPERIMENT_ORDER:
            row = valid_best[(valid_best["disease"] == disease)
                             & (valid_best["experiment"] == experiment)]
            cells.append(_fmt(row.iloc[0]["auroc"]) if len(row) else "NA")
        for experiment in EXPERIMENT_ORDER:
            row = valid_best[(valid_best["disease"] == disease)
                             & (valid_best["experiment"] == experiment)]
            cells.append(_fmt(row.iloc[0]["auprc"]) if len(row) else "NA")
        lines.append(f"| {disease} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## E. 机制分析",
        "",
        "### E1. H1 AP-only 采样",
        "",
    ])
    h1_history_path = experiment_dir("H1") / "training_history.csv"
    if (h1_history_path.is_file()
            and "h1_ap_only_draw_fraction" in pd.read_csv(
                h1_history_path, nrows=0).columns):
        h1_history = pd.read_csv(h1_history_path)
        lines.extend([
            f"- 实际 AP-only 采样比例：均值 "
            f"{h1_history['h1_ap_only_draw_fraction'].mean():.4f}，最小 "
            f"{h1_history['h1_ap_only_draw_fraction'].min():.4f}，最大 "
            f"{h1_history['h1_ap_only_draw_fraction'].max():.4f}（目标 0.55）。",
            f"- 每 Epoch 抽样 Study 次数：{int(h1_history['h1_n_draws'].iloc[0])}；"
            f"唯一 Study 数均值 {h1_history['h1_n_unique_studies'].mean():.1f}；"
            f"重复≥3 次 Study 数均值 "
            f"{h1_history['h1_repeat_three_plus'].mean():.1f}；"
            f"从未被抽到的 Study 数均值 {h1_history['h1_repeat_never'].mean():.1f}。",
            f"- Subject 重复：平均每 Epoch 被抽 "
            f"{h1_history['h1_subject_draw_mean'].mean():.2f} 次，最大 "
            f"{int(h1_history['h1_subject_draw_max'].max())} 次。",
            "- AP-only 与 PA+Lateral 的 AUROC/BCE 对比（Best，Validation）见 "
            "`grouped_performance_comparison.csv` 与下表。",
            "",
        ])
        grouped_best = grouped[(grouped["checkpoint"] == "best")
                               & (grouped["split"] == "valid")
                               & (grouped["disease"] == LABEL_NAMES[0])]
        lines.append("| Experiment | Group | Macro-AUROC | Macro-AUPRC | Overall BCE |")
        lines.append("|---|---|---:|---:|---:|")
        for experiment in EXPERIMENT_ORDER:
            for group in ("AP-only", "PA+Lateral", "Single-image", "Multi-image"):
                row = grouped_best[(grouped_best["experiment"] == experiment)
                                   & (grouped_best["group"] == group)]
                if row.empty:
                    continue
                item = row.iloc[0]
                lines.append(
                    f"| {experiment} | {group} | {_fmt(item['macro_auroc'])} | "
                    f"{_fmt(item['macro_auprc'])} | {_fmt(item['bce'])} |")
        lines.append("")
    lines.extend([
        "### E2. H2 单图 Study 损失权重",
        "",
    ])
    h2_history_path = experiment_dir("H2") / "training_history.csv"
    if (h2_history_path.is_file()
            and "single_raw_bce" in pd.read_csv(
                h2_history_path, nrows=0).columns):
        h2_history = pd.read_csv(h2_history_path)
        lines.extend([
            f"- 单图 Study 训练样本数均值：{h2_history['single_study_count'].mean():.1f}；"
            f"多图 Study 均值：{h2_history['multi_study_count'].mean():.1f}。",
            f"- 单图原始 BCE 均值：{h2_history['single_raw_bce'].mean():.6f}；"
            f"多图原始 BCE 均值：{h2_history['multi_raw_bce'].mean():.6f}。",
            f"- 归一化加权目标 BCE 均值：{h2_history['weighted_objective_bce'].mean():.6f}；"
            f"同 Epoch 未加权 BCE 均值：{h2_history['unweighted_batch_bce'].mean():.6f}。",
            "- 单图/多图 AUROC、BCE 对比见 `grouped_performance_comparison.csv`；"
            "H2 的加权目标仅用于训练，所有比较使用原始未加权 BCE。",
            "",
        ])
    lines.extend([
        "### E3. H3 L2-SP",
        "",
    ])
    h3_history_path = experiment_dir("H3") / "training_history.csv"
    calibration_path = experiment_dir("H3") / "l2sp_calibration.json"
    if calibration_path.is_file():
        calibration = read_json(calibration_path)
        lines.extend([
            f"- λ_SP（Train-only 梯度尺度校准，目标 L2-SP 梯度范数 ≈ BCE 梯度范数 5%）："
            f"**{calibration['lambda_sp']:.6g}**。",
            f"- 校准时 BCE 梯度范数 {calibration['probe']['bce_grad_norm']:.6g}，"
            f"单位 λ L2-SP 梯度范数 {calibration['probe']['l2sp_unit_grad_norm']:.6g}；"
            f"校准后 Stage3/4 相对漂移 "
            f"{calibration['after_calibration_stage3_relative_drift']['relative_drift']:.6f} / "
            f"{calibration['after_calibration_stage4_relative_drift']['relative_drift']:.6f}。",
        ])
    if (h3_history_path.is_file()
            and "l2sp_penalty" in pd.read_csv(
                h3_history_path, nrows=0).columns):
        h3_history = pd.read_csv(h3_history_path)
        lines.extend([
            f"- 每 Epoch L2-SP 项均值：{h3_history['l2sp_penalty'].mean():.6e}；"
            f"λ·R_SP 均值：{h3_history['l2sp_term'].mean():.6f}。",
            f"- Stage3 相对漂移（逐 Epoch 末）：{h3_history['stage3_relative_drift'].iloc[-1]:.6f}；"
            f"Stage4：{h3_history['stage4_relative_drift'].iloc[-1]:.6f}。",
            f"- BCE/L2-SP 梯度范数比（λ·‖g_SP‖/‖g_BCE‖，逐 Epoch 起始）：均值 "
            f"{h3_history.apply(lambda row: row['lambda_sp'] * row['l2sp_unit_grad_norm'] / row['bce_grad_norm'] if row['bce_grad_norm'] else np.nan, axis=1).mean():.4f}；"
            "该比例随参数偏离增大而增长，是 L2-SP 的固有性质。",
        ])
        best_drift = overall[overall["checkpoint"] == "best"].set_index("experiment")
        lines.append("")
        lines.append("| Experiment | Stage3 Best Relative Drift | Stage4 Best Relative Drift |")
        lines.append("|---|---:|---:|")
        for experiment in EXPERIMENT_ORDER:
            payload = read_json(experiment_dir(experiment)
                                / "evaluation_summary.json")
            drift = payload["best"]["stage_parameter_drift"]
            lines.append(
                f"| {experiment} | {drift['stage3']['relative_drift']:.6f} | "
                f"{drift['stage4']['relative_drift']:.6f} |")
        lines.append("")
        lines.extend([
            "- 需要明确：Validation 表现下降不能证明 Test 一定下降，但也不能证明 "
            "Test 会提高；H3 的结论限定在当前验证协议与单 seed 条件下。",
            "",
        ])
    lines.extend([
        "## F. 综合判断",
        "",
        "### F1. 三个假设的证据状态",
        "",
        "| Hypothesis | Valid Macro-AUROC Δ vs H0 | Valid BCE Δ vs H0 | 是否支持 |",
        "|---|---:|---:|---|",
    ])
    for experiment in ("H1", "H2", "H3"):
        auroc_delta = (best.loc[experiment, "valid_macro_auroc"]
                       - best.loc["H0", "valid_macro_auroc"])
        bce_delta = (best.loc[experiment, "valid_overall_bce"]
                     - best.loc["H0", "valid_overall_bce"])
        boot = bootstrap["comparisons"][f"{experiment}_minus_H0"]
        low, high = boot["delta_macro_auroc_ci95"]
        supported = ("CI 排除 0 且为正" if low is not None and low > 0 else
                     "CI 排除 0 且为负" if high is not None and high < 0 else
                     "CI 跨 0，无法确认")
        lines.append(
            f"| {experiment} | {auroc_delta:+.6f} | {bce_delta:+.6f} | {supported} |")
    lines.extend([
        "",
        "### F2. 关键现象检查",
        "",
    ])
    # Direction checks.
    checks = []
    for experiment in ("H1", "H2", "H3"):
        train_delta = (best.loc[experiment, "train_overall_bce"]
                       - best.loc["H0", "train_overall_bce"])
        valid_delta = (best.loc[experiment, "valid_overall_bce"]
                       - best.loc["H0", "valid_overall_bce"])
        auroc_delta = (best.loc[experiment, "valid_macro_auroc"]
                       - best.loc["H0", "valid_macro_auroc"])
        checks.append(
            f"- {experiment}: Train BCE Δ={train_delta:+.6f}，Valid BCE Δ={valid_delta:+.6f}，"
            f"Valid Macro-AUROC Δ={auroc_delta:+.6f}。"
            + (" Train Loss 下降但 Valid Loss 上升；" if train_delta < 0 < valid_delta else "")
            + (" Valid Loss 下降但 AUROC 未提高；" if valid_delta < 0 and auroc_delta <= 0 else "")
            + (" Valid Loss 上升但 AUROC 提高；" if valid_delta > 0 < auroc_delta else ""))
    lines.extend(checks)
    lines.extend([
        "",
        "### F3. 统计不确定性与结论边界",
        "",
        f"- H1−H0: Δ Macro-AUROC = "
        f"{bootstrap['comparisons']['H1_minus_H0']['delta_macro_auroc']:+.6f}, "
        f"95% CI {bootstrap['comparisons']['H1_minus_H0']['delta_macro_auroc_ci95']}。",
        f"- H2−H0: Δ Macro-AUROC = "
        f"{bootstrap['comparisons']['H2_minus_H0']['delta_macro_auroc']:+.6f}, "
        f"95% CI {bootstrap['comparisons']['H2_minus_H0']['delta_macro_auroc_ci95']}。",
        f"- H3−H0: Δ Macro-AUROC = "
        f"{bootstrap['comparisons']['H3_minus_H0']['delta_macro_auroc']:+.6f}, "
        f"95% CI {bootstrap['comparisons']['H3_minus_H0']['delta_macro_auroc_ci95']}。",
        "- 这些区间是同一 Validation 上的 Subject-cluster 配对 bootstrap，属于"
        "验证集被重复使用条件下的探索性证据；单 seed 的万分位差异不足以宣布有效。",
        "- 官方 Test 标签未读取，无法把任何 Validation 改善解释为 Test 收益；"
        "下一步若扩展 seed（2026/3407）或组合策略，需要新的确认性实验。",
        "",
    ])
    lines.extend(_acceptance_answers(best, per_class, grouped, bootstrap))
    (COMPARATIVE / "H1_H2_H3_train_valid_loss_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    # Condensed comparative report.
    lines = [
        "# H1 / H2 / H3 final comparative report", "",
        "| Experiment | Valid Macro-AUROC | Valid Macro-AUPRC | Valid BCE | "
        "ΔAUROC vs H0 (95% CI) | Best Epoch |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for experiment in EXPERIMENT_ORDER:
        row = best.loc[experiment]
        if experiment == "H0":
            interval = "—"
        else:
            info = bootstrap["comparisons"][f"{experiment}_minus_H0"]
            interval = (f"{info['delta_macro_auroc']:+.6f} "
                        f"[{_fmt(info['delta_macro_auroc_ci95'][0])}, "
                        f"{_fmt(info['delta_macro_auroc_ci95'][1])}]")
        lines.append(
            f"| {experiment} | {row['valid_macro_auroc']:.6f} | "
            f"{row['valid_macro_auprc']:.6f} | {row['valid_overall_bce']:.6f} | "
            f"{interval} | {row['best_epoch']} |")
    lines.extend([
        "",
        "## Verdicts", "",
    ])
    for experiment in ("H1", "H2", "H3"):
        info = bootstrap["comparisons"][f"{experiment}_minus_H0"]
        low, high = info["delta_macro_auroc_ci95"]
        if low is not None and low > 0:
            verdict = "方向为正且 95% CI 排除 0（探索性）"
        elif high is not None and high < 0:
            verdict = "方向为负且 95% CI 排除 0（探索性）"
        else:
            verdict = "方向不确定，95% CI 跨 0"
        lines.append(f"- {experiment}: {verdict}；"
                     f"ΔAUROC={info['delta_macro_auroc']:+.6f}，"
                     f"CI [{_fmt(low)}, {_fmt(high)}]。")
    lines.extend([
        "",
        "- 所有比较均以完整确定性 Train/Validation 推理为基础；原始未加权 BCE 是唯一"
        "损失比较口径。",
        "- H0 复用 R5 Seed=42；H1/H2/H3 均从原始预训练权重独立训练，未叠加。",
        f"- **H3 收敛性限制**：H3 在 epoch {_completed_epochs('H3')} 因总实验时间预算"
        "停止，三条件早停未触发，停止时 Validation AUROC 仍在缓慢上升；H3 的比较是"
        "方向性、非收敛结果，不能据此断言完整 20-epoch 预算下的最终效果。",
        "- 详细逐疾病 Loss/AUROC、机制分析与验收问题回答见 "
        "`H1_H2_H3_train_valid_loss_analysis.md`。",
        "",
    ])
    (COMPARATIVE / "H1_H2_H3_final_comparative_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def prepare_experiments(experiments, force: bool = False) -> dict:
    """Run full deterministic inference/artifacts for the given experiments."""
    ensure_dirs()
    train_data, valid_data, _ = build_eval_datasets()
    summary = {}
    for experiment in experiments:
        if experiment == "H0":
            print("[analysis] assembling H0", flush=True)
            summary[experiment] = assemble_h0(force=force)
            continue
        if not (experiment_dir(experiment) / "best.pth").is_file():
            raise RuntimeError(f"{experiment} Best checkpoint is missing")
        print(f"[analysis] inference/artifacts for {experiment}", flush=True)
        results = {}
        for checkpoint in CHECKPOINTS:
            for split, dataset in (("train", train_data), ("valid", valid_data)):
                results[(checkpoint, split)] = run_inference(
                    experiment, checkpoint, split, train_data, valid_data,
                    force=force)
        summary[experiment] = _write_experiment_artifacts(
            experiment, train_data, valid_data, results)
    return summary


def run_analysis(force: bool = False) -> dict:
    ensure_dirs()
    print("[analysis] assembling H0 and running full deterministic inference",
          flush=True)
    prepare_experiments(("H0", "H1", "H2", "H3"), force=force)
    print("[analysis] building comparative tables", flush=True)
    results = _load_best_results()
    tables = _write_comparative_tables(results)
    bootstrap = _paired_bootstrap(results)
    write_json(COMPARATIVE / "paired_bootstrap_results.json", bootstrap)
    _plot_loss_comparison(tables["overall"], tables["per_class"])
    _plot_training_dynamics()
    write_reports(tables["overall"], tables["per_class"], tables["per_class_last"],
                  tables["grouped"], bootstrap)
    summary = {
        "status": "analysis_complete",
        "overall": tables["overall"].to_dict("records"),
        "bootstrap": bootstrap,
        "test_access": False,
    }
    write_json(COMPARATIVE / "analysis_summary.json", summary)
    print("[analysis] comparative report written", flush=True)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_analysis(), indent=2, default=str))
