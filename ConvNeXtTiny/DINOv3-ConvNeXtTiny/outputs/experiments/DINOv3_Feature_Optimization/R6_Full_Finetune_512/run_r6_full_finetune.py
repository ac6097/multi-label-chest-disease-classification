from __future__ import annotations

import copy
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[3]
PROJECT_DIR = WORK_DIR.parent
R5_DIR = HERE.parent / "R5_Global_Only_512"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WORK_DIR))

from dino_analysis import LABEL_NAMES, _columns  # noqa: E402
from dino_data import make_loader, shutdown_loader  # noqa: E402
from dino_experiment import load_pretrained_backbone, save_json_atomic  # noqa: E402
from dino_feature_models import (  # noqa: E402
    build_masked_datasets,
    collate_masked_studies,
    feature_profile_optimizer,
)
from dino_feature_optimization import run_custom_group  # noqa: E402
from dino_freeze_experiment import head_hash  # noqa: E402
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from dino_model import DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from hybrid.common import seed_everything  # noqa: E402


GROUP = "R6_Full_Finetune_512"
SEED = 42
IMAGE_SIZE = 512
EXPECTED_WEIGHT_SHA256 = "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2"
EXPECTED_R5_HEAD_SHA256 = "40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_label_columns():
    return _columns(None)


def feature_groups(model):
    stages = model.backbone.model
    return [
        ("stem", nn.ModuleList([stages.downsample_layers[0]])),
        ("stage1", nn.ModuleList([stages.downsample_layers[1], stages.stages[0]])),
        ("stage2", nn.ModuleList([stages.downsample_layers[2], stages.stages[1]])),
        ("stage3", nn.ModuleList([stages.downsample_layers[3], stages.stages[2]])),
        ("stage4", stages.stages[3]),
        ("final_norm", stages.norm),
        ("global_head", model.classifier),
    ]


def make_r6_builder(expected_head_sha=EXPECTED_R5_HEAD_SHA256):
    def builder(profile, device, runtime, seed):
        if profile != "p2" or int(seed) != SEED:
            raise RuntimeError(f"R6 shared trainer contract changed: profile={profile}, seed={seed}")
        seed_everything(int(seed))
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError("R6 did not load the pinned original general-vision DINOv3 weights")
        model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS).to(device)
        initial_head_sha = head_hash(model)
        if initial_head_sha != expected_head_sha:
            raise RuntimeError(
                "R6 independently initialized Global Head differs from R5's recorded seed-42 head: "
                f"{initial_head_sha}")
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.backbone.set_gradient_checkpointing(bool(runtime["gradient_checkpointing"]))
        if runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        if total != trainable or total != 27_827_818:
            raise RuntimeError(f"R6 full-unfreeze parameter count mismatch: {trainable}/{total}")
        verification.update({
            "model_variant": "DINOv3 ConvNeXt-Tiny Global-only full fine-tuning",
            "loaded_finetuned_checkpoint": False,
            "global_classifier_init_sha256": initial_head_sha,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "training_runtime": {
                key: runtime[key] for key in (
                    "image_budget", "grad_accumulation", "gradient_checkpointing",
                    "channels_last", "num_workers", "pin_memory", "prefetch_factor",
                    "effective_images_per_update", "effective_studies_per_update",
                ) if key in runtime
            },
        })
        trainable_names = [name for name, parameter in model.named_parameters()
                           if parameter.requires_grad]
        if len(trainable_names) != len(list(model.named_parameters())):
            raise RuntimeError("R6 contains a frozen parameter despite full-finetuning mode")
        return model, verification, feature_groups(model), [], trainable_names
    return builder


def forward_global(model, images, _batch):
    return model(images)


def manual_study_mean(image_logits, study_index, num_studies):
    return torch.stack([
        image_logits[study_index == index].float().mean(dim=0)
        for index in range(int(num_studies))
    ])


def actual_backward_preflight(runtime, config, freeze, builder, fallback=False):
    data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    train_data, valid_data, data_config = build_masked_datasets(
        data_path, SEED, IMAGE_SIZE)
    if (len(train_data), int(sum(train_data.image_counts)), len(valid_data),
            int(sum(valid_data.image_counts))) != (18513, 30344, 859, 1422):
        raise RuntimeError("R6 Train/Validation counts differ from the official R5 data manifest")
    if data_config["label_semantics"] != "closed_world":
        raise RuntimeError("R6 requires the confirmed closed-world label semantics")
    if set(train_data.study_keys) & set(valid_data.study_keys):
        raise RuntimeError("R6 Train and Validation Study keys overlap")

    loader, sampler, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=0, pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    model = optimizer = None
    started = time.perf_counter()
    try:
        batches = []
        iterator = iter(loader)
        for _ in range(int(runtime["grad_accumulation"])):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        if not batches:
            raise RuntimeError("R6 actual-runtime preflight produced no Study batches")
        image_total = sum(int(batch["images"].shape[0]) for batch in batches)
        study_total = sum(int(batch["num_studies"]) for batch in batches)
        eligible_total = sum(int(batch["known"].any(dim=1).sum()) for batch in batches)
        target_effective = float(runtime.get("effective_batch_target", 32))
        if abs(image_total - target_effective) / target_effective > 0.20:
            raise RuntimeError(
                f"R6 preflight accumulation window has {image_total} images, outside 20% "
                f"of R5 effective batch target {target_effective}")

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        model, verification, modules, frozen, trainable_names = builder(
            "p2", device, runtime, SEED)
        model.train()
        optimizer = feature_profile_optimizer(model, config)
        optimizer_ids = {id(parameter) for group in optimizer.param_groups
                         for parameter in group["params"]}
        trainable_ids = {id(parameter) for parameter in model.parameters()
                         if parameter.requires_grad}
        if optimizer_ids != trainable_ids:
            raise RuntimeError("R6 optimizer groups do not contain every trainable parameter exactly once")
        group_parameters = {}
        for name, module in modules:
            candidates = [parameter for parameter in module.parameters()
                          if parameter.requires_grad]
            if candidates:
                group_parameters[name] = candidates

        optimizer.zero_grad(set_to_none=True)
        maximum_mean_error = 0.0
        for batch in batches:
            images = batch["images"].to(device, non_blocking=True)
            if runtime["channels_last"]:
                images = images.contiguous(memory_format=torch.channels_last)
            study_index = batch["study_index"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            known = batch["known"].to(device, non_blocking=True)
            image_counts = batch["image_counts"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_logits = forward_global(model, images, batch)
                study_logits = study_logit_mean(
                    image_logits.float(), study_index, batch["num_studies"], image_counts)
                manual = manual_study_mean(image_logits, study_index, batch["num_studies"])
                maximum_mean_error = max(
                    maximum_mean_error,
                    float((study_logits - manual).abs().max().detach().cpu()))
                per_study, eligible = study_bce_per_study(study_logits, labels, known)
                bce_sum = per_study[eligible].sum()
                loss = bce_sum / max(eligible_total, 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("R6 preflight Study BCE is non-finite")
            loss.backward()

        gradient_status = {}
        snapshots = {}
        for name, candidates in group_parameters.items():
            with_grad = [parameter for parameter in candidates if parameter.grad is not None]
            finite_nonzero = [parameter for parameter in with_grad
                              if torch.isfinite(parameter.grad).all()
                              and torch.count_nonzero(parameter.grad).item() > 0]
            gradient_status[name] = bool(finite_nonzero)
            if finite_nonzero:
                representative = max(
                    finite_nonzero,
                    key=lambda parameter: float(parameter.grad.detach().abs().max().cpu()))
                snapshots[name] = (representative, representative.detach().cpu().clone())
        if not all(gradient_status.values()):
            raise RuntimeError(f"R6 preflight has a missing/zero module gradient: {gradient_status}")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        update_status = {}
        for name, (parameter, snapshot) in snapshots.items():
            update_status[name] = not torch.equal(parameter.detach().cpu(), snapshot)
        if not all(update_status.values()):
            raise RuntimeError(f"R6 preflight optimizer step did not change each module group: {update_status}")
        properties = torch.cuda.get_device_properties(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if peak_reserved > int(properties.total_memory * 0.90):
            raise RuntimeError("R6 preflight exceeded the configured 90% GPU memory ceiling")
        result = {
            "passed": True,
            "runtime": runtime,
            "fallback_microbatch_used": bool(fallback),
            "preflight_images": image_total,
            "preflight_studies": study_total,
            "preflight_microbatches": len(batches),
            "optimizer_step_executed_in_disposable_preflight": True,
            "formal_training_initialized_from_original_weights_after_preflight": True,
            "study_logit_mean_max_abs_error": maximum_mean_error,
            "study_bce_finite": True,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad),
            "optimizer_parameter_groups": [
                {"name": group.get("name"), "parameter_count": sum(
                    parameter.numel() for parameter in group["params"])}
                for group in optimizer.param_groups
            ],
            "trainable_group_nonzero_gradients": gradient_status,
            "trainable_group_parameters_changed_after_optimizer_step": update_status,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "elapsed_seconds": time.perf_counter() - started,
            "test_data_accessed": False,
        }
        del batches, iterator
        return result
    finally:
        shutdown_loader(loader)
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def preflight_with_allowed_runtime(config, freeze, runtime, builder):
    try:
        return actual_backward_preflight(runtime, config, freeze, builder, fallback=False), runtime
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        smaller = copy.deepcopy(runtime)
        if int(smaller["image_budget"]) <= 4:
            raise
        old_window = int(smaller["image_budget"]) * int(smaller["grad_accumulation"])
        smaller["image_budget"] = 4
        smaller["grad_accumulation"] = math.ceil(old_window / 4)
        smaller["selected_from_r5_runtime"] = True
        smaller["runtime_change_reason"] = (
            "R6 full-backbone backward preflight exceeded GPU memory at the R5 microbatch; "
            "physical microbatch reduced to 4 and accumulation raised to preserve the "
            "effective image window. Study-level loss remains summed over complete Studies."
        )
        return actual_backward_preflight(smaller, config, freeze, builder, fallback=True), smaller


def check_prediction_frame(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required Best-checkpoint Validation predictions are missing: {path}")
    target_cols, probability_cols = get_label_columns()
    frame = pd.read_csv(path, dtype={"Subject_id": "string", "Study_id": "string"})
    required = {"Subject_id", "Study_id", *target_cols, *probability_cols}
    if required - set(frame.columns):
        raise RuntimeError(f"{path} is missing columns: {sorted(required - set(frame.columns))}")
    if frame[["Subject_id", "Study_id"]].isna().any().any():
        raise RuntimeError(f"{path} contains missing identifiers")
    if frame.duplicated(["Subject_id", "Study_id"]).any():
        raise RuntimeError(f"{path} has duplicated Study keys")
    return frame


def metric_vectors(labels, scores):
    aucs, aps = [], []
    for index in range(labels.shape[1]):
        target = labels[:, index]
        aucs.append(float(roc_auc_score(target, scores[:, index]))
                    if np.unique(target).size == 2 else float("nan"))
        aps.append(float(average_precision_score(target, scores[:, index]))
                    if np.any(target == 1) else float("nan"))
    return np.asarray(aucs), np.asarray(aps)


def r5_r6_comparison(run_dir: Path):
    target_cols, probability_cols = get_label_columns()
    r5 = check_prediction_frame(R5_DIR / "val_best.csv")
    r6 = check_prediction_frame(run_dir / "val_best.csv")
    keys = ["Subject_id", "Study_id"]
    if set(map(tuple, r5[keys].astype(str).to_numpy())) != set(
            map(tuple, r6[keys].astype(str).to_numpy())):
        raise RuntimeError("R5 and R6 Best validation prediction Study sets differ")
    merged = r5.merge(r6, on=keys, how="inner", suffixes=("_r5", "_r6"),
                      validate="one_to_one", sort=False)
    if len(merged) != 859:
        raise RuntimeError(f"Expected 859 aligned Validation Studies, got {len(merged)}")
    labels = merged[[f"{col}_r5" for col in target_cols]].to_numpy(dtype=np.int64)
    labels_r6 = merged[[f"{col}_r6" for col in target_cols]].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, labels_r6):
        raise RuntimeError("R5 and R6 Validation labels differ")
    scores = {
        "R5": merged[[f"{col}_r5" for col in probability_cols]].to_numpy(dtype=np.float64),
        "R6": merged[[f"{col}_r6" for col in probability_cols]].to_numpy(dtype=np.float64),
    }
    point = {name: metric_vectors(labels, matrix) for name, matrix in scores.items()}
    subjects = merged["Subject_id"].astype(str).to_numpy()
    unique = np.asarray(sorted(set(subjects)), dtype=object)
    rows_by_subject = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampling_digest = __import__("hashlib").sha256()
    delta_auroc, delta_auprc = [], []
    class_auc, class_ap = [[] for _ in LABEL_NAMES], [[] for _ in LABEL_NAMES]
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows_by_subject[subject] for subject in sampled])
        sampling_digest.update(indices.astype("<i4", copy=False).tobytes())
        sampling_digest.update(b"\xff")
        boot = {name: metric_vectors(labels[indices], matrix[indices])
                for name, matrix in scores.items()}
        auc_delta = boot["R6"][0] - boot["R5"][0]
        ap_delta = boot["R6"][1] - boot["R5"][1]
        if np.isfinite(auc_delta).any():
            delta_auroc.append(float(np.nanmean(auc_delta)))
        if np.isfinite(ap_delta).any():
            delta_auprc.append(float(np.nanmean(ap_delta)))
        for i in range(len(LABEL_NAMES)):
            if np.isfinite(auc_delta[i]):
                class_auc[i].append(float(auc_delta[i]))
            if np.isfinite(ap_delta[i]):
                class_ap[i].append(float(ap_delta[i]))
    if len(delta_auroc) != BOOTSTRAP_REPLICATES or len(delta_auprc) != BOOTSTRAP_REPLICATES:
        raise RuntimeError("R5/R6 bootstrap did not produce 2000 valid macro repetitions")
    summary = {
        "method": "Subject-cluster paired bootstrap; percentile 95% CI",
        "unit": "Subject_id; all Studies for a sampled Subject are resampled together",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "sampling_indices_sha256": sampling_digest.hexdigest(),
        "aligned_studies": len(merged),
        "subjects": int(len(unique)),
        "one_class_rule": "Skip per-class AUROC when a draw has one label; macro uses available classes.",
        "macro": {
            "R5_auroc": float(np.nanmean(point["R5"][0])),
            "R6_auroc": float(np.nanmean(point["R6"][0])),
            "R5_auprc": float(np.nanmean(point["R5"][1])),
            "R6_auprc": float(np.nanmean(point["R6"][1])),
            "delta_auroc": float(np.nanmean(point["R6"][0] - point["R5"][0])),
            "delta_auroc_ci95": np.quantile(delta_auroc, [0.025, 0.975]).tolist(),
            "delta_auprc": float(np.nanmean(point["R6"][1] - point["R5"][1])),
            "delta_auprc_ci95": np.quantile(delta_auprc, [0.025, 0.975]).tolist(),
        },
        "per_class": {},
    }
    rows = []
    for index, name in enumerate(LABEL_NAMES):
        auc_r5, auc_r6 = point["R5"][0][index], point["R6"][0][index]
        ap_r5, ap_r6 = point["R5"][1][index], point["R6"][1][index]
        delta_auc, delta_ap = auc_r6 - auc_r5, ap_r6 - ap_r5
        summary["per_class"][name] = {
            "positive_studies": int(labels[:, index].sum()),
            "negative_studies": int((labels[:, index] == 0).sum()),
            "r5_auroc": float(auc_r5), "r6_auroc": float(auc_r6),
            "delta_auroc": float(delta_auc),
            "delta_auroc_ci95": np.quantile(class_auc[index], [0.025, 0.975]).tolist(),
            "delta_auroc_valid_replicates": len(class_auc[index]),
            "r5_auprc": float(ap_r5), "r6_auprc": float(ap_r6),
            "delta_auprc": float(delta_ap),
            "delta_auprc_ci95": np.quantile(class_ap[index], [0.025, 0.975]).tolist(),
            "delta_auprc_valid_replicates": len(class_ap[index]),
        }
        rows.append({
            "Disease": name,
            "R5_AUROC": auc_r5, "R6_AUROC": auc_r6, "Delta_AUROC": delta_auc,
            "Delta_AUROC_CI95_Low": summary["per_class"][name]["delta_auroc_ci95"][0],
            "Delta_AUROC_CI95_High": summary["per_class"][name]["delta_auroc_ci95"][1],
            "Delta_AUROC_Valid_Replicates": summary["per_class"][name]["delta_auroc_valid_replicates"],
            "R5_AUPRC": ap_r5, "R6_AUPRC": ap_r6, "Delta_AUPRC": delta_ap,
            "Delta_AUPRC_CI95_Low": summary["per_class"][name]["delta_auprc_ci95"][0],
            "Delta_AUPRC_CI95_High": summary["per_class"][name]["delta_auprc_ci95"][1],
            "Delta_AUPRC_Valid_Replicates": summary["per_class"][name]["delta_auprc_valid_replicates"],
            "Positive_Studies": int(labels[:, index].sum()),
            "Negative_Studies": int((labels[:, index] == 0).sum()),
        })
    save_json_atomic(run_dir / "paired_subject_bootstrap.json", summary)
    per_class = pd.DataFrame(rows)
    per_class.to_csv(run_dir / "r5_vs_r6_per_class.csv", index=False)
    execution = compare_training_execution(run_dir)
    execution.to_csv(run_dir / "optimizer_updates_comparison.csv", index=False)
    write_r5_r6_report(run_dir, summary, per_class, execution)
    return summary


def compare_training_execution(run_dir: Path):
    records = []
    for name, path in (("R5", R5_DIR), ("R6", run_dir)):
        status = read_json(path / "training_status.json")
        log = pd.read_csv(path / "train_log.csv").sort_values("epoch")
        best_epoch = int(status["best_epoch"])
        best_row = log.loc[log["epoch"].astype(int) == best_epoch].iloc[0]
        through_best = int(log.loc[log["epoch"].astype(int) <= best_epoch,
                                   "optimizer_updates"].astype(int).sum())
        runtime = read_json(path / "selected_runtime_config.json")["selected_runtime"]
        verification = read_json(path / "model_verification.json")
        records.append({
            "Run": name,
            "Best_Epoch": best_epoch,
            "Completed_Epochs": int(status["completed_epochs"]),
            "Stop_Reason": status.get("stop_reason"),
            "Best_Macro_AUROC": float(status["best_macro_auroc"]),
            "Best_Train_BCE": float(best_row["train_bce"]),
            "Best_Validation_BCE": float(best_row["valid_loss"]),
            "Final_Train_BCE": float(log.iloc[-1]["train_bce"]),
            "Final_Validation_BCE": float(log.iloc[-1]["valid_loss"]),
            "Optimizer_Updates_Through_Best": through_best,
            "Optimizer_Updates_Through_Stop": int(log["optimizer_updates"].astype(int).sum()),
            "Mean_Images_Per_Second_Through_Best": float(
                log.loc[log["epoch"].astype(int) <= best_epoch, "images_per_s"].mean()),
            "Mean_Images_Per_Second_All_Epochs": float(log["images_per_s"].mean()),
            "Peak_GPU_Reserved_GiB": float(log["peak_gpu_reserved_gib"].max()),
            "Trainable_Parameters": int(verification["trainable_parameters"]),
            "Frozen_Parameters": int(verification["frozen_parameters"]),
            "Image_Budget": int(runtime["image_budget"]),
            "Gradient_Accumulation": int(runtime["grad_accumulation"]),
            "Gradient_Checkpointing": bool(runtime["gradient_checkpointing"]),
            "Channels_Last": bool(runtime["channels_last"]),
        })
    return pd.DataFrame(records)


def write_r5_r6_report(run_dir: Path, summary: dict, metrics: pd.DataFrame,
                       execution: pd.DataFrame):
    macro = summary["macro"]
    lines = [
        "# R5 / R6 训练结果对照", "",
        "R5 是已完成的 512×512 Stage3/4 Global-only 基线；R6 使用同一原始 DINOv3 通用视觉权重、同一 seed、Study 采样、BCE、优化器、学习率、调度和早停规则，独立解冻 Stem 与 Stage1/2。两者均以 Best checkpoint 进行验证比较。", "",
        f"- R5 Macro-AUROC / AUPRC：{macro['R5_auroc']:.6f} / {macro['R5_auprc']:.6f}",
        f"- R6 Macro-AUROC / AUPRC：{macro['R6_auroc']:.6f} / {macro['R6_auprc']:.6f}",
        f"- R6−R5 Macro-AUROC：{macro['delta_auroc']:+.6f}，95% CI [{macro['delta_auroc_ci95'][0]:+.6f}, {macro['delta_auroc_ci95'][1]:+.6f}]",
        f"- R6−R5 Macro-AUPRC：{macro['delta_auprc']:+.6f}，95% CI [{macro['delta_auprc_ci95'][0]:+.6f}, {macro['delta_auprc_ci95'][1]:+.6f}]",
        f"- Subject-cluster paired bootstrap：{summary['replicates']} 次，Seed={summary['seed']}；{summary['aligned_studies']} 个 Study / {summary['subjects']} 个 Subject。",
        "- 置信区间反映同一 Validation 上的配对抽样不确定性；最佳 checkpoint 已通过该 Validation 选择，且多类别比较为探索性结果。", "",
        "## 训练过程对照", "",
        "| Run | Best Epoch | Completed | Stop | Updates through Best / Stop | Mean images/s through Best | Peak reserved GiB | Trainable / Frozen Params |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in execution.itertuples(index=False):
        lines.append(
            f"| {row.Run} | {row.Best_Epoch} | {row.Completed_Epochs} | `{row.Stop_Reason}` | "
            f"{row.Optimizer_Updates_Through_Best:,} / {row.Optimizer_Updates_Through_Stop:,} | "
            f"{row.Mean_Images_Per_Second_Through_Best:.2f} | {row.Peak_GPU_Reserved_GiB:.2f} | "
            f"{row.Trainable_Parameters:,} / {row.Frozen_Parameters:,} |")
    lines += ["", "Train/Validation Loss at Best and Final Epoch are recorded in `optimizer_updates_comparison.csv`; complete Epoch curves are in `train_log.csv` and `train_valid_loss.png`.", "",
        "## 逐类别结果", "",
        "| Disease | R5 AUROC | R6 AUROC | Δ AUROC (95% CI) | R5 AUPRC | R6 AUPRC | Δ AUPRC (95% CI) | Pos/Neg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.Disease} | {row.R5_AUROC:.4f} | {row.R6_AUROC:.4f} | "
            f"{row.Delta_AUROC:+.4f} [{row.Delta_AUROC_CI95_Low:+.4f}, {row.Delta_AUROC_CI95_High:+.4f}] | "
            f"{row.R5_AUPRC:.4f} | {row.R6_AUPRC:.4f} | "
            f"{row.Delta_AUPRC:+.4f} [{row.Delta_AUPRC_CI95_Low:+.4f}, {row.Delta_AUPRC_CI95_High:+.4f}] | "
            f"{row.Positive_Studies}/{row.Negative_Studies} |"
        )
    lines += [
        "", "所有逐类指标均由 R5/R6 各自 Best checkpoint 的 Study Logit Mean 概率计算；Study 按 `(Subject_id, Study_id)` 键严格对齐。Bootstrap 对同一 Subject 下所有 Study 成组重采样；单一类别在某次抽样中缺失阳性或阴性时跳过该类指标，宏平均使用可计算类别。", "",
        "该比较用于估计全量微调相对于 R5 的表现差异，不能消除单次随机种子及 Validation 选模带来的不确定性。", "",
    ]
    (run_dir / "r5_vs_r6_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def save_r6_config(run_dir, runtime, preflight, status):
    config_path = run_dir / "config.json"
    payload = read_json(config_path)
    payload["profile"] = "Full fine-tuning: Stem/Stage1/Stage2/Stage3/Stage4/Final Norm/Global Head trainable"
    payload["trainable_policy"] = "all model parameters are requires_grad=True and included in AdamW groups"
    payload["experiment_config"]["experiment"] = "DINOv3_Feature_Optimization_R6_Full_Finetune"
    payload["experiment_config"]["group"] = GROUP
    payload["experiment_config"]["profile"] = "full_finetune"
    for freeze_payload in (
        payload.get("freeze_config"),
        payload.get("experiment_config", {}).get("freeze"),
    ):
        if isinstance(freeze_payload, dict):
            freeze_payload["profiles"] = ["full_finetune"]
            freeze_payload["experiment"] = "DINOv3_Feature_Optimization_R6_Full_Finetune"
            freeze_payload["trainable_policy"] = "Stem+Stage1+Stage2+Stage3+Stage4+final_norm+global_head"
    payload["training_runtime"] = runtime
    payload["preflight"] = preflight
    payload["test_access"] = False
    payload["test_split_read"] = False
    save_json_atomic(config_path, payload)
    for file_name in ("training_status.json", "training_result.json"):
        path = run_dir / file_name
        if path.is_file():
            record = read_json(path)
            record["profile"] = "full_finetune"
            record["training_runtime"] = runtime
            record["test_data_accessed"] = False
            save_json_atomic(path, record)
    model_verification = read_json(run_dir / "model_verification.json")
    model_verification["profile"] = "full_finetune"
    model_verification["trainable_parameters"] = model_verification["total_parameters"]
    model_verification["frozen_parameters"] = 0
    model_verification["trainable_policy"] = "all Stem, Stage1-4, final normalization, and Global Head parameters"
    model_verification["optimizer_preflight"] = preflight
    save_json_atomic(run_dir / "model_verification.json", model_verification)
    report_lines = [
        "# R6 全量微调模型核验", "",
        f"- 初始化：`facebook/dinov3-convnext-tiny-pretrain-lvd1689m` revision `10d30274b4d445111e2d5bf75ac93bbd94db274b`，SHA-256 `{EXPECTED_WEIGHT_SHA256}`。",
        f"- Global Head 按 seed={SEED} 独立初始化，初始 SHA-256 `{EXPECTED_R5_HEAD_SHA256}`，与 R5 初始分类头一致；没有加载 R5/R3 监督微调权重。",
        "- 可训练范围：Stem、Stage1、Stage2、Stage3、Stage4、Final Norm、Global Head；冻结参数为 0。",
        f"- Total / trainable / frozen parameters：{model_verification['total_parameters']:,} / {model_verification['trainable_parameters']:,} / {model_verification['frozen_parameters']:,}。",
        f"- Optimizer groups：`{json.dumps(preflight['optimizer_parameter_groups'], ensure_ascii=False)}`。",
        f"- R5 对齐训练配置：512×512、AdamW、Study BCE、seed 42、相同 LR/weight decay/warmup+cosine/gradient clip/BF16/Study sampler/augmentation/三条件早停；R6 最大 epoch={status['max_epochs']}。",
        f"- Runtime：image budget={runtime['image_budget']}、accumulation={runtime['grad_accumulation']}、channels-last={runtime['channels_last']}、gradient checkpointing={runtime['gradient_checkpointing']}。",
        f"- 全量微调真实 Forward/Backward/Optimizer Step preflight：通过；{preflight['preflight_images']} images / {preflight['preflight_studies']} Studies / {preflight['preflight_microbatches']} microbatches；Study mean 最大误差 {preflight['study_logit_mean_max_abs_error']:.3g}。",
        f"- 每个可训练模块均有非零梯度且在 disposable preflight optimizer step 后参数发生变化：`{all(preflight['trainable_group_parameters_changed_after_optimizer_step'].values())}`。正式训练重新从原始 DINOv3 权重初始化。",
        f"- Best Epoch={status['best_epoch']}，Macro-AUROC={status['best_macro_auroc']:.6f}；停止原因 `{status['stop_reason']}`。",
        "- 训练和评估只读取官方 Train / Validation；Test 未读取。", "",
    ]
    (run_dir / "model_verification.md").write_text("\n".join(report_lines), encoding="utf-8")
    training_report = run_dir / "training_report.md"
    if training_report.is_file():
        existing = training_report.read_text(encoding="utf-8")
        section = (
            "\n## R6 trainable scope\n\n"
            "R6 is a standalone full-finetuning run. Stem, Stage 1, Stage 2, Stage 3, Stage 4, "
            "final normalization, and Global Head are trainable; no Local Head is present. "
            "All trainable parameters are included in AdamW.\n"
        )
        if "## R6 trainable scope" not in existing:
            training_report.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("R6 requires CUDA with BF16 support, matching the R5 training runtime")
    if not R5_DIR.is_dir() or not (R5_DIR / "best.pth").is_file():
        raise FileNotFoundError("Completed R5 Best checkpoint is required")
    r5_payload = read_json(R5_DIR / "config.json")
    r5_status = read_json(R5_DIR / "training_status.json")
    if not r5_status.get("complete") or int(r5_status["best_epoch"]) != 6:
        raise RuntimeError("R5 status/checkpoint does not match the completed reference experiment")
    if r5_payload["initialization_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R5 original DINOv3 weight SHA does not match the pinned revision")
    r5_runtime_payload = read_json(R5_DIR / "selected_runtime_config.json")
    runtime_payload = copy.deepcopy(r5_runtime_payload)
    runtime = copy.deepcopy(r5_runtime_payload["selected_runtime"])
    runtime["source_group"] = "R5_Global_Only_512"
    runtime["selection_rule"] = "Reused R5 selected runtime; no new throughput or LR search."
    feature = copy.deepcopy(r5_payload["feature_config"])
    config = copy.deepcopy(r5_payload["experiment_config"])
    freeze = copy.deepcopy(r5_payload["freeze_config"])
    if int(config["seed"]) != SEED or int(config["training"]["warmup_epochs"]) != 2:
        raise RuntimeError("R5 seed or training schedule differs from the declared R6 control")
    if int(config["training"]["epochs"]) != 20:
        raise RuntimeError("R6 maximum epochs must remain 20")
    if float(config["training"]["peak_lr_backbone"]) != 1e-5:
        raise RuntimeError("R6 backbone learning rate must remain the R5 value")

    feature = copy.deepcopy(feature)
    feature["outputs"]["root"] = str(HERE.parent)
    config["outputs"]["root"] = str(HERE.parent)
    config["model"]["architecture"] = "DINOv3ConvNeXtTinyGlobalOnlyFullFinetune"
    config["model"]["study_fusion"] = "mean_raw_image_logits_before_sigmoid"
    freeze["experiment"] = "DINOv3_Feature_Optimization_R6_Full_Finetune"
    freeze["outputs_root"] = str(HERE.parent)
    freeze["profiles"] = ["p2"]
    freeze["training"]["max_epochs"] = 20
    freeze["submission"]["selected_model"] = GROUP

    builder = make_r6_builder()
    selected_runtime_path = HERE / "selected_runtime_config.json"
    preflight_path = HERE / "actual_runtime_preflight.json"
    if selected_runtime_path.is_file() and preflight_path.is_file():
        runtime_payload = read_json(selected_runtime_path)
        runtime = copy.deepcopy(runtime_payload["selected_runtime"])
        preflight = read_json(preflight_path)
        if (runtime_payload.get("group") != GROUP or not preflight.get("passed")
                or runtime_payload.get("source_group") != "R5_Global_Only_512"):
            raise RuntimeError("Existing R6 runtime/preflight records do not match this experiment")
        if runtime.get("preflight_passed") is not True:
            raise RuntimeError("Existing R6 runtime record is missing successful full-unfreeze preflight")
    else:
        print("Starting R6 full-unfreeze runtime preflight using the R5-selected Study batch.",
              flush=True)
        preflight, runtime = preflight_with_allowed_runtime(config, freeze, runtime, builder)
        runtime["preflight_passed"] = True
        runtime["preflight_peak_allocated_bytes"] = preflight["peak_allocated_bytes"]
        runtime["preflight_peak_reserved_bytes"] = preflight["peak_reserved_bytes"]
        runtime_payload["group"] = GROUP
        runtime_payload["selected_runtime"] = runtime
        runtime_payload["source_group"] = "R5_Global_Only_512"
        runtime_payload["selection_rule"] = "Strict reuse of R5 runtime; no new benchmark search."
        save_json_atomic(selected_runtime_path, runtime_payload)
        save_json_atomic(preflight_path, preflight)
        preflight_md = [
            "# R6 Actual Runtime Preflight", "",
            f"- Passed: {preflight['passed']}; optimizer step was executed only on the disposable preflight model.",
            f"- Runtime: image budget {runtime['image_budget']}, accumulation {runtime['grad_accumulation']}, channels-last {runtime['channels_last']}, checkpointing {runtime['gradient_checkpointing']}.",
            f"- Preflight window: {preflight['preflight_images']} images / {preflight['preflight_studies']} complete Studies / {preflight['preflight_microbatches']} microbatches.",
            f"- Peak allocated/reserved: {preflight['peak_allocated_bytes'] / (1024 ** 3):.2f}/{preflight['peak_reserved_bytes'] / (1024 ** 3):.2f} GiB.",
            "- All trainable module groups produced nonzero gradients and changed parameters after the optimizer step.",
            "- Formal training builds a new model from the original pinned DINOv3 weights and the same seed-42 Global Head initialization.",
            "",
        ]
        (HERE / "actual_runtime_preflight.md").write_text("\n".join(preflight_md), encoding="utf-8")

    result = run_custom_group(
        feature, config, freeze, "R6", GROUP, IMAGE_SIZE,
        {"repository": r5_payload["initialization_repo"],
         "revision": r5_payload["initialization_revision"],
         "weights_sha256": EXPECTED_WEIGHT_SHA256},
        model_builder=builder,
        optimizer_builder=feature_profile_optimizer,
        batch_forward=forward_global,
        validation_forward=forward_global,
        collate_fn=collate_masked_studies,
        dataset_builder=build_masked_datasets,
        run_context={
            "group": GROUP,
            "architecture": "dinov3_tiny_global_only_full_finetune",
            "image_size": IMAGE_SIZE,
            "head_type": "global_only_gap",
            "pretrained_initialization_sha256": EXPECTED_WEIGHT_SHA256,
            "global_classifier_init_sha256": EXPECTED_R5_HEAD_SHA256,
            "runtime_source": "R5_Global_Only_512/selected_runtime_config.json",
            "trainable_policy": "Stem+Stage1+Stage2+Stage3+Stage4+final_norm+global_head",
            "test_access": False,
        },
        expected_weight_sha256=EXPECTED_WEIGHT_SHA256)
    status = read_json(HERE / "training_status.json")
    if not status.get("complete"):
        raise RuntimeError(f"R6 training has not completed or early-stopped: {status}")
    if status.get("stop_reason") not in {"three_condition_early_stop", "max_epochs_reached"}:
        raise RuntimeError(f"Unexpected R6 stop reason: {status.get('stop_reason')}")
    if not (HERE / "best.pth").is_file() or not (HERE / "last.pth").is_file():
        raise RuntimeError("R6 did not preserve both Best and Last checkpoints")

    save_r6_config(HERE, runtime, preflight, status)
    r5_r6_comparison(HERE)
    print(json.dumps({
        "group": GROUP, "completed_epochs": status["completed_epochs"],
        "best_epoch": status["best_epoch"], "best_macro_auroc": status["best_macro_auroc"],
        "stop_reason": status["stop_reason"], "result": result,
        "output_dir": str(HERE),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
