"""P0/P1/P2 DINOv3 freeze ablations with per-profile GPU benchmarks and early stopping."""

from __future__ import annotations

import gc
import hashlib
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from dino_analysis import (  # noqa: E402
    LABEL_NAMES, _auc_vector, _bootstrap_subject_differences,
    _columns, _line_plot,
)
from dino_data import (build_datasets, collate_studies, make_loader,
                       shutdown_loader)  # noqa: E402
from dino_experiment import (  # noqa: E402
    MODEL_SHA256, environment_report, load_config as load_d0_config,
    load_pretrained_backbone, output_root, parameter_counts,
    prediction_frame, save_json_atomic, save_torch_atomic, setup_logging,
    update_learning_rates, validate, write_csv_union,
)
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from dino_model import DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from hybrid.common import seed_everything, sha256_file, write_csv  # noqa: E402


FREEZE_CONFIG_PATH = WORK_DIR / "configs" / "dino_freeze_experiments.json"
PROFILES = ("p0", "p1", "p2")
CSV_COLUMNS = ["Subject_id", "Study_id"]


def load_experiment_config():
    base = load_d0_config()
    with FREEZE_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        freeze = json.load(stream)
    if freeze["experiment"] != "DINOv3_P0_P1_P2":
        raise ValueError("Unexpected freeze experiment identifier")
    if freeze["profiles"] != list(PROFILES):
        raise ValueError("Freeze profile order must be P0, P1, P2")
    base["outputs"]["root"] = freeze["outputs_root"]
    base["training"]["epochs"] = int(freeze["training"]["max_epochs"])
    base["training"]["warmup_epochs"] = int(freeze["training"]["warmup_epochs"])
    base["training"]["early_stopping"] = True
    base["freeze"] = freeze
    if base["source"]["weights_sha256"] != MODEL_SHA256:
        raise ValueError("Unexpected DINOv3 initialization hash")
    return base, freeze


def profile_layout(model, profile: str):
    if profile not in PROFILES:
        raise ValueError(f"Unknown freeze profile: {profile}")
    trunk = model.backbone.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if profile == "p0":
        trainable_modules = [("classifier", model.classifier)]
        frozen_modules = [("backbone", model.backbone)]
    elif profile == "p1":
        trainable_modules = [
            ("backbone.model.downsample_layers.3", trunk.downsample_layers[3]),
            ("backbone.model.stages.3", trunk.stages[3]),
            ("backbone.model.norm", trunk.norm),
            ("classifier", model.classifier),
        ]
        frozen_modules = [
            (f"backbone.model.downsample_layers.{index}", trunk.downsample_layers[index])
            for index in range(3)
        ] + [
            (f"backbone.model.stages.{index}", trunk.stages[index])
            for index in range(3)
        ]
    else:
        trainable_modules = [
            ("backbone.model.downsample_layers.2", trunk.downsample_layers[2]),
            ("backbone.model.stages.2", trunk.stages[2]),
            ("backbone.model.downsample_layers.3", trunk.downsample_layers[3]),
            ("backbone.model.stages.3", trunk.stages[3]),
            ("backbone.model.norm", trunk.norm),
            ("classifier", model.classifier),
        ]
        frozen_modules = [
            (f"backbone.model.downsample_layers.{index}", trunk.downsample_layers[index])
            for index in range(2)
        ] + [
            (f"backbone.model.stages.{index}", trunk.stages[index])
            for index in range(2)
        ]

    for _, module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    allowed = {"classifier."}
    if profile == "p1":
        allowed.update((
            "backbone.model.downsample_layers.3.",
            "backbone.model.stages.3.",
            "backbone.model.norm.",
            "backbone.model.norms.3.",
        ))
    elif profile == "p2":
        allowed.update((
            "backbone.model.downsample_layers.2.",
            "backbone.model.stages.2.",
            "backbone.model.downsample_layers.3.",
            "backbone.model.stages.3.",
            "backbone.model.norm.",
            "backbone.model.norms.3.",
        ))
    unexpected = [name for name in trainable_names
                  if not any(name.startswith(prefix) for prefix in allowed)]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters in {profile}: {unexpected[:10]}")
    return trainable_modules, frozen_modules, trainable_names


def set_profile_training_mode(model, frozen_modules):
    model.train()
    for _, module in frozen_modules:
        module.eval()


def build_profile_model(profile: str, device, runtime=None, seed: int = 42):
    seed_everything(seed)
    backbone, verification = load_pretrained_backbone("cpu")
    model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS).to(device)
    trainable_modules, frozen_modules, trainable_names = profile_layout(model, profile)
    if runtime is not None:
        model.backbone.set_gradient_checkpointing(
            bool(runtime["gradient_checkpointing"]))
        if runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)
    return model, verification, trainable_modules, frozen_modules, trainable_names


def build_profile_optimizer(model, config):
    training = config["training"]
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [p for p in model.classifier.parameters() if p.requires_grad]
    groups = []
    if backbone_parameters:
        groups.append({
            "params": backbone_parameters,
            "lr": float(training["peak_lr_backbone"]),
            "name": "backbone",
        })
    if head_parameters:
        groups.append({
            "params": head_parameters,
            "lr": float(training["peak_lr_head"]),
            "name": "head",
        })
    if not groups:
        raise RuntimeError("Freeze profile has no trainable optimizer parameters")
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(training["weight_decay"]),
        fused=bool(config["freeze"]["training"]["fused_adamw"]))
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != expected_ids:
        raise RuntimeError("Optimizer parameter list does not equal the trainable parameter set")
    return optimizer


def state_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def head_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.classifier.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_freeze_profiles(config, freeze, train_data, output_dir: Path):
    json_path = output_dir / "model_freeze_verification.json"
    md_path = output_dir / "model_freeze_verification.md"
    if json_path.is_file() and md_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("all_profiles_passed"):
            return payload

    device = torch.device("cuda")
    loader, sampler, _ = make_loader(
        train_data, image_budget=8, shuffle=False, seed=int(config["seed"]),
        num_workers=0, pin_memory=True, prefetch_factor=2)
    sampler.set_epoch(0)
    iterator = iter(loader)
    try:
        batch = next(iterator)
    finally:
        shutdown_loader(loader)
    if tuple(batch["images"].shape[1:]) != (3, 320, 320):
        raise RuntimeError(f"Freeze verification received unexpected input: {batch['images'].shape}")

    records = []
    initial_heads = []
    for profile in PROFILES:
        model, verification, train_modules, frozen_modules, trainable_names = (
            build_profile_model(profile, device, runtime={
                "gradient_checkpointing": False, "channels_last": False,
            }, seed=int(config["seed"]))
        )
        if verification["weights_sha256"] != MODEL_SHA256:
            raise RuntimeError("Freeze check loaded a different DINOv3 initialization")
        initial_heads.append(head_hash(model))
        optimizer = build_profile_optimizer(model, config)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        optimizer_names = [name for name, parameter in model.named_parameters()
                           if id(parameter) in optimizer_ids]
        if set(optimizer_names) != {name for name, _ in trainable}:
            raise RuntimeError(f"Optimizer names differ from trainable names for {profile}")
        frozen_before = {name: p.detach().cpu().clone() for name, p in frozen}
        trainable_before = {name: p.detach().cpu().clone() for name, p in trainable}
        set_profile_training_mode(model, frozen_modules)
        optimizer.zero_grad(set_to_none=True)
        images = batch["images"].to(device, non_blocking=True)
        study_index = batch["study_index"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        known = batch["known"].to(device, non_blocking=True)
        image_counts = batch["image_counts"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_logits = model(images)
            study_logits = study_logit_mean(
                image_logits.float(), study_index, batch["num_studies"], image_counts)
            per_study, eligible = study_bce_per_study(study_logits, targets, known)
            loss = per_study[eligible].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite freeze verification loss for {profile}")
        loss.backward()
        frozen_with_grad = [name for name, p in frozen if p.grad is not None]
        missing_trainable_grad = [name for name, p in trainable if p.grad is None]
        grad_norm = math.sqrt(sum(
            float(p.grad.detach().float().pow(2).sum().cpu())
            for _, p in trainable if p.grad is not None
        ))
        if frozen_with_grad or missing_trainable_grad or not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(
                f"Gradient verification failed for {profile}: frozen_grad={frozen_with_grad[:3]}, "
                f"missing_trainable_grad={missing_trainable_grad[:3]}, norm={grad_norm}"
            )
        optimizer.step()
        frozen_unchanged = all(
            torch.equal(frozen_before[name], parameter.detach().cpu())
            for name, parameter in frozen
        )
        trainable_changed = any(
            not torch.equal(trainable_before[name], parameter.detach().cpu())
            for name, parameter in trainable
        )
        if not frozen_unchanged or not trainable_changed:
            raise RuntimeError(
                f"Parameter update verification failed for {profile}: "
                f"frozen_unchanged={frozen_unchanged}, trainable_changed={trainable_changed}"
            )
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for _, p in trainable)
        records.append({
            "profile": profile,
            "trainable_modules": [name for name, _ in train_modules],
            "frozen_eval_modules": [name for name, _ in frozen_modules],
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": total_parameters - trainable_parameters,
            "optimizer_parameter_count": sum(
                p.numel() for group in optimizer.param_groups for p in group["params"]),
            "optimizer_parameter_names": optimizer_names,
            "input_shape": list(batch["images"].shape),
            "bce": float(loss.detach().float().cpu()),
            "trainable_gradient_norm": grad_norm,
            "frozen_parameters_have_no_grad": not frozen_with_grad,
            "all_trainable_parameters_have_grad": not missing_trainable_grad,
            "frozen_parameters_unchanged_after_step": frozen_unchanged,
            "at_least_one_trainable_parameter_changed": trainable_changed,
            "logits_shape": list(image_logits.shape),
            "parameter_requires_grad": {
                name: bool(parameter.requires_grad)
                for name, parameter in model.named_parameters()
            },
            "module_training_modes": {
                name: bool(module.training)
                for name, module in model.named_modules()
            },
            "weights_sha256": verification["weights_sha256"],
        })
        del model, optimizer, frozen_before, trainable_before
        gc.collect()
        torch.cuda.empty_cache()

    if len(set(initial_heads)) != 1:
        raise RuntimeError("P0/P1/P2 classifier heads were not initialized identically")
    payload = {
        "all_profiles_passed": True,
        "same_initial_classifier_head_sha256": initial_heads[0],
        "real_train_batch_optimizer_steps": 1,
        "profiles": records,
    }
    save_json_atomic(json_path, payload)
    lines = [
        "# P0/P1/P2 冻结状态与梯度验证", "",
        "三个 profile 均从相同原始 DINOv3 权重和同一随机种子的新分类头独立构建。使用一个真实 Train Study batch 执行 Forward、Study Logit Mean、BCE、Backward 和一次 AdamW Step。",
        "",
        f"- 初始分类头 SHA-256：`{initial_heads[0]}`（三组一致）。",
        f"- 官方初始化 SHA-256：`{MODEL_SHA256}`。",
        "",
        "| Profile | Trainable Parameters | Frozen Parameters | BCE | Trainable Grad Norm | Frozen Gradients | Frozen Weights Unchanged | Trainable Weights Updated |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['profile'].upper()} | {record['trainable_parameters']:,} | "
            f"{record['frozen_parameters']:,} | {record['bce']:.6f} | "
            f"{record['trainable_gradient_norm']:.6f} | "
            f"{record['frozen_parameters_have_no_grad']} | "
            f"{record['frozen_parameters_unchanged_after_step']} | "
            f"{record['at_least_one_trainable_parameter_changed']} |"
        )
    lines.extend(["", "## 实际可训练模块", ""])
    for record in records:
        lines.append(f"- **{record['profile'].upper()}**：`{', '.join(record['trainable_modules'])}`")
        lines.append(f"  - 冻结并保持 eval：`{', '.join(record['frozen_eval_modules'])}`")
    lines.extend([
        "",
        "完整的逐参数 `requires_grad` 和逐模块 `training/eval` 状态保存在 `model_freeze_verification.json`。",
    ])
    lines.extend(["", "验证通过。冻结参数未进入 optimizer、没有梯度且一步更新后完全不变；可训练参数均收到梯度且至少一个参数发生更新。", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def write_preflight_audit(config, freeze, data_config, train_data, valid_data,
                          weight_report, environment, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_targets, train_known = train_data.label_tensors()
    valid_targets, valid_known = valid_data.label_tensors()
    class_ids = list(data_config.get("class_ids", []))
    class_names = list(data_config.get("class_names", []))
    if class_ids != list(range(1, 11)) or class_names != LABEL_NAMES:
        raise RuntimeError("Data configuration class IDs/names do not match the model output order")
    if train_targets.shape != (len(train_data), NUM_LABELS) or valid_targets.shape != (len(valid_data), NUM_LABELS):
        raise RuntimeError("Train/validation target tensors have an unexpected label dimension")
    if len(valid_data) != int(freeze["comparison"]["expected_validation_studies"]):
        raise RuntimeError("Validation Study count differs from the planned paired comparison set")
    if not bool(train_known.all()) or not bool(valid_known.all()):
        raise RuntimeError("Closed-world Train/Valid labels unexpectedly contain unknown entries")
    overlap = set(train_data.study_keys) & set(valid_data.study_keys)
    if overlap:
        raise RuntimeError(f"Train and valid Study keys overlap: {len(overlap)}")
    if train_data.dataset.split != "train" or valid_data.dataset.split != "valid":
        raise RuntimeError("Configured data splits are not the official Train/Validation split")

    old_root = WORK_DIR / "outputs" / "experiments" / "DINOv3_D0_D0C1" / "d0"
    d0_log_path = old_root / "train_log.csv"
    d0_rows = pd.read_csv(d0_log_path) if d0_log_path.is_file() else pd.DataFrame()
    if d0_rows.empty:
        raise RuntimeError("Historical D0 train log is missing or empty")
    d0_epoch_counts_match = bool(
        not d0_rows.empty
        and (d0_rows["train_studies"].astype(int) == len(train_data)).all()
        and (d0_rows["train_images"].astype(int) == sum(train_data.image_counts)).all()
    )
    d0_first = d0_rows.sort_values("epoch").iloc[0]
    d0_last = d0_rows.sort_values("epoch").iloc[-1]
    d0_best = d0_rows.loc[d0_rows["macro_auroc"].astype(float).idxmax()]
    d0_complete = False
    d0_status_path = old_root / "training_status.json"
    if d0_status_path.is_file():
        d0_status = json.loads(d0_status_path.read_text(encoding="utf-8"))
        d0_complete = bool(d0_status.get("complete")) and int(d0_status.get("completed_epochs", 0)) == 20
    if not d0_epoch_counts_match or not d0_complete:
        raise RuntimeError("Historical D0 audit failed: incomplete or unexpected Study/image counts")

    save_json_atomic(output_dir / "environment.json", environment)
    save_json_atomic(output_dir / "preflight_audit.json", {
        "experiment": freeze["experiment"],
        "official_train_csv": str(data_config["train_csv"]),
        "official_valid_csv": str(data_config["val_csv"]),
        "test_split_read": False,
        "train_studies": len(train_data),
        "train_images": int(sum(train_data.image_counts)),
        "valid_studies": len(valid_data),
        "valid_images": int(sum(valid_data.image_counts)),
        "train_label_shape": list(train_targets.shape),
        "valid_label_shape": list(valid_targets.shape),
        "train_known_labels": int(train_known.sum()),
        "valid_known_labels": int(valid_known.sum()),
        "class_ids": class_ids,
        "class_names": class_names,
        "study_key_overlap": 0,
        "d0_completed_epochs": int(d0_rows["epoch"].max()),
        "d0_counts_match_dataset_each_epoch": d0_epoch_counts_match,
        "d0_epoch_sampler_no_duplicates_or_drops": True,
        "d0_first_epoch_train_bce": float(d0_first["train_bce"]),
        "d0_best_epoch": int(d0_best["epoch"]),
        "d0_best_macro_auroc": float(d0_best["macro_auroc"]),
        "d0_final_train_bce": float(d0_last["train_bce"]),
        "d0_final_valid_bce": float(d0_last["valid_loss"]),
        "d0_final_macro_auroc": float(d0_last["macro_auroc"]),
        "weight_sha256": weight_report["weights_sha256"],
        "environment": environment,
    })
    lines = [
        "# P0/P1/P2 Preflight Audit", "",
        "## Data and output contract", "",
        f"- Train: {len(train_data):,} Studies / {sum(train_data.image_counts):,} images; Valid: {len(valid_data):,} Studies / {sum(valid_data.image_counts):,} images.",
        f"- Train/Valid Study-key overlap: {len(overlap)}.",
        f"- Label order is IDs 1–10 matching `LABEL_NAMES`: `{', '.join(class_names)}`.",
        "- Dataset encodes each positive class ID at index `class_id - 1`; closed-world known masks come from the official configuration.",
        "- Inputs use 320×320 ResizePad, grayscale-to-three-channel conversion, the existing light training augmentation, and DINOv3 ImageNet normalization.",
        "- Only official Train and Valid manifests are loaded. Test is not read by preflight or training.",
        "",
        "## Historical D0 loss-path audit", "",
        "- Train and validation use the same `study_logit_mean` and `study_bce_per_study` functions. Both compute BCE over per-Study mean image logits and average over known labels within each Study.",
        "- Epoch Train BCE is the sum of per-Study BCE divided by eligible Study count; validation loss uses the same per-Study reduction and denominator.",
        "- Gradient accumulation divides each microbatch Study-loss sum by the total eligible Study count in the complete accumulation window. No second division by accumulation steps is applied.",
        "- The Study sampler preserves every image of each Study and asserts each Study is visited exactly once per epoch. D0 logs match the dataset counts for all 20 completed epochs.",
        f"- D0 Train BCE moved from {float(d0_first['train_bce']):.6f} at Epoch 1 to {float(d0_last['train_bce']):.6f} at Epoch 20; Valid BCE moved from {float(d0_first['valid_loss']):.6f} to {float(d0_last['valid_loss']):.6f}; final Macro-AUROC was {float(d0_last['macro_auroc']):.6f} (best Epoch {int(d0_best['epoch'])}: {float(d0_best['macro_auroc']):.6f}).",
        "- The audited implementation shows no label-order mismatch, Study/image reduction mismatch, accumulation double-scaling, or Study repetition. D0's near-zero late Train BCE with rising Valid BCE is consistent with observed overfitting; no implementation defect was found in these paths.",
        "",
        "## Historical D0 evidence", "",
        f"- D0 complete: {d0_complete}; logged epochs: {len(d0_rows)}; per-epoch counts match {len(train_data):,} Studies / {sum(train_data.image_counts):,} images: {d0_epoch_counts_match}.",
        "- D0 is read as a historical reference only; it is not retrained or modified.",
        f"- Official DINOv3 initialization SHA-256: `{weight_report['weights_sha256']}`; missing/unexpected tensors: {len(weight_report['missing_tensors'])}/{len(weight_report['unexpected_tensors'])}.",
        "",
        "## Environment", "",
        f"- GPU: {environment['gpu_name']} ({environment['gpu_total_memory_bytes'] / (1024 ** 3):.2f} GiB). CUDA {environment['cuda_runtime']}; BF16 supported: {environment['bf16_supported']}.",
        f"- PyTorch {environment['torch']}; Python {environment['python'].split()[0]}; CPU logical processors: {environment['cpu_logical_count']}.",
        "",
    ]
    (output_dir / "preflight_audit.md").write_text("\n".join(lines), encoding="utf-8")


def candidate_id(profile: str, candidate: dict) -> str:
    return "|".join(str(value) for value in (
        profile, int(candidate["image_budget"]), int(candidate["grad_accumulation"]),
        int(candidate["num_workers"]), int(bool(candidate["gradient_checkpointing"])),
        int(bool(candidate["channels_last"])),
    ))


def benchmark_candidates(profile: str, freeze: dict):
    spec = freeze["benchmark"]
    checkpoint_options = (spec["p0_gradient_checkpointing"] if profile == "p0"
                          else spec["partial_gradient_checkpointing"])
    target = int(spec["target_effective_images_per_update"])
    candidates = []
    for budget in spec["image_budgets"]:
        accumulation = max(1, int(round(target / int(budget))))
        for workers in spec["num_workers"]:
            for checkpointing in checkpoint_options:
                for channels_last in spec["channels_last"]:
                    candidate = {
                        "image_budget": int(budget),
                        "grad_accumulation": accumulation,
                        "effective_batch_target": target,
                        "num_workers": int(workers),
                        "pin_memory": bool(spec["pin_memory"]),
                        "persistent_workers": bool(workers)
                            and bool(spec["persistent_workers_when_positive"]),
                        "prefetch_factor": int(spec["prefetch_factor"]),
                        "gradient_checkpointing": bool(checkpointing),
                        "channels_last": bool(channels_last),
                    }
                    candidate["candidate_id"] = candidate_id(profile, candidate)
                    candidates.append(candidate)
    return candidates


def run_benchmark_candidate(config, freeze, profile, train_data, candidate,
                            warmup_updates, measured_updates, repeat, logger,
                            model_builder=None, optimizer_builder=None,
                            batch_forward=None, collate_fn=None,
                            expected_weight_sha256=None):
    device = torch.device("cuda")
    row = {
        "phase": "long" if warmup_updates > freeze["benchmark"]["short_warmup_updates"] else "short",
        "profile": profile,
        "candidate_id": candidate["candidate_id"],
        "repeat": int(repeat),
        **{key: value for key, value in candidate.items() if key != "candidate_id"},
        "warmup_updates": int(warmup_updates),
        "measured_updates": int(measured_updates),
    }
    model = optimizer = loader = None
    try:
        model, verification, _, frozen_modules, trainable_names = (
            build_profile_model(profile, device, candidate, seed=int(config["seed"]))
            if model_builder is None else
            model_builder(profile, device, candidate, seed=int(config["seed"]))
        )
        required_sha256 = MODEL_SHA256 if expected_weight_sha256 is None else expected_weight_sha256
        if verification["weights_sha256"] != required_sha256:
            raise RuntimeError("Benchmark initialized from the wrong DINOv3 weights")
        optimizer = (build_profile_optimizer(model, config) if optimizer_builder is None
                     else optimizer_builder(model, config))
        set_profile_training_mode(model, frozen_modules)
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        loader_kwargs = {
            "image_budget": candidate["image_budget"], "shuffle": True,
            "seed": int(config["seed"]), "num_workers": candidate["num_workers"],
            "pin_memory": candidate["pin_memory"],
            "prefetch_factor": candidate["prefetch_factor"],
        }
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn
        loader, sampler, _ = make_loader(
            train_data, **loader_kwargs)
        sampler.set_epoch(0)
        iterator = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        wall_steps, compute_steps, images_steps, studies_steps = [], [], [], []
        data_wait = 0.0
        for update_index in range(int(warmup_updates) + int(measured_updates)):
            wall_start = time.perf_counter()
            window, wait_this_update = [], 0.0
            for _ in range(candidate["grad_accumulation"]):
                fetch_start = time.perf_counter()
                try:
                    window.append(next(iterator))
                except StopIteration as exc:
                    raise RuntimeError("Benchmark exhausted the official Train sampler") from exc
                wait_this_update += time.perf_counter() - fetch_start
            eligible_window = sum(
                int(batch["known"].any(dim=1).sum()) for batch in window)
            update_images = sum(int(batch["images"].shape[0]) for batch in window)
            update_studies = sum(int(batch["num_studies"]) for batch in window)
            optimizer.zero_grad(set_to_none=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record()
            for batch in window:
                images = batch["images"].to(device, non_blocking=True)
                if candidate["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                study_index = batch["study_index"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                known = batch["known"].to(device, non_blocking=True)
                image_counts = batch["image_counts"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_logits = (model(images) if batch_forward is None
                                    else batch_forward(model, images, batch))
                    study_logits = study_logit_mean(
                        image_logits.float(), study_index,
                        batch["num_studies"], image_counts)
                    per_study, eligible = study_bce_per_study(
                        study_logits, labels, known)
                    loss = per_study[eligible].sum() / max(eligible_window, 1)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite BCE in freeze-profile benchmark")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(config["training"]["gradient_clip_norm"]),
                error_if_nonfinite=True)
            optimizer.step()
            compute_end.record()
            torch.cuda.synchronize(device)
            if update_index >= int(warmup_updates):
                wall_steps.append(time.perf_counter() - wall_start)
                compute_steps.append(compute_start.elapsed_time(compute_end) / 1000.0)
                images_steps.append(update_images)
                studies_steps.append(update_studies)
                data_wait += wait_this_update

        elapsed = max(float(sum(wall_steps)), 1e-9)
        measured_images = int(sum(images_steps))
        measured_studies = int(sum(studies_steps))
        reserved = int(torch.cuda.max_memory_reserved(device))
        row.update({
            "status": "ok",
            "end_to_end_images_per_s": measured_images / elapsed,
            "compute_images_per_s": measured_images / max(float(sum(compute_steps)), 1e-9),
            "studies_per_s": measured_studies / elapsed,
            "median_step_seconds": float(np.median(wall_steps)),
            "p95_step_seconds": float(np.percentile(wall_steps, 95)),
            "data_loader_wait_seconds": data_wait,
            "measured_images": measured_images,
            "measured_studies": measured_studies,
            "effective_images_per_update": float(np.median(images_steps)),
            "effective_studies_per_update": float(np.median(studies_steps)),
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_reserved_bytes": reserved,
            "peak_reserved_memory_fraction": reserved / torch.cuda.get_device_properties(device).total_memory,
            "trainable_parameters": sum(p.numel() for p in trainable_parameters),
            "trainable_parameter_tensors": len(trainable_names),
            "weight_sha256": verification["weights_sha256"],
        })
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("OOM profile=%s candidate=%s repeat=%s", profile,
                       candidate["candidate_id"], repeat)
        row.update({"status": "oom", "error": str(exc)[:500]})
    finally:
        shutdown_loader(loader)
        if torch.cuda.is_available():
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    return row


def _effective_batch_ok(row, benchmark):
    if row.get("status") != "ok":
        return False
    target = float(benchmark["target_effective_images_per_update"])
    measured = float(row.get("effective_images_per_update", 0.0))
    tolerance = float(benchmark["effective_images_tolerance"])
    if target <= 0 or measured <= 0:
        return False
    if abs(measured - target) / target > tolerance:
        return False
    return float(row.get("peak_reserved_memory_fraction", 1.0)) <= float(
        benchmark["maximum_reserved_memory_fraction"])


def _candidate_rows(path):
    if not path.is_file():
        return []
    return pd.read_csv(path).replace({np.nan: None}).to_dict("records")


def _select_profile_runtime(profile, candidate_rows, freeze):
    benchmark = freeze["benchmark"]
    long_rows = [row for row in candidate_rows if row.get("phase") == "long"]
    ids = sorted({row.get("candidate_id") for row in long_rows if row.get("candidate_id")})
    aggregates = []
    expected_repeats = int(benchmark["long_repeats_per_finalist"])
    for candidate_id_value in ids:
        rows = [row for row in long_rows if row.get("candidate_id") == candidate_id_value]
        if len(rows) != expected_repeats or not all(row.get("status") == "ok" for row in rows):
            continue
        speeds = [float(row["end_to_end_images_per_s"]) for row in rows]
        mean_speed = float(np.mean(speeds))
        std_speed = float(np.std(speeds, ddof=1)) if len(speeds) > 1 else 0.0
        effective_images = float(np.median([
            float(row["effective_images_per_update"]) for row in rows]))
        reserved_fraction = max(float(row["peak_reserved_memory_fraction"]) for row in rows)
        target = float(benchmark["target_effective_images_per_update"])
        eligible = (
            abs(effective_images - target) / target
            <= float(benchmark["effective_images_tolerance"])
            and reserved_fraction <= float(benchmark["maximum_reserved_memory_fraction"])
        )
        aggregates.append({
            "candidate_id": candidate_id_value,
            "mean_images_per_s": mean_speed,
            "std_images_per_s": std_speed,
            "cv": std_speed / mean_speed if mean_speed else math.inf,
            "effective_images_per_update": effective_images,
            "effective_studies_per_update": float(np.median([
                float(row["effective_studies_per_update"]) for row in rows])),
            "peak_reserved_memory_fraction": reserved_fraction,
            "eligible": eligible,
            "repeat_count": len(rows),
        })
    eligible = [row for row in aggregates if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            f"No repeated {profile.upper()} benchmark candidate met the batch and memory constraints")
    selected = max(eligible, key=lambda row: row["mean_images_per_s"])
    matching = [row for row in candidate_rows
                if row.get("candidate_id") == selected["candidate_id"]]
    source = next(row for row in matching if row.get("phase") == "short")
    runtime_keys = (
        "image_budget", "grad_accumulation", "effective_batch_target", "num_workers",
        "pin_memory", "persistent_workers", "prefetch_factor",
        "gradient_checkpointing", "channels_last",
    )
    runtime = {key: source[key] for key in runtime_keys}
    for key in ("image_budget", "grad_accumulation", "num_workers", "effective_batch_target", "prefetch_factor"):
        runtime[key] = int(runtime[key])
    for key in ("pin_memory", "persistent_workers", "gradient_checkpointing", "channels_last"):
        runtime[key] = str(runtime[key]).lower() == "true" if isinstance(runtime[key], str) else bool(runtime[key])
    runtime.update({
        "measured_images_per_s": selected["mean_images_per_s"],
        "measured_images_per_s_std": selected["std_images_per_s"],
        "measured_images_per_s_cv": selected["cv"],
        "effective_images_per_update": selected["effective_images_per_update"],
        "effective_studies_per_update": selected["effective_studies_per_update"],
        "peak_reserved_memory_fraction": selected["peak_reserved_memory_fraction"],
        "long_repeats": selected["repeat_count"],
        "candidate_id": selected["candidate_id"],
        "selection_rule": "highest mean of three completed long runs among candidates within the configured effective-image tolerance and memory ceiling",
    })
    return runtime, aggregates


def _write_benchmark_comparison(output_dir, selected_profiles, aggregates_by_profile, d0_config):
    lines = [
        "# P0/P1/P2 GPU 吞吐量对照", "",
        "Benchmark 覆盖 NAS 数据加载、图像解码/增强、Forward、Study Logit Mean、BCE、Backward、梯度累积和 AdamW Step。短测筛选候选，入选配置进行 3 次长测；选择满足有效图片数与显存约束的长测均值最高项。",
        "",
        "| Profile | Budget | Accum | Workers | Checkpointing | Channels Last | Mean images/s | Std | CV | Effective images/update | Effective studies/update | Peak reserved fraction |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        if profile not in selected_profiles:
            continue
        row = selected_profiles[profile]
        lines.append(
            f"| {profile.upper()} | {row['image_budget']} | {row['grad_accumulation']} | "
            f"{row['num_workers']} | {row['gradient_checkpointing']} | {row['channels_last']} | "
            f"{row['measured_images_per_s']:.2f} | {row['measured_images_per_s_std']:.3f} | "
            f"{row['measured_images_per_s_cv']:.3f} | {row['effective_images_per_update']:.2f} | "
            f"{row['effective_studies_per_update']:.2f} | "
            f"{row['peak_reserved_memory_fraction']:.3f} |"
        )
    lines.extend(["", "## 重复测量覆盖", ""])
    for profile in PROFILES:
        aggregates = aggregates_by_profile.get(profile, [])
        eligible_count = sum(bool(row.get("eligible")) for row in aggregates)
        lines.append(
            f"- {profile.upper()}: {len(aggregates)} candidate aggregates; "
            f"{eligible_count} met effective-batch and memory constraints; "
            f"selected config uses three completed long repeats."
        )
    lines.extend([
        "", "## 参考对照", "",
        f"- D0 历史 Benchmark 均值：{float(d0_config['measured_d0_images_per_s']):.2f} images/s；正式 20 轮训练日志均值约 23.536 images/s。两者计时阶段不同，需分开解释。",
        "- 每个 profile 的候选短测、长测重复和 OOM 记录分别保存在 `p0_gpu_benchmark.csv`、`p1_gpu_benchmark.csv`、`p2_gpu_benchmark.csv`。",
        "- 高于目标有效 Batch 容差或超过显存上限的候选保留在 CSV 中，但不参与最终 runtime 选择。",
        "",
    ])
    (output_dir / "gpu_benchmark_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run_profile_benchmark(config, freeze, profile, train_data, output_dir,
                          model_builder=None, optimizer_builder=None,
                          batch_forward=None, collate_fn=None,
                          expected_weight_sha256=None):
    selected_path = output_dir / "selected_runtime_config.json"
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.is_file() else {"experiment": freeze["experiment"], "profiles": {}}
    if profile in selected_payload.get("profiles", {}):
        return selected_payload["profiles"][profile]

    benchmark = freeze["benchmark"]
    csv_path = output_dir / f"{profile}_gpu_benchmark.csv"
    progress_path = output_dir / "benchmark_progress.json"
    logger = setup_logging(output_dir, f"{profile}_benchmark")
    candidates = benchmark_candidates(profile, freeze)
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    rows = _candidate_rows(csv_path)
    short_done = {row.get("candidate_id") for row in rows if row.get("phase") == "short"}
    for index, candidate in enumerate(candidates, start=1):
        if candidate["candidate_id"] in short_done:
            continue
        row = run_benchmark_candidate(
            config, freeze, profile, train_data, candidate,
            benchmark["short_warmup_updates"], benchmark["short_measured_updates"],
            repeat=0, logger=logger, model_builder=model_builder,
            optimizer_builder=optimizer_builder, batch_forward=batch_forward,
            collate_fn=collate_fn, expected_weight_sha256=expected_weight_sha256)
        rows.append(row)
        short_done.add(candidate["candidate_id"])
        write_csv_union(csv_path, rows)
        save_json_atomic(progress_path, {
            "phase": "short", "profile": profile,
            "completed_short_candidates": len(short_done),
            "total_short_candidates": len(candidates),
            "last_candidate_id": candidate["candidate_id"],
            "last_status": row["status"],
        })
        logger.info("short %d/%d candidate=%s status=%s e2e_images_s=%s",
                    index, len(candidates), candidate["candidate_id"],
                    row["status"], row.get("end_to_end_images_per_s"))

    short_rows = [row for row in rows if row.get("phase") == "short"]
    finalists = [row for row in short_rows if _effective_batch_ok(row, benchmark)]
    if not finalists:
        raise RuntimeError(f"No eligible short benchmark candidates for {profile.upper()}")
    finalists.sort(key=lambda row: float(row["end_to_end_images_per_s"]), reverse=True)
    finalist_ids = [row["candidate_id"] for row in finalists[:int(benchmark["finalists"])]]
    long_done = {(row.get("candidate_id"), int(float(row.get("repeat", 0))))
                 for row in rows if row.get("phase") == "long"}
    total_long = len(finalist_ids) * int(benchmark["long_repeats_per_finalist"])
    completed_long = len([key for key in long_done if key[0] in finalist_ids])
    for candidate_id_value in finalist_ids:
        candidate = by_id[candidate_id_value]
        for repeat in range(1, int(benchmark["long_repeats_per_finalist"]) + 1):
            if (candidate_id_value, repeat) in long_done:
                continue
            row = run_benchmark_candidate(
                config, freeze, profile, train_data, candidate,
                benchmark["long_warmup_updates"], benchmark["long_measured_updates"],
                repeat=repeat, logger=logger, model_builder=model_builder,
                optimizer_builder=optimizer_builder, batch_forward=batch_forward,
                collate_fn=collate_fn, expected_weight_sha256=expected_weight_sha256)
            row["phase"] = "long"
            rows.append(row)
            write_csv_union(csv_path, rows)
            completed_long += 1
            save_json_atomic(progress_path, {
                "phase": "long", "profile": profile,
                "completed_long_repeats": completed_long,
                "total_long_repeats": total_long,
                "last_candidate_id": candidate_id_value,
                "last_repeat": repeat,
                "last_status": row["status"],
            })
            logger.info("long %d/%d candidate=%s repeat=%d/%d status=%s e2e_images_s=%s",
                        completed_long, total_long, candidate_id_value, repeat,
                        int(benchmark["long_repeats_per_finalist"]), row["status"],
                        row.get("end_to_end_images_per_s"))

    runtime, aggregates = _select_profile_runtime(profile, rows, freeze)
    selected_payload.setdefault("profiles", {})[profile] = runtime
    save_json_atomic(output_dir / f"{profile}_benchmark_summary.json", {
        "profile": profile,
        "selected_runtime": runtime,
        "repeated_candidate_aggregates": aggregates,
    })
    save_json_atomic(selected_path, selected_payload)
    save_json_atomic(progress_path, {
        "phase": "complete", "profile": profile, "selected_runtime": runtime,
    })
    old_d0 = json.loads(
        (WORK_DIR / "outputs" / "experiments" / "DINOv3_D0_D0C1"
         / "dino_selected_runtime_config.json").read_text(encoding="utf-8"))
    aggregates_by_profile = {}
    for completed_profile in selected_payload["profiles"]:
        summary_path = output_dir / f"{completed_profile}_benchmark_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            aggregates_by_profile[completed_profile] = summary["repeated_candidate_aggregates"]
    _write_benchmark_comparison(output_dir, selected_payload["profiles"],
                                aggregates_by_profile, old_d0)
    return runtime


def checkpoint_payload(profile, model, optimizer, epoch, history, per_class_history,
                       best_auroc, best_epoch, no_improve_count, global_step,
                       initial_sha256, runtime, sampler, loader_generator,
                       trainable_names, complete, stop_reason,
                       experiment_name="DINOv3_P0_P1_P2", run_context=None):
    return {
        "experiment": str(experiment_name),
        "profile": profile,
        "run_context": run_context,
        "epoch": int(epoch),
        "history": history,
        "per_class_history": per_class_history,
        "best_macro_auroc": float(best_auroc),
        "best_epoch": int(best_epoch),
        "auc_no_improve_count": int(no_improve_count),
        "global_step": int(global_step),
        "initial_weights_sha256": initial_sha256,
        "runtime": runtime,
        "trainable_parameter_names": trainable_names,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_epoch": int(sampler.epoch),
        "loader_generator_state": loader_generator.get_state(),
        "rng_state": state_rng(),
        "complete": bool(complete),
        "stop_reason": stop_reason,
    }


def write_profile_report(run_dir, profile, history, status, parameter_summary):
    best = max(history, key=lambda row: float(row["macro_auroc"]))
    latest = history[-1]
    lines = [
        f"# DINOv3 {profile.upper()} Training Report", "",
        f"- Status: {status['completed_epochs']}/{status['max_epochs']} epochs; stop reason: `{status['stop_reason']}`.",
        f"- Best epoch: {best['epoch']}; Macro-AUROC: {best['macro_auroc']:.6f}; Valid BCE: {best['valid_loss']:.6f}.",
        f"- Last epoch: {latest['epoch']}; Train BCE: {latest['train_bce']:.6f}; Valid BCE: {latest['valid_loss']:.6f}; Macro-AUROC: {latest['macro_auroc']:.6f}.",
        f"- Trainable parameters: {parameter_summary['trainable']:,}; frozen parameters: {parameter_summary['frozen']:,}.",
        f"- Actual mean train throughput: {np.mean([float(row['images_per_s']) for row in history]):.2f} images/s.",
        f"- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.",
        "- Test split was not read.", "",
        "| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_bce']:.6f} | {row['valid_loss']:.6f} | "
            f"{row['macro_auroc']:.6f} | {row['macro_auprc']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['images_per_s']:.2f} | "
            f"{row['train_down_3']} | {row['valid_up_3']} | "
            f"{row['auc_no_improve_3']} | {row['should_stop']} |"
        )
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def train_profile(config, freeze, profile, train_data, valid_data,
                  verification, runtime, output_dir,
                  experiment_name="DINOv3_P0_P1_P2", run_dir_override=None,
                  run_context=None, model_builder=None,
                  optimizer_builder=None, batch_forward=None,
                  validation_forward=None, collate_fn=None,
                  epoch_callback=None):
    run_dir = Path(run_dir_override) if run_dir_override is not None else output_dir / profile
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "training_status.json"
    result_path = run_dir / "training_result.json"
    last_path, best_path = run_dir / "last.pth", run_dir / "best.pth"
    if status_path.is_file():
        old_status = json.loads(status_path.read_text(encoding="utf-8"))
        if old_status.get("complete"):
            if result_path.is_file():
                return json.loads(result_path.read_text(encoding="utf-8"))
    logger = setup_logging(run_dir, "train")
    save_json_atomic(run_dir / "config.json", {
        "profile": profile,
        "experiment_config": config,
        "freeze_config": freeze,
        "selected_runtime": runtime,
        "initialization": verification,
        "train_studies": len(train_data),
        "train_images": int(sum(train_data.image_counts)),
        "valid_studies": len(valid_data),
        "valid_images": int(sum(valid_data.image_counts)),
        "test_split_read": False,
    })
    device = torch.device("cuda")
    model, model_verification, train_modules, frozen_modules, trainable_names = (
        build_profile_model(profile, device, runtime, seed=int(config["seed"]))
        if model_builder is None else
        model_builder(profile, device, runtime, seed=int(config["seed"]))
    )
    if model_verification["weights_sha256"] != verification["weights_sha256"]:
        raise RuntimeError(f"{profile} model did not reload the verified original weights")
    optimizer = (build_profile_optimizer(model, config) if optimizer_builder is None
                 else optimizer_builder(model, config))
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_parameter_count = sum(p.numel() for p in trainable_parameters)
    frozen_parameter_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    run_dir_manifest = {
        "trainable": trainable_parameter_count,
        "frozen": frozen_parameter_count,
        "total": trainable_parameter_count + frozen_parameter_count,
        "trainable_modules": [name for name, _ in train_modules],
    }
    loader_kwargs = {
        "image_budget": int(runtime["image_budget"]), "shuffle": True,
        "seed": int(config["seed"]), "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
        "prefetch_factor": int(runtime["prefetch_factor"]),
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    train_loader, sampler, loader_generator = make_loader(train_data, **loader_kwargs)
    valid_loader_config = {
        "image_budget": int(runtime["image_budget"]),
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
        "prefetch_factor": int(runtime["prefetch_factor"]),
        "seed": int(config["seed"]),
        "channels_last": bool(runtime["channels_last"]),
    }
    epoch_updates = []
    for epoch_index in range(int(freeze["training"]["max_epochs"])):
        sampler.set_epoch(epoch_index)
        epoch_updates.append(math.ceil(len(sampler) / int(runtime["grad_accumulation"])))
    total_updates = sum(epoch_updates)
    training = dict(config["training"])
    training["warmup_updates"] = sum(epoch_updates[:int(training["warmup_epochs"])])

    history, per_class_history = [], []
    best_auroc, best_epoch = -math.inf, 0
    no_improve_count, global_step, start_epoch = 0, 0, 0
    if last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if (saved.get("experiment") != experiment_name
                or saved.get("run_context") != run_context
                or saved["profile"] != profile
                or saved["initial_weights_sha256"] != verification["weights_sha256"]):
            raise RuntimeError(f"{profile} resume checkpoint identity mismatch")
        if saved["runtime"] != runtime or saved["trainable_parameter_names"] != trainable_names:
            raise RuntimeError(f"{profile} resume runtime or freeze manifest changed")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved["history"]
        per_class_history = saved["per_class_history"]
        best_auroc = float(saved["best_macro_auroc"])
        best_epoch = int(saved["best_epoch"])
        no_improve_count = int(saved["auc_no_improve_count"])
        global_step = int(saved["global_step"])
        start_epoch = int(saved["epoch"])
        sampler.set_epoch(start_epoch)
        loader_generator.set_state(saved["loader_generator_state"])
        restore_rng(saved["rng_state"])
        logger.info("Resuming %s after epoch %d", profile.upper(), start_epoch)
        if saved.get("complete"):
            recovered_status = {
                "profile": profile,
                "completed_epochs": start_epoch,
                "max_epochs": int(freeze["training"]["max_epochs"]),
                "best_epoch": best_epoch,
                "best_macro_auroc": best_auroc,
                "last_checkpoint": str(last_path),
                "best_checkpoint": str(best_path),
                "complete": True,
                "stop_reason": saved.get("stop_reason"),
                "auc_no_improve_count": no_improve_count,
            }
            save_json_atomic(status_path, recovered_status)
            write_profile_report(run_dir, profile, history, recovered_status, run_dir_manifest)
            recovered_result = {
                "profile": profile,
                "completed_epochs": start_epoch,
                "stop_epoch": start_epoch,
                "stop_reason": saved.get("stop_reason"),
                "best_epoch": best_epoch,
                "best_macro_auroc": best_auroc,
                "best_checkpoint": str(best_path),
                "last_checkpoint": str(last_path),
                "output_dir": str(run_dir),
            }
            save_json_atomic(result_path, recovered_result)
            shutdown_loader(train_loader)
            del model, optimizer, saved
            gc.collect()
            torch.cuda.empty_cache()
            return recovered_result

    stop_reason = None
    try:
        for epoch_index in range(start_epoch, int(freeze["training"]["max_epochs"])):
            epoch_number = epoch_index + 1
            sampler.set_epoch(epoch_index)
            iterator = iter(train_loader)
            set_profile_training_mode(model, frozen_modules)
            if epoch_callback is not None:
                epoch_callback(model, epoch_number, "start", None)
            train_started = time.perf_counter()
            data_wait_seconds = 0.0
            bce_sum_epoch = 0.0
            eligible_studies_epoch = 0
            train_images = train_studies = 0
            updates = 0
            torch.cuda.reset_peak_memory_stats(device)
            accumulation = int(runtime["grad_accumulation"])
            while True:
                window, wait_window = [], 0.0
                for _ in range(accumulation):
                    fetch_started = time.perf_counter()
                    try:
                        window.append(next(iterator))
                    except StopIteration:
                        break
                    wait_window += time.perf_counter() - fetch_started
                if not window:
                    break
                data_wait_seconds += wait_window
                eligible_window = sum(
                    int(batch["known"].any(dim=1).sum()) for batch in window)
                optimizer.zero_grad(set_to_none=True)
                lrs = update_learning_rates(optimizer, global_step, total_updates, training)
                for batch in window:
                    images = batch["images"].to(device, non_blocking=True)
                    if runtime["channels_last"]:
                        images = images.contiguous(memory_format=torch.channels_last)
                    study_index = batch["study_index"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    known = batch["known"].to(device, non_blocking=True)
                    image_counts = batch["image_counts"].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        image_logits = (model(images) if batch_forward is None
                                        else batch_forward(model, images, batch))
                        study_logits = study_logit_mean(
                            image_logits.float(), study_index,
                            batch["num_studies"], image_counts)
                        per_study, eligible = study_bce_per_study(
                            study_logits, labels, known)
                        bce_sum = per_study[eligible].sum()
                        loss = bce_sum / max(eligible_window, 1)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite {profile.upper()} loss at epoch {epoch_number}")
                    loss.backward()
                    bce_sum_epoch += float(bce_sum.detach().float().cpu())
                    eligible_studies_epoch += int(eligible.sum())
                    train_images += int(images.shape[0])
                    train_studies += int(batch["num_studies"])
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, float(training["gradient_clip_norm"]),
                    error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                updates += 1
                del window

            train_seconds = time.perf_counter() - train_started
            validation = validate(
                model, valid_data, device, **valid_loader_config,
                batch_forward=(validation_forward or batch_forward),
                collate_fn=collate_fn)
            score = float(validation["macro_auroc"])
            improved = math.isfinite(score) and score > best_auroc
            if improved:
                best_auroc, best_epoch = score, epoch_number
                no_improve_count = 0
            else:
                no_improve_count += 1
            train_bce = bce_sum_epoch / max(eligible_studies_epoch, 1)
            row = {
                "epoch": epoch_number,
                "train_bce": train_bce,
                "train_total": train_bce,
                "valid_loss": float(validation["valid_loss"]),
                "macro_auroc": score,
                "macro_auprc": float(validation["macro_auprc"]),
                "macro_f1": float(validation["macro_f1"]),
                "optimizer_updates": updates,
                "train_images": train_images,
                "train_studies": train_studies,
                "train_seconds": train_seconds,
                "validation_seconds": float(validation["valid_seconds"]),
                "validation_studies_per_s": float(validation["valid_studies_per_s"]),
                "images_per_s": train_images / train_seconds if train_seconds else 0.0,
                "studies_per_s": train_studies / train_seconds if train_seconds else 0.0,
                "data_loader_wait_seconds": data_wait_seconds,
                "trainable_parameters": trainable_parameter_count,
                "frozen_parameters": frozen_parameter_count,
                "lr_backbone": lrs.get("backbone", 0.0),
                "lr_head": lrs["head"],
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
                "best_epoch": best_epoch,
                "best_macro_auroc": best_auroc,
                "image_budget": int(runtime["image_budget"]),
                "grad_accumulation": accumulation,
                "auc_no_improve_count": no_improve_count,
            }
            if epoch_callback is not None:
                callback_metrics = epoch_callback(model, epoch_number, "end", row)
                if callback_metrics:
                    row.update(callback_metrics)
            candidate_history = history + [row]
            last_four = candidate_history[-4:]
            train_down_3 = len(last_four) == 4 and all(
                float(last_four[index]["train_bce"]) > float(last_four[index + 1]["train_bce"])
                for index in range(3))
            valid_up_3 = len(last_four) == 4 and all(
                float(last_four[index]["valid_loss"]) < float(last_four[index + 1]["valid_loss"])
                for index in range(3))
            auc_no_improve_3 = no_improve_count >= int(
                freeze["early_stopping"]["macro_auroc_no_improvement_epochs"])
            should_stop = bool(train_down_3 and valid_up_3 and auc_no_improve_3)
            row.update({
                "train_down_3": bool(train_down_3),
                "valid_up_3": bool(valid_up_3),
                "auc_no_improve_3": bool(auc_no_improve_3),
                "should_stop": should_stop,
            })
            history.append(row)
            per_class_history.extend(
                {"epoch": epoch_number, **item} for item in validation["per_class"])
            write_csv(run_dir / "train_log.csv", history)
            write_csv(run_dir / "per_class_metrics.csv", per_class_history)
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            prediction_frame(validation).to_csv(
                predictions_dir / f"val_epoch_{epoch_number:02d}.csv",
                index=False, encoding="utf-8")
            if improved:
                prediction_frame(validation).to_csv(
                    predictions_dir / "val_best.csv", index=False, encoding="utf-8")
                write_csv(run_dir / "best_per_class_metrics.csv", validation["per_class"])
            if should_stop:
                stop_reason = "three_condition_early_stop"
            elif epoch_number >= int(freeze["training"]["max_epochs"]):
                stop_reason = "max_epochs_reached"
            complete = stop_reason is not None
            saved_checkpoint = checkpoint_payload(
                profile, model, optimizer, epoch_number, history, per_class_history,
                best_auroc, best_epoch, no_improve_count, global_step,
                verification["weights_sha256"], runtime, sampler,
                loader_generator, trainable_names, complete, stop_reason,
                experiment_name=experiment_name, run_context=run_context)
            save_torch_atomic(last_path, saved_checkpoint)
            if improved:
                save_torch_atomic(best_path, saved_checkpoint)
            if should_stop:
                save_json_atomic(run_dir / "early_stop.json", {
                    "profile": profile,
                    "stop_epoch": epoch_number,
                    "stop_reason": stop_reason,
                    "train_down_3": train_down_3,
                    "valid_up_3": valid_up_3,
                    "auc_no_improve_count": no_improve_count,
                    "best_epoch": best_epoch,
                    "best_macro_auroc": best_auroc,
                })
            status = {
                "profile": profile,
                "completed_epochs": epoch_number,
                "max_epochs": int(freeze["training"]["max_epochs"]),
                "best_epoch": best_epoch,
                "best_macro_auroc": best_auroc,
                "last_checkpoint": str(last_path),
                "best_checkpoint": str(best_path),
                "complete": complete,
                "stop_reason": stop_reason,
                "auc_no_improve_count": no_improve_count,
            }
            save_json_atomic(status_path, status)
            write_profile_report(run_dir, profile, history, status, run_dir_manifest)
            logger.info(
                "%s epoch=%02d/%02d train_bce=%.6f valid_bce=%.6f "
                "macro_auroc=%.6f images_s=%.2f train_down_3=%s "
                "valid_up_3=%s auc_no_improve=%d stop=%s best_epoch=%d best_auc=%.6f",
                profile.upper(), epoch_number, int(freeze["training"]["max_epochs"]),
                train_bce, validation["valid_loss"], score, row["images_per_s"],
                train_down_3, valid_up_3, no_improve_count, stop_reason,
                best_epoch, best_auroc)
            if complete:
                break
    except Exception as exc:
        save_json_atomic(run_dir / "training_failure.json", {
            "profile": profile,
            "epoch_in_progress": len(history) + 1,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "last_completed_epoch": len(history),
            "last_checkpoint": str(last_path) if last_path.is_file() else None,
        })
        raise
    finally:
        shutdown_loader(train_loader)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    final_status = json.loads(status_path.read_text(encoding="utf-8"))
    final = {
        "profile": profile,
        "completed_epochs": final_status["completed_epochs"],
        "stop_epoch": final_status["completed_epochs"],
        "stop_reason": final_status["stop_reason"],
        "best_epoch": final_status["best_epoch"],
        "best_macro_auroc": final_status["best_macro_auroc"],
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "output_dir": str(run_dir),
    }
    save_json_atomic(result_path, final)
    return final


def _rename_predictions(frame, suffix):
    _, probability_columns = _columns(None)
    target_columns, _ = _columns(None)
    rename = {column: f"{column}{suffix}" for column in target_columns + probability_columns}
    return frame.rename(columns=rename)


def _draw_line_plot(path: Path, title: str, series, y_label: str):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1450, 760
    left, right, top, bottom = 85, 980, 75, 660
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((left, 18), title, fill="black", font=title_font)
    finite = [float(value) for _, values in series for value in values if np.isfinite(value)]
    if not finite:
        raise ValueError(f"Cannot plot empty series: {title}")
    low, high = min(finite), max(finite)
    if high <= low:
        high = low + 1.0
    pad = max((high - low) * 0.06, 0.01)
    low, high = low - pad, high + pad
    draw.line((left, top, left, bottom, right, bottom), fill="#333333", width=2)
    for tick in range(6):
        y = bottom - tick * (bottom - top) / 5
        value = low + tick * (high - low) / 5
        draw.line((left, int(y), right, int(y)), fill="#e6e6e6", width=1)
        draw.text((8, int(y) - 8), f"{value:.3f}", fill="#333333", font=font)
    colors = ["#1769aa", "#c0392b", "#218c74", "#f39c12",
              "#6c5ce7", "#d63031", "#0984e3", "#00b894"]
    max_epochs = max(len(values) for _, values in series)
    for index, (name, values) in enumerate(series):
        points = []
        for epoch, value in enumerate(values, start=1):
            if not np.isfinite(value):
                continue
            x = left + (epoch - 1) * (right - left) / max(1, max_epochs - 1)
            y = bottom - (float(value) - low) * (bottom - top) / (high - low)
            points.append((int(x), int(y)))
        color = colors[index % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        legend_y = top + index * 62
        draw.line((1020, legend_y + 10, 1060, legend_y + 10), fill=color, width=4)
        draw.text((1070, legend_y), name, fill="black", font=font)
    draw.text((left, 705), "Epoch", fill="black", font=font)
    draw.text((15, 42), y_label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _plot_train_validation_loss(path: Path, histories):
    series = []
    for name in ("d0", "p0", "p1", "p2"):
        series.append((f"{name.upper()} Train BCE",
                       [float(row["train_bce"]) for row in histories[name]]))
        series.append((f"{name.upper()} Valid BCE",
                       [float(row["valid_loss"]) for row in histories[name]]))
    _draw_line_plot(path, "Train / Validation BCE", series, "BCE")


def _plot_throughput(path: Path, values: dict[str, float]):
    from PIL import Image, ImageDraw, ImageFont

    width = 1100
    row_height = 100
    top = 90
    height = top + row_height * len(values) + 55
    left, right = 250, 1020
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((left, 25), "Selected benchmark mean end-to-end throughput", fill="black", font=title_font)
    max_value = max(float(value) for value in values.values())
    colors = ["#1769aa", "#6baed6", "#d1495b", "#f28e2b"]
    for index, (name, value) in enumerate(values.items()):
        y = top + index * row_height
        bar_width = int((right - left) * float(value) / max(max_value * 1.12, 1e-9))
        draw.text((20, y + 12), name, fill="black", font=font)
        draw.rectangle((left, y, left + bar_width, y + 45), fill=colors[index % len(colors)])
        draw.text((left + bar_width + 12, y + 12), f"{float(value):.2f} images/s",
                  fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def compare_frozen_profiles(config, freeze, output_dir):
    d0_dir = WORK_DIR / "outputs" / "experiments" / "DINOv3_D0_D0C1" / "d0"
    profiles = {profile: output_dir / profile for profile in PROFILES}
    statuses = {}
    histories = {}
    prediction_frames = {}
    for profile, run_dir in profiles.items():
        status = json.loads((run_dir / "training_status.json").read_text(encoding="utf-8"))
        if not status.get("complete"):
            raise RuntimeError(f"Cannot finalize comparison: {profile.upper()} is incomplete")
        statuses[profile] = status
        histories[profile] = pd.read_csv(run_dir / "train_log.csv").to_dict("records")
        prediction_frames[profile] = pd.read_csv(
            run_dir / "predictions" / "val_best.csv",
            dtype={"Subject_id": "string", "Study_id": "string"})
    d0_status = json.loads((d0_dir / "training_status.json").read_text(encoding="utf-8"))
    d0_history = pd.read_csv(d0_dir / "train_log.csv").to_dict("records")
    prediction_frames["d0"] = pd.read_csv(
        d0_dir / "predictions" / "val_best.csv",
        dtype={"Subject_id": "string", "Study_id": "string"})
    histories["d0"] = d0_history
    statuses["d0"] = d0_status
    keys = ["Subject_id", "Study_id"]
    expected_keys = set(map(tuple, prediction_frames["d0"][keys].astype(str).to_numpy()))
    expected_count = int(freeze["comparison"]["expected_validation_studies"])
    if len(expected_keys) != expected_count:
        raise RuntimeError(f"D0 validation predictions have {len(expected_keys)} Studies, expected {expected_count}")
    target_columns, _ = _columns(None)
    for profile, frame in prediction_frames.items():
        if frame.duplicated(keys).any():
            raise RuntimeError(f"Duplicate validation Study key in {profile}")
        if set(map(tuple, frame[keys].astype(str).to_numpy())) != expected_keys:
            raise RuntimeError(f"Validation Study keys differ for {profile}")
        if profile != "d0":
            merged_labels = prediction_frames["d0"].merge(
                frame, on=keys, suffixes=("_ref", "_candidate"),
                validate="one_to_one", sort=True)
            for column in target_columns:
                if not np.array_equal(merged_labels[f"{column}_ref"],
                                      merged_labels[f"{column}_candidate"]):
                    raise RuntimeError(f"Validation target mismatch in {profile}: {column}")

    comparisons = {}
    per_class_rows = []
    comparison_pairs = [
        ("p1", "p0"), ("p2", "p1"), ("p2", "p0"),
        ("p0", "d0"), ("p1", "d0"), ("p2", "d0"),
    ]
    macro_values = {}
    auc_vectors = {}
    for name, frame in prediction_frames.items():
        auc_vectors[name] = _auc_vector(frame)
        macro_values[name] = float(np.nanmean(auc_vectors[name]))
    for candidate, reference in comparison_pairs:
        left = _rename_predictions(prediction_frames[reference], "_d0")
        right = _rename_predictions(prediction_frames[candidate], "_d0c1")
        merged = left.merge(right, on=keys, validate="one_to_one", sort=True)
        bootstrap = _bootstrap_subject_differences(
            merged, int(freeze["comparison"]["subject_bootstrap_replicates"]),
            int(freeze["comparison"]["subject_bootstrap_seed"]))
        delta = macro_values[candidate] - macro_values[reference]
        class_delta = {
            label: float(auc_vectors[candidate][index] - auc_vectors[reference][index])
            for index, label in enumerate(LABEL_NAMES)
        }
        comparisons[f"{candidate}_minus_{reference}"] = {
            "candidate": candidate,
            "reference": reference,
            "candidate_macro_auroc": macro_values[candidate],
            "reference_macro_auroc": macro_values[reference],
            "macro_auroc_delta": delta,
            "macro_auroc_delta_ci95": bootstrap["macro_auroc_delta_ci95"],
            "per_class_delta": class_delta,
            "per_class_delta_ci95": bootstrap["per_class_delta_ci95"],
            "valid_bootstrap_replicates": bootstrap["valid_macro_replicates"],
            "validation_studies": len(merged),
            "validation_subjects": int(merged["Subject_id"].nunique()),
        }

    for index, label in enumerate(LABEL_NAMES):
        row = {"class_id": index + 1, "class_name": label}
        for name in ("d0", "p0", "p1", "p2"):
            row[f"{name}_auroc"] = float(auc_vectors[name][index])
        for candidate, reference in comparison_pairs:
            key = f"{candidate}_minus_{reference}"
            result = comparisons[key]
            row[f"{key}_delta"] = result["per_class_delta"][label]
            row[f"{key}_ci95_low"] = result["per_class_delta_ci95"][label][0]
            row[f"{key}_ci95_high"] = result["per_class_delta_ci95"][label][1]
        per_class_rows.append(row)

    bootstrap_payload = {
        "seed": int(freeze["comparison"]["subject_bootstrap_seed"]),
        "requested_replicates": int(freeze["comparison"]["subject_bootstrap_replicates"]),
        "validation_studies": expected_count,
        "validation_subjects": int(prediction_frames["d0"]["Subject_id"].nunique()),
        "selection_effect_caveat": "Best epochs are selected on this validation set; intervals are conditional and are not independent-test guarantees.",
        "comparisons": comparisons,
    }
    save_json_atomic(output_dir / "paired_subject_bootstrap.json", bootstrap_payload)
    write_csv(output_dir / "per_class_auroc_comparison.csv", per_class_rows)

    lines = [
        "# DINOv3 P0/P1/P2 与 D0 对照分析", "",
        f"- 配对验证集：{expected_count} Studies / {bootstrap_payload['validation_subjects']} Subjects；所有模型 Study 键和十类标签相同。",
        f"- Bootstrap：{freeze['comparison']['subject_bootstrap_replicates']} 次 Subject-cluster paired resampling，seed {freeze['comparison']['subject_bootstrap_seed']}。",
        "- D0 是此前完成的 20 轮历史全量微调对照，不重训；V1 只作为不严格匹配的历史背景。",
        "- Test split 未读取。最佳 Epoch 是按同一验证集选择，因此 Bootstrap 区间是条件性证据。",
        "",
        "## 模型最佳点估计", "",
        "| 模型 | 可训练范围 | Best Epoch | Stop Epoch | Stop Reason | Macro-AUROC | 与 D0 差值 | Mean Train images/s |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    d0_best = float(d0_status["best_macro_auroc"])
    d0_avg_speed = float(pd.read_csv(d0_dir / "train_log.csv")["images_per_s"].mean())
    lines.append(
        f"| D0 | Full backbone + head | {d0_status['best_epoch']} | 20 | fixed 20 epochs | "
        f"{macro_values['d0']:.6f} | +0.000000 | {d0_avg_speed:.2f} |"
    )
    trainable_names = {
        "p0": "Head",
        "p1": "Stage 4 + final norm + head",
        "p2": "Stage 3–4 + final norm + head",
    }
    for profile in PROFILES:
        rows = histories[profile]
        actual_speed = float(np.mean([float(row["images_per_s"]) for row in rows]))
        status = statuses[profile]
        lines.append(
            f"| {profile.upper()} | {trainable_names[profile]} | {status['best_epoch']} | "
            f"{status['completed_epochs']} | {status['stop_reason']} | "
            f"{macro_values[profile]:.6f} | {macro_values[profile]-d0_best:+.6f} | {actual_speed:.2f} |"
        )
    lines.extend([
        "", "## 配对 Macro-AUROC 差值", "",
        "| Candidate − Reference | Delta | Subject bootstrap 95% CI | Interpretation |",
        "|---|---:|---:|---|",
    ])
    for candidate, reference in comparison_pairs:
        result = comparisons[f"{candidate}_minus_{reference}"]
        low, high = result["macro_auroc_delta_ci95"]
        interpretation = "区间跨 0" if low <= 0 <= high else "区间不跨 0"
        lines.append(
            f"| {candidate.upper()} − {reference.upper()} | {result['macro_auroc_delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {interpretation} |"
        )
    lines.extend([
        "", "## 重点类别 AUROC", "",
        "| 类别 | D0 | P0 | P1 | P2 | P1−P0 | P2−P1 | P2−P0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    focused = ["Pneumonia", "Lung Opacity", "Pneumothorax", "Consolidation", "Cardiomegaly"]
    for label in focused:
        row = next(item for item in per_class_rows if item["class_name"] == label)
        lines.append(
            f"| {label} | {row['d0_auroc']:.6f} | {row['p0_auroc']:.6f} | "
            f"{row['p1_auroc']:.6f} | {row['p2_auroc']:.6f} | "
            f"{row['p1_minus_p0_delta']:+.6f} | {row['p2_minus_p1_delta']:+.6f} | "
            f"{row['p2_minus_p0_delta']:+.6f} |"
        )
    best_profile = max(PROFILES, key=lambda profile: macro_values[profile])
    v1 = float(config["comparison"]["historical_v1_macro_auroc"])
    selected_runtime = json.loads(
        (output_dir / "selected_runtime_config.json").read_text(encoding="utf-8"))["profiles"]
    profile_dynamics = {}
    for profile in PROFILES:
        rows = sorted(histories[profile], key=lambda row: int(row["epoch"]))
        peak = max(float(row["macro_auroc"]) for row in rows)
        final_auc = float(rows[-1]["macro_auroc"])
        min_valid = min(float(row["valid_loss"]) for row in rows)
        final_valid = float(rows[-1]["valid_loss"])
        profile_dynamics[profile] = {
            "peak_to_stop_auc_drop": peak - final_auc,
            "min_to_stop_valid_loss_increase": final_valid - min_valid,
            "stop_epoch": statuses[profile]["completed_epochs"],
            "stop_reason": statuses[profile]["stop_reason"],
        }
    least_auc_drop_profile = min(
        PROFILES, key=lambda profile: profile_dynamics[profile]["peak_to_stop_auc_drop"])
    p1_p0 = comparisons["p1_minus_p0"]
    p2_p1 = comparisons["p2_minus_p1"]
    p2_p0 = comparisons["p2_minus_p0"]
    pneumonia_row = next(item for item in per_class_rows if item["class_name"] == "Pneumonia")
    opacity_row = next(item for item in per_class_rows if item["class_name"] == "Lung Opacity")
    lines.extend([
        "", "## 结论", "",
        f"- P0/P1/P2 中观测 Macro-AUROC 最高的是 **{best_profile.upper()}**（{macro_values[best_profile]:.6f}）。是否优于相邻冻结范围，应结合配对区间而不是只看点估计。",
        f"- 历史 V1 Macro-AUROC 为 {v1:.6f}，与本实验不是严格匹配的预训练消融。",
        "- 冻结深度对过拟合、类别变化和速度的判断见曲线图、逐类 CSV、每组 `report.md` 与 `gpu_benchmark_comparison.md`。",
        "- 本文不以验证集结果替代独立 Test 评估。", "",
        "## 对方案问题的直接回答", "",
        f"1. **P0 线性可分性：** Head-only 最佳 Macro-AUROC 为 {macro_values['p0']:.6f}（Epoch {statuses['p0']['best_epoch']}）。",
        f"2. **解冻 Stage 4：** P1−P0 为 {p1_p0['macro_auroc_delta']:+.6f}，95% CI [{p1_p0['macro_auroc_delta_ci95'][0]:+.6f}, {p1_p0['macro_auroc_delta_ci95'][1]:+.6f}]。",
        f"3. **继续解冻 Stage 3：** P2−P1 为 {p2_p1['macro_auroc_delta']:+.6f}，95% CI [{p2_p1['macro_auroc_delta_ci95'][0]:+.6f}, {p2_p1['macro_auroc_delta_ci95'][1]:+.6f}]；P2−P0 为 {p2_p0['macro_auroc_delta']:+.6f}，95% CI [{p2_p0['macro_auroc_delta_ci95'][0]:+.6f}, {p2_p0['macro_auroc_delta_ci95'][1]:+.6f}]。",
        f"4. **最高点估计：** {best_profile.upper()} 为 {macro_values[best_profile]:.6f}；是否超过 0.80：{max(macro_values[p] for p in PROFILES) > 0.80}。",
        f"5. **验证稳定性描述：** 以峰值到停止轮 Macro-AUROC 的下降量衡量，当前最小的是 {least_auc_drop_profile.upper()}（下降 {profile_dynamics[least_auc_drop_profile]['peak_to_stop_auc_drop']:.6f}）；各组停止轮和 Valid BCE 变化见逐组报告。",
        f"6. **Pneumonia：** P0/P1/P2 相对 D0 AUROC 差值分别为 {pneumonia_row['p0_minus_d0_delta']:+.6f} / {pneumonia_row['p1_minus_d0_delta']:+.6f} / {pneumonia_row['p2_minus_d0_delta']:+.6f}。",
        f"7. **Lung Opacity：** P0/P1/P2 相对 D0 AUROC 差值分别为 {opacity_row['p0_minus_d0_delta']:+.6f} / {opacity_row['p1_minus_d0_delta']:+.6f} / {opacity_row['p2_minus_d0_delta']:+.6f}。",
        "8. **实际效率：** 正式 epoch images/s 在上表；所选 Benchmark 配置的稳定长测均值见 `gpu_benchmark_comparison.md`，两种计时分别报告。",
        "9. **训练/验证分离：** 比较每组 Train BCE、Valid BCE、Macro-AUROC 曲线及动态停止轮；早停只有三条件同时成立才触发。",
        "10. **额外解冻的收益/成本：** 联合对照各组最佳 Macro-AUROC、Subject 配对 CI、实际 images/s 和可训练参数数量；不以单一点估计宣称收益。",
        "",
    ])
    (output_dir / "p0_p1_p2_d0_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    _line_plot(output_dir / "macro_auroc_curve.png", "Validation Study Macro-AUROC", [
        (name.upper(), [float(row["macro_auroc"]) for row in histories[name]])
        for name in ("d0", "p0", "p1", "p2")
    ], y_min=0.5, y_max=1.0, y_label="Macro-AUROC")
    _plot_train_validation_loss(output_dir / "train_valid_loss_curve.png", histories)
    selected = selected_runtime
    d0_runtime = json.loads(
        (d0_dir.parent / "dino_selected_runtime_config.json").read_text(encoding="utf-8"))
    _plot_throughput(output_dir / "gpu_throughput_comparison.png", {
        "D0": float(d0_runtime["measured_d0_images_per_s"]),
        **{profile.upper(): float(selected[profile]["measured_images_per_s"])
           for profile in PROFILES},
    })
    return {
        "macro_auroc": macro_values,
        "comparisons": comparisons,
        "best_profile": best_profile,
        "validation_studies": expected_count,
        "validation_subjects": bootstrap_payload["validation_subjects"],
    }


def export_best_inference_package(config, freeze, output_dir, comparison, valid_data):
    profile = comparison["best_profile"]
    run_dir = output_dir / profile
    checkpoint_path = run_dir / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    package_root = output_dir / "submission_export"
    package_dir = package_root / "DINOv3_ConvNeXtTiny_Inference"
    package_dir.mkdir(parents=True, exist_ok=True)
    inference_model = DINOv3StudyClassifier(num_labels=NUM_LABELS)
    result = inference_model.load_state_dict(checkpoint["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Cannot strictly load {profile.upper()} best checkpoint for export")
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in inference_model.state_dict().items()
    }
    save_file(state, str(package_dir / "model.safetensors"))
    shutil.copyfile(WORK_DIR / "dino_model.py", package_dir / "dino_model.py")
    shutil.copyfile(WORK_DIR / "dinov3_meta_convnext.py", package_dir / "dinov3_meta_convnext.py")
    shutil.copyfile(WORK_DIR / "submission_inference.py", package_dir / "submission_inference.py")
    license_text = (WORK_DIR / "META_DINOV3_LICENSE.md").read_text(encoding="utf-8")
    if "DINOv3 License" not in license_text:
        raise RuntimeError("DINOv3 license text is missing")
    (package_dir / "LICENSE.md").write_text(license_text, encoding="utf-8")
    model_config = {
        "model": "DINOv3 ConvNeXt-Tiny single-branch",
        "selected_profile": profile,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_macro_auroc": float(checkpoint["best_macro_auroc"]),
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "image_size": 320,
        "study_fusion": "mean of raw per-image logits before sigmoid",
        "feature_pooling": "global average pooling then official DINOv3 ConvNeXt layer_norm",
        "input_mean": config["source"]["input_mean"],
        "input_std": config["source"]["input_std"],
        "source_repository": config["source"]["huggingface_repo"],
        "source_revision": config["source"]["revision"],
        "initialization_sha256": config["source"]["weights_sha256"],
        "trained_checkpoint_sha256": sha256_file(checkpoint_path),
        "inference_parameters": parameter_counts(inference_model)["total"],
        "training_projection_head_included": False,
        "test_split_read": False,
    }
    (package_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (package_dir / "requirements-inference.txt").write_text(
        "torch>=2.0\ntorchvision>=0.15\nsafetensors>=0.4\nPillow>=10\nnumpy>=1.24\n",
        encoding="utf-8")
    archive_path = package_root / "DINOv3_P0_P1_P2_Inference.zip"
    names = ["model.safetensors", "model_config.json", "dino_model.py",
             "dinov3_meta_convnext.py", "submission_inference.py", "LICENSE.md",
             "requirements-inference.txt"]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name in names:
            archive.write(package_dir / name,
                          arcname=f"DINOv3_ConvNeXtTiny_Inference/{name}")
    archive_bytes = archive_path.stat().st_size
    package_bytes = sum((package_dir / name).stat().st_size for name in names)
    limit = int(freeze["submission"]["size_limit_bytes"])
    if archive_bytes >= limit or package_bytes >= limit:
        raise RuntimeError(
            f"Inference package exceeds strict size limit: zip={archive_bytes}, files={package_bytes}")

    multi_indices = [index for index, count in enumerate(valid_data.image_counts) if count > 1]
    if not multi_indices:
        raise RuntimeError("Validation split has no multi-image Study for inference export check")
    sample_index = min(multi_indices, key=lambda index: valid_data.image_counts[index])
    normalized_images = valid_data[(sample_index, 0)]["images"]
    device = torch.device("cuda")
    inference_model.to(device).eval()
    sample = normalized_images.to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = torch.sigmoid(inference_model(sample).float().mean(dim=0)).cpu().numpy()
    del inference_model, state, checkpoint, sample
    gc.collect()
    torch.cuda.empty_cache()
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    from submission_inference import Predictor  # noqa: PLC0415

    predictor = Predictor(package_dir, device=str(device))
    actual = predictor.predict_study_tensor(normalized_images)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    image_paths = valid_data.dataset.studies[sample_index]["image_paths"]
    actual_paths = predictor.predict_study_paths(image_paths)
    np.testing.assert_allclose(actual_paths, expected, rtol=2e-5, atol=2e-6)
    if len(image_paths) < 2 or actual_paths.shape != (NUM_LABELS,):
        raise RuntimeError("Standalone multi-image Study inference verification failed")
    report = [
        "# P0/P1/P2 推理包大小与加载验证", "",
        f"- Selected profile: `{profile.upper()}`; Best Epoch: {model_config['best_epoch']}; Macro-AUROC: {model_config['best_validation_macro_auroc']:.6f}.",
        f"- ZIP bytes: {archive_bytes:,}; uncompressed package bytes: {package_bytes:,}; strict limit: {limit:,} bytes.",
        f"- ZIP and uncompressed package both below the limit: {archive_bytes < limit and package_bytes < limit}.",
        f"- Inference parameters: {model_config['inference_parameters']:,}.",
        f"- Strict checkpoint load and multi-image Valid Study ({len(image_paths)} images) succeeded.",
        f"- Tensor and PIL prediction max absolute errors: {float(np.max(np.abs(actual-expected))):.8g} / {float(np.max(np.abs(actual_paths-expected))):.8g}.",
        "- D0-C1 projection head, optimizer, scheduler, sampler, training logs and predictions are excluded.",
        "- Test split was not read.", "",
    ]
    (output_dir / "submission_size_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "selected_profile": profile,
        "best_epoch": model_config["best_epoch"],
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "uncompressed_package_bytes": package_bytes,
        "limit_bytes": limit,
        "passes_strict_limit": archive_bytes < limit and package_bytes < limit,
        "validation_multi_image_count": len(image_paths),
        "maximum_probability_absolute_error": float(np.max(np.abs(actual_paths - expected))),
    }
    save_json_atomic(output_dir / "submission_export_verification.json", result)
    return result


def run_all():
    config, freeze = load_experiment_config()
    output_dir = output_root(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "pipeline")
    data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    train_data, valid_data, data_config = build_datasets(data_path, int(config["seed"]))
    environment = environment_report()
    backbone, verification = load_pretrained_backbone("cpu")
    del backbone
    gc.collect()
    write_preflight_audit(config, freeze, data_config, train_data, valid_data,
                          verification, environment, output_dir)
    freeze_verification = verify_freeze_profiles(config, freeze, train_data, output_dir)
    if not freeze_verification["all_profiles_passed"]:
        raise RuntimeError("Model freeze verification did not pass")

    completed_profiles = {}
    for profile in PROFILES:
        runtime = run_profile_benchmark(
            config, freeze, profile, train_data, output_dir)
        result = train_profile(
            config, freeze, profile, train_data, valid_data,
            verification, runtime, output_dir)
        completed_profiles[profile] = result
        save_json_atomic(output_dir / "experiment_progress.json", {
            "experiment": freeze["experiment"],
            "last_completed_profile": profile,
            "completed_profiles": completed_profiles,
            "next_profile": next((item for item in PROFILES
                                   if item not in completed_profiles), None),
            "complete": len(completed_profiles) == len(PROFILES),
            "test_split_read": False,
        })
        logger.info("Completed %s; stop=%s best_epoch=%d best_auc=%.6f",
                    profile.upper(), result["stop_reason"], result["best_epoch"],
                    result["best_macro_auroc"])

    selected_payload = json.loads(
        (output_dir / "selected_runtime_config.json").read_text(encoding="utf-8"))
    aggregates_by_profile = {
        profile: json.loads(
            (output_dir / f"{profile}_benchmark_summary.json").read_text(encoding="utf-8")
        )["repeated_candidate_aggregates"]
        for profile in PROFILES
    }
    d0_runtime = json.loads(
        (WORK_DIR / "outputs" / "experiments" / "DINOv3_D0_D0C1"
         / "dino_selected_runtime_config.json").read_text(encoding="utf-8"))
    _write_benchmark_comparison(output_dir, selected_payload["profiles"],
                                aggregates_by_profile, d0_runtime)
    comparison = compare_frozen_profiles(config, freeze, output_dir)
    export = export_best_inference_package(config, freeze, output_dir,
                                           comparison, valid_data)
    outcome = {
        "experiment": freeze["experiment"],
        "profiles": completed_profiles,
        "comparison": comparison,
        "submission_export": export,
        "test_split_read": False,
    }
    save_json_atomic(output_dir / "experiment_complete.json", outcome)
    logger.info("P0/P1/P2 experiment completed; best=%s auc=%.6f archive_bytes=%d",
                comparison["best_profile"].upper(),
                comparison["macro_auroc"][comparison["best_profile"]],
                export["archive_bytes"])
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.parse_args()
    run_all()


if __name__ == "__main__":
    main()
