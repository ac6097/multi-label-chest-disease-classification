from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
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
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[3]
PROJECT_DIR = WORK_DIR.parent
R5_DIR = HERE.parent / "R5_Global_Only_512"
OUTPUT_ROOT = WORK_DIR / "outputs/experiments/DINOv3_Feature_Optimization"
GROUP = "R7_ResidualTransfer_512"
IMAGE_SIZE = 512
SEED = 42
EXPECTED_WEIGHT_SHA256 = "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
PRELIGHT_VERSION = "r7-zero-gamma-equivalence-and-two-step-preflight-v1"
LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
TARGET_COLS = [f"target_{name.replace(' ', '_')}" for name in LABELS]
PROB_COLS = [f"prob_{name.replace(' ', '_')}" for name in LABELS]
LOGIT_COLS = [f"logit_{name.replace(' ', '_')}" for name in LABELS]

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WORK_DIR))

from dino_analysis import LABEL_NAMES, _columns  # noqa: E402
from dino_data import make_loader, shutdown_loader  # noqa: E402
from dino_experiment import load_pretrained_backbone, save_json_atomic  # noqa: E402
from dino_feature_models import (  # noqa: E402
    build_masked_datasets,
    collate_masked_studies,
)
from dino_feature_optimization import run_custom_group  # noqa: E402
from dino_freeze_experiment import head_hash, set_profile_training_mode  # noqa: E402
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from hybrid.common import seed_everything  # noqa: E402
from r7_model import R7Stage3Stage4ResidualClassifier  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def key_frame(path: Path):
    frame = pd.read_csv(path, dtype={"Subject_id": "string", "Study_id": "string"})
    target_cols, probability_cols = _columns(None)
    required = {"Subject_id", "Study_id", *target_cols, *probability_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame[["Subject_id", "Study_id"]].isna().any().any():
        raise ValueError(f"{path} has missing Study identifiers")
    if frame.duplicated(["Subject_id", "Study_id"]).any():
        raise ValueError(f"{path} has duplicated Study keys")
    return frame


def metric_vectors(labels: np.ndarray, scores: np.ndarray):
    aucs, aps = [], []
    for class_index in range(labels.shape[1]):
        y = labels[:, class_index]
        p = scores[:, class_index]
        aucs.append(float(roc_auc_score(y, p))
                    if np.unique(y).size == 2 else float("nan"))
        aps.append(float(average_precision_score(y, p))
                    if np.any(y == 1) else float("nan"))
    return np.asarray(aucs, dtype=np.float64), np.asarray(aps, dtype=np.float64)


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None], 0
    return np.quantile(values, [0.025, 0.975]).tolist(), int(len(values))


class ScalarGradientRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.nonzero_count = 0
        self.sum_abs = 0.0
        self.max_abs = 0.0

    def capture(self, gradient: torch.Tensor):
        value = float(gradient.detach().abs().max().cpu())
        self.count += 1
        self.nonzero_count += int(value > 0.0)
        self.sum_abs += value
        self.max_abs = max(self.max_abs, value)
        return gradient

    def snapshot(self):
        return {
            "count": int(self.count),
            "mean_abs": self.sum_abs / self.count if self.count else 0.0,
            "max_abs": float(self.max_abs),
            "nonzero_fraction": self.nonzero_count / self.count if self.count else 0.0,
        }


def r7_profile_layout(model):
    trunk = model.backbone.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules = [
        ("backbone.model.downsample_layers.2", trunk.downsample_layers[2]),
        ("backbone.model.stages.2", trunk.stages[2]),
        ("backbone.model.downsample_layers.3", trunk.downsample_layers[3]),
        ("backbone.model.stages.3", trunk.stages[3]),
        ("backbone.model.norm", trunk.norm),
        ("classifier", model.classifier),
        ("transfer", model.transfer),
    ]
    frozen_modules = [
        ("backbone.model.downsample_layers.0", trunk.downsample_layers[0]),
        ("backbone.model.downsample_layers.1", trunk.downsample_layers[1]),
        ("backbone.model.stages.0", trunk.stages[0]),
        ("backbone.model.stages.1", trunk.stages[1]),
    ]
    for _, module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    allowed = (
        "backbone.model.downsample_layers.2.", "backbone.model.stages.2.",
        "backbone.model.downsample_layers.3.", "backbone.model.stages.3.",
        "backbone.model.norm.", "backbone.model.norms.3.", "classifier.", "transfer.",
    )
    unexpected = [name for name in trainable_names
                  if not any(name.startswith(prefix) for prefix in allowed)]
    if unexpected:
        raise RuntimeError(f"R7 has unexpected trainable parameters: {unexpected[:10]}")
    return trainable_modules, frozen_modules, trainable_names


def make_r7_builder(runtime, expected_head_sha, gradient_recorder):
    def builder(profile, device, active_runtime, seed):
        if profile != "p2" or int(seed) != SEED:
            raise RuntimeError(f"R7 must keep the R5 P2 freeze profile and seed {SEED}")
        seed_everything(SEED)
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError("R7 original DINOv3 pretrained weight SHA-256 mismatch")
        model = R7Stage3Stage4ResidualClassifier(backbone, num_labels=NUM_LABELS).to(device)
        initial_head_sha = head_hash(model)
        if initial_head_sha != expected_head_sha:
            raise RuntimeError(
                "R7 Global Head initialization does not match the independently seeded R5 head: "
                f"{initial_head_sha}")
        trainable_modules, frozen_modules, trainable_names = r7_profile_layout(model)
        model.backbone.set_gradient_checkpointing(
            bool(active_runtime["gradient_checkpointing"]))
        if active_runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        frozen = total - trainable
        if (total, trainable, frozen) != (28_122_731, 26_887_691, 1_235_040):
            raise RuntimeError(
                f"R7 total/trainable/frozen parameters differ from the declared architecture: "
                f"{total}/{trainable}/{frozen}")
        verification.update({
            "model_variant": "DINOv3 ConvNeXt-Tiny Global-only + Stage3-to-Stage4 residual transfer",
            "loaded_finetuned_checkpoint": False,
            "global_classifier_init_sha256": initial_head_sha,
            "r7_new_branch": {
                "operation": "AvgPool2d(2,2) -> Conv2d(384,768,1,bias=False) -> gamma * projection",
                "gamma_init": 0.0,
                "projection_parameters": sum(p.numel() for p in model.transfer.projection.parameters()),
                "gamma_parameters": model.transfer.gamma.numel(),
            },
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "training_runtime": {
                key: active_runtime[key] for key in (
                    "image_budget", "grad_accumulation", "gradient_checkpointing",
                    "channels_last", "num_workers", "pin_memory", "prefetch_factor",
                    "effective_images_per_update", "effective_studies_per_update",
                ) if key in active_runtime
            },
        })
        gradient_recorder.reset()
        model._r7_gamma_gradient_hook = model.transfer.gamma.register_hook(
            gradient_recorder.capture)
        return model, verification, trainable_modules, frozen_modules, trainable_names
    return builder


def r7_optimizer(model, config):
    training = config["training"]
    decay = float(training["weight_decay"])
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [p for p in model.classifier.parameters() if p.requires_grad]
    projection_parameters = [p for p in model.transfer.projection.parameters()
                             if p.requires_grad]
    gamma_parameters = [model.transfer.gamma] if model.transfer.gamma.requires_grad else []
    groups = [
        {"params": backbone_parameters, "lr": float(training["peak_lr_backbone"]),
         "weight_decay": decay, "name": "backbone", "r7_role": "r5_backbone_trainable"},
        {"params": head_parameters + projection_parameters,
         "lr": float(training["peak_lr_head"]), "weight_decay": decay,
         "name": "head", "r7_role": "classifier_and_projection"},
        {"params": gamma_parameters, "lr": float(training["peak_lr_head"]),
         "weight_decay": 0.0, "name": "head", "r7_role": "gamma_no_weight_decay"},
    ]
    if any(not group["params"] for group in groups):
        raise RuntimeError("R7 optimizer has an empty parameter group")
    optimizer = torch.optim.AdamW(
        groups, fused=bool(config["freeze"]["training"]["fused_adamw"]))
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    observed = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError("R7 optimizer groups do not contain every trainable parameter exactly once")
    return optimizer


def r7_audit_forward(model, images, _batch):
    components = model.forward_components(images, include_gamma_zero_logits=True)
    expected = {
        "stage3": (384, 32, 32), "stage4": (768, 16, 16),
        "projected_stage3": (768, 16, 16), "residual_contribution": (768, 16, 16),
        "fused_stage4": (768, 16, 16),
    }
    shapes = {}
    for key, shape in expected.items():
        actual = tuple(components[key].shape[1:])
        shapes[key] = list(components[key].shape)
        if actual != shape:
            raise RuntimeError(f"R7 {key} shape {actual} != {shape}")
    if float(components["gamma"].detach().float().cpu()) != 0.0:
        raise RuntimeError("R7 must start with gamma exactly zero")
    equivalence = float((components["fused_logits"]
                         - components["gamma_zero_logits"]).abs().max().detach().cpu())
    if equivalence > 1e-6:
        raise RuntimeError(f"R7 initial gamma-zero logits differ from raw Stage4 path: {equivalence}")
    return {
        "logits": components["fused_logits"],
        "r7_shapes": shapes,
        "gamma_init": 0.0,
        "gamma_zero_logits_max_abs_error": equivalence,
        "residual_projection_parameters": sum(
            parameter.numel() for parameter in model.transfer.projection.parameters()),
    }


def manual_study_mean(image_logits, study_index, num_studies):
    return torch.stack([
        image_logits[study_index == index].float().mean(dim=0)
        for index in range(int(num_studies))
    ])


def capture_sample_validation_batch(valid_data, runtime):
    loader, sampler, _ = make_loader(
        valid_data, image_budget=int(runtime["image_budget"]), shuffle=False,
        seed=SEED, num_workers=0, pin_memory=True,
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    try:
        for batch in loader:
            if (int(batch["images"].shape[0]) > 0
                    and bool((batch["image_counts"] > 1).any())):
                return {
                    "images": batch["images"].detach().cpu(),
                    "study_index": batch["study_index"].detach().cpu(),
                    "image_counts": batch["image_counts"].detach().cpu(),
                    "num_studies": int(batch["num_studies"]),
                    "subject_id": [str(value) for value in batch["subject_id"]],
                    "study_id": [str(value) for value in batch["study_id"]],
                    "image_ids": batch["image_ids"],
                }
        raise RuntimeError("Could not load a Validation batch containing a multi-image Study")
    finally:
        shutdown_loader(loader)


def equivalence_pair(expected_head_sha):
    seed_everything(SEED)
    r5_backbone, r5_verification = load_pretrained_backbone("cpu")
    r5 = DINOv3StudyClassifier(r5_backbone, num_labels=NUM_LABELS).cpu().eval()
    seed_everything(SEED)
    r7_backbone, r7_verification = load_pretrained_backbone("cpu")
    r7 = R7Stage3Stage4ResidualClassifier(r7_backbone, num_labels=NUM_LABELS).cpu().eval()
    r5_head = head_hash(r5)
    r7_head = head_hash(r7)
    if r5_head != expected_head_sha or r7_head != expected_head_sha:
        raise RuntimeError(
            f"Seeded Global Head mismatch: R5={r5_head}, R7={r7_head}, expected={expected_head_sha}")
    if (r5_verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256
            or r7_verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256):
        raise RuntimeError("R5/R7 equivalence models did not use the pinned original DINOv3 weights")
    return r5, r7, {"r5": r5_verification, "r7": r7_verification,
                    "global_head_sha256": r5_head}


def max_abs(left, right):
    return float((left.float() - right.float()).abs().max().detach().cpu())


def test_initialization_equivalence(valid_data, runtime, expected_head_sha):
    sample = capture_sample_validation_batch(valid_data, runtime)
    images_cpu = sample["images"].float().contiguous()
    if images_cpu.shape[0] < 2:
        raise RuntimeError("Initialization equivalence requires a multi-image Validation batch")
    r5, r7, verification = equivalence_pair(expected_head_sha)
    cpu_results = {}
    with torch.inference_mode():
        single = images_cpu[:1]
        logits5_single = r5(single)
        logits7_single = r7(single)
        cpu_results["single_image_logits_max_abs_error"] = max_abs(
            logits5_single, logits7_single)
        logits5 = r5(images_cpu)
        components7 = r7.forward_components(images_cpu, include_gamma_zero_logits=True)
        logits7 = components7["fused_logits"]
        study5 = study_logit_mean(
            logits5, sample["study_index"], sample["num_studies"], sample["image_counts"])
        study7 = study_logit_mean(
            logits7, sample["study_index"], sample["num_studies"], sample["image_counts"])
        cpu_results["multi_image_logits_max_abs_error"] = max_abs(logits5, logits7)
        cpu_results["study_logit_mean_max_abs_error"] = max_abs(study5, study7)
        cpu_results["study_probability_max_abs_error"] = max_abs(
            torch.sigmoid(study5), torch.sigmoid(study7))
        cpu_results["stage3_shape"] = list(components7["stage3"].shape)
        cpu_results["stage4_shape"] = list(components7["stage4"].shape)
        cpu_results["projection_shape"] = list(components7["projected_stage3"].shape)
        cpu_results["gamma_init"] = float(components7["gamma"].detach().cpu())
    if max(cpu_results[key] for key in (
            "single_image_logits_max_abs_error", "multi_image_logits_max_abs_error",
            "study_logit_mean_max_abs_error", "study_probability_max_abs_error")) > 1e-6:
        raise RuntimeError(f"CPU FP32 gamma-zero equivalence failed: {cpu_results}")

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("R7 equivalence requires the R5 CUDA BF16 runtime")
    device = torch.device("cuda")
    r5.to(device, memory_format=torch.channels_last).eval()
    r7.to(device, memory_format=torch.channels_last).eval()
    images_gpu = images_cpu.to(device).contiguous(memory_format=torch.channels_last)
    sample_index_gpu = sample["study_index"].to(device)
    counts_gpu = sample["image_counts"].to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits5_single = r5(images_gpu[:1])
        logits7_single = r7(images_gpu[:1])
        logits5 = r5(images_gpu)
        components7 = r7.forward_components(images_gpu, include_gamma_zero_logits=True)
        logits7 = components7["fused_logits"]
        study5 = study_logit_mean(logits5, sample_index_gpu, sample["num_studies"], counts_gpu)
        study7 = study_logit_mean(logits7, sample_index_gpu, sample["num_studies"], counts_gpu)
    gpu_results = {
        "single_image_logits_max_abs_error": max_abs(logits5_single, logits7_single),
        "multi_image_logits_max_abs_error": max_abs(logits5, logits7),
        "study_logit_mean_max_abs_error": max_abs(study5, study7),
        "study_probability_max_abs_error": max_abs(torch.sigmoid(study5), torch.sigmoid(study7)),
        "amp_dtype": "bfloat16",
    }
    if max(value for key, value in gpu_results.items() if key.endswith("error")) > 1e-4:
        raise RuntimeError(f"CUDA BF16 gamma-zero equivalence failed: {gpu_results}")
    result = {
        "passed": True,
        "method": "R5 and R7 independently initialized from the same pinned original weights and seed; R7 gamma held at zero",
        "input_source": "official Validation only; no Test access",
        "validation_sample_studies": sample["study_id"],
        "validation_sample_image_counts": sample["image_counts"].tolist(),
        "pretrained_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "global_head_init_sha256": expected_head_sha,
        "cpu_fp32": cpu_results,
        "gpu_bf16": gpu_results,
        "tolerance_cpu_fp32": 1e-6,
        "tolerance_gpu_bf16": 1e-4,
        "test_data_accessed": False,
    }
    del r5, r7, images_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return result


def r7_parameter_groups_report(model, optimizer, runtime, config):
    counts = []
    for index, group in enumerate(optimizer.param_groups):
        counts.append({
            "group_index": index,
            "name": group["name"],
            "role": group["r7_role"],
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "learning_rate_peak": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        })
    return {
        "optimizer": config["training"]["optimizer"],
        "fused_adamw": bool(config["freeze"]["training"]["fused_adamw"]),
        "groups": counts,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                     if parameter.requires_grad),
        "frozen_parameters": sum(parameter.numel() for parameter in model.parameters()
                                  if not parameter.requires_grad),
        "runtime": runtime,
        "gamma_weight_decay": 0.0,
        "projection_weight_decay": float(config["training"]["weight_decay"]),
        "test_data_accessed": False,
    }


def r7_actual_preflight(train_data, config, freeze, runtime, builder, optimizer_builder):
    loader, sampler, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=0, pin_memory=bool(runtime["pin_memory"]),
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    iterator = iter(loader)
    batches = []
    model = optimizer = None
    try:
        for _ in range(2 * int(runtime["grad_accumulation"])):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        accumulation = int(runtime["grad_accumulation"])
        if len(batches) < 2 * accumulation:
            raise RuntimeError("R7 preflight could not form two full gradient-accumulation windows")
        windows = (batches[:accumulation], batches[accumulation:2 * accumulation])
        window_counts = [sum(int(batch["images"].shape[0]) for batch in window)
                         for window in windows]
        target = float(runtime["effective_batch_target"])
        if any(abs(count - target) / target > 0.2 for count in window_counts):
            raise RuntimeError(f"R7 preflight effective image windows differ from R5: {window_counts}")

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        model, verification, train_modules, frozen_modules, trainable_names = builder(
            "p2", device, runtime, SEED)
        set_profile_training_mode(model, frozen_modules)
        optimizer = optimizer_builder(model, config)
        optimizer_ids = [id(parameter) for group in optimizer.param_groups
                         for parameter in group["params"]]
        expected_ids = {id(parameter) for parameter in model.parameters()
                        if parameter.requires_grad}
        if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != expected_ids:
            raise RuntimeError("R7 preflight optimizer groups do not exactly cover trainable parameters")
        frozen_snapshots = {
            name: next(module.parameters()).detach().cpu().clone()
            for name, module in frozen_modules
        }
        transfer_before = model.transfer.projection.weight.detach().cpu().clone()
        gamma_initial = float(model.transfer.gamma.detach().cpu())
        if gamma_initial != 0.0:
            raise RuntimeError(f"R7 preflight gamma initial value is not zero: {gamma_initial}")

        max_logit_error = 0.0
        max_loss_error = 0.0
        window_losses = []
        gradient_groups = {}
        frozen_gradients_absent = True
        gamma_first_grad = 0.0
        projection_first_grad = 0.0
        gamma_after_first_step = None
        projection_second_grad = 0.0
        gamma_second_grad = 0.0
        max_window_loss_error = 0.0
        for window_index, window in enumerate(windows, start=1):
            eligible_total = sum(int(batch["known"].any(dim=1).sum()) for batch in window)
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            study_logits_cpu, labels_cpu = [], []
            for batch in window:
                images = batch["images"].to(device, non_blocking=True)
                if runtime["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                study_index = batch["study_index"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                known = batch["known"].to(device, non_blocking=True)
                image_counts = batch["image_counts"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_logits = model(images)
                    study_logits = study_logit_mean(
                        image_logits.float(), study_index, batch["num_studies"], image_counts)
                    manual_logits = manual_study_mean(
                        image_logits, study_index, batch["num_studies"])
                    max_logit_error = max(
                        max_logit_error,
                        float((study_logits - manual_logits).abs().max().detach().cpu()))
                    per_study, eligible = study_bce_per_study(study_logits, labels, known)
                    manual_per_study = F.binary_cross_entropy_with_logits(
                        manual_logits, labels, reduction="none").mean(dim=1)
                    max_loss_error = max(
                        max_loss_error,
                        float((per_study[eligible] - manual_per_study[eligible])
                              .abs().max().detach().cpu()))
                    bce_sum = per_study[eligible].sum()
                    loss = bce_sum / max(eligible_total, 1)
                if not torch.isfinite(loss):
                    raise FloatingPointError("R7 actual preflight produced non-finite Study BCE")
                loss.backward()
                accumulated_loss += float(bce_sum.detach().float().cpu()) / max(eligible_total, 1)
                study_logits_cpu.append(study_logits.detach().float().cpu())
                labels_cpu.append(labels.detach().float().cpu())

            concatenated_study_logits = torch.cat(study_logits_cpu, dim=0)
            concatenated_labels = torch.cat(labels_cpu, dim=0)
            manual_window_loss = float(F.binary_cross_entropy_with_logits(
                concatenated_study_logits, concatenated_labels,
                reduction="none").mean(dim=1).mean())
            max_window_loss_error = max(
                max_window_loss_error, abs(accumulated_loss - manual_window_loss))
            if max_window_loss_error > 1e-6:
                raise RuntimeError(
                    f"R7 accumulation-window Study BCE differs from the manual mean: "
                    f"{max_window_loss_error}")

            trainable_grad_groups = {}
            for module_name, module in train_modules:
                grads = [parameter.grad for parameter in module.parameters()
                         if parameter.requires_grad and parameter.grad is not None]
                trainable_grad_groups[module_name] = bool(
                    any(torch.isfinite(grad).all() and torch.count_nonzero(grad).item() > 0
                        for grad in grads))
            if not all(trainable_grad_groups.values()):
                raise RuntimeError(f"R7 preflight found a zero-gradient trainable group: {trainable_grad_groups}")
            for _, module in frozen_modules:
                if any(parameter.grad is not None for parameter in module.parameters()):
                    frozen_gradients_absent = False
            if not frozen_gradients_absent:
                raise RuntimeError("A frozen R5 Stem/Stage1/Stage2 parameter received gradient")

            gamma_grad = model.transfer.gamma.grad
            gamma_grad_value = float(gamma_grad.detach().abs().max().cpu()) if gamma_grad is not None else 0.0
            projection_grad = model.transfer.projection.weight.grad
            projection_grad_value = (float(projection_grad.detach().norm().cpu())
                                     if projection_grad is not None else 0.0)
            if window_index == 1:
                gamma_first_grad = gamma_grad_value
                projection_first_grad = projection_grad_value
                if gamma_first_grad <= 0.0:
                    raise RuntimeError("R7 gamma did not receive nonzero gradient at gamma=0")
                if projection_first_grad > 1e-10:
                    raise RuntimeError(
                        "R7 projection gradient at gamma=0 was unexpectedly nonzero; inspect fusion implementation")
            else:
                gamma_second_grad = gamma_grad_value
                projection_second_grad = projection_grad_value
                if gamma_second_grad <= 0.0 or projection_second_grad <= 0.0:
                    raise RuntimeError(
                        f"R7 branch did not learn after gamma moved: gamma_grad={gamma_second_grad}, "
                        f"projection_grad={projection_second_grad}")

            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True)
            optimizer.step()
            window_losses.append(accumulated_loss)
            if window_index == 1:
                gamma_after_first_step = float(model.transfer.gamma.detach().cpu())
                if abs(gamma_after_first_step) <= 0.0:
                    raise RuntimeError("R7 gamma stayed exactly zero after its first optimizer step")
        frozen_unchanged = {
            name: torch.equal(next(module.parameters()).detach().cpu(), snapshot)
            for (name, module), snapshot in zip(frozen_modules, frozen_snapshots.values())
        }
        if not all(frozen_unchanged.values()):
            raise RuntimeError(f"R7 preflight changed frozen modules: {frozen_unchanged}")
        projection_after = model.transfer.projection.weight.detach().cpu()
        if torch.equal(transfer_before, projection_after):
            raise RuntimeError("R7 Projection Conv weights did not change after gamma became nonzero")
        properties = torch.cuda.get_device_properties(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if peak_reserved > int(properties.total_memory * 0.90):
            raise RuntimeError("R7 preflight exceeded the configured 90% GPU memory ceiling")
        return {
            "passed": True,
            "preflight_version": PRELIGHT_VERSION,
            "runtime": runtime,
            "preflight_window_images": window_counts,
            "preflight_microbatches_per_update": accumulation,
            "preflight_studies_per_update": [
                sum(int(batch["num_studies"]) for batch in window) for window in windows],
            "disposable_optimizer_steps": 2,
            "formal_training_restarts_from_original_pretrained_weights": True,
            "study_logit_mean_max_abs_error": max_logit_error,
            "study_bce_per_study_max_abs_error": max_loss_error,
            "study_bce_accumulation_window_max_abs_error": max_window_loss_error,
            "study_loss_window_values": window_losses,
            "loss_finite": True,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad),
            "trainable_groups_nonzero_gradients": trainable_grad_groups,
            "frozen_groups_gradient_absent": frozen_gradients_absent,
            "frozen_groups_unchanged": frozen_unchanged,
            "gamma_initial": gamma_initial,
            "gamma_first_step_gradient_abs": gamma_first_grad,
            "gamma_after_first_step": gamma_after_first_step,
            "projection_first_step_gradient_norm_expected_zero": projection_first_grad,
            "gamma_second_step_gradient_abs": gamma_second_grad,
            "projection_second_step_gradient_norm_nonzero": projection_second_grad,
            "projection_parameters_changed_after_gamma_nonzero": True,
            "optimizer_parameter_groups": [
                {"name": group["name"], "role": group["r7_role"],
                 "parameter_count": sum(p.numel() for p in group["params"]),
                 "peak_lr": float(group["lr"]),
                 "weight_decay": float(group["weight_decay"])}
                for group in optimizer.param_groups
            ],
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "test_data_accessed": False,
        }
    finally:
        shutdown_loader(loader)
        del optimizer, model, batches, iterator
        gc.collect()
        torch.cuda.empty_cache()


class R7EpochDiagnostics:
    def __init__(self, output_path: Path, fixed_images: torch.Tensor,
                 runtime: dict, gradient_recorder: ScalarGradientRecorder):
        self.path = output_path
        self.fixed_images = fixed_images.detach().cpu()
        self.runtime = runtime
        self.gradient_recorder = gradient_recorder
        self.gamma_start = 0.0
        self.rows = (pd.read_csv(output_path).to_dict("records")
                     if output_path.is_file() else [])

    def __call__(self, model, epoch, phase, _row):
        if phase == "start":
            self.gamma_start = float(model.transfer.gamma.detach().float().cpu())
            self.gradient_recorder.reset()
            return None
        gamma_end = float(model.transfer.gamma.detach().float().cpu())
        gamma_grad = model.transfer.gamma.grad
        projection_grad = model.transfer.projection.weight.grad
        gamma_last_grad = float(gamma_grad.detach().abs().max().cpu()) if gamma_grad is not None else 0.0
        projection_last_grad = float(projection_grad.detach().norm().cpu()) if projection_grad is not None else 0.0

        device = next(model.parameters()).device
        images = self.fixed_images.to(device, non_blocking=True)
        if self.runtime["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        modes = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16):
                components = model.forward_components(images)
                base_norm = components["stage4"].float().flatten(1).norm(dim=1)
                residual_norm = components["residual_contribution"].float().flatten(1).norm(dim=1)
                ratios = residual_norm / (base_norm + 1e-12)
                cosines = F.cosine_similarity(
                    components["stage4"].float().flatten(1),
                    components["residual_contribution"].float().flatten(1), dim=1)
                ratios = ratios.float().cpu()
                cosines = cosines.float().cpu()
        finally:
            for module, training in modes:
                module.training = training
        grad_stats = self.gradient_recorder.snapshot()
        row = {
            "epoch": int(epoch),
            "scope": "fixed_validation_batch",
            "gamma_start": self.gamma_start,
            "gamma_end": gamma_end,
            "gamma_gradient_hook_count": grad_stats["count"],
            "gamma_gradient_hook_mean_abs": grad_stats["mean_abs"],
            "gamma_gradient_hook_max_abs": grad_stats["max_abs"],
            "gamma_gradient_hook_nonzero_fraction": grad_stats["nonzero_fraction"],
            "gamma_gradient_last_optimizer_update_abs": gamma_last_grad,
            "projection_gradient_last_optimizer_update_norm": projection_last_grad,
            "projection_gradient_nonzero": bool(projection_last_grad > 0.0),
            "residual_ratio_mean": float(ratios.mean()),
            "residual_ratio_median": float(ratios.median()),
            "residual_ratio_p90": float(torch.quantile(ratios, 0.9)),
            "residual_ratio_max": float(ratios.max()),
            "residual_stage4_cosine_mean": float(cosines.mean()),
            "residual_stage4_cosine_median": float(cosines.median()),
            "diagnostic_images": int(len(ratios)),
            "lr_gamma": float(_row["lr_head"]),
        }
        self.rows = [entry for entry in self.rows if int(entry.get("epoch", -1)) != int(epoch)]
        self.rows.append(row)
        pd.DataFrame(self.rows).sort_values(["epoch", "scope"]).to_csv(
            self.path, index=False, encoding="utf-8")
        return {key: value for key, value in row.items()
                if key not in {"scope", "epoch"}}


def load_checkpoint_model(checkpoint_path: Path, expected_epoch: int,
                          runtime: dict, device: torch.device):
    seed_everything(SEED)
    backbone, verification = load_pretrained_backbone("cpu")
    if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R7 inference did not load the pinned original DINOv3 weights")
    model = R7Stage3Stage4ResidualClassifier(backbone, num_labels=NUM_LABELS)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if state is None:
        raise RuntimeError(f"R7 Best checkpoint has no model_state: {checkpoint_path}")
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError(
            f"R7 Best checkpoint epoch {payload.get('epoch')} != recorded Best {expected_epoch}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"R7 Best strict load failed: {result}")
    model.to(device)
    if runtime["channels_last"]:
        model.to(memory_format=torch.channels_last)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def load_r5_model(checkpoint_path: Path, expected_epoch: int,
                  runtime: dict, device: torch.device):
    seed_everything(SEED)
    backbone, verification = load_pretrained_backbone("cpu")
    if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R5 reference inference did not load original DINOv3 weights")
    model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError("R5 checkpoint epoch differs from its recorded Best epoch")
    result = model.load_state_dict(payload["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"R5 Best strict load failed: {result}")
    model.to(device)
    if runtime["channels_last"]:
        model.to(memory_format=torch.channels_last)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_best_inference_ablation(valid_data, r5_status, r7_status,
                                runtime, run_dir: Path):
    device = torch.device("cuda")
    r5_model = load_r5_model(
        R5_DIR / "best.pth", int(r5_status["best_epoch"]), runtime, device)
    r7_model, r7_checkpoint = load_checkpoint_model(
        run_dir / "best.pth", int(r7_status["best_epoch"]), runtime, device)
    gamma_best = float(r7_model.transfer.gamma.detach().float().cpu())
    if not math.isfinite(gamma_best):
        raise FloatingPointError("R7 Best gamma is non-finite")
    r7_model_gamma0 = r7_model

    loader, sampler, _ = make_loader(
        valid_data, image_budget=int(runtime["image_budget"]), shuffle=False,
        seed=SEED, num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    subject_ids, study_ids, image_counts, label_rows = [], [], [], []
    r5_logits, r7_fused_logits, r7_gamma0_logits = [], [], []
    ratio_chunks, cosine_chunks = [], []
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch["images"].to(device, non_blocking=True)
                if runtime["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                study_index = batch["study_index"].to(device, non_blocking=True)
                counts = batch["image_counts"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_logits_r5 = r5_model(images)
                    components = r7_model_gamma0.forward_components(
                        images, include_gamma_zero_logits=True)
                    logits_fused = components["fused_logits"]
                    logits_gamma0 = components["gamma_zero_logits"]
                    studies_r5 = study_logit_mean(
                        image_logits_r5.float(), study_index,
                        batch["num_studies"], counts)
                    studies_fused = study_logit_mean(
                        logits_fused.float(), study_index,
                        batch["num_studies"], counts)
                    studies_gamma0 = study_logit_mean(
                        logits_gamma0.float(), study_index,
                        batch["num_studies"], counts)
                r5_logits.append(studies_r5.cpu().numpy())
                r7_fused_logits.append(studies_fused.cpu().numpy())
                r7_gamma0_logits.append(studies_gamma0.cpu().numpy())
                label_rows.append(batch["labels"].numpy().astype(np.uint8, copy=False))
                subject_ids.extend(str(value) for value in batch["subject_id"])
                study_ids.extend(str(value) for value in batch["study_id"])
                image_counts.extend(int(value) for value in batch["image_counts"].tolist())
                base_norm = components["stage4"].float().flatten(1).norm(dim=1)
                residual_norm = components["residual_contribution"].float().flatten(1).norm(dim=1)
                ratio_chunks.append((residual_norm / (base_norm + 1e-12)).cpu().numpy())
                cosine_chunks.append(F.cosine_similarity(
                    components["stage4"].float().flatten(1),
                    components["residual_contribution"].float().flatten(1), dim=1).cpu().numpy())
    finally:
        shutdown_loader(loader)

    subjects = np.asarray(subject_ids, dtype="U64")
    studies = np.asarray(study_ids, dtype="U64")
    counts = np.asarray(image_counts, dtype=np.int16)
    labels = np.concatenate(label_rows, axis=0).astype(np.uint8, copy=False)
    scores = {
        "R5": np.concatenate(r5_logits, axis=0),
        "R7_Fused": np.concatenate(r7_fused_logits, axis=0),
        "R7_gamma0": np.concatenate(r7_gamma0_logits, axis=0),
    }
    probs = {name: 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
             for name, logits in scores.items()}
    keys = list(zip(subjects.astype(str), studies.astype(str)))
    if len(keys) != 859 or len(set(keys)) != len(keys):
        raise RuntimeError(f"R7 inference expected 859 unique Validation Studies, got {len(keys)}")
    expected_keys = list(valid_data.study_keys)
    if set(keys) != set(expected_keys):
        raise RuntimeError("R7 inference Study keys differ from the official Validation set")

    r5_saved = pd.read_csv(R5_DIR / "val_best.csv",
                           dtype={"Subject_id": "string", "Study_id": "string"})
    r7_saved = pd.read_csv(run_dir / "val_best.csv",
                           dtype={"Subject_id": "string", "Study_id": "string"})
    base = pd.DataFrame({"Subject_id": subjects, "Study_id": studies})
    r5_aligned = base.merge(r5_saved, on=["Subject_id", "Study_id"], how="left",
                            validate="one_to_one", sort=False)
    r7_aligned = base.merge(r7_saved, on=["Subject_id", "Study_id"], how="left",
                            validate="one_to_one", sort=False)
    target_cols, probability_cols = _columns(None)
    for saved, model_name in ((r5_aligned, "R5"), (r7_aligned, "R7")):
        target_saved = saved[target_cols].to_numpy(dtype=np.uint8)
        if not np.array_equal(target_saved, labels):
            raise RuntimeError(f"{model_name} saved validation labels differ from inference labels")
    r5_saved_probabilities = r5_aligned[probability_cols].to_numpy(dtype=np.float64)
    r7_saved_probabilities = r7_aligned[probability_cols].to_numpy(dtype=np.float64)
    r5_probability_delta = np.abs(r5_saved_probabilities - probs["R5"])
    r7_probability_delta = np.abs(r7_saved_probabilities - probs["R7_Fused"])
    max_r5_error = float(r5_probability_delta.max())
    max_r7_error = float(r7_probability_delta.max())
    r5_saved_auroc, r5_saved_auprc = metric_vectors(labels, r5_saved_probabilities)
    r5_replayed_auroc, r5_replayed_auprc = metric_vectors(labels, probs["R5"])
    r5_row, r5_class = np.unravel_index(
        int(np.argmax(r5_probability_delta)), r5_probability_delta.shape)
    r5_replay_diagnostic = {
        "saved_prediction_file": str(R5_DIR / "val_best.csv"),
        "status": "matched" if max_r5_error <= 1e-5 else "probability_mismatch",
        "max_probability_abs_error": max_r5_error,
        "probability_cells_over_1e-5": int((r5_probability_delta > 1e-5).sum()),
        "saved_macro_auroc": float(np.nanmean(r5_saved_auroc)),
        "replayed_macro_auroc": float(np.nanmean(r5_replayed_auroc)),
        "saved_macro_auprc": float(np.nanmean(r5_saved_auprc)),
        "replayed_macro_auprc": float(np.nanmean(r5_replayed_auprc)),
        "max_abs_per_class_auroc_delta": float(
            np.nanmax(np.abs(r5_saved_auroc - r5_replayed_auroc))),
        "max_abs_per_class_auprc_delta": float(
            np.nanmax(np.abs(r5_saved_auprc - r5_replayed_auprc))),
        "worst_prediction_cell": {
            "subject_id": str(subjects[r5_row]),
            "study_id": str(studies[r5_row]),
            "disease": LABELS[r5_class],
            "saved_probability": float(r5_saved_probabilities[r5_row, r5_class]),
            "replayed_probability": float(probs["R5"][r5_row, r5_class]),
        },
        "max_abs_error_by_disease": {
            label: float(r5_probability_delta[:, index].max())
            for index, label in enumerate(LABELS)
        },
        "paired_comparison_uses": "fresh R5 Best checkpoint inference on the same current Validation loader as R7",
        "cause_resolved": False,
    }
    if max_r7_error > 1e-5:
        r5_row, r5_class = np.unravel_index(
            int(np.argmax(r5_probability_delta)), r5_probability_delta.shape)
        raise RuntimeError(
            f"R7 Best checkpoint Validation replay mismatch: R7={max_r7_error}; "
            f"R5 historical prediction replay diagnostic="
            f"{json.dumps(r5_replay_diagnostic, ensure_ascii=False)}")
    if max_r5_error > 1e-5:
        print(
            "R5 historical prediction replay discrepancy; using fresh R5 Best inference "
            "for same-loader paired comparisons: "
            f"{json.dumps(r5_replay_diagnostic, ensure_ascii=False)}",
            flush=True)

    output = pd.DataFrame({
        "Subject_id": subjects, "Study_id": studies,
        "Image_Count": counts,
    })
    for class_index, label in enumerate(LABELS):
        key = label.replace(" ", "_")
        output[f"target_{key}"] = labels[:, class_index]
        for model_name in ("R5", "R7_Fused", "R7_gamma0"):
            output[f"logit_{model_name}_{key}"] = scores[model_name][:, class_index]
            output[f"prob_{model_name}_{key}"] = probs[model_name][:, class_index]
    output.to_csv(run_dir / "r7_validation_predictions.csv", index=False, encoding="utf-8")

    draws, draw_hash = make_subject_draws(subjects)
    ablation, per_class, bootstrap = summarize_ablation(
        labels, probs, subjects, draws, draw_hash)
    ablation.to_csv(run_dir / "r7_inference_ablation.csv", index=False, encoding="utf-8")
    per_class.to_csv(run_dir / "r7_vs_r5_per_class.csv", index=False, encoding="utf-8")
    write_json(run_dir / "paired_subject_bootstrap.json", bootstrap)

    ratio = np.concatenate(ratio_chunks).astype(np.float64)
    cosine = np.concatenate(cosine_chunks).astype(np.float64)
    branch_best = {
        "epoch": int(r7_status["best_epoch"]),
        "scope": "all_Validation_images_at_R7_Best",
        "gamma": gamma_best,
        "residual_ratio_mean": float(np.mean(ratio)),
        "residual_ratio_median": float(np.median(ratio)),
        "residual_ratio_p90": float(np.quantile(ratio, 0.9)),
        "residual_ratio_max": float(np.max(ratio)),
        "residual_stage4_cosine_mean": float(np.mean(cosine)),
        "residual_stage4_cosine_median": float(np.median(cosine)),
        "validation_images": int(len(ratio)),
        "test_data_accessed": False,
    }
    branch_path = run_dir / "r7_branch_diagnostics.csv"
    existing = pd.read_csv(branch_path) if branch_path.is_file() else pd.DataFrame()
    if not existing.empty and "scope" in existing.columns:
        existing = existing[existing["scope"] != "best_full_validation"]
    final_row = {
        "epoch": int(r7_status["best_epoch"]),
        "scope": "best_full_validation",
        "gamma_start": gamma_best, "gamma_end": gamma_best,
        "gamma_gradient_hook_count": np.nan,
        "gamma_gradient_hook_mean_abs": np.nan,
        "gamma_gradient_hook_max_abs": np.nan,
        "gamma_gradient_hook_nonzero_fraction": np.nan,
        "gamma_gradient_last_optimizer_update_abs": np.nan,
        "projection_gradient_last_optimizer_update_norm": np.nan,
        "projection_gradient_nonzero": np.nan,
        "residual_ratio_mean": branch_best["residual_ratio_mean"],
        "residual_ratio_median": branch_best["residual_ratio_median"],
        "residual_ratio_p90": branch_best["residual_ratio_p90"],
        "residual_ratio_max": branch_best["residual_ratio_max"],
        "residual_stage4_cosine_mean": branch_best["residual_stage4_cosine_mean"],
        "residual_stage4_cosine_median": branch_best["residual_stage4_cosine_median"],
        "diagnostic_images": branch_best["validation_images"],
        "lr_gamma": float(r7_status.get("best_lr_head", 0.0)),
    }
    pd.concat([existing, pd.DataFrame([final_row])], ignore_index=True).to_csv(
        branch_path, index=False, encoding="utf-8")

    del r5_model, r7_model_gamma0
    gc.collect()
    torch.cuda.empty_cache()
    replay = {
        "R5_max_probability_abs_error_vs_saved_val_best": max_r5_error,
        "R7_fused_max_probability_abs_error_vs_saved_val_best": max_r7_error,
        "R5_saved_prediction_replay": r5_replay_diagnostic,
        "aligned_validation_studies": int(len(labels)),
        "r7_best_gamma": gamma_best,
        "best_checkpoint_epoch": int(r7_status["best_epoch"]),
        "branch_diagnostics_best_validation": branch_best,
        "test_data_accessed": False,
    }
    write_json(run_dir / "validation_replay_and_branch_summary.json", replay)
    return ablation, per_class, bootstrap, replay


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


def summarize_ablation(labels, probabilities, subjects, draws, draw_hash):
    names = ("R5", "R7_Fused", "R7_gamma0")
    point = {name: metric_vectors(labels, probabilities[name]) for name in names}
    bootstrap_values = {}
    for name in names:
        auc_values = np.full((BOOTSTRAP_REPLICATES, labels.shape[1]), np.nan, dtype=np.float32)
        ap_values = np.full_like(auc_values, np.nan)
        macro_auc = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float32)
        macro_ap = np.full_like(macro_auc, np.nan)
        for draw_index, indices in enumerate(draws):
            aucs, aps = [], []
            y_sample, p_sample = labels[indices], probabilities[name][indices]
            for class_index in range(labels.shape[1]):
                y = y_sample[:, class_index]
                if np.unique(y).size == 2:
                    value = roc_auc_score(y, p_sample[:, class_index])
                    auc_values[draw_index, class_index] = value
                    aucs.append(value)
                if np.any(y == 1):
                    value = average_precision_score(y, p_sample[:, class_index])
                    ap_values[draw_index, class_index] = value
                    aps.append(value)
            if aucs:
                macro_auc[draw_index] = float(np.mean(aucs))
            if aps:
                macro_ap[draw_index] = float(np.mean(aps))
        bootstrap_values[name] = {
            "auroc": auc_values, "auprc": ap_values,
            "macro_auroc": macro_auc, "macro_auprc": macro_ap,
        }

    comparisons = {
        "R7_Fused_minus_R5": ("R5", "R7_Fused"),
        "R7_Fused_minus_R7_gamma0": ("R7_gamma0", "R7_Fused"),
        "R7_gamma0_minus_R5": ("R5", "R7_gamma0"),
    }
    comparison_results = {}
    for comparison, (left, right) in comparisons.items():
        diff_auc_macro = bootstrap_values[right]["macro_auroc"] - bootstrap_values[left]["macro_auroc"]
        diff_ap_macro = bootstrap_values[right]["macro_auprc"] - bootstrap_values[left]["macro_auprc"]
        auc_ci, auc_valid = ci95(diff_auc_macro)
        ap_ci, ap_valid = ci95(diff_ap_macro)
        per_class = {}
        for class_index, disease in enumerate(LABEL_NAMES):
            class_auc = (bootstrap_values[right]["auroc"][:, class_index]
                         - bootstrap_values[left]["auroc"][:, class_index])
            class_ap = (bootstrap_values[right]["auprc"][:, class_index]
                        - bootstrap_values[left]["auprc"][:, class_index])
            class_auc_ci, class_auc_n = ci95(class_auc)
            class_ap_ci, class_ap_n = ci95(class_ap)
            per_class[disease] = {
                "delta_auroc": float(point[right][0][class_index] - point[left][0][class_index]),
                "delta_auroc_ci95": class_auc_ci,
                "delta_auroc_valid_replicates": class_auc_n,
                "delta_auprc": float(point[right][1][class_index] - point[left][1][class_index]),
                "delta_auprc_ci95": class_ap_ci,
                "delta_auprc_valid_replicates": class_ap_n,
            }
        comparison_results[comparison] = {
            "left": left, "right": right,
            "delta_macro_auroc": float(np.nanmean(point[right][0]) - np.nanmean(point[left][0])),
            "delta_macro_auroc_ci95": auc_ci,
            "macro_auroc_valid_replicates": auc_valid,
            "delta_macro_auprc": float(np.nanmean(point[right][1]) - np.nanmean(point[left][1])),
            "delta_macro_auprc_ci95": ap_ci,
            "macro_auprc_valid_replicates": ap_valid,
            "per_class": per_class,
        }

    model_results = {}
    rows = []
    for name in names:
        model_results[name] = {
            "macro_auroc": float(np.nanmean(point[name][0])),
            "macro_auprc": float(np.nanmean(point[name][1])),
            "per_class": {},
        }
        for index, disease in enumerate(LABEL_NAMES):
            auc_ci, auc_n = ci95(bootstrap_values[name]["auroc"][:, index])
            ap_ci, ap_n = ci95(bootstrap_values[name]["auprc"][:, index])
            model_results[name]["per_class"][disease] = {
                "auroc": float(point[name][0][index]), "auroc_ci95": auc_ci,
                "auroc_valid_replicates": auc_n,
                "auprc": float(point[name][1][index]), "auprc_ci95": ap_ci,
                "auprc_valid_replicates": ap_n,
                "positive_studies": int(labels[:, index].sum()),
                "negative_studies": int((labels[:, index] == 0).sum()),
            }
    bootstrap = {
        "method": "Subject-cluster paired bootstrap; percentile 95% CI",
        "sampling_unit": "Subject_id; all Studies belonging to a sampled Subject move together",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "sampling_indices_sha256": draw_hash,
        "aligned_validation_studies": int(len(labels)),
        "validation_subjects": int(len(set(subjects))),
        "one_class_rule": "Skip AUROC for a resample with only one label; skip AP if no positive Study; macro uses available classes.",
        "models": model_results,
        "comparisons": comparison_results,
        "limitations": (
            "Validation has been used for historical checkpoint selection and multiple disease/model comparisons; "
            "these intervals are exploratory and do not replace an independent test set."),
        "test_data_accessed": False,
    }
    per_class_rows = []
    for index, disease in enumerate(LABEL_NAMES):
        row = {
            "Disease": disease,
            "Positive_Studies": int(labels[:, index].sum()),
            "Negative_Studies": int((labels[:, index] == 0).sum()),
        }
        for name in names:
            row[f"{name}_AUROC"] = float(point[name][0][index])
            row[f"{name}_AUPRC"] = float(point[name][1][index])
        for comparison, info in comparison_results.items():
            stats = info["per_class"][disease]
            prefix = comparison.replace("_minus_", "_minus_")
            row[f"{prefix}_AUROC_Delta"] = stats["delta_auroc"]
            row[f"{prefix}_AUROC_CI95_Low"] = stats["delta_auroc_ci95"][0]
            row[f"{prefix}_AUROC_CI95_High"] = stats["delta_auroc_ci95"][1]
            row[f"{prefix}_AUPRC_Delta"] = stats["delta_auprc"]
            row[f"{prefix}_AUPRC_CI95_Low"] = stats["delta_auprc_ci95"][0]
            row[f"{prefix}_AUPRC_CI95_High"] = stats["delta_auprc_ci95"][1]
        per_class_rows.append(row)
    macro_row = {"Disease": "Macro", "Positive_Studies": None, "Negative_Studies": None}
    for name in names:
        macro_row[f"{name}_AUROC"] = model_results[name]["macro_auroc"]
        macro_row[f"{name}_AUPRC"] = model_results[name]["macro_auprc"]
    for comparison, info in comparison_results.items():
        macro_row[f"{comparison}_AUROC_Delta"] = info["delta_macro_auroc"]
        macro_row[f"{comparison}_AUROC_CI95_Low"] = info["delta_macro_auroc_ci95"][0]
        macro_row[f"{comparison}_AUROC_CI95_High"] = info["delta_macro_auroc_ci95"][1]
        macro_row[f"{comparison}_AUPRC_Delta"] = info["delta_macro_auprc"]
        macro_row[f"{comparison}_AUPRC_CI95_Low"] = info["delta_macro_auprc_ci95"][0]
        macro_row[f"{comparison}_AUPRC_CI95_High"] = info["delta_macro_auprc_ci95"][1]
    ablation_frame = pd.DataFrame([macro_row, *per_class_rows])
    r7_vs_r5 = pd.DataFrame(per_class_rows)
    keep = [column for column in r7_vs_r5.columns
            if column in {"Disease", "Positive_Studies", "Negative_Studies", "R5_AUROC",
                          "R7_Fused_AUROC", "R5_AUPRC", "R7_Fused_AUPRC"}
            or column.startswith("R7_Fused_minus_R5_")]
    r7_vs_r5 = r7_vs_r5[keep]
    return ablation_frame, r7_vs_r5, bootstrap


def draw_training_curves(run_dir: Path):
    runs = {
        "R5": R5_DIR,
        "R6": R5_DIR.parent / "R6_Full_Finetune_512",
        "R7": run_dir,
    }
    histories = {name: pd.read_csv(path / "train_log.csv").sort_values("epoch")
                 for name, path in runs.items()}
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in histories.items():
        ax.plot(history["epoch"], history["train_bce"], marker="o", label=f"{name} Train")
        ax.plot(history["epoch"], history["valid_loss"], marker="s", linestyle="--",
                label=f"{name} Validation")
    ax.set(xlabel="Epoch", ylabel="Study BCE", title="R5 / R6 / R7 Train and Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / "train_valid_loss.png", dpi=180)
    plt.close(fig)

    for metric, ylabel, title, file_name in (
        ("macro_auroc", "Macro-AUROC", "R5 / R6 / R7 Validation Macro-AUROC",
         "macro_auroc_curve.png"),
        ("macro_auprc", "Macro-AUPRC", "R5 / R6 / R7 Validation Macro-AUPRC",
         "macro_auprc_curve.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, history in histories.items():
            ax.plot(history["epoch"], history[metric], marker="o", label=name)
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / file_name, dpi=180)
        plt.close(fig)


def draw_branch_curves(run_dir: Path):
    frame = pd.read_csv(run_dir / "r7_branch_diagnostics.csv")
    frame = frame[frame["scope"] == "fixed_validation_batch"].sort_values("epoch")
    if frame.empty:
        raise RuntimeError("R7 gamma/residual diagnostics are empty")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["epoch"], frame["gamma_start"], marker="o", label="gamma start")
    ax.plot(frame["epoch"], frame["gamma_end"], marker="s", label="gamma end")
    ax.set(xlabel="Epoch", ylabel="gamma", title="R7 Residual Transfer Gamma")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "gamma_vs_epoch.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in (
        ("residual_ratio_mean", "Mean"),
        ("residual_ratio_median", "Median"),
        ("residual_ratio_p90", "P90"),
    ):
        ax.plot(frame["epoch"], frame[column], marker="o", label=label)
    ax.set(xlabel="Epoch", ylabel="||gamma*T(F3)|| / ||F4||", title="R7 Residual Strength")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "residual_strength_vs_epoch.png", dpi=180)
    plt.close(fig)


def write_r7_reports(run_dir, status, equivalence, preflight,
                     branch_summary, ablation, r7_vs_r5, bootstrap):
    parameter_groups = read_json(run_dir / "optimizer_parameter_groups.json")
    replay_summary = read_json(run_dir / "validation_replay_and_branch_summary.json")
    r5_replay = replay_summary["R5_saved_prediction_replay"]
    r7_replay_error = replay_summary["R7_fused_max_probability_abs_error_vs_saved_val_best"]
    size_metrics = read_json(run_dir / "r7_model_size_metrics.json")
    inference_cost = read_json(run_dir / "r7_inference_cost.json")
    package_replay = inference_cost["standalone_package_replay"]
    latency = inference_cost["latency"]
    r5_history = pd.read_csv(R5_DIR / "train_log.csv")
    r7_history = pd.read_csv(run_dir / "train_log.csv")
    r5_best_history = r5_history[r5_history["epoch"] <= int(read_json(R5_DIR / "training_status.json")["best_epoch"])]
    r7_best_history = r7_history[r7_history["epoch"] <= int(status["best_epoch"])]
    r5_best_throughput = float(r5_best_history["images_per_s"].mean())
    r7_best_throughput = float(r7_best_history["images_per_s"].mean())
    r5_best_updates = int(r5_best_history["optimizer_updates"].sum())
    r7_best_updates = int(r7_best_history["optimizer_updates"].sum())
    r5_peak_reserved = float(r5_history["peak_gpu_reserved_gib"].max())
    r7_peak_reserved = float(r7_history["peak_gpu_reserved_gib"].max())
    branch_history = pd.read_csv(run_dir / "r7_branch_diagnostics.csv")
    fixed_history = branch_history[branch_history["scope"] == "fixed_validation_batch"].sort_values("epoch")
    gamma_final = float(fixed_history.iloc[-1]["gamma_end"])
    macro = bootstrap["comparisons"]["R7_Fused_minus_R5"]
    direct = bootstrap["comparisons"]["R7_Fused_minus_R7_gamma0"]
    gamma0 = bootstrap["comparisons"]["R7_gamma0_minus_R5"]
    gamma = float(branch_summary["gamma"])
    lines = [
        "# R7 Residual Transfer Final Comparison", "",
        f"- Macro-AUROC: R5 `{bootstrap['models']['R5']['macro_auroc']:.6f}`; R7 Fused `{bootstrap['models']['R7_Fused']['macro_auroc']:.6f}`; R7 gamma=0 `{bootstrap['models']['R7_gamma0']['macro_auroc']:.6f}`.",
        f"- Macro-AUPRC: R5 `{bootstrap['models']['R5']['macro_auprc']:.6f}`; R7 Fused `{bootstrap['models']['R7_Fused']['macro_auprc']:.6f}`; R7 gamma=0 `{bootstrap['models']['R7_gamma0']['macro_auprc']:.6f}`.",
        f"- R7 Fused−R5 Macro-AUROC: `{macro['delta_macro_auroc']:+.6f}` 95% CI `{macro['delta_macro_auroc_ci95']}`.",
        f"- R7 Fused−R7 gamma=0 Macro-AUROC: `{direct['delta_macro_auroc']:+.6f}` 95% CI `{direct['delta_macro_auroc_ci95']}`.",
        f"- R7 Fused−R5 Macro-AUPRC: `{macro['delta_macro_auprc']:+.6f}` 95% CI `{macro['delta_macro_auprc_ci95']}`.",
        f"- R7 gamma=0−R5 Macro-AUROC: `{gamma0['delta_macro_auroc']:+.6f}` 95% CI `{gamma0['delta_macro_auroc_ci95']}`.",
        f"- R7 Best Epoch `{status['best_epoch']}` of {status['completed_epochs']}; stop `{status['stop_reason']}`; Best checkpoint gamma `{gamma:.8g}`, final-stop gamma `{gamma_final:.8g}`.",
        f"- Best Validation replay max probability error vs saved R7 prediction: `{r7_replay_error:.3g}`.",
        f"- R5 saved-prediction replay: max `|Δp|={r5_replay['max_probability_abs_error']:.6f}`; saved/replayed Macro-AUROC `{r5_replay['saved_macro_auroc']:.6f}` / `{r5_replay['replayed_macro_auroc']:.6f}`; max per-class AUROC delta `{r5_replay['max_abs_per_class_auroc_delta']:.6f}`. Same Study keys and labels matched; paired comparisons use fresh R5 Best inference from the same Validation loader as R7. The score-level discrepancy's cause remains unresolved.",
        "", "## Per-disease results", "",
        "| Disease | Pos/Neg | R5 AUROC | R7 Fused AUROC | R7 gamma=0 AUROC | R7−R5 AUROC Δ (95% CI) | Fused−gamma0 AUROC Δ (95% CI) | R5 AUPRC | R7 Fused AUPRC | R7 gamma=0 AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r7_vs_r5.itertuples(index=False):
        direct_row = bootstrap["comparisons"]["R7_Fused_minus_R7_gamma0"]["per_class"][row.Disease]
        r7r5_row = bootstrap["comparisons"]["R7_Fused_minus_R5"]["per_class"][row.Disease]
        lines.append(
            f"| {row.Disease} | {row.Positive_Studies}/{row.Negative_Studies} | "
            f"{row.R5_AUROC:.4f} | {row.R7_Fused_AUROC:.4f} | "
            f"{bootstrap['models']['R7_gamma0']['per_class'][row.Disease]['auroc']:.4f} | "
            f"{r7r5_row['delta_auroc']:+.4f} [{r7r5_row['delta_auroc_ci95'][0]:+.4f}, {r7r5_row['delta_auroc_ci95'][1]:+.4f}] | "
            f"{direct_row['delta_auroc']:+.4f} [{direct_row['delta_auroc_ci95'][0]:+.4f}, {direct_row['delta_auroc_ci95'][1]:+.4f}] | "
            f"{row.R5_AUPRC:.4f} | {row.R7_Fused_AUPRC:.4f} | "
            f"{bootstrap['models']['R7_gamma0']['per_class'][row.Disease]['auprc']:.4f} |")
    lines += [
        "", "## Residual branch diagnostics", "",
        f"- Best-epoch gamma: `{gamma:.8g}`.",
        f"- Full Validation image residual ratio mean / median / P90 / max: `{branch_summary['residual_ratio_mean']:.6f}` / `{branch_summary['residual_ratio_median']:.6f}` / `{branch_summary['residual_ratio_p90']:.6f}` / `{branch_summary['residual_ratio_max']:.6f}`.",
        f"- Stage4-residual cosine mean / median: `{branch_summary['residual_stage4_cosine_mean']:.6f}` / `{branch_summary['residual_stage4_cosine_median']:.6f}`.",
        f"- Disposable two-step preflight: initial gamma gradient `{preflight['gamma_first_step_gradient_abs']:.6g}`; gamma after first step `{preflight['gamma_after_first_step']:.8g}`; projection gradient at zero gamma `{preflight['projection_first_step_gradient_norm_expected_zero']:.6g}`; projection gradient after gamma moved `{preflight['projection_second_step_gradient_norm_nonzero']:.6g}`.",
        "- `r7_branch_diagnostics.csv` records gamma, per-epoch gamma-gradient statistics, fixed Validation-batch residual ratios/cosines, and the Best checkpoint full-Validation residual summary.",
        "", "## Training and resource use", "",
        f"- R7 trainable/frozen parameters: `{preflight['trainable_parameter_count']:,}` / `{parameter_groups['frozen_parameters']:,}`.",
        f"- Added branch parameters: projection `{parameter_groups['r7_projection_parameters']:,}`, gamma `{parameter_groups['r7_gamma_parameters']}`.",
        f"- Best checkpoint bytes: `{(run_dir / 'best.pth').stat().st_size:,}`.",
        f"- Standalone inference archive: `{size_metrics['archive_bytes']:,}` bytes (strict < `{size_metrics['size_limit_bytes']:,}` passed); model-only safetensors `{size_metrics['inference_weights_bytes']:,}` bytes.",
        f"- Per-Study package Validation inference: Macro-AUROC `{package_replay['macro_auroc']:.6f}`, Macro-AUPRC `{package_replay['macro_auprc']:.6f}`; mean forward latency `{latency['mean_model_forward_ms_per_study']:.2f}` ms/Study; end-to-end `{latency['mean_end_to_end_ms_per_study']:.2f}` ms/Study (`{latency['images_per_second_end_to_end']:.2f}` images/s, including NAS reads and preprocessing).",
        f"- Standalone per-Study versus batched Validation replay max `|Δp|={package_replay['max_probability_abs_error_vs_batched_validation']:.6f}`, Macro-AUROC `{package_replay['macro_auroc']:.6f}` vs `{package_replay['batched_macro_auroc']:.6f}`; batch-shape precision is a plausible contributor, not a proven explanation for the R5 historical replay difference.",
        f"- Formal Train throughput through each Best Epoch: R5 `{r5_best_throughput:.2f}` vs R7 `{r7_best_throughput:.2f}` images/s (`{(r7_best_throughput / r5_best_throughput - 1.0) * 100.0:+.1f}%`); cumulative optimizer updates `{r5_best_updates}` vs `{r7_best_updates}`. Peak reserved training memory: R5 `{r5_peak_reserved:.2f}` vs R7 `{r7_peak_reserved:.2f}` GiB.",
        "- Train/Validation plots compare R5, R6, and R7; the R7 early-stop rule and R5 runtime were reused.",
        "", "## Interpretation", "",
        "`R7 Fused−R7 gamma=0` compares two forward paths using the same trained R7 weights and isolates the inference-time direct branch contribution. `R7 gamma=0−R5` compares separately trained models and cannot be attributed causally to the branch because optimization trajectories differ.",
        "The branch can provide a disease-specific effect without improving Macro-AUROC. Validation was used for Best-checkpoint selection and contains prior model comparisons; all paired intervals are exploratory and exposed to multiple-comparison risk. These measurements do not prove that Stage 4 irreversibly loses information.",
        "",
    ]
    (run_dir / "r7_final_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    differences = {
        "R7_Fused_minus_R5": macro,
        "R7_Fused_minus_R7_gamma0": direct,
        "R7_gamma0_minus_R5": gamma0,
    }
    stage3_pneumonia = bootstrap["comparisons"]["R7_Fused_minus_R5"]["per_class"].get("Pneumonia")
    r5 = bootstrap["models"]["R5"]["macro_auroc"]
    r7 = bootstrap["models"]["R7_Fused"]["macro_auroc"]
    r7_zero = bootstrap["models"]["R7_gamma0"]["macro_auroc"]
    final_lines = [
        "# R7 Final Report: Stage 3 to Stage 4 Residual Feature Transfer", "",
        "## Executive conclusion", "",
        f"R7 Best Epoch {status['best_epoch']} achieved Macro-AUROC `{r7:.6f}` versus fresh R5 Best inference `{r5:.6f}`; R7−R5 delta `{macro['delta_macro_auroc']:+.6f}` with Subject-cluster 95% CI `{macro['delta_macro_auroc_ci95']}`. The interval is below zero, indicating a small paired decrease in this Validation run; repeated Validation use and the unresolved R5 score replay discrepancy limit the strength of that inference.",
        f"Within the same R7 Best weights, disabling gamma yields Macro-AUROC `{r7_zero:.6f}`; Fused−gamma=0 delta `{direct['delta_macro_auroc']:+.6f}` with CI `{direct['delta_macro_auroc_ci95']}`. This directly measures the inference contribution of the branch in the trained R7 model.",
        "", "## Required questions", "",
        f"1. R7 Best Macro-AUROC: **{r7:.6f}** at Epoch {status['best_epoch']}; Macro-AUPRC: **{bootstrap['models']['R7_Fused']['macro_auprc']:.6f}**.",
        f"2. Relative to the historically reported R5 Best Macro-AUROC `{r5_replay['saved_macro_auroc']:.6f}`, R7 is `{r7 - r5_replay['saved_macro_auroc']:+.6f}`. Paired comparisons use fresh same-loader R5 Best inference `{r5:.6f}`, giving a delta of `{r7 - r5:+.6f}` with CI `{macro['delta_macro_auroc_ci95']}` entirely below zero; this run provides no evidence of an AUROC gain and points to a small decrease.",
        "3–5. Per-disease AUROC/AUPRC, sample sizes and paired intervals are in `r7_inference_ablation.csv` and `r7_vs_r5_per_class.csv`. Pneumonia/Consolidation/Lung Opacity are called out in those tables; the result should be read by class and with the CI.",
        "6. Potential degradation in historically strong classes is also shown per disease in the same table; no class-wise significance claim is made without considering the ten-class multiple-comparison scope.",
        f"7. Best-epoch gamma is `{gamma:.8g}`; final-stop gamma is `{gamma_final:.8g}`. Its per-epoch trajectory is in `gamma_vs_epoch.png` and `r7_branch_diagnostics.csv`.",
        f"8. On all Validation images at Best, residual ratio `||gamma*T(F3)|| / ||F4||` mean/median/P90/max is `{branch_summary['residual_ratio_mean']:.6f}` / `{branch_summary['residual_ratio_median']:.6f}` / `{branch_summary['residual_ratio_p90']:.6f}` / `{branch_summary['residual_ratio_max']:.6f}`; cosine statistics are in the branch diagnostics CSV.",
        f"9. R7 Fused−R7 gamma=0 Macro-AUROC delta is `{direct['delta_macro_auroc']:+.6f}`, 95% CI `{direct['delta_macro_auroc_ci95']}`; this is the inference-time direct branch effect on identical R7 weights.",
        f"10. R7−R5 Macro-AUROC paired 95% CI `{macro['delta_macro_auroc_ci95']}` is entirely negative for the fresh same-loader predictions; this suggests a small decrease in this run, subject to the R5 replay discrepancy, repeated Validation use and multiple-comparison limits. For Pneumonia, the point change is `{stage3_pneumonia['delta_auroc']:+.6f}` with CI `{stage3_pneumonia['delta_auroc_ci95']}`.",
        f"11. R7 adds `{int(parameter_groups['r7_projection_parameters'] + parameter_groups['r7_gamma_parameters']):,}` trainable parameters. Best checkpoint size is `{(run_dir / 'best.pth').stat().st_size:,}` bytes; inference ZIP size is `{size_metrics['archive_bytes']:,}` bytes (limit passed). Standalone per-Study mean model forward latency is `{latency['mean_model_forward_ms_per_study']:.2f}` ms; peak inference allocation is `{latency['peak_gpu_allocated_gib']:.3f}` GiB. Peak reserved training memory is R5 `{r5_peak_reserved:.2f}` / R7 `{r7_peak_reserved:.2f}` GiB; mean Train throughput through Best is R5 `{r5_best_throughput:.2f}` / R7 `{r7_best_throughput:.2f}` images/s, with Best-epoch cumulative updates `{r5_best_updates}` / `{r7_best_updates}`.",
        f"12. R7 Train BCE fell to `{float(pd.read_csv(run_dir / 'train_log.csv').iloc[-1]['train_bce']):.6f}` while Validation BCE rose to `{float(pd.read_csv(run_dir / 'train_log.csv').iloc[-1]['valid_loss']):.6f}` at stop; Best was Epoch {status['best_epoch']}, consistent with late-epoch overfitting under the observed curves.",
        "13. R7 did not improve over fresh R5 inference; its shared residual route produced a small negative Macro-AUROC difference, and disabling gamma on the same R7 weights changes Macro-AUROC by a near-zero amount with a CI crossing zero. This does not prove the route is generally harmful or that Stage 4 loses information.",
        "14. The current result supports stopping the shared residual route. Any disease-selective transfer should be a separately pre-registered future experiment with an independent evaluation set. No attention or disease gate was added in R7.",
        "", "## Limitations", "",
        f"R5's fresh Validation inference differs from the saved R5 probability table by max `|Δp|={r5_replay['max_probability_abs_error']:.6f}`; its Macro-AUROC differs by `{r5_replay['replayed_macro_auroc'] - r5_replay['saved_macro_auroc']:+.6f}` and max per-class AUROC by `{r5_replay['max_abs_per_class_auroc_delta']:.6f}`. Model weights, official Study keys and labels match, but the score-level difference's cause remains unresolved. Standalone per-Study R7 inference also differs slightly from batched Validation inference (`max |Δp|={package_replay['max_probability_abs_error_vs_batched_validation']:.6f}`); this is consistent with batch-shape precision variation but does not prove the R5 cause. Paired comparisons use fresh R5 and R7 predictions generated together on the same Validation loader.",
        "All model selection and inference comparisons use the previously reused Validation set. Subject-cluster bootstrap is paired and uses common draws, but repeated Validation use, multiple diseases, one training seed and Best-epoch selection limit inferential strength. `R7 gamma=0−R5` cannot isolate the causal training-time effect of the new branch. The analysis did not read Test.",
        "",
    ]
    (run_dir / "r7_final_report.md").write_text("\n".join(final_lines), encoding="utf-8")


def optimizer_parameter_groups(model, optimizer=None):
    projection_count = sum(parameter.numel() for parameter in model.transfer.projection.parameters())
    gamma_count = model.transfer.gamma.numel()
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "frozen_parameters": sum(parameter.numel() for parameter in model.parameters()
                                  if not parameter.requires_grad),
        "r7_projection_parameters": projection_count,
        "r7_gamma_parameters": gamma_count,
    }


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("R7 requires the R5 CUDA BF16 training runtime")
    r5_payload = read_json(R5_DIR / "config.json")
    r5_status = read_json(R5_DIR / "training_status.json")
    if not r5_status.get("complete") or int(r5_status["best_epoch"]) != 6:
        raise RuntimeError("R7 requires the completed R5 Best checkpoint and run record")
    if r5_payload["initialization_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R5 does not use the pinned original DINOv3 general-vision checkpoint")
    r5_verify = read_json(R5_DIR / "model_verification.json")
    expected_head_sha = r5_verify["weight_verification"]["global_classifier_init_sha256"]
    if expected_head_sha != "40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9":
        raise RuntimeError(f"Unexpected R5 original Global Head hash: {expected_head_sha}")

    runtime_payload = read_json(R5_DIR / "selected_runtime_config.json")
    runtime = copy.deepcopy(runtime_payload["selected_runtime"])
    feature = copy.deepcopy(r5_payload["feature_config"])
    config = copy.deepcopy(r5_payload["experiment_config"])
    freeze = copy.deepcopy(r5_payload["freeze_config"])
    if int(config["seed"]) != SEED or int(config["training"]["epochs"]) != 20:
        raise RuntimeError("R7 seed/epoch plan must match R5")
    if int(config["training"]["warmup_epochs"]) != 2:
        raise RuntimeError("R7 warmup must match R5")
    for key in ("peak_lr_backbone", "peak_lr_head", "weight_decay", "gradient_clip_norm"):
        if key not in config["training"]:
            raise RuntimeError(f"R5 training configuration is missing {key}")
    if runtime != read_json(R5_DIR / "selected_runtime_config.json")["selected_runtime"]:
        raise RuntimeError("R7 selected runtime must be an exact copy of the completed R5 runtime")

    config["model"]["architecture"] = "DINOv3ConvNeXtTinyR7Stage3Stage4ResidualTransfer"
    config["model"]["study_fusion"] = "mean_raw_image_logits_before_sigmoid"
    config["model"]["r7_transfer"] = {
        "placement": "Stage3 output to Stage4 output before original Norm/GAP/Global Head",
        "operation": "AvgPool2d(2,2) -> Conv2d(384,768,1,bias=False)",
        "fusion": "F4_fused = F4 + gamma * T(F3)",
        "gamma_initial": 0.0,
        "new_attention_or_local_head": False,
    }
    config["outputs"]["root"] = str(HERE)
    freeze["experiment"] = "DINOv3_Feature_Optimization_R7_ResidualTransfer"
    freeze["outputs_root"] = str(HERE.parent)
    freeze["profiles"] = ["p2"]
    freeze["training"]["max_epochs"] = 20
    freeze["submission"]["selected_model"] = GROUP
    config["freeze"] = copy.deepcopy(freeze)
    feature["outputs"]["root"] = str(HERE.parent)

    r7_runtime_payload = copy.deepcopy(runtime_payload)
    r7_runtime_payload["group"] = GROUP
    r7_runtime_payload["source_group"] = "R5_Global_Only_512"
    r7_runtime_payload["selection_rule"] = "Exact R5 selected runtime reuse; no new runtime, throughput, or LR search."
    r7_runtime_path = HERE / "selected_runtime_config.json"
    if not r7_runtime_path.is_file():
        save_json_atomic(r7_runtime_path, r7_runtime_payload)
    else:
        saved_runtime = read_json(r7_runtime_path)
        if (saved_runtime.get("group") != GROUP
                or saved_runtime.get("selected_runtime") != runtime):
            raise RuntimeError("Existing R7 selected runtime differs from the controlled R5 runtime")

    data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    train_data, valid_data, data_config = build_masked_datasets(
        data_path, SEED, IMAGE_SIZE)
    if (len(train_data), int(sum(train_data.image_counts)), len(valid_data),
            int(sum(valid_data.image_counts))) != (18513, 30344, 859, 1422):
        raise RuntimeError("R7 Train/Validation counts differ from the R5 official manifest")
    if (data_config.get("label_semantics") != "closed_world"
            or data_config.get("class_ids") != list(range(1, 11))
            or data_config.get("class_names") != LABEL_NAMES):
        raise RuntimeError("R7 data label semantics/order differ from R5")
    if set(train_data.study_keys) & set(valid_data.study_keys):
        raise RuntimeError("R7 Train/Validation Study keys overlap")
    train_subjects = {str(subject) for subject, _ in train_data.study_keys}
    valid_subjects = {str(subject) for subject, _ in valid_data.study_keys}
    if train_subjects & valid_subjects:
        raise RuntimeError("R7 Train/Validation Subject leakage detected")

    init_test_path = HERE / "initialization_equivalence_test.json"
    if init_test_path.is_file():
        equivalence = read_json(init_test_path)
        if (equivalence.get("passed") is not True
                or equivalence.get("pretrained_weight_sha256") != EXPECTED_WEIGHT_SHA256
                or equivalence.get("global_head_init_sha256") != expected_head_sha):
            raise RuntimeError("Existing R7 initialization equivalence record is invalid")
    else:
        print("Running R7 CPU FP32 and GPU BF16 zero-gamma initialization equivalence tests.",
              flush=True)
        equivalence = test_initialization_equivalence(
            valid_data, runtime, expected_head_sha)
        save_json_atomic(init_test_path, equivalence)

    gradient_recorder = ScalarGradientRecorder()
    builder = make_r7_builder(runtime, expected_head_sha, gradient_recorder)
    preflight_path = HERE / "r7_preflight.json"
    if preflight_path.is_file():
        preflight = read_json(preflight_path)
        if (preflight.get("passed") is not True
                or preflight.get("preflight_version") != PRELIGHT_VERSION
                or preflight.get("runtime") != runtime):
            raise RuntimeError("Existing R7 preflight does not match this architecture/runtime")
    else:
        print("Running R7 Stage-size, gamma/projection-gradient, optimizer and 8GB backward-step preflight.",
              flush=True)
        preflight = r7_actual_preflight(
            train_data, config, freeze, runtime, builder, r7_optimizer)
        save_json_atomic(preflight_path, preflight)

    seed_everything(SEED)
    model_for_groups, _, modules, frozen, names = builder("p2", torch.device("cuda"), runtime, SEED)
    optimizer_for_groups = r7_optimizer(model_for_groups, {
        **config, "freeze": freeze,
    })
    group_report = r7_parameter_groups_report(model_for_groups, optimizer_for_groups, runtime, {
        **config, "freeze": freeze,
    })
    group_report.update(optimizer_parameter_groups(model_for_groups))
    group_report["trainable_modules"] = [name for name, _ in modules]
    group_report["frozen_modules"] = [name for name, _ in frozen]
    group_report["trainable_parameter_name_count"] = len(names)
    save_json_atomic(HERE / "optimizer_parameter_groups.json", group_report)
    del optimizer_for_groups, model_for_groups
    gc.collect()
    torch.cuda.empty_cache()

    fixed_loader, fixed_sampler, _ = make_loader(
        valid_data, image_budget=int(runtime["image_budget"]), shuffle=False,
        seed=SEED, num_workers=0, pin_memory=bool(runtime["pin_memory"]),
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    fixed_sampler.set_epoch(0)
    try:
        fixed_batch = next(iter(fixed_loader))
        fixed_images = fixed_batch["images"].detach().cpu()
    finally:
        shutdown_loader(fixed_loader)
    branch_diagnostics = R7EpochDiagnostics(
        HERE / "r7_branch_diagnostics.csv", fixed_images, runtime, gradient_recorder)

    print("Starting R7 training from the original pinned DINOv3 weights; no R5/R6 fine-tuned weights are loaded.",
          flush=True)
    result = run_custom_group(
        feature, config, freeze, "R7", GROUP, IMAGE_SIZE,
        {"repository": r5_payload["initialization_repo"],
         "revision": r5_payload["initialization_revision"],
         "weights_sha256": EXPECTED_WEIGHT_SHA256},
        model_builder=builder,
        optimizer_builder=r7_optimizer,
        batch_forward=lambda model, images, _batch: model(images),
        validation_forward=lambda model, images, _batch: model(images),
        collate_fn=collate_masked_studies,
        audit_forward=r7_audit_forward,
        backbone_name="DINOv3 ConvNeXt-Tiny R7 Stage3→Stage4 Residual Transfer",
        expected_stage3_channels=384,
        dataset_builder=build_masked_datasets,
        run_context={
            "group": GROUP,
            "architecture": "dinov3_tiny_global_only_stage3_stage4_residual_transfer",
            "image_size": IMAGE_SIZE,
            "head_type": "original_global_gap_linear_10",
            "pretrained_initialization_sha256": EXPECTED_WEIGHT_SHA256,
            "global_classifier_init_sha256": expected_head_sha,
            "runtime_source": "R5_Global_Only_512/selected_runtime_config.json",
            "freeze_profile": "R5_P2_Stage3_Stage4",
            "added_branch": "AvgPool2d(2,2)->Conv2d(384,768,1,bias=False)->gamma*projection",
            "gamma_init": 0.0,
            "test_access": False,
        },
        expected_weight_sha256=EXPECTED_WEIGHT_SHA256,
        epoch_callback=branch_diagnostics)
    status = read_json(HERE / "training_status.json")
    if (not status.get("complete")
            or status.get("stop_reason") not in {"three_condition_early_stop", "max_epochs_reached"}):
        raise RuntimeError(f"R7 training did not complete by the planned stop rule: {status}")
    if not (HERE / "best.pth").is_file() or not (HERE / "last.pth").is_file():
        raise RuntimeError("R7 Best/Last checkpoints are missing after training")

    config_path = HERE / "config.json"
    config_payload = read_json(config_path)
    config_payload["profile"] = "R7: R5 P2 trainable range + Stage3→Stage4 residual transfer"
    config_payload["model_variant"] = "Global-only GAP + lightweight shared Stage3 residual transfer"
    config_payload["trainable_policy"] = (
        "R5 trainable Stage3/Stage4/final Norm/Global Head plus transfer projection and gamma; "
        "Stem/Stage1/Stage2 frozen")
    config_payload["initialization_equivalence_test"] = equivalence
    config_payload["r7_preflight"] = preflight
    config_payload["optimizer_parameter_groups"] = group_report
    config_payload["test_access"] = False
    config_payload["test_split_read"] = False
    save_json_atomic(config_path, config_payload)

    status["profile"] = "r7_residual_transfer"
    status["test_data_accessed"] = False
    save_json_atomic(HERE / "training_status.json", status)
    result_path = HERE / "training_result.json"
    if result_path.is_file():
        result_payload = read_json(result_path)
        result_payload["profile"] = "r7_residual_transfer"
        result_payload["test_data_accessed"] = False
        save_json_atomic(result_path, result_payload)

    architecture_lines = [
        "# R7 Architecture and Initialization Report", "",
        "- Backbone: official general-vision DINOv3 ConvNeXt-Tiny; same pinned revision and SHA-256 as R5.",
        f"- Original pretrained backbone SHA-256: `{EXPECTED_WEIGHT_SHA256}`.",
        f"- Global Head independently initialized with seed {SEED}; initial hash matches R5: `{expected_head_sha}`.",
        "- Original R5 Global Head and backbone parameter names are unchanged; the added state namespace is `transfer.*`.",
        "- Main path: Stage3 → original Downsample → Stage4.",
        "- Transfer path: Stage3 `[B,384,32,32]` → AvgPool2d(2,2) → Conv2d(384,768,1,bias=False) → gamma scaling.",
        "- Fusion: `F4_fused = F4 + gamma*T(F3)` before the original Stage4 pooled-token normalization and Global Linear(768,10) head.",
        "- No added activation, attention, local head, auxiliary loss, multiscale module, or pretraining phase.",
        f"- Gamma initialization equivalence CPU FP32: `{equivalence['cpu_fp32']}`.",
        f"- Gamma initialization equivalence CUDA BF16: `{equivalence['gpu_bf16']}`.",
        f"- Added parameters: Projection {group_report['r7_projection_parameters']:,}; gamma {group_report['r7_gamma_parameters']}.",
        f"- Total/trainable/frozen parameters: {group_report['total_parameters']:,}/{group_report['trainable_parameters']:,}/{group_report['frozen_parameters']:,}.",
        f"- R5 selected runtime reused exactly: image budget {runtime['image_budget']}, accumulation {runtime['grad_accumulation']}, channels-last {runtime['channels_last']}, checkpointing {runtime['gradient_checkpointing']}; no new benchmark or LR search.",
        "- Study Logit Mean, Study BCE, ResizePad, augmentation, normalization, split and three-condition stop rule follow the R5 implementation.",
        f"- Actual two-step runtime preflight passed at {preflight['peak_reserved_bytes'] / (1024**3):.2f} GiB peak reserved memory; initial gamma and delayed projection gradients were verified.",
        "- Only official Train/Validation were accessed; Test was not read.", "",
    ]
    (HERE / "architecture_report.md").write_text("\n".join(architecture_lines), encoding="utf-8")
    training_report = HERE / "training_report.md"
    if training_report.is_file():
        original = training_report.read_text(encoding="utf-8")
        addendum = (
            "\n## R7 residual branch\n\n"
            "R7 retains the R5 Stage3/Stage4 freeze range and original Global Head, adding only "
            "AvgPool2d(2,2) + Conv2d(384,768,1,bias=False) with scalar gamma initialized at zero. "
            "Per-epoch branch activity is in `r7_branch_diagnostics.csv`.\n"
        )
        if "## R7 residual branch" not in original:
            training_report.write_text(original.rstrip() + "\n" + addendum, encoding="utf-8")

    r5_status = read_json(R5_DIR / "training_status.json")
    ablation, r7_vs_r5, bootstrap, replay = run_best_inference_ablation(
        valid_data, r5_status, status, runtime, HERE)
    draw_training_curves(HERE)
    draw_branch_curves(HERE)
    branch_rows = pd.read_csv(HERE / "r7_branch_diagnostics.csv")
    best_diag_rows = branch_rows[(branch_rows["epoch"].astype(int) == int(status["best_epoch"]))
                                 & (branch_rows["scope"] == "fixed_validation_batch")]
    if len(best_diag_rows) != 1:
        raise RuntimeError("R7 Best Epoch branch diagnostics are missing or duplicated")
    best_diag = best_diag_rows.iloc[0].to_dict()
    best_diag.update(read_json(HERE / "validation_replay_and_branch_summary.json")[
        "branch_diagnostics_best_validation"])
    write_r7_reports(HERE, status, equivalence, preflight, best_diag,
                     ablation, r7_vs_r5, bootstrap)
    print(json.dumps({
        "status": "complete", "output_dir": str(HERE),
        "best_epoch": status["best_epoch"],
        "best_macro_auroc": status["best_macro_auroc"],
        "best_checkpoint_bytes": (HERE / "best.pth").stat().st_size,
        "gamma_best": float(best_diag["gamma"]),
        "R7_fused_minus_R5_macro_auc_ci95": bootstrap["comparisons"]["R7_Fused_minus_R5"]["delta_macro_auroc_ci95"],
        "R7_fused_minus_gamma0_macro_auc_ci95": bootstrap["comparisons"]["R7_Fused_minus_R7_gamma0"]["delta_macro_auroc_ci95"],
        "test_data_accessed": False,
        "result": result,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
