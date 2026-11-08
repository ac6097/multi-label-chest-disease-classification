"""R8/R9/R10 controlled experiment orchestration.

The three runs reuse, unchanged, the R5 training machinery:

    run_custom_group -> train_profile  (ConvNeXtTiny/DINOv3-ConvNeXtTiny/dino_*.py)

so the data pipeline, Study Logit Mean, Study BCE, optimizer/schedule, BF16,
channels-last, whole-Study sampler, validation and the three-condition early
stopping are identical to R5/R7.  Only the classifier topology differs, and
every run starts from the pinned original DINOv3 general-vision weights.

This module owns: R5 contract loading, run configuration, freeze layout,
optimizer parameter groups, model audit, initialization equivalence, delayed
gradient checks, per-epoch branch diagnostics, architecture artifacts and the
top-level orchestration for one experiment.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dino_analysis import LABEL_NAMES  # noqa: E402
from dino_experiment import load_pretrained_backbone, save_json_atomic  # noqa: E402
from dino_feature_models import (  # noqa: E402
    build_masked_datasets,
    collate_masked_studies,
)
from dino_feature_optimization import run_custom_group  # noqa: E402
from dino_freeze_experiment import (  # noqa: E402
    head_hash,
    set_profile_training_mode,
)
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from dino_model import DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from hybrid.common import seed_everything, write_csv  # noqa: E402
from stage_fusion_models import (  # noqa: E402
    ALPHA_INIT,
    ARCHITECTURES,
    D_MODEL,
    FEATURE_DIM,
    NUM_HEADS,
    THETA_INIT,
    build_model,
    new_branch_parameter_names,
)

WORK_DIR = HERE
OUTPUT_ROOT = HERE / "outputs" / "experiments" / "DINOv3_Feature_Optimization"
R5_DIR = OUTPUT_ROOT / "R5_Global_Only_512"
R7_DIR = OUTPUT_ROOT / "R7_ResidualTransfer_512"
EXPERIMENT_NAME = "DINOv3_Feature_Optimization"
PROFILE = "p2"
SEED = 42
IMAGE_SIZE = 512
EXPECTED_WEIGHT_SHA256 = (
    "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2")
EXPECTED_HEAD_SHA = (
    "40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9")
R5_TOTAL_PARAMETERS = 27_827_818
R5_TRAINABLE_PARAMETERS = 26_592_778
R5_FROZEN_PARAMETERS = 1_235_040

GROUP_NAMES = {
    "R8": "R8_DiseaseSpecific_StageLogitFusion_512",
    "R9": "R9_DiseaseSpecific_QueryAttention_512",
    "R10": "R10_Stage4Guided_CrossStageAttention_512",
}
EXPECTED_NEW_PARAMETERS = {"R8": 4_628, "R9": 101_908, "R10": 201_876}

# New-branch module namespaces, in the order they are created on top of the
# original `backbone.*` + `classifier.*` state dict.
NEW_BRANCH_NAMES = {
    "R8": ("stage3_branch", "fusion_alpha"),
    "R9": ("query_attention", "fusion_alpha"),
    "R10": ("query_attention", "fusion_alpha"),
}
BRANCH_DESCRIPTIONS = {
    "R8": ("LayerNorm(384) + Linear(384,10) on Stage3 GAP, zero-initialized head, "
           "z_k = z4_k + sigmoid(theta_k) * z3_k"),
    "R9": ("10 learnable disease queries cross-attend 32x32 Stage3 tokens "
           "(d_model=128, 4 heads); per-disease zero-initialized logit head; "
           "z_k = z4_k + sigmoid(theta_k) * z_attn_k"),
    "R10": ("R9 plus q_k(x) = e_k + W_q LayerNorm(g4(x)) with zero-initialized "
            "W_q: Stage4 global semantics condition Stage3 token selection"),
}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    save_json_atomic(Path(path), payload)


def load_r5_contract() -> dict:
    """Load and verify the completed R5 run that all three experiments reuse."""
    config_payload = read_json(R5_DIR / "config.json")
    status = read_json(R5_DIR / "training_status.json")
    verification = read_json(R5_DIR / "model_verification.json")
    runtime_payload = read_json(R5_DIR / "selected_runtime_config.json")
    if not status.get("complete") or int(status["best_epoch"]) != 6:
        raise RuntimeError("R8/R9/R10 require the completed R5 Best checkpoint (epoch 6)")
    if config_payload["initialization_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R5 initialization is not the pinned DINOv3 general-vision weight")
    head_sha = verification["weight_verification"]["global_classifier_init_sha256"]
    if head_sha != EXPECTED_HEAD_SHA:
        raise RuntimeError(f"Unexpected R5 original Global Head hash: {head_sha}")
    runtime = copy.deepcopy(runtime_payload["selected_runtime"])
    if int(runtime["image_budget"]) != 8 or int(runtime["grad_accumulation"]) != 4:
        raise RuntimeError("R5 selected runtime is not the expected 8/4 controlled runtime")
    return {
        "config_payload": config_payload,
        "status": status,
        "model_verification": verification,
        "runtime_payload": runtime_payload,
        "runtime": runtime,
        "head_sha": head_sha,
        "feature": copy.deepcopy(config_payload["feature_config"]),
        "experiment_config": copy.deepcopy(config_payload["experiment_config"]),
        "freeze": copy.deepcopy(config_payload["freeze_config"]),
    }


def build_run_configs(experiment: str, contract: dict) -> tuple[dict, dict, dict]:
    config = contract["experiment_config"]
    freeze = contract["freeze"]
    feature = contract["feature"]
    config = copy.deepcopy(config)
    freeze = copy.deepcopy(freeze)
    feature = copy.deepcopy(feature)
    config["model"]["architecture"] = ARCHITECTURES[experiment]
    config["model"]["stage_fusion"] = {
        "experiment": experiment,
        "main_path": "unchanged R5 Stage4 -> official pooled token norm -> Linear(768,10)",
        "auxiliary_path": BRANCH_DESCRIPTIONS[experiment],
        "alpha_initial": ALPHA_INIT,
        "alpha_parameterization": "sigmoid(theta_k), theta initialized at logit(0.1)",
        "auxiliary_loss": False,
        "study_fusion": "mean_raw_image_logits_before_sigmoid",
    }
    config["outputs"]["root"] = str(OUTPUT_ROOT / GROUP_NAMES[experiment])
    config["outputs"]["export_zip"] = (
        f"submission_export/{experiment}_Inference.zip")
    config["outputs"]["submission_size_limit_bytes"] = 500_000_000
    freeze["experiment"] = f"DINOv3_Feature_Optimization_{experiment}"
    freeze["outputs_root"] = str(OUTPUT_ROOT)
    freeze["profiles"] = [PROFILE]
    freeze["training"]["max_epochs"] = 20
    freeze["submission"]["selected_model"] = GROUP_NAMES[experiment]
    config["freeze"] = copy.deepcopy(freeze)
    feature["outputs"]["root"] = str(OUTPUT_ROOT)
    return feature, config, freeze


def freeze_layout(experiment: str, model: torch.nn.Module):
    """R5 P2 trainable range plus the declared new branch (nothing else)."""
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
    ]
    new_names = NEW_BRANCH_NAMES[experiment]
    for name in new_names:
        if name == "fusion_alpha":
            continue
        trainable_modules.append((name, getattr(model, name)))
    frozen_modules = [
        ("backbone.model.downsample_layers.0", trunk.downsample_layers[0]),
        ("backbone.model.downsample_layers.1", trunk.downsample_layers[1]),
        ("backbone.model.stages.0", trunk.stages[0]),
        ("backbone.model.stages.1", trunk.stages[1]),
    ]
    for _, module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    # theta starts with an exactly zero gradient because the zero-initialized
    # auxiliary head emits zero logits at step 0; the one-step per-module
    # gradient audit in run_custom_group therefore cannot observe it.  It is
    # enabled explicitly here and its delayed gradient is verified separately
    # by run_gradient_check (two/three optimizer steps + isolated backward).
    model.fusion_alpha.theta.requires_grad_(True)
    allowed = (
        "backbone.model.downsample_layers.2.", "backbone.model.stages.2.",
        "backbone.model.downsample_layers.3.", "backbone.model.stages.3.",
        "backbone.model.norm.", "backbone.model.norms.3.", "classifier.",
        *(f"{name}." for name in new_names),
    )
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    unexpected = [name for name in trainable_names
                  if not any(name.startswith(prefix) for prefix in allowed)]
    if unexpected:
        raise RuntimeError(
            f"{experiment} has unexpected trainable parameters: {unexpected[:10]}")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    frozen = total - trainable
    if (total, trainable, frozen) != (
            R5_TOTAL_PARAMETERS + EXPECTED_NEW_PARAMETERS[experiment],
            R5_TRAINABLE_PARAMETERS + EXPECTED_NEW_PARAMETERS[experiment],
            R5_FROZEN_PARAMETERS):
        raise RuntimeError(
            f"{experiment} parameter counts differ from the declared architecture: "
            f"{total}/{trainable}/{frozen}")
    return trainable_modules, frozen_modules, trainable_names


def make_builder(experiment: str, runtime: dict):
    """Return the model builder consumed by run_custom_group/train_profile."""

    def builder(profile, device, active_runtime, seed):
        if profile != PROFILE or int(seed) != SEED:
            raise RuntimeError(
                f"{experiment} must keep the R5 P2 freeze profile and seed {SEED}")
        seed_everything(SEED)
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError(
                f"{experiment} original DINOv3 pretrained weight SHA-256 mismatch")
        model = build_model(experiment, backbone).to(device)
        initial_head_sha = head_hash(model)
        if initial_head_sha != EXPECTED_HEAD_SHA:
            raise RuntimeError(
                f"{experiment} Global Head initialization does not match the seeded "
                f"R5 head: {initial_head_sha}")
        trainable_modules, frozen_modules, trainable_names = freeze_layout(
            experiment, model)
        model.backbone.set_gradient_checkpointing(
            bool(active_runtime["gradient_checkpointing"]))
        if active_runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        verification.update({
            "model_variant": f"DINOv3 ConvNeXt-Tiny {ARCHITECTURES[experiment]}",
            "loaded_finetuned_checkpoint": False,
            "global_classifier_init_sha256": initial_head_sha,
            "added_branch": {
                "experiment": experiment,
                "description": BRANCH_DESCRIPTIONS[experiment],
                "parameter_namespaces": list(NEW_BRANCH_NAMES[experiment]),
                "new_parameters": EXPECTED_NEW_PARAMETERS[experiment],
                "alpha_initial": ALPHA_INIT,
                "theta_initial": THETA_INIT,
                "stage3_channels": 384,
                "stage4_channels": 768,
                "d_model": D_MODEL if experiment in {"R9", "R10"} else None,
                "num_heads": NUM_HEADS if experiment in {"R9", "R10"} else None,
                "stage4_query_conditioning": experiment == "R10",
            },
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "training_runtime": {
                key: active_runtime[key] for key in (
                    "image_budget", "grad_accumulation", "gradient_checkpointing",
                    "channels_last", "num_workers", "pin_memory", "prefetch_factor",
                    "effective_images_per_update", "effective_studies_per_update",
                ) if key in active_runtime
            },
        })
        return model, verification, trainable_modules, frozen_modules, trainable_names

    return builder


def make_optimizer(experiment: str):
    """R5 optimizer parameter groups plus the new branch head / alpha groups."""
    new_names = NEW_BRANCH_NAMES[experiment]

    def builder(model, config):
        training = config["training"]
        decay = float(training["weight_decay"])
        backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
        head_parameters = [p for name, p in model.named_parameters()
                           if p.requires_grad and name.startswith("classifier.")]
        branch_parameters = []
        alpha_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "fusion_alpha.theta":
                alpha_parameters.append(parameter)
            elif any(name.startswith(f"{namespace}.") for namespace in new_names):
                branch_parameters.append(parameter)
        groups = [
            {"params": backbone_parameters,
             "lr": float(training["peak_lr_backbone"]),
             "weight_decay": decay, "name": "backbone"},
            {"params": head_parameters + branch_parameters,
             "lr": float(training["peak_lr_head"]),
             "weight_decay": decay, "name": "head"},
            {"params": alpha_parameters,
             "lr": float(training["peak_lr_head"]),
             "weight_decay": 0.0, "name": "head"},
        ]
        if any(not group["params"] for group in groups):
            raise RuntimeError(f"{experiment} optimizer has an empty parameter group")
        optimizer = torch.optim.AdamW(
            groups, fused=bool(config["freeze"]["training"]["fused_adamw"]))
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        observed = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise RuntimeError(
                f"{experiment} optimizer groups do not contain every trainable "
                "parameter exactly once")
        return optimizer

    return builder


def make_audit_forward(experiment: str):
    """One-batch architecture audit used by run_custom_group's model verification."""

    def audit_forward(model, images, _batch):
        components = model.forward_components(images)
        logits = components["logits"]
        if tuple(logits.shape) != (int(images.shape[0]), NUM_LABELS):
            raise RuntimeError(f"{experiment} logits shape is {tuple(logits.shape)}")
        alpha = components["alpha"].detach().float().cpu()
        if not torch.allclose(alpha, torch.full_like(alpha, ALPHA_INIT), atol=1e-6):
            raise RuntimeError(f"{experiment} alpha initialization is not {ALPHA_INIT}")
        equivalence = float(
            (logits - components["stage4_logits"]).abs().max().detach().cpu())
        if equivalence > 1e-6:
            raise RuntimeError(
                f"{experiment} zero-initialized branch changes the initial logits: "
                f"{equivalence}")
        stage3 = components["stage3"]
        payload = {
            "logits": logits,
            "stage3_shape": list(stage3.shape),
            "stage4_shape": list(components["stage4"].shape),
            "alpha_init_max_abs_error": float(
                (alpha - ALPHA_INIT).abs().max()),
            "initial_zero_branch_logit_max_abs_error": equivalence,
        }
        if experiment == "R8":
            payload["stage3_logits_shape"] = list(components["stage3_logits"].shape)
        else:
            payload["attended_shape"] = list(components["attended"].shape)
        if experiment == "R10":
            conditioning = components["conditioning"]
            payload["conditioning_norm_max"] = float(
                conditioning.detach().float().norm(dim=-1).max().cpu())
        return payload

    return audit_forward


def capture_fixed_validation_batch(valid_data, runtime: dict):
    """One deterministic multi-image Validation batch for per-epoch diagnostics."""
    from dino_data import make_loader, shutdown_loader

    loader, sampler, _ = make_loader(
        valid_data, image_budget=int(runtime["image_budget"]), shuffle=False,
        seed=SEED, num_workers=0, pin_memory=True,
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    try:
        for batch in loader:
            if int(batch["images"].shape[0]) > 0 and bool(
                    (batch["image_counts"] > 1).any()):
                return {
                    "images": batch["images"].detach().cpu(),
                    "study_index": batch["study_index"].detach().cpu(),
                    "image_counts": batch["image_counts"].detach().cpu(),
                    "num_studies": int(batch["num_studies"]),
                    "labels": batch["labels"].detach().cpu(),
                    "known": batch["known"].detach().cpu(),
                    "subject_id": [str(value) for value in batch["subject_id"]],
                    "study_id": [str(value) for value in batch["study_id"]],
                    "image_ids": batch["image_ids"],
                }
    finally:
        shutdown_loader(loader)
    raise RuntimeError("Could not load a multi-image Validation batch")


def _build_pair(experiment: str):
    """Independently seeded R5 Global-only and candidate models (CPU)."""
    seed_everything(SEED)
    r5_backbone, r5_verification = load_pretrained_backbone("cpu")
    r5_model = DINOv3StudyClassifier(r5_backbone, num_labels=NUM_LABELS).cpu().eval()
    seed_everything(SEED)
    candidate_backbone, candidate_verification = load_pretrained_backbone("cpu")
    candidate = build_model(experiment, candidate_backbone).cpu().eval()
    if r5_verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("R5 equivalence model did not use the pinned DINOv3 weights")
    if candidate_verification["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Candidate equivalence model did not use the pinned DINOv3 weights")
    if head_hash(r5_model) != EXPECTED_HEAD_SHA or head_hash(candidate) != EXPECTED_HEAD_SHA:
        raise RuntimeError("Seeded Global Head does not match the R5 checkpoint contract")
    return r5_model, candidate


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu())


def test_initialization_equivalence(experiment: str, valid_data, runtime: dict) -> dict:
    """Initial logits/Study logits/probabilities must equal the seeded R5 model."""
    sample = capture_fixed_validation_batch(valid_data, runtime)
    images_cpu = sample["images"].float().contiguous()
    if images_cpu.shape[0] < 2:
        raise RuntimeError("Initialization equivalence needs a multi-image batch")
    r5_model, candidate = _build_pair(experiment)
    results = {"experiment": experiment, "head_sha256": EXPECTED_HEAD_SHA,
               "pretrained_weight_sha256": EXPECTED_WEIGHT_SHA256}
    with torch.inference_mode():
        logits5 = r5_model(images_cpu)
        components = candidate.forward_components(images_cpu)
        logits_candidate = components["logits"]
        study5 = study_logit_mean(
            logits5, sample["study_index"], sample["num_studies"], sample["image_counts"])
        study_candidate = study_logit_mean(
            logits_candidate, sample["study_index"], sample["num_studies"],
            sample["image_counts"])
        cpu = {
            "image_logits_max_abs_error": _max_abs(logits5, logits_candidate),
            "study_logits_max_abs_error": _max_abs(study5, study_candidate),
            "study_probabilities_max_abs_error": _max_abs(
                torch.sigmoid(study5), torch.sigmoid(study_candidate)),
            "stage3_shape": list(components["stage3"].shape),
            "alpha_init": float(components["alpha"].mean().detach().cpu()),
            "initial_branch_contribution_max_abs": float(
                (logits_candidate - components["stage4_logits"]).abs().max().detach().cpu()),
        }
    results["cpu_fp32"] = cpu
    tolerance = 1e-6
    if max(cpu["image_logits_max_abs_error"], cpu["study_logits_max_abs_error"],
           cpu["study_probabilities_max_abs_error"]) > tolerance:
        raise RuntimeError(f"{experiment} CPU FP32 initialization equivalence failed: {cpu}")

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage-fusion experiments require the R5 CUDA BF16 runtime")
    device = torch.device("cuda")
    r5_model.to(device, memory_format=torch.channels_last).eval()
    candidate.to(device, memory_format=torch.channels_last).eval()
    images_gpu = images_cpu.to(device).contiguous(memory_format=torch.channels_last)
    sample_index = sample["study_index"].to(device)
    counts = sample["image_counts"].to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits5 = r5_model(images_gpu)
        components = candidate.forward_components(images_gpu)
        study5 = study_logit_mean(logits5, sample_index, sample["num_studies"], counts)
        study_candidate = study_logit_mean(
            components["logits"], sample_index, sample["num_studies"], counts)
    gpu = {
        "image_logits_max_abs_error": _max_abs(logits5, components["logits"]),
        "study_logits_max_abs_error": _max_abs(study5, study_candidate),
        "study_probabilities_max_abs_error": _max_abs(
            torch.sigmoid(study5), torch.sigmoid(study_candidate)),
        "amp_dtype": "bfloat16",
    }
    results["gpu_bf16"] = gpu
    if max(gpu["image_logits_max_abs_error"], gpu["study_logits_max_abs_error"],
           gpu["study_probabilities_max_abs_error"]) > 1e-5:
        raise RuntimeError(f"{experiment} CUDA BF16 initialization equivalence failed: {gpu}")
    results["passed"] = True
    del r5_model, candidate
    torch.cuda.empty_cache()
    return results


def _grad_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not name.startswith(prefix):
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def _current_grads(model: torch.nn.Module, experiment: str) -> dict:
    probes = {
        "classifier": "classifier.",
        "fusion_alpha_theta": "fusion_alpha.theta",
        "stage4_backbone_stage3": "backbone.model.stages.3.",
        "stage4_backbone_downsample3": "backbone.model.downsample_layers.3.",
        "stage3_backbone_stage2": "backbone.model.stages.2.",
        "stage3_backbone_downsample2": "backbone.model.downsample_layers.2.",
    }
    if experiment == "R8":
        probes.update({
            "stage3_branch_head": "stage3_branch.head.",
            "stage3_branch_norm": "stage3_branch.norm.",
        })
    else:
        probes.update({
            "query_token_norm": "query_attention.token_norm.",
            "query_key_proj": "query_attention.key_proj.",
            "query_value_proj": "query_attention.value_proj.",
            "disease_queries": "query_attention.disease_queries",
            "aux_head_weight": "query_attention.aux_head_weight",
            "aux_head_bias": "query_attention.aux_head_bias",
        })
        if experiment == "R10":
            probes.update({
                "condition_norm": "query_attention.condition_norm.",
                "condition_proj": "query_attention.condition_proj.",
            })
    return {name: _grad_norm(model, prefix) for name, prefix in probes.items()}


def _gradient_step(model, optimizer, batch, device, channels_last: bool,
                   frozen_modules) -> float:
    from torch import autocast

    set_profile_training_mode(model, frozen_modules)
    optimizer.zero_grad(set_to_none=True)
    images = batch["images"].to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    study_index = batch["study_index"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    known = batch["known"].to(device, non_blocking=True)
    image_counts = batch["image_counts"].to(device, non_blocking=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        image_logits = model(images)
        study_logits = study_logit_mean(
            image_logits.float(), study_index, batch["num_studies"], image_counts)
        per_study, eligible = study_bce_per_study(study_logits, labels, known)
        loss = per_study[eligible].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Gradient-check Study BCE is not finite")
    loss.backward()
    return float(loss.detach().float().cpu())


def _conditioning_path_probe(model, batch, device, channels_last: bool,
                             frozen_modules) -> dict:
    """Prove the trained R10 model moved W_q, uses it, and backprops to Stage4.

    Comparing the norms of two large backbone gradients is numerically fragile
    here (the conditioning contribution is many orders smaller than the main
    path), so three direct FP32 measurements are used instead: the conditioning
    term magnitude, the on/off logit difference, and the gradient of the
    auxiliary logits with respect to the Stage4 trunk normalization weight
    (a path that only exists through W_q).
    """
    set_profile_training_mode(model, frozen_modules)
    model.zero_grad(set_to_none=True)
    images = batch["images"].to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    components = model.forward_components(images)
    with torch.no_grad():
        components_off = model.forward_components(images, use_conditioning=False)
        conditioning_norm = float(components["conditioning"].abs().max())
        logit_difference = float(
            (components["logits"] - components_off["logits"]).abs().max())
    stage4_target = model.backbone.model.norm.weight
    gradient = torch.autograd.grad(
        components["stage3_logits"].sum(), stage4_target, retain_graph=False,
        allow_unused=True)[0]
    model.zero_grad(set_to_none=True)
    gradient_norm = 0.0 if gradient is None else float(gradient.detach().norm())
    return {
        "conditioning_abs_max": conditioning_norm,
        "condition_proj_weight_abs_max": float(
            model.query_attention.condition_proj.weight.detach().abs().max()),
        "logits_max_abs_change_without_conditioning": logit_difference,
        "auxiliary_to_stage4_trunk_grad_norm": gradient_norm,
        "conditioning_nonzero": bool(conditioning_norm > 0.0),
        "conditioning_changes_prediction": bool(logit_difference > 0.0),
        "auxiliary_gradient_reaches_stage4_trunk": bool(gradient_norm > 0.0),
    }


def _auxiliary_path_isolation(model, batch, device, channels_last: bool,
                              frozen_modules) -> dict:
    """Backprop through the auxiliary logits only (real Stage3 path proof).

    The Stage3 trunk also receives gradients from the unhanged Stage4 main
    path, so total Stage3 gradients cannot prove the auxiliary branch is wired.
    This isolated backward starts from z3 / z_attn directly: the classifier
    gradient must stay zero and the Stage3 trunk gradient must be nonzero.
    """
    from torch import autocast

    set_profile_training_mode(model, frozen_modules)
    model.zero_grad(set_to_none=True)
    images = batch["images"].to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    study_index = batch["study_index"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    known = batch["known"].to(device, non_blocking=True)
    image_counts = batch["image_counts"].to(device, non_blocking=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        components = model.forward_components(images)
        auxiliary_study = study_logit_mean(
            components["stage3_logits"].float(), study_index,
            batch["num_studies"], image_counts)
        per_study, eligible = study_bce_per_study(auxiliary_study, labels, known)
        loss = per_study[eligible].mean()
    loss.backward()
    payload = {
        "auxiliary_loss": float(loss.detach().float().cpu()),
        "stage3_branch_head_grad_norm": _grad_norm(
            model, "stage3_branch.head." if hasattr(model, "stage3_branch")
            else "query_attention.aux_head_weight"),
        "stage3_backbone_grad_norm": _grad_norm(model, "backbone.model.stages.2."),
        "stage3_downsample_grad_norm": _grad_norm(
            model, "backbone.model.downsample_layers.2."),
        "main_path_classifier_grad_norm": _grad_norm(model, "classifier."),
        "gradient_reaches_stage3_trunk": bool(
            _grad_norm(model, "backbone.model.stages.2.") > 0.0),
    }
    model.zero_grad(set_to_none=True)
    return payload


def run_gradient_check(experiment: str, contract: dict, config: dict, freeze: dict,
                       runtime: dict, train_data) -> dict:
    """Real-batch preflight: shape/memory audit and the delayed-gradient contract.

    Zero-initialized auxiliary logit heads must still receive gradient at the
    first step, while theta (and, for R9/R10, the Query/Key/Value subtree) is
    expected to stay at zero until the head has moved.  R10 needs a second step
    before the Stage4 conditioning projection can receive gradient, after which
    a dedicated probe verifies that the conditioning is non-zero, changes the
    prediction and back-propagates into the Stage4 trunk.
    """
    from dino_data import make_loader, shutdown_loader

    if not torch.cuda.is_available():
        raise RuntimeError("Gradient check requires CUDA")
    device = torch.device("cuda")
    seed_everything(SEED)
    backbone, verification = load_pretrained_backbone("cpu")
    model = build_model(experiment, backbone).to(device)
    trainable_modules, frozen_modules, trainable_names = freeze_layout(experiment, model)
    model.backbone.set_gradient_checkpointing(
        bool(runtime["gradient_checkpointing"]))
    if runtime["channels_last"]:
        model.to(memory_format=torch.channels_last)
    optimizer = make_optimizer(experiment)(model, config)
    set_profile_training_mode(model, frozen_modules)
    loader, sampler, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=0, pin_memory=True,
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    sampler.set_epoch(0)
    batches = []
    try:
        iterator = iter(loader)
        for _ in range(3):
            batches.append(next(iterator))
    finally:
        shutdown_loader(loader)

    torch.cuda.reset_peak_memory_stats(device)
    steps = []
    number_of_steps = 2
    for step_index in range(1, number_of_steps + 1):
        batch = batches[(step_index - 1) % len(batches)]
        loss = _gradient_step(model, optimizer, batch, device,
                              bool(runtime["channels_last"]), frozen_modules)
        grads = _current_grads(model, experiment)
        steps.append({"step": step_index, "loss": loss, "grad_norms": grads})
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    auxiliary_isolation = _auxiliary_path_isolation(
        model, batches[0], device, bool(runtime["channels_last"]), frozen_modules)
    conditioning_probe = None
    if experiment == "R10":
        conditioning_probe = _conditioning_path_probe(
            model, batches[0], device, bool(runtime["channels_last"]),
            frozen_modules)

    def nonzero(step: dict, name: str) -> bool:
        return step["grad_norms"][name] > 0.0

    checks = {}
    first, second = steps[0], steps[1]
    checks["stage3_auxiliary_path_reaches_stage3_trunk"] = bool(
        auxiliary_isolation["gradient_reaches_stage3_trunk"])
    checks["stage3_auxiliary_path_leaves_main_classifier_gradient_zero"] = bool(
        auxiliary_isolation["main_path_classifier_grad_norm"] == 0.0)
    if experiment == "R8":
        checks.update({
            "stage3_branch_head_gradient_at_step1": nonzero(first, "stage3_branch_head"),
            "theta_gradient_at_step1_is_zero": not nonzero(first, "fusion_alpha_theta"),
            "stage3_branch_norm_gradient_at_step1_is_zero": not nonzero(
                first, "stage3_branch_norm"),
            "theta_gradient_at_step2_is_nonzero": nonzero(second, "fusion_alpha_theta"),
            "stage3_branch_norm_gradient_at_step2_is_nonzero": nonzero(
                second, "stage3_branch_norm"),
            "stage4_main_path_gradient_at_step1": nonzero(first, "classifier"),
        })
    else:
        checks.update({
            "aux_head_gradient_at_step1": nonzero(first, "aux_head_weight"),
            "theta_gradient_at_step1_is_zero": not nonzero(first, "fusion_alpha_theta"),
            "key_proj_gradient_at_step1_is_zero": not nonzero(first, "query_key_proj"),
            "value_proj_gradient_at_step1_is_zero": not nonzero(first, "query_value_proj"),
            "disease_queries_gradient_at_step1_is_zero": not nonzero(
                first, "disease_queries"),
            "theta_gradient_at_step2_is_nonzero": nonzero(second, "fusion_alpha_theta"),
            "key_proj_gradient_at_step2_is_nonzero": nonzero(second, "query_key_proj"),
            "value_proj_gradient_at_step2_is_nonzero": nonzero(second, "query_value_proj"),
            "disease_queries_gradient_at_step2_is_nonzero": nonzero(
                second, "disease_queries"),
            "stage4_main_path_gradient_at_step1": nonzero(first, "classifier"),
        })
        if experiment == "R10":
            checks.update({
                "condition_proj_gradient_at_step1_is_zero": not nonzero(
                    first, "condition_proj"),
                "condition_proj_gradient_at_step2_is_nonzero": nonzero(
                    second, "condition_proj"),
                "conditioning_nonzero_after_two_steps": bool(
                    conditioning_probe["conditioning_nonzero"]),
                "conditioning_changes_prediction": bool(
                    conditioning_probe["conditioning_changes_prediction"]),
                "auxiliary_gradient_reaches_stage4_trunk": bool(
                    conditioning_probe["auxiliary_gradient_reaches_stage4_trunk"]),
            })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{experiment} delayed-gradient contract failed: {failed}")
    payload = {
        "experiment": experiment,
        "passed": True,
        "device": torch.cuda.get_device_name(0),
        "image_budget": int(runtime["image_budget"]),
        "grad_accumulation": int(runtime["grad_accumulation"]),
        "channels_last": bool(runtime["channels_last"]),
        "gradient_checkpointing": bool(runtime["gradient_checkpointing"]),
        "number_of_optimizer_steps": number_of_steps,
        "checks": checks,
        "steps": steps,
        "auxiliary_path_isolation": auxiliary_isolation,
        "conditioning_probe": conditioning_probe,
        "peak_reserved_gib": peak_reserved,
        "memory_limit_gib": 0.9 * (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)),
        "test_data_accessed": False,
    }
    if peak_reserved > payload["memory_limit_gib"]:
        raise RuntimeError(
            f"{experiment} preflight peak reserved memory {peak_reserved:.2f} GiB "
            f"exceeds the 90% device budget")
    payload["preflight_passed"] = True
    del model, optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return payload


class StageFusionEpochDiagnostics:
    """Per-epoch branch/attention diagnostics written to CSV/PNG artifacts."""

    def __init__(self, experiment: str, run_dir: Path, sample: dict, runtime: dict):
        self.experiment = experiment
        self.run_dir = Path(run_dir)
        self.images = sample["images"].detach().cpu()
        self.runtime = runtime
        self.branch_path = self.run_dir / "branch_diagnostics.csv"
        self.alpha_path = self.run_dir / "alpha_by_epoch.csv"
        self.contribution_path = self.run_dir / "stage3_logit_contribution.csv"
        self.attention_path = self.run_dir / "attention_statistics.csv"
        self.query_similarity_path = self.run_dir / "query_similarity.csv"
        self.conditioning_path = self.run_dir / "stage4_query_conditioning.csv"
        self.conditioned_attention_path = self.run_dir / "conditioned_vs_unconditioned_attention.csv"
        self.branch_rows = (pd.read_csv(self.branch_path).to_dict("records")
                            if self.branch_path.is_file() else [])
        self.alpha_rows = (pd.read_csv(self.alpha_path).to_dict("records")
                           if self.alpha_path.is_file() else [])
        self.contribution_rows = (pd.read_csv(self.contribution_path).to_dict("records")
                                  if self.contribution_path.is_file() else [])
        self.attention_rows = (pd.read_csv(self.attention_path).to_dict("records")
                               if self.attention_path.is_file() else [])
        self.query_rows = (pd.read_csv(self.query_similarity_path).to_dict("records")
                           if self.query_similarity_path.is_file() else [])
        self.conditioning_rows = (pd.read_csv(self.conditioning_path).to_dict("records")
                                  if self.conditioning_path.is_file() else [])
        self.conditioned_attention_rows = (
            pd.read_csv(self.conditioned_attention_path).to_dict("records")
            if self.conditioned_attention_path.is_file() else [])

    def _forward(self, model):
        device = next(model.parameters()).device
        images = self.images.to(device, non_blocking=True)
        if self.runtime["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        modes = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16):
                kwargs = ({'return_attention_weights': True}
                          if self.experiment in {"R9", "R10"} else {})
                components = model.forward_components(images, **kwargs)
        finally:
            for module, training in modes:
                module.training = training
        return components

    def __call__(self, model, epoch, phase, row):
        if phase == "start":
            self._theta_start = model.fusion_alpha.theta.detach().float().cpu().clone()
            return None
        components = self._forward(model)
        epoch_number = int(epoch)
        alpha = components["alpha"].detach().float().cpu()
        stage4_logits = components["stage4_logits"].float().cpu()
        stage3_logits = components["stage3_logits"].float().cpu()
        logits = components["logits"].float().cpu()
        residual = alpha.unsqueeze(0) * stage3_logits
        delta = logits - stage4_logits
        mean_abs_stage4 = stage4_logits.abs().mean(dim=0)
        mean_abs_stage3 = stage3_logits.abs().mean(dim=0)
        mean_abs_residual = residual.abs().mean(dim=0)
        mean_abs_delta = delta.abs().mean(dim=0)
        grad_norms = _current_grads(model, self.experiment)
        theta_end = model.fusion_alpha.theta.detach().float().cpu()
        theta_start = getattr(self, "_theta_start", theta_end.clone())
        lr_head = float((row or {}).get("lr_head", float("nan")))

        self.alpha_rows = [entry for entry in self.alpha_rows
                           if int(entry["epoch"]) != epoch_number]
        self.alpha_rows.append({
            "epoch": epoch_number,
            "theta_start": float(theta_start.mean()),
            "theta_end": float(theta_end.mean()),
            "theta_abs_mean": float(theta_end.abs().mean()),
            "lr_head": lr_head,
            **{f"alpha_{LABEL_NAMES[index].replace(' ', '_')}": float(alpha[index])
               for index in range(NUM_LABELS)},
        })
        self.branch_rows = [entry for entry in self.branch_rows
                            if int(entry["epoch"]) != epoch_number]
        self.contribution_rows = [entry for entry in self.contribution_rows
                                  if int(entry["epoch"]) != epoch_number]
        for index, name in enumerate(LABEL_NAMES):
            self.branch_rows.append({
                "epoch": epoch_number,
                "disease": name,
                "alpha": float(alpha[index]),
                "mean_abs_stage4_logit": float(mean_abs_stage4[index]),
                "mean_abs_stage3_logit": float(mean_abs_stage3[index]),
                "mean_abs_residual_term": float(mean_abs_residual[index]),
                "mean_abs_logit_delta": float(mean_abs_delta[index]),
                "residual_to_stage4_ratio": float(
                    mean_abs_residual[index] / (mean_abs_stage4[index] + 1e-12)),
                "stage3_logit_grad_norm": float(
                    grad_norms.get("stage3_branch_head",
                                   grad_norms.get("aux_head_weight", 0.0))),
                "theta_grad_norm": float(grad_norms["fusion_alpha_theta"]),
                "stage3_backbone_grad_norm": float(
                    grad_norms["stage3_backbone_stage2"]),
                "classifier_grad_norm": float(grad_norms["classifier"]),
                "lr_head": lr_head,
            })
            self.contribution_rows.append({
                "epoch": epoch_number,
                "disease": name,
                "alpha": float(alpha[index]),
                "mean_abs_stage3_logit": float(mean_abs_stage3[index]),
                "mean_abs_residual_term": float(mean_abs_residual[index]),
                "mean_abs_logit_delta": float(mean_abs_delta[index]),
                "mean_signed_logit_delta": float(delta[:, index].mean()),
            })
        write_csv(self.branch_path, self.branch_rows)
        write_csv(self.alpha_path, self.alpha_rows)
        write_csv(self.contribution_path, self.contribution_rows)

        attention_summary = {}
        if self.experiment in {"R9", "R10"}:
            weights = components["attention_weights"]
            attention_summary = self._attention_diagnostics(
                model, components, epoch_number, lr_head)
        self._plot_branch_curves()
        return {
            "alpha_mean": float(alpha.mean()),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            **{f"attention_{key}": value for key, value in attention_summary.items()},
        }

    def _attention_diagnostics(self, model, components, epoch_number, lr_head):
        weights = components["attention_weights"].float().cpu()  # [B,H,10,T]
        batch_size, heads, labels, tokens = weights.shape
        if (labels, tokens) != (NUM_LABELS, 32 * 32):
            raise RuntimeError(
                f"Unexpected attention weights shape {tuple(weights.shape)}")
        flat = weights.mean(dim=(0, 1))  # [10, 1024]
        flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(flat * flat.clamp_min(1e-12).log()).sum(dim=-1)
        max_weight = flat.max(dim=-1).values
        sorted_weights = torch.sort(flat, dim=-1, descending=True).values
        top5 = sorted_weights[:, : max(1, int(round(tokens * 0.05)))].sum(dim=-1)
        top10 = sorted_weights[:, : max(1, int(round(tokens * 0.10)))].sum(dim=-1)
        uniform_entropy = float(math.log(tokens))
        queries = components["queries"].float().cpu().mean(dim=0)  # [10,128]
        query_norm = F.normalize(queries, dim=-1)
        query_cos = torch.matmul(query_norm, query_norm.T)
        attention_norm = F.normalize(flat, dim=-1)
        attention_cos = torch.matmul(attention_norm, attention_norm.T)
        mask = ~torch.eye(NUM_LABELS, dtype=torch.bool)
        attention_js = self._pairwise_js(flat)
        grad_norms = _current_grads(model, self.experiment)
        self.attention_rows = [entry for entry in self.attention_rows
                               if int(entry["epoch"]) != epoch_number]
        for index, name in enumerate(LABEL_NAMES):
            self.attention_rows.append({
                "epoch": epoch_number,
                "disease": name,
                "entropy": float(entropy[index]),
                "normalized_entropy": float(entropy[index] / uniform_entropy),
                "max_weight": float(max_weight[index]),
                "top5pct_mass": float(top5[index]),
                "top10pct_mass": float(top10[index]),
                "query_norm": float(queries[index].norm()),
                "attention_mean_offdiag_cosine": float(
                    attention_cos[index][mask[index]].mean()),
                "key_proj_grad_norm": float(grad_norms.get("query_key_proj", 0.0)),
                "value_proj_grad_norm": float(grad_norms.get("query_value_proj", 0.0)),
                "query_grad_norm": float(grad_norms.get("disease_queries", 0.0)),
                "aux_head_grad_norm": float(grad_norms.get("aux_head_weight", 0.0)),
                "theta_grad_norm": float(grad_norms["fusion_alpha_theta"]),
                "stage3_backbone_grad_norm": float(
                    grad_norms["stage3_backbone_stage2"]),
                "lr_head": lr_head,
            })
        self.query_rows = [entry for entry in self.query_rows
                           if int(entry["epoch"]) != epoch_number]
        for first in range(NUM_LABELS):
            for second in range(first + 1, NUM_LABELS):
                self.query_rows.append({
                    "epoch": epoch_number,
                    "disease_a": LABEL_NAMES[first],
                    "disease_b": LABEL_NAMES[second],
                    "query_cosine": float(query_cos[first, second]),
                    "attention_cosine": float(attention_cos[first, second]),
                    "attention_js_divergence": float(attention_js[first, second]),
                })
        write_csv(self.attention_path, self.attention_rows)
        write_csv(self.query_similarity_path, self.query_rows)

        conditioning_summary = {}
        if self.experiment == "R10":
            conditioning = components["conditioning"].float().cpu()
            query_offset = conditioning.mean(dim=1)  # [B,128]
            disease_norm = float(model.query_attention.disease_queries.detach()
                                 .float().cpu().norm(dim=-1).mean())
            condition_norm = float(query_offset.norm(dim=-1).mean())
            ratio = condition_norm / (disease_norm + 1e-12)
            attention_with = components["attention_weights"].float().cpu()
            no_cond_components = self._forward_conditioning_off(model)
            attention_without = no_cond_components["attention_weights"].float().cpu()
            attention_l1 = float(
                (attention_with.mean(dim=(0, 1)) - attention_without.mean(dim=(0, 1)))
                .abs().sum(dim=-1).mean())
            conditioning_summary = {
                "condition_norm_mean": condition_norm,
                "disease_embedding_norm_mean": disease_norm,
                "condition_to_disease_norm_ratio": ratio,
                "attention_l1_change_with_conditioning": attention_l1,
            }
            self.conditioning_rows = [entry for entry in self.conditioning_rows
                                      if int(entry["epoch"]) != epoch_number]
            self.conditioning_rows.append({
                "epoch": epoch_number,
                "disease": "all_mean",
                "condition_norm_mean": condition_norm,
                "disease_embedding_norm_mean": disease_norm,
                "condition_to_disease_norm_ratio": ratio,
                "query_offset_norm_mean": condition_norm,
                "condition_proj_grad_norm": float(
                    grad_norms.get("condition_proj", 0.0)),
                "condition_norm_grad_norm": float(
                    grad_norms.get("condition_norm", 0.0)),
                "weighted_aux_logit_change": float(
                    (components["stage3_logits"] - no_cond_components["stage3_logits"])
                    .abs().mean().detach().cpu()),
            })
            self.conditioned_attention_rows = [
                entry for entry in self.conditioned_attention_rows
                if int(entry["epoch"]) != epoch_number]
            for index, name in enumerate(LABEL_NAMES):
                row_with = attention_with.mean(dim=(0, 1))[index]
                row_without = attention_without.mean(dim=(0, 1))[index]
                row_with = row_with / row_with.sum().clamp_min(1e-12)
                row_without = row_without / row_without.sum().clamp_min(1e-12)
                self.conditioned_attention_rows.append({
                    "epoch": epoch_number,
                    "disease": name,
                    "attention_l1_with_vs_without": float(
                        (row_with - row_without).abs().sum()),
                    "attention_max_weight_with": float(row_with.max()),
                    "attention_max_weight_without": float(row_without.max()),
                    "entropy_with": float(
                        -(row_with * row_with.clamp_min(1e-12).log()).sum()),
                    "entropy_without": float(
                        -(row_without * row_without.clamp_min(1e-12).log()).sum()),
                })
            write_csv(self.conditioning_path, self.conditioning_rows)
            write_csv(self.conditioned_attention_path, self.conditioned_attention_rows)
        return {
            "attention_entropy_mean": float(entropy.mean()),
            "attention_normalized_entropy_mean": float(entropy.mean() / uniform_entropy),
            "attention_max_weight_mean": float(max_weight.mean()),
            "attention_top10_mass_mean": float(top10.mean()),
            "query_offdiag_cosine_mean": float(query_cos[mask].mean()),
            "attention_offdiag_cosine_mean": float(attention_cos[mask].mean()),
            **conditioning_summary,
        }

    def _forward_conditioning_off(self, model):
        device = next(model.parameters()).device
        images = self.images.to(device, non_blocking=True)
        if self.runtime["channels_last"]:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16):
            return model.forward_components(
                images, return_attention_weights=True, use_conditioning=False)

    @staticmethod
    def _pairwise_js(rows: torch.Tensor) -> torch.Tensor:
        rows = rows.clamp_min(1e-12)
        rows = rows / rows.sum(dim=-1, keepdim=True)
        result = torch.zeros(NUM_LABELS, NUM_LABELS)
        logs = rows.log()
        for first in range(NUM_LABELS):
            for second in range(first + 1, NUM_LABELS):
                middle = 0.5 * (rows[first] + rows[second])
                log_middle = middle.log()
                value = 0.5 * (rows[first] * (logs[first] - log_middle)).sum() + \
                    0.5 * (rows[second] * (logs[second] - log_middle)).sum()
                result[first, second] = value
                result[second, first] = value
        return result

    def _plot_branch_curves(self):
        if not self.alpha_rows:
            return
        frame = pd.DataFrame(self.alpha_rows).sort_values("epoch")
        plt.figure(figsize=(9, 4.8))
        for index, name in enumerate(LABEL_NAMES):
            column = f"alpha_{name.replace(' ', '_')}"
            if column in frame:
                plt.plot(frame["epoch"], frame[column], label=name, linewidth=1.2)
        plt.axhline(ALPHA_INIT, color="black", linestyle="--", linewidth=0.8,
                    label=f"init {ALPHA_INIT}")
        plt.xlabel("Epoch")
        plt.ylabel("alpha = sigmoid(theta)")
        plt.title(f"{self.experiment} per-disease fusion coefficient")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(self.run_dir / "alpha_by_disease.png", dpi=140)
        plt.close()
        contribution = pd.DataFrame(self.contribution_rows)
        if not contribution.empty:
            plt.figure(figsize=(9, 4.8))
            grouped = contribution.groupby("disease")["mean_abs_residual_term"]
            for name, values in grouped:
                plt.plot(sorted(contribution["epoch"].unique()), values.values,
                         label=name, linewidth=1.2)
            plt.xlabel("Epoch")
            plt.ylabel("mean |alpha * auxiliary logit|")
            plt.title(f"{self.experiment} branch contribution by disease")
            plt.legend(fontsize=6, ncol=2)
            plt.tight_layout()
            plt.savefig(self.run_dir / "branch_contribution_curves.png", dpi=140)
            plt.close()


def draw_training_curves(run_dir: Path):
    log_path = Path(run_dir) / "train_log.csv"
    if not log_path.is_file():
        return
    frame = pd.read_csv(log_path)
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(frame["epoch"], frame["macro_auroc"], marker="o", label="Macro-AUROC")
    axes[0].plot(frame["epoch"], frame["macro_auprc"], marker="s", label="Macro-AUPRC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation metric")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(frame["epoch"], frame["train_bce"], marker="o", label="Train BCE")
    axes[1].plot(frame["epoch"], frame["valid_loss"], marker="s", label="Valid BCE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.suptitle(Path(run_dir).name)
    figure.tight_layout()
    figure.savefig(Path(run_dir) / "training_curves.png", dpi=140)
    plt.close(figure)


def draw_architecture_diagram(experiment: str, path: Path):
    figure, axis = plt.subplots(figsize=(10, 6.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)
    axis.axis("off")

    def box(x, y, width, height, text, color):
        rectangle = plt.Rectangle((x, y), width, height, facecolor=color,
                                  edgecolor="#333333", linewidth=1.2)
        axis.add_patch(rectangle)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                  fontsize=8.5, wrap=True)

    def arrow(x1, y1, x2, y2, color="#333333"):
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    box(0.3, 8.2, 2.2, 0.9, "Image\n3x512x512", "#e8f0fe")
    box(0.3, 5.4, 2.2, 1.6, "DINOv3 ConvNeXt-Tiny\nStem/Stage1/2 frozen\nStage3 trainable", "#dbe7fb")
    box(0.3, 2.6, 2.2, 1.4, "Stage4 trainable\n[B,768,16,16]", "#cfe3fb")
    arrow(1.4, 8.2, 1.4, 7.0)
    arrow(1.4, 5.4, 1.4, 4.0)
    box(3.4, 2.3, 2.3, 2.0,
        "Original R5 Global path\npooled-token LayerNorm\nGAP -> Linear(768,10)\nz4", "#d9f2d9")
    arrow(2.5, 3.3, 3.4, 3.3)
    box(3.4, 6.2, 2.3, 1.4, "Stage3 features\n[B,384,32,32]", "#fdf2d0")
    arrow(2.5, 6.2, 3.4, 6.6)
    if experiment == "R8":
        box(6.4, 5.9, 3.1, 2.0,
            "R8 auxiliary branch\nLayerNorm(384)\nGAP -> Linear(384,10)\nzero-initialized head -> z3", "#fde8d8")
        arrow(5.7, 6.9, 6.4, 6.9)
        box(6.4, 2.4, 3.1, 1.6,
            "Disease-specific fusion\nz_k = z4_k + sigmoid(theta_k) * z3_k\ntheta_k init logit(0.1)", "#f7d9d9")
        arrow(5.7, 3.3, 6.4, 3.2)
        arrow(7.95, 5.9, 7.95, 4.0)
    else:
        condition_text = ("Stage4 global g4\n-> LayerNorm -> W_q (zero init)\n" if experiment == "R10" else "")
        box(6.4, 6.1, 3.1, 2.0,
            f"R9/R10 query attention\n{condition_text}e_k + W_q LN(g4)\n"
            "CrossAttention over 1024 Stage3 tokens\n"
            "d_model=128, heads=4\nzero-init per-disease logit head -> z_attn",
            "#fde8d8")
        arrow(5.7, 6.9, 6.4, 6.9)
        if experiment == "R10":
            arrow(2.5, 2.9, 6.4, 6.2, color="#8a6d3b")
            axis.text(3.9, 4.9, "Stage4 global feature\nconditions the query", fontsize=7.5,
                      color="#8a6d3b")
        box(6.4, 2.4, 3.1, 1.6,
            "Disease-specific fusion\nz_k = z4_k + sigmoid(theta_k) * z_attn,k", "#f7d9d9")
        arrow(5.7, 3.3, 6.4, 3.2)
        arrow(7.95, 6.1, 7.95, 4.0)
    box(6.4, 0.4, 3.1, 1.1, "Study Logit Mean -> Sigmoid -> Study AUROC", "#eeeeee")
    arrow(7.95, 2.4, 7.95, 1.5)
    axis.set_title(f"{experiment} architecture (R5 main path unchanged)", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_architecture_report(experiment: str, run_dir: Path, contract: dict,
                              gradient_check: dict, equivalence: dict,
                              group_report: dict):
    head_description = {
        "R8": "original Global Head + zero-init Stage3 GAP auxiliary head",
        "R9": "original Global Head + disease-query Stage3 cross-attention head",
        "R10": "original Global Head + Stage4-conditioned disease-query cross-attention head",
    }[experiment]
    lines = [
        f"# {experiment} Architecture and Initialization Report", "",
        "- Backbone: official general-vision DINOv3 ConvNeXt-Tiny; same pinned "
        "revision and SHA-256 as R5.",
        f"- Original pretrained backbone SHA-256: `{EXPECTED_WEIGHT_SHA256}`.",
        f"- Original Global Head independently initialized with seed {SEED}; "
        f"initial hash matches R5: `{contract['head_sha']}`.",
        "- Original R5 parameter names and main path are unchanged; the added "
        f"state namespaces are `{', '.join(NEW_BRANCH_NAMES[experiment])}`.",
        "- Main path: Stage3 -> original Downsample -> Stage4 -> official "
        "pooled-token normalization -> Linear(768,10) -> z4.",
        f"- Added branch: {BRANCH_DESCRIPTIONS[experiment]}.",
        f"- Head: {head_description}.",
        "- No R7 shared residual transfer, no auxiliary loss, no multiscale "
        "module, no extra Stage3 structure, no Study-level pooling change.",
        f"- New parameters: {EXPECTED_NEW_PARAMETERS[experiment]:,}; "
        f"total/trainable/frozen: {group_report['total_parameters']:,}/"
        f"{group_report['trainable_parameters']:,}/{group_report['frozen_parameters']:,}.",
        f"- R5 selected runtime reused exactly: image budget "
        f"{contract['runtime']['image_budget']}, accumulation "
        f"{contract['runtime']['grad_accumulation']}, channels-last "
        f"{contract['runtime']['channels_last']}, gradient checkpointing "
        f"{contract['runtime']['gradient_checkpointing']}; no new benchmark or LR search.",
        "- Runtime note: because Studies are indivisible, the measured effective "
        f"batch is {contract['runtime'].get('effective_images_per_update')} images / "
        f"{contract['runtime'].get('effective_studies_per_update')} Studies per "
        "optimizer update instead of exactly 32 images. The LR schedule "
        "(2 warmup epochs + cosine over the remaining 18 planned epochs) is "
        "anchored to the 20-epoch update budget, so an early stop leaves LR above "
        "its configured minimum; this is identical to R5/R7 and is not a "
        "R8/R9/R10-specific change.",
        "- Study Logit Mean, Study BCE, ResizePad, augmentation, normalization, "
        "split, seed and the three-condition stop rule follow the R5 implementation.",
        f"- Initialization equivalence: CPU FP32 max |logit| error "
        f"{equivalence['cpu_fp32']['image_logits_max_abs_error']:.3e}, CUDA BF16 "
        f"{equivalence['gpu_bf16']['image_logits_max_abs_error']:.3e}.",
        f"- Delayed-gradient contract verified over "
        f"{gradient_check['number_of_optimizer_steps']} real optimizer steps; "
        f"preflight peak reserved {gradient_check['peak_reserved_gib']:.2f} GiB.",
        "- Only official Train/Validation were accessed; Test was not read.", "",
        "## Delayed gradient checks", "",
        "| Check | Passed |", "|---|---|",
    ]
    for name, passed in gradient_check["checks"].items():
        lines.append(f"| {name} | {passed} |")
    (run_dir / "architecture_report.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")


def _copy_equivalence(equivalence: dict, run_dir: Path):
    write_json(run_dir / "initial_equivalence_test.json", equivalence)
    write_json(run_dir / "initialization_equivalence_test.json", equivalence)


def refresh_group_artifacts(experiment: str) -> Path:
    """Regenerate report artifacts for a finished group from on-disk evidence.

    Used by the analysis entry so post-training reports always come from the
    current code even when the training process imported an older revision.
    """
    group = GROUP_NAMES[experiment]
    run_dir = OUTPUT_ROOT / group
    if not (run_dir / "training_status.json").is_file():
        raise RuntimeError(f"{experiment} has no training_status.json")
    contract = load_r5_contract()
    gradient_check = read_json(run_dir / "gradient_check.json")
    equivalence = read_json(run_dir / "initial_equivalence_test.json")
    group_report = read_json(run_dir / "optimizer_parameter_groups.json")
    write_architecture_report(experiment, run_dir, contract, gradient_check,
                              equivalence, group_report)
    draw_architecture_diagram(experiment, run_dir / "architecture_diagram.png")
    draw_training_curves(run_dir)
    train_log = run_dir / "train_log.csv"
    if train_log.is_file():
        shutil.copy2(train_log, run_dir / "training_log.csv")
    val_best = run_dir / "predictions" / "val_best.csv"
    if val_best.is_file():
        pd.read_csv(val_best).to_csv(run_dir / "val_best_predictions.csv", index=False)
    return run_dir


def run_experiment(experiment: str, skip_existing: bool = True) -> dict:
    """Prepare, verify, train and post-process one stage-fusion experiment."""
    if experiment not in GROUP_NAMES:
        raise ValueError(f"Unknown experiment: {experiment}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{experiment} requires the R5 CUDA BF16 training runtime")
    group = GROUP_NAMES[experiment]
    run_dir = OUTPUT_ROOT / group
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "training_status.json"
    if skip_existing and status_path.is_file():
        existing = read_json(status_path)
        if existing.get("complete"):
            print(f"{experiment} is already complete: {existing}", flush=True)
            return existing

    contract = load_r5_contract()
    feature, config, freeze = build_run_configs(experiment, contract)
    runtime = contract["runtime"]

    # Pin the exact R5 runtime so run_custom_group never re-benchmarks.
    selected_path = run_dir / "selected_runtime_config.json"
    runtime_payload = copy.deepcopy(contract["runtime_payload"])
    runtime_payload["group"] = group
    runtime_payload["source_group"] = "R5_Global_Only_512"
    runtime_payload["selection_rule"] = (
        "Exact R5 selected runtime reuse; no new runtime, throughput or LR search.")
    if selected_path.is_file():
        saved = read_json(selected_path)
        if saved.get("selected_runtime") != runtime:
            raise RuntimeError(
                f"Existing {experiment} selected runtime differs from the R5 runtime")
    else:
        write_json(selected_path, runtime_payload)

    data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    train_data, valid_data, data_config = build_masked_datasets(
        data_path, SEED, IMAGE_SIZE)
    if (len(train_data), int(sum(train_data.image_counts)), len(valid_data),
            int(sum(valid_data.image_counts))) != (18513, 30344, 859, 1422):
        raise RuntimeError(f"{experiment} Train/Validation counts differ from the R5 manifest")
    if (data_config.get("label_semantics") != "closed_world"
            or data_config.get("class_ids") != list(range(1, 11))
            or data_config.get("class_names") != LABEL_NAMES):
        raise RuntimeError(f"{experiment} label semantics/order differ from R5")
    if set(train_data.study_keys) & set(valid_data.study_keys):
        raise RuntimeError(f"{experiment} Train/Validation Study keys overlap")
    train_subjects = {str(subject) for subject, _ in train_data.study_keys}
    valid_subjects = {str(subject) for subject, _ in valid_data.study_keys}
    if train_subjects & valid_subjects:
        raise RuntimeError(f"{experiment} Train/Validation Subject leakage detected")

    equivalence_path = run_dir / "initial_equivalence_test.json"
    if equivalence_path.is_file():
        equivalence = read_json(equivalence_path)
        if (equivalence.get("passed") is not True
                or equivalence.get("pretrained_weight_sha256") != EXPECTED_WEIGHT_SHA256
                or equivalence.get("head_sha256") != EXPECTED_HEAD_SHA):
            raise RuntimeError(f"Existing {experiment} initialization equivalence is invalid")
    else:
        print(f"Running {experiment} CPU FP32 / CUDA BF16 initialization equivalence.",
              flush=True)
        equivalence = test_initialization_equivalence(experiment, valid_data, runtime)
        _copy_equivalence(equivalence, run_dir)

    gradient_path = run_dir / "gradient_check.json"
    if gradient_path.is_file():
        gradient_check = read_json(gradient_path)
        if gradient_check.get("passed") is not True:
            raise RuntimeError(f"Existing {experiment} gradient check did not pass")
    else:
        print(f"Running {experiment} real-batch delayed-gradient and memory preflight.",
              flush=True)
        gradient_check = run_gradient_check(
            experiment, contract, config, freeze, runtime, train_data)
        write_json(gradient_path, gradient_check)

    builder = make_builder(experiment, runtime)
    optimizer_builder = make_optimizer(experiment)
    seed_everything(SEED)
    model_for_groups, _, modules, frozen, names = builder(
        PROFILE, torch.device("cuda"), runtime, SEED)
    optimizer = optimizer_builder(model_for_groups, config)
    group_report = {
        "experiment": experiment,
        "group": group,
        "total_parameters": sum(p.numel() for p in model_for_groups.parameters()),
        "trainable_parameters": sum(p.numel() for p in model_for_groups.parameters()
                                    if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model_for_groups.parameters()
                                 if not p.requires_grad),
        "new_parameters": EXPECTED_NEW_PARAMETERS[experiment],
        "trainable_modules": [name for name, _ in modules],
        "frozen_modules": [name for name, _ in frozen],
        "trainable_parameter_name_count": len(names),
        "optimizer_groups": [
            {"name": str(group_item.get("name")),
             "learning_rate": float(group_item["lr"]),
             "weight_decay": float(group_item["weight_decay"]),
             "parameter_count": int(sum(p.numel() for p in group_item["params"]))}
            for group_item in optimizer.param_groups
        ],
    }
    write_json(run_dir / "optimizer_parameter_groups.json", group_report)
    del optimizer, model_for_groups
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    sample = capture_fixed_validation_batch(valid_data, runtime)
    diagnostics = StageFusionEpochDiagnostics(experiment, run_dir, sample, runtime)

    print(f"Starting {experiment} training from the original pinned DINOv3 weights; "
          "no R5/R6/R7 fine-tuned weights are loaded.", flush=True)
    result = run_custom_group(
        feature, config, freeze, experiment, group, IMAGE_SIZE,
        {"repository": contract["config_payload"]["initialization_repo"],
         "revision": contract["config_payload"]["initialization_revision"],
         "weights_sha256": EXPECTED_WEIGHT_SHA256},
        model_builder=builder,
        optimizer_builder=optimizer_builder,
        batch_forward=lambda model, images, _batch: model(images),
        validation_forward=lambda model, images, _batch: model(images),
        collate_fn=collate_masked_studies,
        audit_forward=make_audit_forward(experiment),
        backbone_name=f"DINOv3 ConvNeXt-Tiny {ARCHITECTURES[experiment]}",
        expected_stage3_channels=384,
        dataset_builder=build_masked_datasets,
        run_context={
            "group": group,
            "architecture": ARCHITECTURES[experiment],
            "image_size": IMAGE_SIZE,
            "head_type": (
                "original_global_gap_linear10_plus_disease_specific_stage3_aux"),
            "pretrained_initialization_sha256": EXPECTED_WEIGHT_SHA256,
            "global_classifier_init_sha256": contract["head_sha"],
            "runtime_source": "R5_Global_Only_512/selected_runtime_config.json",
            "freeze_profile": "R5_P2_Stage3_Stage4",
            "added_branch": BRANCH_DESCRIPTIONS[experiment],
            "alpha_init": ALPHA_INIT,
            "auxiliary_loss": False,
            "test_access": False,
        },
        expected_weight_sha256=EXPECTED_WEIGHT_SHA256,
        epoch_callback=diagnostics)
    status = read_json(run_dir / "training_status.json")
    if (not status.get("complete")
            or status.get("stop_reason") not in {"three_condition_early_stop",
                                                 "max_epochs_reached"}):
        raise RuntimeError(f"{experiment} training did not complete: {status}")
    if not (run_dir / "best.pth").is_file() or not (run_dir / "last.pth").is_file():
        raise RuntimeError(f"{experiment} Best/Last checkpoints are missing")

    config_path = run_dir / "config.json"
    config_payload = read_json(config_path)
    config_payload.update({
        "experiment": EXPERIMENT_NAME,
        "group": group,
        "image_size": IMAGE_SIZE,
        "backbone": "DINOv3 ConvNeXt-Tiny (official general-vision pretrained)",
        "initialization_repo": contract["config_payload"]["initialization_repo"],
        "initialization_revision": contract["config_payload"]["initialization_revision"],
        "initialization_sha256": EXPECTED_WEIGHT_SHA256,
        "model_variant": ARCHITECTURES[experiment],
        "model_description": BRANCH_DESCRIPTIONS[experiment],
        "head_description": (
            "original R5 Global Head retained; added independent Stage3 auxiliary "
            "path with ten disease-specific alpha fusion coefficients"),
        "alpha_initial": ALPHA_INIT,
        "theta_initial": THETA_INIT,
        "auxiliary_loss": False,
        "trainable_policy": (
            "R5 P2 Stage3/Stage4/final Norm/Global Head plus the declared auxiliary "
            "branch; Stem/Stage1/Stage2 frozen"),
        "initialization_equivalence_test": equivalence,
        "gradient_check": gradient_check,
        "optimizer_parameter_groups": group_report,
        "test_access": False,
        "test_split_read": False,
    })
    write_json(config_path, config_payload)
    status["profile"] = f"{experiment.lower()}_stage_fusion"
    status["test_data_accessed"] = False
    write_json(status_path, status)
    result_path = run_dir / "training_result.json"
    if result_path.is_file():
        result_payload = read_json(result_path)
        result_payload["profile"] = f"{experiment.lower()}_stage_fusion"
        result_payload["test_data_accessed"] = False
        write_json(result_path, result_payload)

    # Keep the trainer-generated report, then add the architecture artifacts.
    architecture_path = run_dir / "architecture_report.md"
    write_architecture_report(experiment, run_dir, contract, gradient_check,
                              equivalence, group_report)
    draw_architecture_diagram(experiment, run_dir / "architecture_diagram.png")
    draw_training_curves(run_dir)
    train_log = run_dir / "train_log.csv"
    if train_log.is_file():
        shutil.copy2(train_log, run_dir / "training_log.csv")
    val_best = run_dir / "predictions" / "val_best.csv"
    if val_best.is_file():
        pd.read_csv(val_best).to_csv(run_dir / "val_best_predictions.csv", index=False)
    training_report = run_dir / "training_report.md"
    if training_report.is_file():
        original = training_report.read_text(encoding="utf-8")
        addendum = (
            f"\n## {experiment} auxiliary branch\n\n"
            f"{BRANCH_DESCRIPTIONS[experiment]}\n\n"
            "The original R5 Global Head and main path are unchanged; the added "
            f"state namespaces are `{', '.join(NEW_BRANCH_NAMES[experiment])}`. "
            "Per-epoch branch diagnostics are in `branch_diagnostics.csv`, "
            "`alpha_by_epoch.csv`, `stage3_logit_contribution.csv`"
            + (", `attention_statistics.csv`, `query_similarity.csv`"
               if experiment in {"R9", "R10"} else "")
            + (", `stage4_query_conditioning.csv`" if experiment == "R10" else "")
            + ".\n"
        )
        if f"## {experiment} auxiliary branch" not in original:
            training_report.write_text(original.rstrip() + "\n" + addendum,
                                       encoding="utf-8")
    print(json.dumps({
        "status": "training_complete",
        "experiment": experiment,
        "output_dir": str(run_dir),
        "best_epoch": status["best_epoch"],
        "best_macro_auroc": status["best_macro_auroc"],
        "stop_reason": status["stop_reason"],
        "test_data_accessed": False,
        "training_result": result,
    }, ensure_ascii=False), flush=True)
    return status
