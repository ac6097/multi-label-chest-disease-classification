"""Resolution ablations for DINOv3 ConvNeXt-Tiny using the completed P2 baseline."""

from __future__ import annotations

import copy
import gc
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from safetensors.torch import save_file
from PIL import Image

WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
PROJECT_DIR = WORK_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.transforms import ResizePad  # noqa: E402
from dino_analysis import LABEL_NAMES, _auc_vector, _bootstrap_subject_differences, _columns  # noqa: E402
from dino_data import build_datasets, make_loader  # noqa: E402
from dino_experiment import (  # noqa: E402
    environment_report,
    load_pretrained_backbone,
    save_json_atomic,
    setup_logging,
)
from dino_freeze_experiment import (  # noqa: E402
    _rename_predictions,
    build_profile_model,
    build_profile_optimizer,
    head_hash,
    load_experiment_config,
    profile_layout,
    run_profile_benchmark,
    set_profile_training_mode,
    train_profile,
)
from dino_feature_models import (  # noqa: E402
    DINOv3GlobalLocalClassifier,
    build_r3_model,
    build_masked_datasets,
    build_small_model,
    collate_masked_studies,
    feature_profile_layout,
    feature_profile_optimizer,
    load_small_pretrained_backbone,
    r3_batch_forward,
)
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from hybrid.common import seed_everything, write_csv  # noqa: E402


FEATURE_CONFIG_PATH = WORK_DIR / "configs" / "dino_feature_optimization.json"
OUTPUT_ROOT = WORK_DIR / "outputs" / "experiments" / "DINOv3_Feature_Optimization"
RESOLUTIONS = (("r1", "R1_448_GAP", 448), ("r2", "R2_512_GAP", 512))
EXPERIMENT_NAME = "DINOv3_Feature_Optimization"
PROFILE = "p2"
PACKAGE_INFERENCE_PROBABILITY_TOLERANCE = 1e-2


def r3_audit_forward(model, images, batch):
    masks = batch["valid_masks"].to(images.device, non_blocking=True)
    result = model(images, masks, return_components=True)
    alpha = float(result["alpha"].detach().cpu())
    if not (0.0 < alpha <= model.maximum_alpha):
        raise RuntimeError(f"R3 fusion alpha escaped its constraint: {alpha}")
    invalid_selected = result["topk_grid"] & ~result["valid_grid"][:, None]
    invalid_count = int(invalid_selected.sum().item())
    if invalid_count:
        raise RuntimeError(f"R3 Top-K selected {invalid_count} ResizePad-invalid positions")
    if not all(count > 0 for count in result["valid_counts"]):
        raise RuntimeError("R3 ResizePad mask produced an empty Stage3 valid region")
    if any(selected <= 0 or selected > valid
           for selected, valid in zip(result["selected_counts"], result["valid_counts"])):
        raise RuntimeError("R3 Top-K count is inconsistent with valid Stage3 cells")
    return {
        "logits": result["logits"],
        "initial_alpha": alpha,
        "valid_stage3_positions_min": min(result["valid_counts"]),
        "selected_topk_positions_min": min(result["selected_counts"]),
        "invalid_topk_positions": invalid_count,
        "stage3_response_shape": list(result["response_maps"].shape),
    }


def load_feature_config():
    with FEATURE_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        feature = json.load(stream)
    config, freeze = load_experiment_config()
    if feature["experiment"] != EXPERIMENT_NAME:
        raise ValueError("Unexpected feature-optimization experiment identifier")
    if feature["source"]["tiny_repo"] != config["source"]["huggingface_repo"]:
        raise ValueError("R1/R2 must use the original Tiny DINOv3 initialization")
    if feature["source"]["tiny_sha256"] != config["source"]["weights_sha256"]:
        raise ValueError("Feature experiment Tiny weight hash differs from the P2 baseline")
    freeze["benchmark"] = copy.deepcopy(feature["benchmark"])
    return feature, config, freeze


def save_group_config(run_dir: Path, feature: dict, config: dict, freeze: dict,
                      data_config: dict, image_size: int, train_data, valid_data,
                      backbone_name="DINOv3 ConvNeXt-Tiny",
                      weight_verification=None):
    weight_verification = weight_verification or {
        "repository": feature["source"]["tiny_repo"],
        "revision": feature["source"]["tiny_revision"],
        "weights_sha256": feature["source"]["tiny_sha256"],
    }
    payload = {
        "experiment": EXPERIMENT_NAME,
        "group": run_dir.name,
        "image_size": int(image_size),
        "backbone": backbone_name,
        "initialization_repo": weight_verification["repository"],
        "initialization_revision": weight_verification["revision"],
        "initialization_sha256": weight_verification["weights_sha256"],
        "profile": "P2: freeze Stem/Stage1/Stage2; train Stage3/Stage4/final norm/head",
        "experiment_config": config,
        "feature_config": feature,
        "freeze_config": freeze,
        "data_config": data_config,
        "train_studies": len(train_data),
        "train_images": int(sum(train_data.image_counts)),
        "validation_studies": len(valid_data),
        "validation_images": int(sum(valid_data.image_counts)),
        "test_access": False,
    }
    save_json_atomic(run_dir / "config.json", payload)


def verify_resizepad(image_size: int):
    records = []
    resize_pad = ResizePad(image_size)
    for width, height in ((2544, 3056), (3056, 2544), (2021, 2021)):
        image = Image.new("RGB", (width, height), (127, 127, 127))
        transformed = resize_pad(image)
        if transformed.size != (image_size, image_size):
            raise RuntimeError(f"ResizePad emitted {transformed.size}, expected square {image_size}")
        scale = image_size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        expected_ratio = width / height
        actual_ratio = resized_width / resized_height
        if abs(actual_ratio - expected_ratio) > max(1 / resized_width, 1 / resized_height):
            raise RuntimeError(f"ResizePad changed aspect ratio for {width}x{height}")
        pad_width = image_size - resized_width
        pad_height = image_size - resized_height
        left, top = pad_width // 2, pad_height // 2
        right, bottom = pad_width - left, pad_height - top
        if abs(left - right) > 1 or abs(top - bottom) > 1:
            raise RuntimeError("ResizePad padding is not centered")
        records.append({
            "source_size": [width, height],
            "canvas_size": [image_size, image_size],
            "resized_size": [resized_width, resized_height],
            "padding_ltrb": [left, top, right, bottom],
            "aspect_ratio_error": abs(actual_ratio - expected_ratio),
        })
    return records


def verify_profile_at_resolution(config: dict, train_data, image_size: int,
                                  weight_verification: dict,
                                  model_builder=None, optimizer_builder=None,
                                  batch_forward=None, collate_fn=None,
                                  audit_forward=None,
                                  expected_stage3_channels: int = 384):
    device = torch.device("cuda")
    loader_kwargs = {
        "image_budget": 1, "shuffle": False, "seed": int(config["seed"]),
        "num_workers": 0, "pin_memory": True, "prefetch_factor": 2,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader, sampler, _ = make_loader(train_data, **loader_kwargs)
    model = optimizer = None
    try:
        runtime = {"gradient_checkpointing": False, "channels_last": False}
        model, verification, train_modules, frozen_modules, trainable_names = (
            build_profile_model(PROFILE, device, runtime, seed=int(config["seed"]))
            if model_builder is None else
            model_builder(PROFILE, device, runtime, seed=int(config["seed"]))
        )
        if verification["weights_sha256"] != weight_verification["weights_sha256"]:
            raise RuntimeError("Resolution audit did not load the verified original Tiny weights")
        batch = next(iter(loader))
        images = batch["images"].to(device, non_blocking=True)
        if tuple(images.shape[-2:]) != (image_size, image_size):
            raise RuntimeError(f"Dataset emitted unexpected resolution {tuple(images.shape[-2:])}")
        if not torch.isfinite(images).all():
            raise RuntimeError("Normalized image batch contains NaN or Inf")

        with torch.no_grad():
            pooled, stage3, stage4 = model.backbone.forward_features_with_stage3(images)
        expected_shapes = {
            "stage3": [int(images.shape[0]), int(expected_stage3_channels),
                       image_size // 16, image_size // 16],
            "stage4": [int(images.shape[0]), 768, image_size // 32, image_size // 32],
        }
        if list(stage3.shape) != expected_shapes["stage3"]:
            raise RuntimeError(f"Stage3 shape {list(stage3.shape)} != {expected_shapes['stage3']}")
        if list(stage4.shape) != expected_shapes["stage4"]:
            raise RuntimeError(f"Stage4 shape {list(stage4.shape)} != {expected_shapes['stage4']}")
        if audit_forward is None:
            audit_logits = model.classifier(pooled)
            extra_checks = {}
        else:
            audit_result = audit_forward(model, images, batch)
            audit_logits = audit_result["logits"]
            extra_checks = {key: value for key, value in audit_result.items()
                            if key != "logits"}
        if audit_logits.shape != (int(images.shape[0]), 10):
            raise RuntimeError("GAP classifier output shape is not [B,10]")

        profile_layout_result = {
            "trainable_modules": [name for name, _ in train_modules],
            "frozen_eval_modules": [name for name, _ in frozen_modules],
        }
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters()
                                   if parameter.requires_grad)
        frozen_parameters = total_parameters - trainable_parameters
        trainable = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in model.named_parameters()
                  if not parameter.requires_grad]
        optimizer = (build_profile_optimizer(model, config) if optimizer_builder is None
                     else optimizer_builder(model, config))
        optimizer_ids = {id(parameter) for group in optimizer.param_groups
                         for parameter in group["params"]}
        if optimizer_ids != {id(parameter) for _, parameter in trainable}:
            raise RuntimeError("Optimizer parameter set differs from the unfrozen parameter set")

        set_profile_training_mode(model, frozen_modules)
        frozen_snapshots = {}
        for name, module in frozen_modules:
            if module.training:
                raise RuntimeError(f"Frozen module was not in eval mode: {name}")
            parameter = next(module.parameters(), None)
            if parameter is not None:
                frozen_snapshots[name] = (parameter, parameter.detach().cpu().clone())
        optimizer.zero_grad(set_to_none=True)
        study_index = batch["study_index"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        known = batch["known"].to(device, non_blocking=True)
        image_counts = batch["image_counts"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_logits = (model(images) if batch_forward is None
                            else batch_forward(model, images, batch))
            study_logits = study_logit_mean(
                image_logits.float(), study_index, batch["num_studies"], image_counts)
            per_study, eligible = study_bce_per_study(study_logits, labels, known)
            loss = per_study[eligible].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Resolution audit produced a non-finite Study BCE")
        loss.backward()
        frozen_with_grad = [name for name, parameter in frozen if parameter.grad is not None]
        grad_groups = {}
        trainable_snapshots = {}
        for module_name, module in train_modules:
            module_parameters = [parameter for parameter in module.parameters()
                                 if parameter.requires_grad]
            nonzero = [parameter for parameter in module_parameters
                       if parameter.grad is not None
                       and torch.count_nonzero(parameter.grad).item() > 0]
            grad_groups[module_name] = bool(nonzero)
            if nonzero:
                parameter = nonzero[0]
                trainable_snapshots[module_name] = (
                    parameter, parameter.detach().cpu().clone())
        if frozen_with_grad or not all(grad_groups.values()):
            raise RuntimeError(
                f"Resolution gradient audit failed: frozen_grad={frozen_with_grad[:3]}, "
                f"trainable_groups={grad_groups}")
        optimizer.step()
        frozen_unchanged = all(
            torch.equal(parameter.detach().cpu(), snapshot)
            for parameter, snapshot in frozen_snapshots.values())
        trainable_changed = {
            name: not torch.equal(parameter.detach().cpu(), snapshot)
            for name, (parameter, snapshot) in trainable_snapshots.items()
        }
        if not frozen_unchanged or not all(trainable_changed.values()):
            raise RuntimeError(
                f"Resolution update audit failed: frozen_unchanged={frozen_unchanged}, "
                f"trainable_changed={trainable_changed}")
        return {
            "resolution": image_size,
            "weight_verification": verification,
            "dataset_batch_shape": list(images.shape),
            "stage_shapes": {
                "stage3": list(stage3.shape), "stage4": list(stage4.shape),
            },
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": frozen_parameters,
            "trainable_modules": profile_layout_result["trainable_modules"],
            "frozen_eval_modules": profile_layout_result["frozen_eval_modules"],
            "trainable_module_gradients": grad_groups,
            "frozen_parameters_have_no_grad": not frozen_with_grad,
            "frozen_parameter_snapshot_unchanged": frozen_unchanged,
            "trainable_parameter_snapshot_changed": trainable_changed,
            "study_bce": float(loss.detach().float().cpu()),
            "resizepad_checks": verify_resizepad(image_size),
            "extra_checks": extra_checks,
            "trainable_parameter_names": trainable_names,
            "train_studies": len(train_data),
        }
    finally:
        from dino_data import shutdown_loader
        shutdown_loader(loader)
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def write_model_verification(run_dir: Path, payload: dict):
    save_json_atomic(run_dir / "model_verification.json", payload)
    module_rows = [
        f"| `{name}` | Trainable | {payload['trainable_module_gradients'].get(name, False)} |"
        for name in payload["trainable_modules"]
    ]
    lines = [
        f"# {run_dir.name} model and preprocessing verification", "",
        f"- Input size: {payload['resolution']}×{payload['resolution']}; Test split read: false.",
        f"- Original DINOv3 weights SHA-256: `{payload['weight_verification']['weights_sha256']}`.",
        f"- Sample Study batch tensor: `{payload['dataset_batch_shape']}`.",
        f"- Stage 3: `{payload['stage_shapes']['stage3']}`; Stage 4: `{payload['stage_shapes']['stage4']}`.",
        f"- Parameters: total {payload['total_parameters']:,}, trainable {payload['trainable_parameters']:,}, frozen {payload['frozen_parameters']:,}.",
        "", "## Trainable module gradient audit", "",
        "| Module | State | Non-zero gradient |", "|---|---|---|", *module_rows,
        "", f"- Frozen modules stayed unchanged after one optimizer step: {payload['frozen_parameter_snapshot_unchanged']}.",
        f"- Representative trainable parameters changed after one optimizer step: {all(payload['trainable_parameter_snapshot_changed'].values())}.",
        f"- Study BCE from the real Train batch: {payload['study_bce']:.6f}.",
        "", "## ResizePad checks", "",
        "| Source W×H | Canvas | Resized W×H | Padding L/T/R/B | Aspect ratio error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["resizepad_checks"]:
        w, h = row["source_size"]
        rw, rh = row["resized_size"]
        lines.append(
            f"| {w}×{h} | {row['canvas_size'][0]}×{row['canvas_size'][1]} | "
            f"{rw}×{rh} | {row['padding_ltrb']} | {row['aspect_ratio_error']:.7f} |"
        )
    if payload.get("extra_checks"):
        lines.extend(["", "## Group-specific checks", ""])
        for key, value in payload["extra_checks"].items():
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "Aspect ratio is preserved by scaling the longest side to the target and centering padding on the other axis.", ""])
    (run_dir / "model_verification.md").write_text("\n".join(lines), encoding="utf-8")


def benchmark_freeze_config(freeze: dict, phase: str, selected_budget: int | None = None):
    result = copy.deepcopy(freeze)
    benchmark = result["benchmark"]
    if phase == "screen":
        benchmark["image_budgets"] = list(benchmark["image_budgets"])
        benchmark["num_workers"] = list(benchmark["screen_workers"])
        benchmark["partial_gradient_checkpointing"] = list(
            benchmark["screen_gradient_checkpointing"])
        benchmark["channels_last"] = list(benchmark["screen_channels_last"])
    elif phase == "refine":
        if selected_budget is None:
            raise ValueError("Refinement phase requires the screened image budget")
        benchmark["image_budgets"] = [int(selected_budget)]
        benchmark["num_workers"] = list(benchmark["refine_workers"])
        benchmark["partial_gradient_checkpointing"] = list(
            benchmark["refine_gradient_checkpointing"])
        benchmark["channels_last"] = list(benchmark["refine_channels_last"])
    else:
        raise ValueError(f"Unknown benchmark phase: {phase}")
    return result


def read_selected_runtime(stage_dir: Path):
    path = stage_dir / "selected_runtime_config.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("profiles", {}).get(PROFILE)


def run_two_stage_benchmark(config, freeze, train_data, run_dir: Path,
                            model_builder=None, optimizer_builder=None,
                            batch_forward=None, collate_fn=None,
                            expected_weight_sha256=None):
    screen_dir = run_dir / "benchmark_screen"
    refine_dir = run_dir / "benchmark_refine"
    screen_runtime = read_selected_runtime(screen_dir)
    if screen_runtime is None:
        screen_freeze = benchmark_freeze_config(freeze, "screen")
        run_profile_benchmark(
            config, screen_freeze, PROFILE, train_data, screen_dir,
            model_builder=model_builder, optimizer_builder=optimizer_builder,
            batch_forward=batch_forward, collate_fn=collate_fn,
            expected_weight_sha256=expected_weight_sha256)
        screen_runtime = read_selected_runtime(screen_dir)
    if screen_runtime is None:
        raise RuntimeError("Screening benchmark did not select a stable runtime configuration")

    runtime = read_selected_runtime(refine_dir)
    if runtime is None:
        refine_freeze = benchmark_freeze_config(
            freeze, "refine", selected_budget=int(screen_runtime["image_budget"]))
        run_profile_benchmark(
            config, refine_freeze, PROFILE, train_data, refine_dir,
            model_builder=model_builder, optimizer_builder=optimizer_builder,
            batch_forward=batch_forward, collate_fn=collate_fn,
            expected_weight_sha256=expected_weight_sha256)
        runtime = read_selected_runtime(refine_dir)
    if runtime is None:
        raise RuntimeError("Refinement benchmark did not select a stable runtime configuration")

    csv_frames = []
    for label, stage_dir in (("screen", screen_dir), ("refine", refine_dir)):
        csv_path = stage_dir / f"{PROFILE}_gpu_benchmark.csv"
        if csv_path.is_file():
            frame = pd.read_csv(csv_path)
            frame.insert(0, "search_stage", label)
            csv_frames.append(frame)
    if csv_frames:
        pd.concat(csv_frames, ignore_index=True, sort=False).to_csv(
            run_dir / "gpu_benchmark.csv", index=False, encoding="utf-8")

    selected_payload = {
        "experiment": EXPERIMENT_NAME,
        "group": run_dir.name,
        "image_size": int(config["data"]["image_size"]),
        "profile": "p2",
        "screen_runtime": screen_runtime,
        "selected_runtime": runtime,
        "selection_rule": (
            "Enumerated every requested image budget with a fixed stable loader profile; "
            "then compared workers, gradient checkpointing and channels_last at the "
            "screened image budget using repeated long measurements."
        ),
    }
    save_json_atomic(run_dir / "selected_runtime_config.json", selected_payload)
    lines = [
        f"# {run_dir.name} GPU throughput benchmark", "",
        "The screen covered all planned image budgets with 4 workers, BF16, gradient checkpointing and channels_last. The refinement stage compared the worker count, checkpointing and memory format at the selected image budget. Every candidate used real Study batches, full preprocessing, Study Logit Mean, BCE, backward, optimizer step, and CUDA synchronization.",
        "", "## Selection", "",
        "| Phase | Budget | Accumulation | Workers | Checkpointing | Channels Last | Long mean images/s | CV | Peak reserved fraction |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for label, value in (("Screen", screen_runtime), ("Refine", runtime)):
        lines.append(
            f"| {label} | {value['image_budget']} | {value['grad_accumulation']} | "
            f"{value['num_workers']} | {value['gradient_checkpointing']} | "
            f"{value['channels_last']} | {value['measured_images_per_s']:.2f} | "
            f"{value['measured_images_per_s_cv']:.4f} | "
            f"{value['peak_reserved_memory_fraction']:.3f} |"
        )
    lines.extend([
        "", "## Measurement details", "",
        f"- Screen candidate rows: {len(pd.read_csv(screen_dir / f'{PROFILE}_gpu_benchmark.csv'))}.",
        f"- Refinement candidate rows: {len(pd.read_csv(refine_dir / f'{PROFILE}_gpu_benchmark.csv'))}.",
        f"- Selected runtime: budget {runtime['image_budget']}, accumulation {runtime['grad_accumulation']}, workers {runtime['num_workers']}, checkpointing {runtime['gradient_checkpointing']}, channels_last {runtime['channels_last']}.",
        "- Benchmark timing reports end-to-end time separately from GPU compute time, includes actual NAS image loading and preprocessing, and synchronizes CUDA before stopping each timer.",
        "- Formal epoch throughput is recorded in `train_log.csv` and can differ from short and repeated benchmark measurements.",
        "- Test split was not read.", "",
    ])
    (run_dir / "gpu_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    return runtime


def plot_training_curves(run_dir: Path):
    log_path = run_dir / "train_log.csv"
    if not log_path.is_file():
        return
    history = pd.read_csv(log_path)
    if history.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_bce"], label="Train BCE")
    plt.plot(history["epoch"], history["valid_loss"], label="Valid BCE")
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.title(f"{run_dir.name}: Train and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "train_valid_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["macro_auroc"], label="Macro-AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-AUROC")
    plt.title(f"{run_dir.name}: Validation Macro-AUROC")
    plt.tight_layout()
    plt.savefig(run_dir / "macro_auroc_curve.png", dpi=160)
    plt.close()


def write_group_training_report(run_dir: Path, label: str, backbone: str,
                                image_size: int, head: str):
    log_path = run_dir / "train_log.csv"
    status_path = run_dir / "training_status.json"
    if not log_path.is_file() or not status_path.is_file():
        return
    history = pd.read_csv(log_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    best = history.loc[history["epoch"] == int(status["best_epoch"])].iloc[0]
    last = history.iloc[-1]
    lines = [
        f"# {label} training report", "",
        f"- Backbone: {backbone}; input: {image_size}×{image_size}; head: {head}.",
        f"- Stop: epoch {status['completed_epochs']}/{status['max_epochs']} (`{status['stop_reason']}`).",
        f"- Best Epoch: {status['best_epoch']}; Macro-AUROC: {status['best_macro_auroc']:.6f}.",
        f"- Best Train BCE / Valid BCE: {best['train_bce']:.6f} / {best['valid_loss']:.6f}.",
        f"- Last Train BCE / Valid BCE / Macro-AUROC: {last['train_bce']:.6f} / {last['valid_loss']:.6f} / {last['macro_auroc']:.6f}.",
        f"- Mean formal train throughput: {history['images_per_s'].mean():.2f} images/s.",
        f"- Mean DataLoader wait: {history['data_loader_wait_seconds'].mean():.1f} seconds/epoch.",
        f"- Trainable / frozen parameters: {int(last['trainable_parameters']):,} / {int(last['frozen_parameters']):,}.",
        "- Study-level BCE uses the arithmetic mean of raw image logits within each Study.",
        "- Test split was not read.",
        "", "| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for _, row in history.iterrows():
        lines.append(
            f"| {int(row['epoch'])} | {row['train_bce']:.6f} | {row['valid_loss']:.6f} | "
            f"{row['macro_auroc']:.6f} | {row['macro_auprc']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['images_per_s']:.2f} | {bool(row['train_down_3'])} | "
            f"{bool(row['valid_up_3'])} | {bool(row['auc_no_improve_3'])} | "
            f"{bool(row['should_stop'])} |"
        )
    lines.append("")
    (run_dir / "training_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_resolution_report(feature, completed):
    baseline_dir = WORK_DIR / "outputs" / "experiments" / "DINOv3_P0_P1_P2" / "p2"
    baseline_predictions = baseline_dir / "predictions" / "val_best.csv"
    if not baseline_predictions.is_file():
        raise FileNotFoundError(f"Historical P2 validation predictions missing: {baseline_predictions}")
    predictions = {"R0": pd.read_csv(
        baseline_predictions, dtype={"Subject_id": "string", "Study_id": "string"})}
    statuses = {
        "R0": {
            "best_epoch": 5,
            "stop_epoch": 8,
            "macro_auroc": 0.794882408546734,
            "images_per_s": 23.58,
            "image_size": 320,
        }
    }
    for key, folder, image_size in RESOLUTIONS:
        if key not in completed:
            continue
        run_dir = OUTPUT_ROOT / folder
        prediction_path = run_dir / "predictions" / "val_best.csv"
        status_path = run_dir / "training_status.json"
        if not prediction_path.is_file() or not status_path.is_file():
            continue
        frame = pd.read_csv(prediction_path, dtype={"Subject_id": "string", "Study_id": "string"})
        status = json.loads(status_path.read_text(encoding="utf-8"))
        log = pd.read_csv(run_dir / "train_log.csv")
        best = log.loc[log["epoch"] == int(status["best_epoch"])].iloc[0]
        predictions[key.upper()] = frame
        statuses[key.upper()] = {
            "best_epoch": int(status["best_epoch"]),
            "stop_epoch": int(status["completed_epochs"]),
            "stop_reason": status["stop_reason"],
            "macro_auroc": float(status["best_macro_auroc"]),
            "images_per_s": float(log["images_per_s"].mean()),
            "image_size": int(image_size),
            "train_bce_best": float(best["train_bce"]),
            "valid_bce_best": float(best["valid_loss"]),
        }

    pairs = []
    comparison_results = {}
    target_columns, _ = _columns(None)
    key_columns = ["Subject_id", "Study_id"]
    for candidate, reference in (("R1", "R0"), ("R2", "R0"), ("R2", "R1")):
        if candidate not in predictions or reference not in predictions:
            continue
        left = predictions[reference].copy()
        right = predictions[candidate].copy()
        if left.duplicated(key_columns).any() or right.duplicated(key_columns).any():
            raise RuntimeError("Resolution comparison has duplicate validation Study keys")
        left = _rename_predictions(left, "_d0")
        right = _rename_predictions(right, "_d0c1")
        merged = left.merge(right, on=key_columns, validate="one_to_one", sort=True)
        for column in target_columns:
            if not np.array_equal(merged[f"{column}_d0"], merged[f"{column}_d0c1"]):
                raise RuntimeError(f"Validation labels differ for {candidate} in {column}")
        bootstrap = _bootstrap_subject_differences(
            merged, int(feature["comparison"]["subject_bootstrap_replicates"]),
            int(feature["comparison"]["subject_bootstrap_seed"]))
        macro_values = {
            reference: float(np.nanmean(_auc_vector(predictions[reference]))),
            candidate: float(np.nanmean(_auc_vector(predictions[candidate]))),
        }
        candidate_vector = _auc_vector(predictions[candidate])
        reference_vector = _auc_vector(predictions[reference])
        per_class = {
            name: float(candidate_vector[index] - reference_vector[index])
            for index, name in enumerate(LABEL_NAMES)
        }
        comparison_results[f"{candidate}_minus_{reference}"] = {
            "candidate": candidate,
            "reference": reference,
            "candidate_macro_auroc": macro_values[candidate],
            "reference_macro_auroc": macro_values[reference],
            "macro_auroc_delta": macro_values[candidate] - macro_values[reference],
            "macro_auroc_delta_ci95": bootstrap["macro_auroc_delta_ci95"],
            "per_class_delta": per_class,
            "per_class_delta_ci95": bootstrap["per_class_delta_ci95"],
            "valid_bootstrap_replicates": bootstrap["valid_macro_replicates"],
            "validation_studies": len(merged),
            "validation_subjects": int(merged["Subject_id"].nunique()),
        }
        for class_index, class_name in enumerate(LABEL_NAMES):
            pairs.append({
                "candidate": candidate,
                "reference": reference,
                "class_name": class_name,
                "candidate_auroc": float(candidate_vector[class_index]),
                "reference_auroc": float(reference_vector[class_index]),
                "delta": per_class[class_name],
            })

    save_json_atomic(OUTPUT_ROOT / "paired_subject_bootstrap.json", {
        "seed": int(feature["comparison"]["subject_bootstrap_seed"]),
        "requested_replicates": int(feature["comparison"]["subject_bootstrap_replicates"]),
        "comparisons": comparison_results,
        "selection_effect_caveat": (
            "Best epochs and the R3/R4 path are selected using this validation set; "
            "intervals are conditional and are not independent-test guarantees."
        ),
        "test_split_read": False,
    })
    if pairs:
        pd.DataFrame(pairs).to_csv(
            OUTPUT_ROOT / "per_class_auroc_comparison.csv", index=False, encoding="utf-8")

    lines = [
        "# DINOv3 resolution ablation: 320 vs 448 vs 512", "",
        "R0 is the completed P2 DINOv3 ConvNeXt-Tiny GAP baseline. R1 and R2 use the same raw Tiny initialization, stage-freeze policy, Study Logit Mean, BCE, seed and early-stopping criteria. Test split was not read.",
        "", "## Best checkpoints", "",
        "| Group | Resolution | Best Epoch | Stop Epoch | Stop Reason | Macro-AUROC | Formal train images/s | Train BCE at best | Valid BCE at best |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for group in ("R0", "R1", "R2"):
        if group not in statuses:
            continue
        row = statuses[group]
        lines.append(
            f"| {group} | {row['image_size']} | {row['best_epoch']} | "
            f"{row['stop_epoch']} | {row.get('stop_reason', 'three_condition_early_stop')} | "
            f"{row['macro_auroc']:.6f} | {row['images_per_s']:.2f} | "
            f"{row.get('train_bce_best', float('nan')):.6f} | "
            f"{row.get('valid_bce_best', float('nan')):.6f} |"
        )
    lines.extend(["", "## Subject-cluster paired Bootstrap", ""])
    if comparison_results:
        lines.extend([
            "| Difference | Delta Macro-AUROC | Subject bootstrap 95% CI |", "|---|---:|---:|",
        ])
        for name, row in comparison_results.items():
            low, high = row["macro_auroc_delta_ci95"]
            lines.append(
                f"| {name.replace('_minus_', ' − ')} | {row['macro_auroc_delta']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] |"
            )
    else:
        lines.append("R1/R2 paired comparisons are pending both resolution groups completing.")
    lines.extend([
        "", "## Interpretation limits", "",
        f"- Bootstrap replicates: {feature['comparison']['subject_bootstrap_replicates']}; seed: {feature['comparison']['subject_bootstrap_seed']}; validation Studies: 859.",
        "- The best epoch is selected using the same validation set. Confidence intervals are conditional and do not replace an independent Test evaluation.",
        "- Per-class resolution differences are in `per_class_auroc_comparison.csv`.",
        "- Test split was not read.", "",
    ])
    report_path = OUTPUT_ROOT / "resolution_ablation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return statuses, comparison_results


def run_resolution(feature, base_config, base_freeze, label, folder, image_size,
                   weight_verification):
    run_dir = OUTPUT_ROOT / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir, "pipeline")
    config = copy.deepcopy(base_config)
    config["data"]["image_size"] = int(image_size)
    config["outputs"]["root"] = str(run_dir)
    config["feature_experiment"] = EXPERIMENT_NAME
    freeze = copy.deepcopy(base_freeze)
    data_config_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    train_data, valid_data, data_config = build_datasets(
        data_config_path, int(feature["seed"]), image_size=int(image_size))
    if len(valid_data) != int(feature["comparison"]["expected_validation_studies"]):
        raise RuntimeError(
            f"{label} has {len(valid_data)} validation Studies, expected "
            f"{feature['comparison']['expected_validation_studies']}")
    if train_data.dataset.split != "train" or valid_data.dataset.split != "valid":
        raise RuntimeError("Only official Train and Validation splits may be used")
    if data_config.get("class_ids") != list(range(1, 11)) or data_config.get("class_names") != LABEL_NAMES:
        raise RuntimeError("The ten-label mapping differs from the completed P2 baseline")
    if set(train_data.study_keys) & set(valid_data.study_keys):
        raise RuntimeError("Train and Validation Study keys overlap")
    save_group_config(run_dir, feature, config, freeze, data_config,
                      image_size, train_data, valid_data)

    audit_path = run_dir / "model_verification.json"
    audit = None
    if audit_path.is_file():
        candidate = json.loads(audit_path.read_text(encoding="utf-8"))
        if (int(candidate.get("resolution", -1)) == int(image_size)
                and candidate.get("weight_verification", {}).get("weights_sha256")
                == weight_verification["weights_sha256"]):
            audit = candidate
    if audit is None:
        audit = verify_profile_at_resolution(
            config, train_data, int(image_size), weight_verification)
        write_model_verification(run_dir, audit)
    logger.info("%s data/model audit passed at %dx%d", label, image_size, image_size)

    selected_path = run_dir / "selected_runtime_config.json"
    runtime = None
    if selected_path.is_file():
        saved_selected = json.loads(selected_path.read_text(encoding="utf-8"))
        runtime = saved_selected.get("selected_runtime")
    if runtime is None:
        runtime = run_two_stage_benchmark(config, freeze, train_data, run_dir)
        saved_selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not runtime:
        raise RuntimeError(f"{label} has no selected benchmark runtime")

    result = train_profile(
        config, freeze, PROFILE, train_data, valid_data,
        weight_verification, runtime, run_dir,
        experiment_name=EXPERIMENT_NAME,
        run_dir_override=run_dir,
        run_context={
            "image_size": int(image_size),
            "architecture": "convnext_tiny",
            "group": label,
        })
    write_group_training_report(
        run_dir, label, "DINOv3 ConvNeXt-Tiny", image_size, "GAP + Linear(10)")
    if (run_dir / "predictions" / "val_best.csv").is_file():
        import shutil
        shutil.copy2(run_dir / "predictions" / "val_best.csv", run_dir / "val_best.csv")
    plot_training_curves(run_dir)
    logger.info(
        "Completed %s: stop=%s best_epoch=%d best_auc=%.6f",
        label, result["stop_reason"], result["best_epoch"], result["best_macro_auroc"])
    return result


def verify_r3_forward(model, images, batch):
    masks = batch["valid_masks"].to(images.device, non_blocking=True)
    result = model(images, masks, return_components=True)
    invalid_selected = result["topk_grid"] & ~result["valid_grid"][:, None]
    invalid_count = int(invalid_selected.sum().item())
    if invalid_count:
        raise RuntimeError(f"R3 Top-K selected {invalid_count} invalid padded positions")
    alpha = float(result["alpha"].detach().float().cpu())
    if not (0.0 < alpha <= model.maximum_alpha):
        raise RuntimeError(f"R3 learned fusion alpha is outside its bound: {alpha}")
    if not all(count > 0 for count in result["valid_counts"]):
        raise RuntimeError("R3 valid mask produced an empty Stage3 region")
    return {
        "logits": result["logits"],
        "initial_alpha": alpha,
        "valid_stage3_positions_min": min(result["valid_counts"]),
        "selected_topk_positions_min": min(result["selected_counts"]),
        "invalid_topk_positions": invalid_count,
        "stage3_response_shape": list(result["response_maps"].shape),
    }


def prepare_small_weights(feature):
    verification_path = OUTPUT_ROOT / "small_weight_verification.json"
    repo_id = feature["source"]["small_repo"]
    if verification_path.is_file():
        pinned = json.loads(verification_path.read_text(encoding="utf-8"))
        backbone, verification = load_small_pretrained_backbone(
            repo_id, pinned["revision"], device="cpu",
            expected_sha256=pinned["weights_sha256"], local_files_only=True)
    else:
        backbone, verification = load_small_pretrained_backbone(
            repo_id, feature["source"]["small_revision"], device="cpu")
        save_json_atomic(verification_path, verification)
    del backbone
    gc.collect()
    torch.cuda.empty_cache()
    return verification


def run_custom_group(feature, base_config, base_freeze, label, folder,
                     image_size, weight_verification, model_builder,
                     optimizer_builder=None, batch_forward=None,
                     validation_forward=None, collate_fn=None,
                     audit_forward=None, backbone_name="DINOv3 ConvNeXt-Tiny",
                     expected_stage3_channels=384, dataset_builder=None,
                     run_context=None, expected_weight_sha256=None,
                     epoch_callback=None):
    run_dir = OUTPUT_ROOT / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir, "pipeline")
    config = copy.deepcopy(base_config)
    config["data"]["image_size"] = int(image_size)
    config["outputs"]["root"] = str(run_dir)
    config["feature_experiment"] = EXPERIMENT_NAME
    freeze = copy.deepcopy(base_freeze)
    data_config_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    if dataset_builder is None:
        train_data, valid_data, data_config = build_datasets(
            data_config_path, int(feature["seed"]), image_size=int(image_size))
    else:
        train_data, valid_data, data_config = dataset_builder(
            data_config_path, int(feature["seed"]), int(image_size))
    if len(valid_data) != int(feature["comparison"]["expected_validation_studies"]):
        raise RuntimeError(
            f"{label} has {len(valid_data)} validation Studies, expected "
            f"{feature['comparison']['expected_validation_studies']}")
    if train_data.dataset.split != "train" or valid_data.dataset.split != "valid":
        raise RuntimeError("Custom feature group may only use official Train and Validation")
    if data_config.get("class_ids") != list(range(1, 11)) or data_config.get("class_names") != LABEL_NAMES:
        raise RuntimeError("Custom feature group label mapping differs from the P2 baseline")
    if set(train_data.study_keys) & set(valid_data.study_keys):
        raise RuntimeError("Train and Validation Study keys overlap")
    save_group_config(
        run_dir, feature, config, freeze, data_config, image_size,
        train_data, valid_data, backbone_name=backbone_name,
        weight_verification=weight_verification)

    audit_path = run_dir / "model_verification.json"
    audit = None
    if audit_path.is_file():
        candidate = json.loads(audit_path.read_text(encoding="utf-8"))
        if (int(candidate.get("resolution", -1)) == int(image_size)
                and candidate.get("weight_verification", {}).get("weights_sha256")
                == weight_verification["weights_sha256"]):
            audit = candidate
    if audit is None:
        audit = verify_profile_at_resolution(
            config, train_data, int(image_size), weight_verification,
            model_builder=model_builder,
            optimizer_builder=optimizer_builder,
            batch_forward=batch_forward,
            collate_fn=collate_fn,
            audit_forward=audit_forward,
            expected_stage3_channels=int(expected_stage3_channels))
        write_model_verification(run_dir, audit)
    logger.info("%s data/model audit passed at %dx%d", label, image_size, image_size)

    selected_path = run_dir / "selected_runtime_config.json"
    selected_payload = (json.loads(selected_path.read_text(encoding="utf-8"))
                       if selected_path.is_file() else None)
    runtime = selected_payload.get("selected_runtime") if selected_payload else None
    if runtime is None:
        runtime = run_two_stage_benchmark(
            config, freeze, train_data, run_dir,
            model_builder=model_builder,
            optimizer_builder=optimizer_builder,
            batch_forward=batch_forward,
            collate_fn=collate_fn,
            expected_weight_sha256=(expected_weight_sha256
                                    or weight_verification["weights_sha256"]))

    result = train_profile(
        config, freeze, PROFILE, train_data, valid_data,
        weight_verification, runtime, run_dir,
        experiment_name=EXPERIMENT_NAME,
        run_dir_override=run_dir,
        run_context=run_context,
        model_builder=model_builder,
        optimizer_builder=optimizer_builder,
        batch_forward=batch_forward,
        validation_forward=validation_forward,
        collate_fn=collate_fn,
        epoch_callback=epoch_callback)
    write_group_training_report(
        run_dir, label, backbone_name, image_size,
        "GAP + mask-aware Stage3 Top-K" if label == "R3" else "GAP + Linear(10)")
    if (run_dir / "predictions" / "val_best.csv").is_file():
        import shutil
        shutil.copy2(run_dir / "predictions" / "val_best.csv", run_dir / "val_best.csv")
    plot_training_curves(run_dir)
    logger.info("Completed %s: stop=%s best_epoch=%d best_auc=%.6f",
                label, result["stop_reason"], result["best_epoch"],
                result["best_macro_auroc"])
    return result


def select_feature_resolution(feature):
    resolution_report = OUTPUT_ROOT / "resolution_ablation.md"
    if not resolution_report.is_file():
        raise RuntimeError("R3/R4 require the completed R1/R2 paired resolution analysis")
    details = {}
    for key, folder, size in RESOLUTIONS:
        group = OUTPUT_ROOT / folder
        status_path = group / "training_status.json"
        runtime_path = group / "selected_runtime_config.json"
        if not status_path.is_file() or not runtime_path.is_file():
            raise RuntimeError(f"Cannot select R3/R4 resolution before {key.upper()} completes")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))["selected_runtime"]
        if not status.get("complete"):
            raise RuntimeError(f"Cannot select R3/R4 resolution: {key.upper()} is incomplete")
        details[key.upper()] = {
            "image_size": int(size),
            "best_macro_auroc": float(status["best_macro_auroc"]),
            "best_epoch": int(status["best_epoch"]),
            "formal_train_images_per_s": float(pd.read_csv(group / "train_log.csv")["images_per_s"].mean()),
            "selected_benchmark_images_per_s": float(runtime["measured_images_per_s"]),
            "selected_runtime": runtime,
        }
    selected = max(details, key=lambda key: details[key]["best_macro_auroc"])
    payload = {
        "selected_resolution_group": selected,
        "selected_image_size": details[selected]["image_size"],
        "selection_rule": "Choose the highest observed validation Macro-AUROC among R1/R2; report benchmark and formal epoch throughput alongside the choice.",
        "groups": details,
        "validation_selection_caveat": "Resolution and best epoch are selected on the same Validation set; paired intervals are conditional.",
        "test_split_read": False,
    }
    save_json_atomic(OUTPUT_ROOT / "r3_r4_resolution_selection.json", payload)
    return payload


def _paired_comparison(candidate_frame: pd.DataFrame, reference_frame: pd.DataFrame,
                       candidate_name: str, reference_name: str,
                       feature: dict):
    keys = ["Subject_id", "Study_id"]
    if candidate_frame.duplicated(keys).any() or reference_frame.duplicated(keys).any():
        raise RuntimeError("Paired validation predictions contain duplicate Study keys")
    left = _rename_predictions(reference_frame, "_d0")
    right = _rename_predictions(candidate_frame, "_d0c1")
    merged = left.merge(right, on=keys, validate="one_to_one", sort=True)
    target_columns, _ = _columns(None)
    for column in target_columns:
        if not np.array_equal(merged[f"{column}_d0"], merged[f"{column}_d0c1"]):
            raise RuntimeError(f"Paired validation labels differ for {candidate_name}: {column}")
    bootstrap = _bootstrap_subject_differences(
        merged, int(feature["comparison"]["subject_bootstrap_replicates"]),
        int(feature["comparison"]["subject_bootstrap_seed"]))
    candidate_auc = _auc_vector(candidate_frame)
    reference_auc = _auc_vector(reference_frame)
    candidate_macro = float(np.nanmean(candidate_auc))
    reference_macro = float(np.nanmean(reference_auc))
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "candidate_macro_auroc": candidate_macro,
        "reference_macro_auroc": reference_macro,
        "macro_auroc_delta": candidate_macro - reference_macro,
        "macro_auroc_delta_ci95": bootstrap["macro_auroc_delta_ci95"],
        "per_class_delta": {
            name: float(candidate_auc[i] - reference_auc[i])
            for i, name in enumerate(LABEL_NAMES)
        },
        "candidate_per_class_auroc": {
            name: float(candidate_auc[i]) for i, name in enumerate(LABEL_NAMES)
        },
        "reference_per_class_auroc": {
            name: float(reference_auc[i]) for i, name in enumerate(LABEL_NAMES)
        },
        "per_class_delta_ci95": bootstrap["per_class_delta_ci95"],
        "valid_bootstrap_replicates": bootstrap["valid_macro_replicates"],
        "validation_studies": int(len(merged)),
        "validation_subjects": int(merged["Subject_id"].nunique()),
    }


def record_paired_comparison(name: str, result: dict):
    path = OUTPUT_ROOT / "paired_subject_bootstrap.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "seed": 42, "requested_replicates": 2000, "comparisons": {},
        "test_split_read": False,
    }
    payload.setdefault("comparisons", {})[name] = result
    save_json_atomic(path, payload)

    rows = []
    for label, delta in result["per_class_delta"].items():
        rows.append({
            "candidate": result["candidate"],
            "reference": result["reference"],
            "class_name": label,
            "candidate_auroc": result["candidate_per_class_auroc"][label],
            "reference_auroc": result["reference_per_class_auroc"][label],
            "delta": delta,
        })
    comparison_path = OUTPUT_ROOT / "per_class_auroc_comparison.csv"
    if comparison_path.is_file():
        prior = pd.read_csv(comparison_path)
        prior = prior[~((prior["candidate"] == result["candidate"])
                        & (prior["reference"] == result["reference"]))]
        frame = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True)
    else:
        frame = pd.DataFrame(rows)
    frame.to_csv(comparison_path, index=False, encoding="utf-8")


def write_paired_report(path: Path, title: str, candidate_name: str,
                        reference_name: str, result: dict,
                        extra_lines: list[str] | None = None):
    low, high = result["macro_auroc_delta_ci95"]
    lines = [
        f"# {title}", "",
        f"- Candidate: `{candidate_name}`; reference: `{reference_name}`.",
        f"- Macro-AUROC: {result['candidate_macro_auroc']:.6f} vs {result['reference_macro_auroc']:.6f}.",
        f"- Difference: {result['macro_auroc_delta']:+.6f}; subject-bootstrap 95% CI [{low:+.6f}, {high:+.6f}].",
        f"- Paired validation: {result['validation_studies']} Studies / {result['validation_subjects']} Subjects; {result['valid_bootstrap_replicates']} valid bootstrap replicates.",
        "- Best epoch selection is based on this same Validation set; intervals are conditional and do not replace an independent Test evaluation.",
        "- Test split was not read.",
    ]
    if extra_lines:
        lines.extend(["", "## Additional measurements", "", *extra_lines])
    lines.extend(["", "## Per-class AUROC differences", "",
                  "| Class | Delta | 95% CI |", "|---|---:|---:|"])
    for label, delta in result["per_class_delta"].items():
        ci = result["per_class_delta_ci95"][label]
        if ci[0] is None:
            ci_text = "NA"
        else:
            ci_text = f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        lines.append(f"| {label} | {delta:+.6f} | {ci_text} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_r3_visualizations(run_dir: Path, feature: dict, image_size: int,
                            valid_data, weight_verification: dict, runtime: dict):
    device = torch.device("cuda")
    model, verification, _, frozen_modules, _ = build_r3_model(
        PROFILE, device, runtime, int(feature["seed"]), feature)
    checkpoint = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    if checkpoint["initial_weights_sha256"] != weight_verification["weights_sha256"]:
        raise RuntimeError("R3 visualization checkpoint has the wrong initialization")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    out_dir = run_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    selections = {}
    label_indices = {
        "Pneumonia": 3,
        "Lung Opacity": 7,
        "Pneumothorax": 1,
        "Consolidation": 2,
    }
    for label, label_index in label_indices.items():
        study_index = next((i for i, study in enumerate(valid_data.studies)
                            if float(study["target"][label_index]) == 1.0), None)
        if study_index is None:
            continue
        item = valid_data[study_index]
        images = item["images"][:1].to(device)
        masks = item["valid_masks"][:1].to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            components = model(images, masks, return_components=True)
        response = components["response_maps"][0, label_index].float()
        response = F.interpolate(
            response[None, None], size=(image_size, image_size),
            mode="bilinear", align_corners=False)[0, 0]
        valid_mask = masks[0, 0].float()
        response = response.masked_fill(valid_mask <= 0.5, 0.0)
        topk = components["topk_grid"][0, label_index].float()
        topk = F.interpolate(
            topk[None, None], size=(image_size, image_size),
            mode="nearest")[0, 0]
        response_array = response.cpu().numpy()
        valid_array = valid_mask.cpu().numpy() > 0.5
        finite_values = response_array[valid_array]
        low = float(np.min(finite_values)) if finite_values.size else 0.0
        high = float(np.max(finite_values)) if finite_values.size else 1.0
        heat = np.clip((response_array - low) / max(high - low, 1e-8), 0, 1)
        heat[~valid_array] = 0.0
        topk_array = topk.cpu().numpy() > 0.5

        image_path = valid_data.studies[study_index]["image_paths"][0]
        with Image.open(image_path) as source:
            original = ResizePad(image_size)(source.convert("RGB"))
        original = np.asarray(T.Grayscale(num_output_channels=1)(original)).squeeze()
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("ResizePad image")
        axes[1].imshow(original, cmap="gray")
        axes[1].imshow(heat, cmap="inferno", alpha=0.50, vmin=0, vmax=1)
        axes[1].set_title(f"{label} response")
        axes[2].imshow(original, cmap="gray")
        axes[2].contour(topk_array.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2)
        axes[2].set_title("Top-K valid locations")
        axes[3].imshow(valid_array, cmap="gray")
        axes[3].set_title("ResizePad valid mask")
        for axis in axes:
            axis.axis("off")
        global_logit = float(components["global_logits"][0, label_index].float().cpu())
        local_logit = float(components["local_logits"][0, label_index].float().cpu())
        fused_logit = float(components["logits"][0, label_index].float().cpu())
        alpha = float(components["alpha"].float().cpu())
        fig.suptitle(
            f"Subject {item['subject_id']} / Study {item['study_id']} | target=1 | "
            f"global={global_logit:.3f}, local={local_logit:.3f}, fused={fused_logit:.3f}, alpha={alpha:.3f}")
        fig.tight_layout()
        path = out_dir / f"{label.replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        selections[label] = {
            "subject_id": item["subject_id"],
            "study_id": item["study_id"],
            "image_path": str(image_path),
            "global_logit": global_logit,
            "local_logit": local_logit,
            "fused_logit": fused_logit,
            "alpha": alpha,
            "valid_stage3_positions": components["valid_counts"][0],
            "selected_topk_positions": components["selected_counts"][0],
            "visualization": str(path),
        }
    save_json_atomic(run_dir / "visualization_manifest.json", {
        "resolution": image_size,
        "selection_rule": "First positive Validation Study for each listed label.",
        "selections": selections,
        "test_split_read": False,
        "local_response_interpretation": (
            "Response maps visualize model activations only; no lesion localization labels were used."
        ),
    })
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return selections


def write_r3_analysis(feature, resolution_choice, r3_result):
    run_dir = OUTPUT_ROOT / "R3_Global_Local"
    reference_group = resolution_choice["selected_resolution_group"]
    reference_dir = OUTPUT_ROOT / resolution_folder(reference_group)
    candidate_frame = pd.read_csv(
        run_dir / "predictions" / "val_best.csv",
        dtype={"Subject_id": "string", "Study_id": "string"})
    reference_frame = pd.read_csv(
        reference_dir / "predictions" / "val_best.csv",
        dtype={"Subject_id": "string", "Study_id": "string"})
    result = _paired_comparison(candidate_frame, reference_frame,
                               "R3", reference_group, feature)
    record_paired_comparison(
        f"R3_minus_{reference_group}", result)
    status = json.loads((run_dir / "training_status.json").read_text(encoding="utf-8"))
    log = pd.read_csv(run_dir / "train_log.csv")
    best = log.loc[log["epoch"] == int(status["best_epoch"])].iloc[0]
    with_best = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    alpha_logit = with_best["model_state"]["alpha.logit"].float()
    alpha = float((float(feature["r3"]["maximum_alpha"]) * torch.sigmoid(alpha_logit)).item())
    r3_audit = json.loads((run_dir / "model_verification.json").read_text(encoding="utf-8"))
    baseline_audit_path = reference_dir / "model_verification.json"
    r3_parameters = int(r3_audit["total_parameters"])
    baseline_parameters = int(json.loads(baseline_audit_path.read_text(encoding="utf-8"))["total_parameters"])
    lines = [
        f"- Best Epoch: {status['best_epoch']}; stop epoch: {status['completed_epochs']} (`{status['stop_reason']}`).",
        f"- Final learned alpha: {alpha:.6f}; initial alpha: {feature['r3']['initial_alpha']:.6f}; bound: [0, {feature['r3']['maximum_alpha']:.2f}].",
        f"- Top-K ratio: {feature['r3']['topk_ratio']:.2f}; selected resolution: {resolution_choice['selected_image_size']}.",
        f"- Additional parameters versus GAP: {r3_parameters - baseline_parameters:,}.",
        f"- Mean formal train throughput: {float(log['images_per_s'].mean()):.2f} images/s.",
        f"- Validation seconds at best epoch: {float(best['validation_seconds']):.2f}.",
        f"- Global classifier initialization SHA-256 matches the R0 seed: {r3_audit['weight_verification'].get('global_classifier_init_sha256', 'not recorded')}.",
    ]
    write_paired_report(
        OUTPUT_ROOT / "r3_global_local_analysis.md",
        "R3 Global GAP plus Local Disease Response",
        "R3", reference_group, result, lines)
    return result


def write_r4_analysis(feature, resolution_choice):
    run_dir = OUTPUT_ROOT / "R4_ConvNeXt_Small"
    reference_group = resolution_choice["selected_resolution_group"]
    reference_dir = OUTPUT_ROOT / resolution_folder(reference_group)
    candidate_frame = pd.read_csv(
        run_dir / "predictions" / "val_best.csv",
        dtype={"Subject_id": "string", "Study_id": "string"})
    reference_frame = pd.read_csv(
        reference_dir / "predictions" / "val_best.csv",
        dtype={"Subject_id": "string", "Study_id": "string"})
    result = _paired_comparison(candidate_frame, reference_frame,
                               "R4", reference_group, feature)
    record_paired_comparison(
        f"R4_minus_{reference_group}", result)
    status = json.loads((run_dir / "training_status.json").read_text(encoding="utf-8"))
    log = pd.read_csv(run_dir / "train_log.csv")
    verification = json.loads((run_dir / "model_verification.json").read_text(encoding="utf-8"))
    export_meta_path = run_dir / "model_size_metrics.json"
    extra = [
        f"- Small Best Epoch: {status['best_epoch']}; stop epoch: {status['completed_epochs']} (`{status['stop_reason']}`).",
        f"- Small initialization SHA-256: `{verification['weight_verification']['weights_sha256']}`.",
        f"- Trainable parameters: {verification['trainable_parameters']:,}; total: {verification['total_parameters']:,}.",
        f"- Mean formal train throughput: {float(log['images_per_s'].mean()):.2f} images/s.",
        f"- Resolution: {resolution_choice['selected_image_size']}.",
    ]
    if export_meta_path.is_file():
        size_meta = json.loads(export_meta_path.read_text(encoding="utf-8"))
        extra.extend([
            f"- FP32 model bytes: {size_meta.get('fp32_model_bytes')}",
            f"- Inference package bytes: {size_meta.get('archive_bytes')}",
            f"- Mean inference ms/Study: {size_meta.get('mean_inference_ms_per_study')}",
            f"- Peak inference GPU memory GiB: {size_meta.get('peak_inference_gpu_gib')}",
        ])
    write_paired_report(
        OUTPUT_ROOT / "tiny_vs_small_analysis.md",
        "R4 DINOv3 ConvNeXt-Small vs matched Tiny GAP",
        "R4", reference_group, result, extra)
    return result


def write_gpu_throughput_analysis(feature):
    rows = []
    for group in ("R1_448_GAP", "R2_512_GAP", "R3_Global_Local", "R4_ConvNeXt_Small"):
        run_dir = OUTPUT_ROOT / group
        selected_path = run_dir / "selected_runtime_config.json"
        train_log_path = run_dir / "train_log.csv"
        if not selected_path.is_file() or not train_log_path.is_file():
            continue
        selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected_runtime"]
        log = pd.read_csv(train_log_path)
        group_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        experiment_data = group_config.get("data_config") or group_config.get(
            "experiment_config", {}).get("data", {})
        rows.append({
            "group": group,
            "image_size": int(selected.get("image_size", 0)
                              or experiment_data.get("image_size", 0)),
            "budget": selected["image_budget"],
            "accumulation": selected["grad_accumulation"],
            "workers": selected["num_workers"],
            "checkpointing": selected["gradient_checkpointing"],
            "channels_last": selected["channels_last"],
            "benchmark_long_mean_images_s": selected["measured_images_per_s"],
            "formal_epoch_mean_images_s": float(log["images_per_s"].mean()),
            "mean_dataloader_wait_s": float(log["data_loader_wait_seconds"].mean()),
            "peak_gpu_reserved_gib": float(log["peak_gpu_reserved_gib"].max()),
        })
    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUT_ROOT / "gpu_throughput_analysis.csv", index=False, encoding="utf-8")
    lines = [
        "# DINOv3 feature optimization throughput analysis", "",
        "Benchmark long-measurement throughput and formal epoch throughput are reported separately. Long candidates include CUDA synchronization and actual Study batches; formal epochs include every official Train Study and all of its images.",
        "", "| Group | Resolution | Budget | Accum | Workers | Checkpointing | Channels Last | Long mean images/s | Formal mean images/s | Mean loader wait (s/epoch) | Peak reserved GiB |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['image_size']} | {row['budget']} | "
            f"{row['accumulation']} | {row['workers']} | {row['checkpointing']} | "
            f"{row['channels_last']} | {row['benchmark_long_mean_images_s']:.2f} | "
            f"{row['formal_epoch_mean_images_s']:.2f} | "
            f"{row['mean_dataloader_wait_s']:.1f} | {row['peak_gpu_reserved_gib']:.2f} |"
        )
    lines.extend(["", "Per-resolution candidate timings and the reason for the selected runtime are in each group's `gpu_benchmark.md`.", ""])
    (OUTPUT_ROOT / "gpu_throughput_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8")


def write_feature_final_comparison(feature):
    rows = []
    groups = [
        ("R0", None, 320, 0.794882408546734, 23.58, "P2 baseline", 8),
        ("R1", "R1_448_GAP", 448, None, None, "", None),
        ("R2", "R2_512_GAP", 512, None, None, "", None),
        ("R3", "R3_Global_Local", None, None, None, "", None),
        ("R4", "R4_ConvNeXt_Small", None, None, None, "", None),
    ]
    for label, folder, fixed_size, fixed_auc, fixed_speed, fixed_stop, fixed_epoch in groups:
        if folder is None:
            rows.append({"group": label, "image_size": fixed_size,
                         "best_auc": fixed_auc, "mean_images_s": fixed_speed,
                         "stop_reason": fixed_stop, "stop_epoch": fixed_epoch,
                         "status": "historical baseline"})
            continue
        run_dir = OUTPUT_ROOT / folder
        status_path = run_dir / "training_status.json"
        log_path = run_dir / "train_log.csv"
        if not status_path.is_file() or not log_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        log = pd.read_csv(log_path)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        experiment_data = config.get("data_config") or config.get(
            "experiment_config", {}).get("data", {})
        rows.append({
            "group": label,
            "image_size": int(experiment_data.get("image_size", config.get("image_size", 0))),
            "best_auc": float(status["best_macro_auroc"]),
            "mean_images_s": float(log["images_per_s"].mean()),
            "stop_reason": status["stop_reason"],
            "stop_epoch": int(status["completed_epochs"]),
            "status": "complete" if status.get("complete") else "in progress",
        })
    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUT_ROOT / "feature_optimization_summary.csv", index=False, encoding="utf-8")
    lines = [
        "# DINOv3 feature optimization final comparison", "",
        "| Group | Backbone | Resolution | Head | Best Macro-AUROC | Mean train images/s | Stop Epoch | Stop reason | Status |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        backbone = "ConvNeXt-Small" if row["group"] == "R4" else "ConvNeXt-Tiny"
        head = "GAP+Local" if row["group"] == "R3" else "GAP"
        lines.append(
            f"| {row['group']} | {backbone} | {row['image_size']} | {head} | "
            f"{row['best_auc']:.6f} | {row['mean_images_s']:.2f} | "
            f"{row['stop_epoch']} | {row['stop_reason']} | {row['status']} |"
        )
    lines.extend([
        "", "## Statistical comparisons", "",
        "Paired Subject-cluster confidence intervals are in `paired_subject_bootstrap.json`; comparisons whose interval includes zero are not described as clear improvements.",
        "", "## Evaluation scope", "",
        "All reported model selection and analysis uses the official Train/Validation split. Test split was not read.", "",
    ])
    (OUTPUT_ROOT / "feature_optimization_final_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8")


def export_feature_package(feature, group: str, folder: str):
    run_dir = OUTPUT_ROOT / folder
    status = json.loads((run_dir / "training_status.json").read_text(encoding="utf-8"))
    if not status.get("complete"):
        raise RuntimeError(f"Cannot export incomplete feature group {group}")
    checkpoint = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    group_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    verification = json.loads((run_dir / "model_verification.json").read_text(encoding="utf-8"))
    experiment_config = group_config.get("experiment_config", {})
    data_config = group_config.get("data_config") or experiment_config.get("data", {})
    image_size = int(data_config.get("image_size", group_config.get("image_size", 0)))
    if image_size <= 0:
        raise RuntimeError(f"Could not resolve inference resolution for {group}")
    weight_verification = verification["weight_verification"]
    is_small = group == "R4"
    is_local = group == "R3"
    arch = "convnext_small" if is_small else "convnext_tiny"
    model_config = {
        "group": group,
        "backbone": "DINOv3 ConvNeXt-Small" if is_small else "DINOv3 ConvNeXt-Tiny",
        "architecture": arch,
        "head_type": "global_local" if is_local else "gap",
        "image_size": image_size,
        "num_labels": 10,
        "label_names": LABEL_NAMES,
        "layer_norm_eps": float(weight_verification.get("layer_norm_eps", 1e-6)),
        "layer_scale_init_value": float(weight_verification.get("layer_scale_init_value", 1e-6)),
        "drop_path_rate": float(weight_verification.get("drop_path_rate", 0.0)),
        "input_mean": weight_verification.get("input_mean", [0.485, 0.456, 0.406]),
        "input_std": weight_verification.get("input_std", [0.229, 0.224, 0.225]),
        "initialization_repository": weight_verification["repository"],
        "initialization_revision": weight_verification["revision"],
        "initialization_sha256": weight_verification["weights_sha256"],
        "best_epoch": int(status["best_epoch"]),
        "best_validation_macro_auroc": float(status["best_macro_auroc"]),
        "channels_last": bool(
            group_config.get("selected_runtime", {}).get("channels_last", False)),
        "inference_parameters": int(sum(t.numel() for t in checkpoint["model_state"].values())),
    }
    if is_local:
        model_config["r3"] = feature["r3"]

    package_root = run_dir / "submission_export"
    package_root.mkdir(parents=True, exist_ok=True)
    package_dir = package_root / f"{group}_Inference"
    package_dir.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in checkpoint["model_state"].items()
    }
    save_file(state, str(package_dir / "model.safetensors"))
    save_json_atomic(package_dir / "model_config.json", {"model": model_config})
    dependencies = [
        (WORK_DIR / "dino_feature_inference.py", "dino_feature_inference.py"),
        (WORK_DIR / "dino_feature_models.py", "dino_feature_models.py"),
        (WORK_DIR / "dino_model.py", "dino_model.py"),
        (WORK_DIR / "dinov3_meta_convnext.py", "dinov3_meta_convnext.py"),
        (WORK_DIR / "dino_data.py", "dino_data.py"),
        (WORK_DIR / "dino_losses.py", "dino_losses.py"),
        (WORK_DIR.parent / "datasets" / "__init__.py", "datasets/__init__.py"),
        (WORK_DIR.parent / "datasets" / "dataset.py", "datasets/dataset.py"),
        (WORK_DIR.parent / "datasets" / "sampler.py", "datasets/sampler.py"),
        (WORK_DIR.parent / "datasets" / "transforms.py", "datasets/transforms.py"),
        (WORK_DIR.parent / "hybrid" / "__init__.py", "hybrid/__init__.py"),
        (WORK_DIR.parent / "hybrid" / "common.py", "hybrid/common.py"),
        (WORK_DIR / "META_DINOV3_LICENSE.md", "LICENSE.md"),
    ]
    for source, relative in dependencies:
        if not source.is_file():
            raise FileNotFoundError(f"Required inference source is missing: {source}")
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (package_dir / "requirements-inference.txt").write_text(
        "torch>=2.0\ntorchvision>=0.15\nsafetensors>=0.4\nPillow>=10\n"
        "numpy>=1.24\npandas>=2.0\nPyYAML>=6\n",
        encoding="utf-8")

    if is_local:
        data_path = (WORK_DIR / experiment_config["data"]["source_config"]).resolve()
        _, valid_data, _ = build_masked_datasets(
            data_path, int(feature["seed"]), image_size)
    else:
        data_path = (WORK_DIR / experiment_config["data"]["source_config"]).resolve()
        _, valid_data, _ = build_datasets(
            data_path, int(feature["seed"]), image_size=image_size)
    predictions = pd.read_csv(
        run_dir / "predictions" / "val_best.csv",
        dtype={"Subject_id": "string", "Study_id": "string"})
    keys = ["Subject_id", "Study_id"]
    studies = valid_data.studies
    study_index = next((index for index, study in enumerate(studies)
                        if len(study["image_paths"]) >= 2), 0)
    study = studies[study_index]
    expected_row = predictions[
        (predictions["Subject_id"].astype(str) == str(study["subject_id"]))
        & (predictions["Study_id"].astype(str) == str(study["study_id"]))]
    if len(expected_row) != 1:
        raise RuntimeError(f"Could not find the multi-image validation Study for {group}")
    _, probability_columns = _columns(None)
    expected = expected_row[probability_columns].iloc[0].astype(float).to_numpy().tolist()
    payload = {
        "package_dir": str(package_dir),
        "paths": [str(path) for path in study["image_paths"]],
        "expected": expected,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    inference_code = (
        "import json,sys,numpy as np; "
        "from dino_feature_inference import FeaturePredictor; "
        "p=json.loads(sys.argv[1]); model=FeaturePredictor(p['package_dir'],device=p['device']); "
        "actual=model.predict_study_paths(p['paths']); expected=np.asarray(p['expected'],dtype=float); "
        "print(json.dumps({'shape':list(actual.shape),'max_abs_error':float(np.max(np.abs(actual-expected))),"
        "'inference_ms':float(model.last_inference_ms),'peak_gpu_gib':float(model.last_peak_gpu_gib)}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", inference_code, json.dumps(payload)],
        cwd=package_dir, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False)
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Standalone inference failed for {group}: "
            f"{diagnostic[-2000:] or f'exit code {completed.returncode}'}")
    inference_result = json.loads(completed.stdout.strip().splitlines()[-1])
    if inference_result["shape"] != [10]:
        raise RuntimeError(f"Standalone inference produced {inference_result['shape']} for {group}")
    if inference_result["max_abs_error"] > PACKAGE_INFERENCE_PROBABILITY_TOLERANCE:
        raise RuntimeError(
            f"Standalone multi-image inference differs from saved Validation predictions for {group}: "
            f"max_abs_error={inference_result['max_abs_error']}, "
            f"tolerance={PACKAGE_INFERENCE_PROBABILITY_TOLERANCE}")

    archive_path = package_root / f"{group}_Inference.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for file_path in sorted(package_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(package_dir))
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Inference archive CRC failed at {bad_member}")
        uncompressed_archive_bytes = sum(info.file_size for info in archive.infolist())
    archive_bytes = archive_path.stat().st_size
    package_bytes = sum(item.stat().st_size for item in package_dir.rglob("*")
                        if item.is_file())
    limit = int(feature["outputs"]["submission_size_limit_bytes"])
    result = {
        "group": group,
        "selected_backbone": model_config["backbone"],
        "head_type": model_config["head_type"],
        "best_epoch": model_config["best_epoch"],
        "best_macro_auroc": model_config["best_validation_macro_auroc"],
        "fp32_model_bytes": (package_dir / "model.safetensors").stat().st_size,
        "package_bytes": package_bytes,
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "archive_uncompressed_bytes": uncompressed_archive_bytes,
        "limit_bytes": limit,
        "passes_strict_limit": archive_bytes < limit and package_bytes < limit,
        "validation_subject_id": str(study["subject_id"]),
        "validation_study_id": str(study["study_id"]),
        "validation_image_count": int(len(study["image_paths"])),
        "max_probability_abs_error": inference_result["max_abs_error"],
        "validation_probability_tolerance": PACKAGE_INFERENCE_PROBABILITY_TOLERANCE,
        "standalone_inference_ms": inference_result["inference_ms"],
        "peak_inference_gpu_gib": inference_result["peak_gpu_gib"],
        "test_split_read": False,
    }
    if not result["passes_strict_limit"]:
        raise RuntimeError(f"{group} inference package exceeds the strict size limit")
    save_json_atomic(run_dir / "model_size_metrics.json", result)
    return result


def export_completed_models(feature, progress):
    export_results = {}
    group_map = {
        "r1": ("R1", "R1_448_GAP"),
        "r2": ("R2", "R2_512_GAP"),
        "r3": ("R3", "R3_Global_Local"),
        "r4": ("R4", "R4_ConvNeXt_Small"),
    }
    for key, (label, folder) in group_map.items():
        if key not in progress.get("completed_groups", {}):
            continue
        run_dir = OUTPUT_ROOT / folder
        metric_path = run_dir / "model_size_metrics.json"
        if metric_path.is_file():
            export_results[key] = json.loads(metric_path.read_text(encoding="utf-8"))
            continue
        export_results[key] = export_feature_package(feature, label, folder)
    rows = []
    for item in export_results.values():
        rows.append(item)
    lines = [
        "# DINOv3 feature optimization submission package sizes", "",
        f"- Strict limit: {feature['outputs']['submission_size_limit_bytes']:,} bytes.",
        "- Each package is FP32, stores only inference weights and required code, loads strictly, and validates a multi-image Validation Study through the standalone entrypoint.",
        "- The probability check uses a 1e-2 absolute tolerance for BF16 batch-shape numerical differences; actual maximum absolute error is reported per package.",
        "- Test split was not read.", "",
        "| Group | Backbone/head | FP32 weights bytes | ZIP bytes | Uncompressed package bytes | Passes limit | Validation images | Max probability abs error / tolerance | Mean inference ms/Study | Peak GPU GiB |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            f"| {item['group']} | {item['selected_backbone']} / {item['head_type']} | "
            f"{item['fp32_model_bytes']:,} | {item['archive_bytes']:,} | "
            f"{item['archive_uncompressed_bytes']:,} | {item['passes_strict_limit']} | "
            f"{item['validation_image_count']} | "
            f"{item['max_probability_abs_error']:.8g} / "
            f"{item['validation_probability_tolerance']:.1g} | "
            f"{item['standalone_inference_ms']:.2f} | {item['peak_inference_gpu_gib']:.3f} |"
        )
    lines.append("")
    (OUTPUT_ROOT / "submission_size_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    save_json_atomic(OUTPUT_ROOT / "submission_package_verification.json", export_results)
    return export_results


def make_r3_builder(feature):
    reference_verification = json.loads(
        (WORK_DIR / "outputs" / "experiments" / "DINOv3_P0_P1_P2"
         / "model_freeze_verification.json").read_text(encoding="utf-8"))
    expected_head_hash = reference_verification["same_initial_classifier_head_sha256"]

    def builder(profile, device, runtime, seed):
        model, verification, train_modules, frozen_modules, names = build_r3_model(
            profile, device, runtime, seed, feature)
        actual_head_hash = head_hash(model)
        if actual_head_hash != expected_head_hash:
            raise RuntimeError(
                "R3 global classifier initialization differs from the matched GAP baseline")
        verification["global_classifier_init_sha256"] = actual_head_hash
        return model, verification, train_modules, frozen_modules, names
    return builder


def choose_r3_r4_resolution(feature):
    choice = select_feature_resolution(feature)
    selection_path = OUTPUT_ROOT / "r3_r4_resolution_selection.json"
    if selection_path.is_file():
        saved_choice = json.loads(selection_path.read_text(encoding="utf-8"))
        if saved_choice == choice:
            return choice
    save_json_atomic(selection_path, choice)
    return choice


def resolution_folder(group: str) -> str:
    for key, folder, _ in RESOLUTIONS:
        if key.upper() == group:
            return folder
    raise ValueError(f"Unknown resolution group: {group}")


def make_r3_builder(feature):
    reference_path = (WORK_DIR / "outputs" / "experiments" / "DINOv3_P0_P1_P2"
                      / "model_freeze_verification.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_head_hash = reference["same_initial_classifier_head_sha256"]

    def builder(profile, device, runtime, seed):
        built = build_r3_model(profile, device, runtime, seed, feature)
        model, verification, train_modules, frozen_modules, names = built
        actual_head_hash = head_hash(model)
        if actual_head_hash != expected_head_hash:
            raise RuntimeError(
                "R3 global classifier initialization differs from the P2 GAP seed")
        verification["global_classifier_init_sha256"] = actual_head_hash
        return model, verification, train_modules, frozen_modules, names

    return builder


def run_r3_stage(feature, base_config, base_freeze, progress, resolution_choice,
                 tiny_verification):
    group = "r3"
    folder = "R3_Global_Local"
    run_dir = OUTPUT_ROOT / folder
    size = int(resolution_choice["selected_image_size"])
    config = copy.deepcopy(base_config)
    config["data"]["image_size"] = size
    config["outputs"]["root"] = str(run_dir)
    config["model"]["architecture"] = "DINOv3ConvNeXtTinyGlobalLocal"
    config["model"]["pooling"] = "Stage4 GAP plus mask-aware Stage3 Top-K disease logits"
    config["model"]["r3"] = feature["r3"]
    freeze = copy.deepcopy(base_freeze)
    builder = make_r3_builder(feature)
    result = run_custom_group(
        feature, config, freeze, "R3", folder, size, tiny_verification,
        model_builder=builder,
        optimizer_builder=feature_profile_optimizer,
        batch_forward=r3_batch_forward,
        validation_forward=r3_batch_forward,
        collate_fn=collate_masked_studies,
        audit_forward=verify_r3_forward,
        backbone_name="DINOv3 ConvNeXt-Tiny",
        expected_stage3_channels=384,
        dataset_builder=build_masked_datasets,
        run_context={
            "group": "R3_Global_Local",
            "architecture": "dinov3_tiny_gap_plus_masked_stage3_topk",
            "image_size": size,
            "topk_ratio": float(feature["r3"]["topk_ratio"]),
            "initial_alpha": float(feature["r3"]["initial_alpha"]),
            "maximum_alpha": float(feature["r3"]["maximum_alpha"]),
            "mask_valid_fraction_threshold": float(
                feature["r3"]["mask_valid_fraction_threshold"]),
        },
        expected_weight_sha256=tiny_verification["weights_sha256"],
    )
    progress.setdefault("completed_groups", {})[group] = result
    progress["next_group"] = "r4_small"
    write_progress(progress)
    data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    _, valid_data, _ = build_masked_datasets(data_path, int(feature["seed"]), size)
    runtime = json.loads(
        (run_dir / "selected_runtime_config.json").read_text(encoding="utf-8"))[
            "selected_runtime"]
    visualizations = write_r3_visualizations(
        run_dir, feature, size, valid_data, tiny_verification, runtime)
    comparison = write_r3_analysis(feature, resolution_choice, result)
    progress["r3_analysis_complete"] = True
    save_json_atomic(OUTPUT_ROOT / "r3_visualization_manifest.json", visualizations)
    write_progress(progress)
    return result, comparison


def small_model_builder(feature, verification):
    def builder(profile, device, runtime, seed):
        return build_small_model(
            profile, device, runtime, seed, feature,
            pinned_verification=verification)
    return builder


def run_r4_stage(feature, base_config, base_freeze, progress,
                 resolution_choice, limitation_dir):
    group = "r4"
    folder = "R4_ConvNeXt_Small"
    run_dir = OUTPUT_ROOT / folder
    size = int(resolution_choice["selected_image_size"])
    try:
        small_verification = prepare_small_weights(feature)
    except Exception as exc:
        limitation_dir.mkdir(parents=True, exist_ok=True)
        save_json_atomic(limitation_dir / "r4_small_weight_access.json", {
            "group": "R4",
            "repository": feature["source"]["small_repo"],
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "resolution": size,
            "test_split_read": False,
        })
        progress["r4_status"] = "stopped_weight_access_or_verification_failure"
        progress["next_group"] = "r4_small"
        write_progress(progress)
        return None

    config = copy.deepcopy(base_config)
    config["data"]["image_size"] = size
    config["outputs"]["root"] = str(run_dir)
    config["source"]["huggingface_repo"] = feature["source"]["small_repo"]
    config["source"]["revision"] = small_verification["revision"]
    config["source"]["weights_sha256"] = small_verification["weights_sha256"]
    config["model"].update({
        "architecture": "DINOv3ConvNeXtSmall",
        "depths": [3, 3, 27, 3],
        "dims": [96, 192, 384, 768],
        "pooling": "global_average_pool_then_official_layer_norm",
    })
    freeze = copy.deepcopy(base_freeze)
    builder = small_model_builder(feature, small_verification)
    try:
        result = run_custom_group(
            feature, config, freeze, "R4", folder, size, small_verification,
            model_builder=builder,
            optimizer_builder=build_profile_optimizer,
            backbone_name="DINOv3 ConvNeXt-Small",
            expected_stage3_channels=384,
            run_context={
                "group": "R4_ConvNeXt_Small",
                "architecture": "convnext_small",
                "image_size": size,
                "repository": small_verification["repository"],
                "revision": small_verification["revision"],
                "weights_sha256": small_verification["weights_sha256"],
            },
            expected_weight_sha256=small_verification["weights_sha256"],
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        limitation_dir.mkdir(parents=True, exist_ok=True)
        save_json_atomic(limitation_dir / "r4_small_training_limit.json", {
            "group": "R4",
            "resolution": size,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "small_weights": small_verification,
            "test_split_read": False,
        })
        progress["r4_status"] = "stopped_after_recorded_runtime_limit"
        progress["next_group"] = "r4_small"
        write_progress(progress)
        return None
    progress.setdefault("completed_groups", {})[group] = result
    progress["r4_status"] = "complete"
    progress["next_group"] = None
    write_progress(progress)
    resolution_choice["small_weight_verification"] = small_verification
    comparison = write_r4_analysis(feature, resolution_choice)
    return result, comparison


def write_progress(payload):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_json_atomic(OUTPUT_ROOT / "experiment_progress.json", payload)


def run_all():
    feature, config, freeze = load_feature_config()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pipeline_logger = setup_logging(OUTPUT_ROOT, "pipeline")
    pipeline_logger.info("Starting or resuming %s", EXPERIMENT_NAME)
    environment = environment_report()
    save_json_atomic(OUTPUT_ROOT / "environment.json", environment)
    backbone, weight_verification = load_pretrained_backbone("cpu")
    del backbone
    gc.collect()
    save_json_atomic(OUTPUT_ROOT / "tiny_weight_verification.json", weight_verification)

    progress = {
        "experiment": EXPERIMENT_NAME,
        "completed_groups": {},
        "next_group": "r1",
        "complete": False,
        "test_split_read": False,
    }
    progress_path = OUTPUT_ROOT / "experiment_progress.json"
    if progress_path.is_file():
        old = json.loads(progress_path.read_text(encoding="utf-8"))
        if old.get("experiment") != EXPERIMENT_NAME or old.get("test_split_read"):
            raise RuntimeError("Feature-optimization progress identity mismatch")
        progress.update(old)

    for key, folder, image_size in RESOLUTIONS:
        completed = progress.get("completed_groups", {})
        run_dir = OUTPUT_ROOT / folder
        status_path = run_dir / "training_status.json"
        result_path = run_dir / "training_result.json"
        if key in completed and status_path.is_file() and result_path.is_file():
            saved_status = json.loads(status_path.read_text(encoding="utf-8"))
            if saved_status.get("complete"):
                pipeline_logger.info("Skipping completed %s", key.upper())
                continue
        try:
            result = run_resolution(
                feature, config, freeze, key.upper(), folder, image_size,
                weight_verification)
        except torch.cuda.OutOfMemoryError as exc:
            limitation_dir = OUTPUT_ROOT / "technical_limitations"
            limitation_dir.mkdir(parents=True, exist_ok=True)
            save_json_atomic(limitation_dir / f"{key}_cuda_oom.json", {
                "group": key.upper(),
                "image_size": int(image_size),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "test_split_read": False,
            })
            progress["blocked_at"] = key
            progress["next_group"] = key
            write_progress(progress)
            pipeline_logger.exception("Stopping at %s after CUDA OOM", key.upper())
            raise
        except RuntimeError as exc:
            limitation_dir = OUTPUT_ROOT / "technical_limitations"
            limitation_dir.mkdir(parents=True, exist_ok=True)
            save_json_atomic(limitation_dir / f"{key}_runtime_failure.json", {
                "group": key.upper(),
                "image_size": int(image_size),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "test_split_read": False,
            })
            progress["blocked_at"] = key
            progress["next_group"] = key
            write_progress(progress)
            pipeline_logger.exception("Stopping at %s after runtime failure", key.upper())
            raise
        progress.setdefault("completed_groups", {})[key] = result
        remaining = [item[0] for item in RESOLUTIONS
                     if item[0] not in progress["completed_groups"]]
        progress["next_group"] = remaining[0] if remaining else None
        write_progress(progress)
        if result["stop_reason"] is None:
            raise RuntimeError(f"{key.upper()} returned without a valid stop reason")

    if progress.get("resolution_analysis_complete"):
        required_outputs = (
            OUTPUT_ROOT / "resolution_ablation.md",
            OUTPUT_ROOT / "paired_subject_bootstrap.json",
            OUTPUT_ROOT / "per_class_auroc_comparison.csv",
        )
        missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
        if missing_outputs:
            raise FileNotFoundError(
                f"Completed resolution analysis is missing outputs: {missing_outputs}")
        paired_payload = json.loads(
            (OUTPUT_ROOT / "paired_subject_bootstrap.json").read_text(encoding="utf-8"))
        required_comparisons = {"R1_minus_R0", "R2_minus_R0", "R2_minus_R1"}
        missing_comparisons = required_comparisons - set(
            paired_payload.get("comparisons", {}))
        if missing_comparisons:
            raise RuntimeError(
                f"Completed resolution analysis is missing comparisons: "
                f"{sorted(missing_comparisons)}")
        statuses = {}
        resolution_comparisons = paired_payload["comparisons"]
        pipeline_logger.info("Reusing completed R1/R2 paired resolution analysis")
    else:
        statuses, resolution_comparisons = write_resolution_report(
            feature, progress.get("completed_groups", {}))
        progress["resolution_analysis_complete"] = True
        write_progress(progress)
    resolution_choice = choose_r3_r4_resolution(feature)
    progress["selected_resolution"] = resolution_choice["selected_image_size"]
    progress["selected_resolution_group"] = resolution_choice["selected_resolution_group"]
    progress["next_group"] = "r3_global_local"
    write_progress(progress)

    r3_result = None
    try:
        r3_result = run_r3_stage(
            feature, config, freeze, progress, resolution_choice,
            weight_verification)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        limitation_dir = OUTPUT_ROOT / "technical_limitations"
        limitation_dir.mkdir(parents=True, exist_ok=True)
        save_json_atomic(limitation_dir / "r3_runtime_limit.json", {
            "group": "R3_Global_Local",
            "image_size": int(resolution_choice["selected_image_size"]),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "test_split_read": False,
        })
        progress["r3_status"] = "stopped_after_recorded_runtime_limit"
        progress["next_group"] = "r4_small"
        write_progress(progress)
        pipeline_logger.exception("R3 stopped after a recorded runtime limit")

    r4_result = None
    try:
        r4_result = run_r4_stage(
            feature, config, freeze, progress, resolution_choice,
            OUTPUT_ROOT / "technical_limitations")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        limitation_dir = OUTPUT_ROOT / "technical_limitations"
        limitation_dir.mkdir(parents=True, exist_ok=True)
        save_json_atomic(limitation_dir / "r4_runtime_limit.json", {
            "group": "R4_ConvNeXt_Small",
            "image_size": int(resolution_choice["selected_image_size"]),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "test_split_read": False,
        })
        progress["r4_status"] = "stopped_after_recorded_runtime_limit"
        progress["next_group"] = "final_analysis"
        write_progress(progress)
        pipeline_logger.exception("R4 stopped after a recorded runtime limit")

    write_gpu_throughput_analysis(feature)
    write_feature_final_comparison(feature)
    required_groups_complete = (
        bool(progress.get("completed_groups", {}).get("r3"))
        and bool(progress.get("completed_groups", {}).get("r4")))
    progress["complete"] = False
    progress["next_group"] = "submission_package_export"
    progress["test_split_read"] = False
    write_progress(progress)
    package_results = export_completed_models(feature, progress)
    packages_complete = all(
        item.get("passes_strict_limit", False) for item in package_results.values())
    progress["submission_packages_complete"] = packages_complete
    progress["submission_packages"] = package_results
    progress["complete"] = required_groups_complete and packages_complete
    progress["next_group"] = None if progress["complete"] else "submission_package_export"
    write_progress(progress)
    pipeline_logger.info(
        "Feature optimization stages: R1/R2 complete, R3=%s, R4=%s",
        "complete" if r3_result else progress.get("r3_status", "pending"),
        "complete" if r4_result else progress.get("r4_status", "pending"))
    return {
        "progress": progress,
        "resolution_analysis": statuses,
        "resolution_comparisons": resolution_comparisons,
        "r3_result": r3_result,
        "r4_result": r4_result,
        "submission_packages": package_results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.parse_args()
    run_all()


if __name__ == "__main__":
    main()
