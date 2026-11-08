from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[3]
PROJECT_DIR = WORK_DIR.parent
REPO_DIR = PROJECT_DIR.parent
DIAGNOSTICS_DIR = WORK_DIR / "outputs/experiments/DINOv3_Feature_Diagnostics"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(DIAGNOSTICS_DIR))

from dino_data import EpochTaggedStudyDataset, make_loader, shutdown_loader
from dino_experiment import load_pretrained_backbone, save_json_atomic
from dino_analysis import LABEL_NAMES
from dino_feature_models import (
    MaskedStudyDataset,
    build_masked_datasets,
    collate_masked_studies,
    feature_profile_optimizer,
)
from dino_feature_inference import FeaturePredictor
from dino_feature_optimization import (
    export_feature_package,
    load_feature_config,
    run_custom_group,
)
from dino_freeze_experiment import head_hash, profile_layout, set_profile_training_mode
from dino_losses import study_bce_per_study, study_logit_mean
from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier, NUM_LABELS
from hybrid.common import seed_everything
import run_diagnostics as diagnostics


R3_DIR = WORK_DIR / "outputs/experiments/DINOv3_Feature_Optimization/R3_Global_Local"
R3_RUNTIME_PATH = R3_DIR / "selected_runtime_config.json"
R3_VERIFICATION_PATH = R3_DIR / "model_verification.json"
GROUP = "R5_Global_Only_512"
IMAGE_SIZE = 512
PROFILE = "p2"
SEED = 42
EXPECTED_BACKBONE_SHA256 = "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def runtime_from_r3():
    if not R3_RUNTIME_PATH.is_file() or not R3_VERIFICATION_PATH.is_file():
        raise FileNotFoundError("R3 runtime and model verification records are required")
    r3_config = read_json(R3_DIR / "config.json")
    r3_status = read_json(R3_DIR / "training_status.json")
    r3_log = __import__("pandas").read_csv(R3_DIR / "train_log.csv")
    runtime_payload = read_json(R3_RUNTIME_PATH)
    runtime = runtime_payload["selected_runtime"]
    if int(r3_config["image_size"]) != IMAGE_SIZE:
        raise RuntimeError("R3 is not the required 512x512 reference")
    if int(r3_config["experiment_config"]["seed"]) != SEED:
        raise RuntimeError("R3 training seed differs from the requested R5 seed")
    if not r3_status.get("complete") or int(r3_status["best_epoch"]) != 6:
        raise RuntimeError(f"Unexpected R3 status: {r3_status}")
    expected_runtime = {
        "image_budget": 8,
        "grad_accumulation": 4,
        "gradient_checkpointing": False,
        "channels_last": True,
    }
    runtime_mismatch = {
        key: (runtime.get(key), expected)
        for key, expected in expected_runtime.items()
        if runtime.get(key) != expected
    }
    if runtime_mismatch:
        raise RuntimeError(f"R3 selected runtime differs from the experiment plan: {runtime_mismatch}")
    cumulative_updates = int(r3_log.loc[r3_log["epoch"] <= 6, "optimizer_updates"].sum())
    if cumulative_updates != 6037:
        raise RuntimeError(f"R3 cumulative updates through Best Epoch changed: {cumulative_updates}")
    if r3_config["initialization_sha256"] != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError("R3 backbone initialization SHA-256 differs from the pinned Tiny weights")
    head_sha = read_json(R3_VERIFICATION_PATH)["weight_verification"][
        "global_classifier_init_sha256"]
    if head_sha != "40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9":
        raise RuntimeError(f"Unexpected R3 initial Global Head SHA-256: {head_sha}")
    return r3_config, r3_status, r3_log, runtime_payload, runtime, head_sha


def make_r5_builder(runtime: dict, expected_head_sha: str):
    def builder(profile, device, _runtime, seed):
        if profile != PROFILE or int(seed) != SEED:
            raise RuntimeError(f"R5 model must use profile={PROFILE}, seed={SEED}")
        seed_everything(int(seed))
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_BACKBONE_SHA256:
            raise RuntimeError("R5 did not load the pinned official DINOv3 Tiny weights")
        model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS).to(device)
        trainable_modules, frozen_modules, trainable_names = profile_layout(model, profile)

        # Use the selected R3 runtime even during the profile audit.
        model.backbone.set_gradient_checkpointing(
            bool(runtime["gradient_checkpointing"]))
        if runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)

        actual_head_sha = head_hash(model)
        if actual_head_sha != expected_head_sha:
            raise RuntimeError(
                "R5 initial Global Head differs from the recorded R3 Global Head: "
                f"expected={expected_head_sha}, actual={actual_head_sha}"
            )
        verification["global_classifier_init_sha256"] = actual_head_sha
        verification["model_variant"] = "DINOv3 ConvNeXt-Tiny Global-only"
        verification["loaded_finetuned_checkpoint"] = False
        verification["training_runtime"] = {
            key: runtime[key] for key in (
                "image_budget", "grad_accumulation", "gradient_checkpointing",
                "channels_last", "num_workers", "pin_memory", "prefetch_factor",
                "effective_images_per_update", "effective_studies_per_update",
            )
        }

        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        frozen = total - trainable
        if (total, trainable, frozen) != (27827818, 26592778, 1235040):
            raise RuntimeError(
                f"Unexpected Global-only parameter counts: {total}/{trainable}/{frozen}")
        if any("local_classifier" in name or name.startswith("alpha.")
               for name, _ in model.named_parameters()):
            raise RuntimeError("R5 model unexpectedly contains Local Head or alpha parameters")

        lines = [
            "# R5 initialization verification", "",
            "- Initial weights: official general-vision DINOv3 ConvNeXt-Tiny, loaded from the local pinned revision.",
            f"- Repository / revision: `{verification['repository']}` / `{verification['revision']}`.",
            f"- Pretrained weight SHA-256: `{verification['weights_sha256']}`.",
            "- No R2/R3 supervised-finetuned checkpoint was used to initialize the backbone.",
            f"- Seed: {seed}; Global Linear head initialized by the same construction order as R3.",
            f"- Initial Global Head SHA-256: `{actual_head_sha}`.",
            f"- R3 recorded initial Global Head SHA-256: `{expected_head_sha}`.",
            f"- Hash match: {actual_head_sha == expected_head_sha}.",
            "- Model contains only the DINOv3 backbone and Global Linear(768,10) head; Local Head and alpha are absent.",
            f"- Parameters total/trainable/frozen: {total:,}/{trainable:,}/{frozen:,}.",
            "",
        ]
        write_markdown(HERE / "initialization_verification.md", lines)
        return model, verification, trainable_modules, frozen_modules, trainable_names
    return builder


def r5_batch_forward(model, images, _batch):
    return model(images)


def r5_audit_forward(model, images, batch):
    images = images.contiguous(memory_format=torch.channels_last)
    logits = model(images)
    study_index = batch["study_index"].to(images.device)
    image_counts = batch["image_counts"].to(images.device)
    study_logits = study_logit_mean(
        logits.float(), study_index, batch["num_studies"], image_counts)
    manual = torch.stack([
        logits[study_index == index].float().mean(dim=0)
        for index in range(int(batch["num_studies"]))
    ])
    maximum_error = float((study_logits - manual).abs().max().detach().cpu())
    if maximum_error > 1e-6:
        raise RuntimeError(f"Study Logit Mean mismatch in R5 audit: {maximum_error}")
    return {"logits": logits, "study_mean_max_abs_error": maximum_error}


def run_actual_runtime_preflight(runtime, builder, training_config, freeze_config):
    result_path = HERE / "actual_runtime_preflight.json"
    if result_path.is_file():
        previous = read_json(result_path)
        if (previous.get("passed") and previous.get("runtime") == runtime
                and previous.get("train_studies") == 18513
                and previous.get("train_images") == 30344):
            return previous

    data_path = (PROJECT_DIR / "configs" / "hybrid_data.yaml").resolve()
    train_data, valid_data, data_config = build_masked_datasets(
        data_path, SEED, IMAGE_SIZE)
    if len(train_data) != 18513 or int(sum(train_data.image_counts)) != 30344:
        raise RuntimeError("R5 Train dataset differs from the R3 Study/image counts")
    if len(valid_data) != 859 or int(sum(valid_data.image_counts)) != 1422:
        raise RuntimeError("R5 Validation dataset differs from the official 859/1422 counts")
    if data_config["label_semantics"] != "closed_world":
        raise RuntimeError("R5 must preserve closed-world label semantics")
    if set(train_data.study_keys) & set(valid_data.study_keys):
        raise RuntimeError("Train and Validation Study keys overlap")

    loader, sampler, _ = make_loader(
        train_data,
        image_budget=int(runtime["image_budget"]),
        shuffle=True,
        seed=SEED,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies,
    )
    sampler.set_epoch(0)
    device = torch.device("cuda")
    model, _, train_modules, frozen_modules, _ = builder(
        PROFILE, device, runtime, SEED)
    optimizer = feature_profile_optimizer(model, {
        "training": training_config,
        "freeze": freeze_config,
    })
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters()
              if not parameter.requires_grad]
    frozen_before = {name: parameter.detach().cpu().clone()
                     for name, parameter in frozen}
    optimizer.zero_grad(set_to_none=True)
    set_profile_training_mode(model, frozen_modules)
    window = []
    try:
        iterator = iter(loader)
        for _ in range(int(runtime["grad_accumulation"])):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if len(window) != int(runtime["grad_accumulation"]):
            raise RuntimeError(f"Preflight received only {len(window)} microbatches")
        eligible_window = sum(int(batch["known"].any(dim=1).sum()) for batch in window)
        if eligible_window <= 0:
            raise RuntimeError("R5 preflight accumulation window has no eligible Studies")
        total_images = total_studies = 0
        consistency_errors = []
        for batch in window:
            images = batch["images"].to(device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
            study_index = batch["study_index"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            known = batch["known"].to(device, non_blocking=True)
            image_counts = batch["image_counts"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(images)
                study_logits = study_logit_mean(
                    logits.float(), study_index, batch["num_studies"], image_counts)
                manual = torch.stack([
                    logits[study_index == index].float().mean(dim=0)
                    for index in range(int(batch["num_studies"]))
                ])
                error = float((study_logits - manual).abs().max().detach().cpu())
                per_study, eligible = study_bce_per_study(study_logits, labels, known)
                loss = per_study[eligible].sum() / eligible_window
            if error > 1e-6 or not torch.isfinite(loss):
                raise RuntimeError(f"R5 preflight loss/fusion check failed: {error}")
            consistency_errors.append(error)
            loss.backward()
            total_images += int(images.shape[0])
            total_studies += int(batch["num_studies"])

        gradient_groups = {}
        for name, module in train_modules:
            nonzero = [parameter for parameter in module.parameters()
                       if parameter.requires_grad and parameter.grad is not None
                       and torch.count_nonzero(parameter.grad).item() > 0]
            gradient_groups[name] = bool(nonzero)
        frozen_with_grad = [name for name, parameter in frozen if parameter.grad is not None]
        if not all(gradient_groups.values()) or frozen_with_grad:
            raise RuntimeError(
                f"R5 preflight gradients invalid: groups={gradient_groups}, "
                f"frozen_gradients={frozen_with_grad[:10]}")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            trainable, float(training_config["gradient_clip_norm"])
        ).detach().cpu())
        frozen_unchanged = all(
            torch.equal(frozen_before[name], parameter.detach().cpu())
            for name, parameter in frozen)
        if not frozen_unchanged:
            raise RuntimeError("R5 preflight changed frozen parameters without an optimizer step")
        learning_rates = {group.get("name", str(index)): float(group["lr"])
                          for index, group in enumerate(optimizer.param_groups)}
        expected_updates_per_epoch = math.ceil(
            len(sampler) / int(runtime["grad_accumulation"]))
        result = {
            "passed": True,
            "train_studies": len(train_data),
            "train_images": int(sum(train_data.image_counts)),
            "validation_studies": len(valid_data),
            "validation_images": int(sum(valid_data.image_counts)),
            "image_size": IMAGE_SIZE,
            "runtime": runtime,
            "preflight_microbatches": len(window),
            "preflight_images": total_images,
            "preflight_studies": total_studies,
            "optimizer_updates_per_epoch_from_sampler": expected_updates_per_epoch,
            "optimizer_step_executed_in_this_preflight": False,
            "study_logit_mean_max_abs_error": max(consistency_errors),
            "study_bce_finite": True,
            "trainable_gradient_groups_nonzero": gradient_groups,
            "frozen_gradients_absent": not frozen_with_grad,
            "frozen_parameters_unchanged": frozen_unchanged,
            "global_gradient_norm_before_clip": gradient_norm,
            "optimizer_learning_rates": learning_rates,
            "channels_last": True,
            "test_data_accessed": False,
        }
        save_json_atomic(result_path, result)
        write_markdown(HERE / "actual_runtime_preflight.md", [
            "# R5 actual-runtime forward/backward preflight", "",
            f"- R3 runtime reused: image budget {runtime['image_budget']}, accumulation {runtime['grad_accumulation']}, channels-last {runtime['channels_last']}, checkpointing {runtime['gradient_checkpointing']}.",
            f"- First accumulation window: {len(window)} microbatches, {total_images} images / {total_studies} Studies.",
            f"- Expected optimizer updates per full epoch from Study sampler: {expected_updates_per_epoch}.",
            f"- Study Logit Mean max absolute difference against explicit per-Study mean: {max(consistency_errors):.3g}.",
            f"- Trainable groups have nonzero gradients: {all(gradient_groups.values())}; frozen modules have no gradients: {not frozen_with_grad}.",
            f"- Frozen weights unchanged: {frozen_unchanged}; no optimizer step was executed in this preflight.",
            f"- Learning rates checked: `{learning_rates}`.",
            "- This disposable backward preflight is excluded from the formal R5 epoch and update counts.",
            "- Test split was not read.", "",
        ])
        return result
    finally:
        shutdown_loader(loader)
        optimizer.zero_grad(set_to_none=True)
        del model, optimizer, train_data, valid_data
        gc.collect()
        torch.cuda.empty_cache()


def save_global_only_inference(run_dir: Path, feature: dict):
    best_path = run_dir / "best.pth"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "best_epoch" not in checkpoint:
        raise RuntimeError("Best checkpoint is missing inference weights or epoch metadata")
    status = read_json(run_dir / "training_status.json")
    if (int(checkpoint["best_epoch"]) != int(status["best_epoch"])
            or abs(float(checkpoint["best_macro_auroc"])
                   - float(status["best_macro_auroc"])) > 1e-12):
        raise RuntimeError("Best checkpoint and completed training status disagree")
    state = checkpoint["model_state"]
    unexpected = [name for name in state if "local_classifier" in name or name.startswith("alpha.")]
    if unexpected:
        raise RuntimeError(f"Global-only checkpoint contains Local/alpha tensors: {unexpected[:5]}")
    expected = {name for name in state if name.startswith("backbone.") or name.startswith("classifier.")}
    if expected != set(state):
        raise RuntimeError("Global-only checkpoint contains unrecognized parameters")
    weights_path = run_dir / "global_only_inference.pth"
    temporary = weights_path.with_suffix(".pth.tmp")
    torch.save({name: tensor.detach().cpu().contiguous() for name, tensor in state.items()},
               temporary)
    os.replace(temporary, weights_path)
    strict_model = DINOv3StudyClassifier(
        DINOv3ConvNeXtBackbone(
            layer_norm_eps=1e-6,
            layer_scale_init_value=1e-6,
            drop_path_rate=0.0,
            architecture="convnext_tiny",
        ),
        num_labels=NUM_LABELS,
    )
    strict_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    load_result = strict_model.load_state_dict(strict_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"R5 global-only inference state failed strict load: {load_result}")
    del strict_model, strict_state
    gc.collect()
    config = {
        "group": GROUP,
        "head_type": "gap",
        "backbone": "DINOv3 ConvNeXt-Tiny",
        "architecture": "convnext_tiny",
        "image_size": IMAGE_SIZE,
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "study_fusion": "mean raw image logits before sigmoid",
        "channels_last": True,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_macro_auroc": float(checkpoint["best_macro_auroc"]),
        "pretrained_initialization_sha256": EXPECTED_BACKBONE_SHA256,
        "global_classifier_init_sha256": read_json(run_dir / "model_verification.json")[
            "weight_verification"]["global_classifier_init_sha256"],
        "state_tensor_count": len(state),
        "state_parameter_count": int(sum(tensor.numel() for tensor in state.values())),
        "contains_optimizer_or_scheduler": False,
        "strict_state_dict_load_passed": True,
    }
    save_json_atomic(run_dir / "global_only_inference_config.json", config)
    del checkpoint, state
    return weights_path


def validate_r5_inference_package(run_dir: Path, runtime: dict):
    package_dir = run_dir / "submission_export" / "R5_Inference"
    predictor = FeaturePredictor(package_dir, device="cuda")
    model_config = predictor.config["model"]
    if model_config["head_type"] != "gap" or bool(model_config["channels_last"]) != bool(runtime["channels_last"]):
        raise RuntimeError("R5 inference package metadata does not match the trained Global-only runtime")
    if "r3" in model_config:
        raise RuntimeError("R5 inference package unexpectedly contains Local branch configuration")

    valid_base, official, labels = diagnostics.official_dataset()
    expected_frame, expected_scores = diagnostics.read_predictions(
        run_dir / "predictions" / "val_best.csv", official, labels)
    dataset = EpochTaggedStudyDataset(
        MaskedStudyDataset(valid_base, IMAGE_SIZE, train=False), SEED)
    loader, _, _ = make_loader(
        dataset,
        image_budget=int(runtime["image_budget"]),
        shuffle=False,
        seed=SEED,
        num_workers=int(runtime["num_workers"]),
        pin_memory=True,
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies,
    )
    model = predictor.model.eval()
    rows = {}
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch["images"].to("cuda", non_blocking=True)
                if predictor.channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=torch.cuda.is_bf16_supported(),
                ):
                    image_logits = model(images)
                study_logits = study_logit_mean(
                    image_logits.float(),
                    batch["study_index"].to("cuda", non_blocking=True),
                    batch["num_studies"],
                    batch["image_counts"].to("cuda", non_blocking=True),
                )
                probs = torch.sigmoid(study_logits.float()).cpu().numpy()
                for index, key in enumerate(zip(batch["subject_id"], batch["study_id"])):
                    rows[(str(key[0]), str(key[1]))] = probs[index]
    finally:
        shutdown_loader(loader)

    ordered_keys = list(zip(official["Subject_id"].astype(str),
                            official["Study_id"].astype(str)))
    if set(rows) != set(ordered_keys) or len(rows) != len(ordered_keys):
        raise RuntimeError("R5 inference package did not produce exactly one probability vector per Validation Study")
    reproduced = np.asarray([rows[key] for key in ordered_keys], dtype=np.float64)
    replay = diagnostics.validate_checkpoint_reproduction(
        "R5", expected_scores, reproduced, labels)
    if (replay["macro_auroc_abs_error"] > 1e-5
            or replay["max_probability_abs_error"] > 0.02):
        raise RuntimeError(f"R5 package failed full Validation metric replay: {replay}")
    result = {
        "passed": True,
        "validation_studies": len(official),
        "validation_images": int(sum(valid_base.image_counts)),
        "head_type": model_config["head_type"],
        "channels_last": predictor.channels_last,
        "checkpoint_replay": replay,
        "test_data_accessed": False,
    }
    save_json_atomic(run_dir / "validation_inference_replay.json", result)
    print(f"R5 standalone Validation replay passed: "
          f"Macro-AUROC={replay['reproduced_macro_auroc']:.6f}, "
          f"max |Δp|={replay['max_probability_abs_error']:.8g}", flush=True)
    return result


def main():
    feature, config, freeze = load_feature_config()
    r3_config, r3_status, r3_log, r3_runtime_payload, runtime, head_sha = runtime_from_r3()
    if EXPECTED_BACKBONE_SHA256 != feature["source"]["tiny_sha256"]:
        raise RuntimeError("R5 feature config Tiny SHA does not match pinned R3 source")
    verified_backbone, weight_verification = load_pretrained_backbone("cpu")
    if weight_verification["weights_sha256"] != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError("Loaded pretrained backbone does not match the pinned R3 source SHA")
    del verified_backbone
    gc.collect()
    if int(freeze["training"]["max_epochs"]) != 20:
        raise RuntimeError("R5 maximum epochs must remain 20")
    if int(config["training"]["warmup_epochs"]) != 2:
        raise RuntimeError("R5 warmup must remain the R3 two-epoch warmup")

    R5_RUN_DIR = OUTPUT_ROOT = WORK_DIR / "outputs" / "experiments" / "DINOv3_Feature_Optimization" / GROUP
    R5_RUN_DIR.mkdir(parents=True, exist_ok=True)
    r3_verification = read_json(R3_VERIFICATION_PATH)
    r3_freeze = r3_config["freeze_config"]
    control_audit = {
        "reference_group": "R3_Global_Local",
        "reference_best_epoch": int(r3_status["best_epoch"]),
        "reference_best_macro_auroc": float(r3_status["best_macro_auroc"]),
        "reference_completed_epochs": int(r3_status["completed_epochs"]),
        "reference_stop_reason": r3_status["stop_reason"],
        "seed": int(r3_config["experiment_config"]["seed"]),
        "image_size": int(r3_config["image_size"]),
        "source_repository": r3_config["initialization_repo"],
        "source_revision": r3_config["initialization_revision"],
        "source_weights_sha256": r3_config["initialization_sha256"],
        "global_head_initial_sha256": head_sha,
        "training": r3_config["experiment_config"]["training"],
        "selected_runtime": runtime,
        "freeze_profile": r3_config["profile"],
        "trainable_modules": r3_verification["trainable_modules"],
        "frozen_eval_modules": r3_verification["frozen_eval_modules"],
        "train_studies": int(r3_config["train_studies"]),
        "train_images": int(r3_config["train_images"]),
        "validation_studies": int(r3_config["validation_studies"]),
        "validation_images": int(r3_config["validation_images"]),
        "optimizer_updates_by_epoch": [
            {"epoch": int(row.epoch), "updates": int(row.optimizer_updates),
             "lr_backbone": float(row.lr_backbone), "lr_head": float(row.lr_head)}
            for row in r3_log.itertuples(index=False)
        ],
        "optimizer_updates_through_best_epoch": int(
            r3_log.loc[r3_log.epoch <= int(r3_status["best_epoch"]),
                       "optimizer_updates"].sum()),
        "early_stopping": r3_freeze["early_stopping"],
        "test_accessed_in_r3": False,
    }
    save_json_atomic(R5_RUN_DIR / "r3_control_audit.json", control_audit)
    selected_runtime_path = R5_RUN_DIR / "selected_runtime_config.json"
    if selected_runtime_path.is_file():
        existing_runtime = read_json(selected_runtime_path)["selected_runtime"]
        if existing_runtime != runtime:
            raise RuntimeError("Existing R5 selected runtime differs from the R3 controlled runtime")
    else:
        runtime_payload = copy.deepcopy(r3_runtime_payload)
        runtime_payload["group"] = GROUP
        runtime_payload["source_group"] = "R3_Global_Local"
        runtime_payload["selection_rule"] = (
            "Directly reused R3 selected runtime for controlled comparison; no new benchmark search."
        )
        save_json_atomic(selected_runtime_path, runtime_payload)

    r5_feature = copy.deepcopy(feature)
    r5_feature.pop("r3", None)
    r5_config = copy.deepcopy(config)
    r5_config["seed"] = int(r3_config["experiment_config"]["seed"])
    r5_config["training"] = copy.deepcopy(r3_config["experiment_config"]["training"])
    r5_freeze = copy.deepcopy(r3_config["freeze_config"])
    r5_freeze["experiment"] = "DINOv3_Feature_Optimization_R5"
    r5_freeze["outputs_root"] = str(R5_RUN_DIR)
    r5_freeze["profiles"] = [PROFILE]
    if "submission" in r5_freeze:
        r5_freeze["submission"]["selected_model"] = GROUP
    r5_config["freeze"] = copy.deepcopy(r5_freeze)
    r5_config["data"]["image_size"] = IMAGE_SIZE
    r5_config["model"]["architecture"] = "DINOv3ConvNeXtTinyGlobalOnly"
    r5_config["model"]["pooling"] = (
        "Stage4 GAP followed by official DINOv3 final normalization and Linear(768,10)"
    )
    r5_config["model"]["study_fusion"] = "mean_raw_image_logits_before_sigmoid"
    r5_config["model"].pop("r3", None)
    r5_config["outputs"]["root"] = str(R5_RUN_DIR)
    r5_config["feature_experiment"] = "DINOv3_Feature_Optimization"

    builder = make_r5_builder(runtime, head_sha)
    preflight = run_actual_runtime_preflight(
        runtime, builder, r5_config["training"], r5_freeze)
    builder_calls = builder
    result = run_custom_group(
        r5_feature,
        r5_config,
        r5_freeze,
        GROUP,
        GROUP,
        IMAGE_SIZE,
        weight_verification,
        model_builder=builder_calls,
        optimizer_builder=feature_profile_optimizer,
        batch_forward=r5_batch_forward,
        validation_forward=r5_batch_forward,
        collate_fn=collate_masked_studies,
        audit_forward=r5_audit_forward,
        backbone_name="DINOv3 ConvNeXt-Tiny",
        expected_stage3_channels=384,
        dataset_builder=build_masked_datasets,
        run_context={
            "group": GROUP,
            "architecture": "dinov3_tiny_global_only",
            "image_size": IMAGE_SIZE,
            "head_type": "global_only_gap",
            "pretrained_initialization_sha256": EXPECTED_BACKBONE_SHA256,
            "global_classifier_init_sha256": head_sha,
            "runtime_source": "R3_Global_Local/selected_runtime_config.json",
            "no_local_head": True,
        },
        expected_weight_sha256=EXPECTED_BACKBONE_SHA256,
    )
    status = read_json(R5_RUN_DIR / "training_status.json")
    if not status.get("complete"):
        raise RuntimeError(f"R5 did not complete or stop by the configured rule: {status}")
    if status.get("stop_reason") not in {"three_condition_early_stop", "max_epochs_reached"}:
        raise RuntimeError(f"R5 stopped for an unexpected reason: {status.get('stop_reason')}")

    inference_weights = save_global_only_inference(R5_RUN_DIR, r5_feature)
    group_config_path = R5_RUN_DIR / "config.json"
    group_config = read_json(group_config_path)
    saved_runtime = group_config.get("selected_runtime")
    if saved_runtime is not None and saved_runtime != runtime:
        raise RuntimeError("R5 config.json runtime metadata does not match the reused R3 runtime")
    group_config["selected_runtime"] = runtime
    group_config["model_variant"] = "Global-only; no Local Head/alpha/Auxiliary Loss"
    group_config["runtime_metadata_source"] = str(R5_RUN_DIR / "selected_runtime_config.json")
    save_json_atomic(group_config_path, group_config)

    size_metrics_path = R5_RUN_DIR / "model_size_metrics.json"
    size_metrics = (read_json(size_metrics_path) if size_metrics_path.is_file() else None)
    if not size_metrics or not size_metrics.get("passes_strict_limit"):
        package_result = export_feature_package(
            r5_feature, group="R5", folder=GROUP)
    else:
        package_result = size_metrics
    if not package_result.get("passes_strict_limit"):
        raise RuntimeError("R5 inference package did not pass the strict 500 MB limit")
    package_config = read_json(R5_RUN_DIR / "submission_export/R5_Inference/model_config.json")["model"]
    if not package_config.get("channels_last") or package_config.get("head_type") != "gap":
        raise RuntimeError("R5 exported inference config has inconsistent Global-only/runtime metadata")
    validation_replay = validate_r5_inference_package(R5_RUN_DIR, runtime)

    write_markdown(R5_RUN_DIR / "initialization_verification.md", [
        "# R5 初始化核验", "",
        f"- 官方通用视觉预训练权重：`{r3_config['initialization_repo']}` revision `{r3_config['initialization_revision']}`。",
        f"- 权重 SHA-256：`{EXPECTED_BACKBONE_SHA256}`；通过与 R3 相同的本地缓存和 SHA 校验流程加载。",
        "- Backbone 从原始通用视觉预训练权重初始化，没有加载 R2/R3 胸片监督微调权重。",
        f"- Global Head seed={SEED}，构造顺序与 R3 相同；初始 SHA-256 `{head_sha}` 与 R3 记录一致。",
        f"- R5 Best Epoch={status['best_epoch']}，Macro-AUROC={status['best_macro_auroc']:.6f}。",
        f"- 权重专用文件：`{inference_weights.name}`；仅包含 Backbone 和 Global Linear(768,10) 参数，不包含 Optimizer/Scheduler 状态。",
        "",
    ])
    write_markdown(R5_RUN_DIR / "r5_execution_summary.md", [
        "# R5 Global-only execution summary", "",
        f"- Status: complete; stop reason: `{status['stop_reason']}` at epoch {status['completed_epochs']}/{status['max_epochs']}.",
        f"- Best epoch: {status['best_epoch']}; Macro-AUROC: {status['best_macro_auroc']:.6f}.",
        f"- Runtime copied from R3: budget={runtime['image_budget']}, accumulation={runtime['grad_accumulation']}, channels-last={runtime['channels_last']}, checkpointing={runtime['gradient_checkpointing']}; no runtime benchmark search.",
        f"- Actual-runtime disposable backward preflight passed on {preflight['preflight_images']} images / {preflight['preflight_studies']} Studies across {preflight['preflight_microbatches']} accumulation microbatches; no optimizer step was counted in preflight.",
        f"- R3 through Best Epoch had {sum(int(v) for v in r3_log.loc[r3_log['epoch'] <= 6, 'optimizer_updates'])} formal optimizer updates.",
        f"- R5 trainable/frozen parameters: {read_json(R5_RUN_DIR / 'model_verification.json')['trainable_parameters']:,} / {read_json(R5_RUN_DIR / 'model_verification.json')['frozen_parameters']:,}.",
        f"- Standalone inference package: `{package_result['archive_path']}` ({package_result['archive_bytes']:,} bytes; strict size check passed).",
        f"- Full Validation inference replay passed: Macro-AUROC {validation_replay['checkpoint_replay']['reproduced_macro_auroc']:.6f}; max probability error {validation_replay['checkpoint_replay']['max_probability_abs_error']:.8g}.",
        "- Only official Train and Validation were used. Test was not read by training or model selection.",
        "",
    ])
    print(f"R5 completed: epoch={status['completed_epochs']}, best_epoch={status['best_epoch']}, "
          f"best_macro_auroc={status['best_macro_auroc']:.6f}", flush=True)
    print(f"R5 output directory: {R5_RUN_DIR}", flush=True)

    from analyze_r5_comparison import run_analysis
    run_analysis()


if __name__ == "__main__":
    main()
