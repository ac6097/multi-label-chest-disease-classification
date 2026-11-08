"""Training implementation for H1/H2/H3 generalization-optimization experiments.

The loop mirrors R5's ``train_profile`` (same loss reduction, accumulation,
optimizer, LR schedule, clipping, validation, three-condition early stopping and
checkpoint format) with only the declared intervention enabled.
"""

from __future__ import annotations

import copy
import gc
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from genopt_common import (
    DATA_CONFIG_PATH,
    EXPERIMENT_ROOT,
    EXPERIMENTS,
    IMAGE_SIZE,
    LABEL_NAMES,
    PROFILE,
    R5_DIR,
    SEED,
    APWeightedStudyBatchSampler,
    build_h_datasets,
    class_bce_breakdown,
    class_bce_batch_accumulator,
    accumulate_class_bce,
    finalize_class_bce,
    l2sp_parameter_names,
    l2sp_penalty,
    l2sp_reference,
    load_r5_reference,
    make_h_builder,
    make_h_loader,
    merge_class_bce,
    r5_batch_forward,
    read_json,
    stage_relative_drift,
    study_view_metadata,
    write_json,
)
from dino_data import make_loader, shutdown_loader
from dino_experiment import (
    prediction_frame,
    save_json_atomic,
    save_torch_atomic,
    update_learning_rates,
    validate,
)
from dino_feature_models import collate_masked_studies, feature_profile_optimizer
from dino_freeze_experiment import (
    checkpoint_payload,
    restore_rng,
    set_profile_training_mode,
    state_rng,
)
from dino_losses import study_bce_per_study, study_logit_mean
from hybrid.common import write_csv


def run_dir_for(experiment: str) -> Path:
    return EXPERIMENT_ROOT / EXPERIMENTS[experiment]["label"]


def _experiment_context(experiment: str, optimization: dict) -> dict:
    reference = load_r5_reference()
    return {
        "experiment": experiment,
        "label": EXPERIMENTS[experiment]["label"],
        "optimization": optimization,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "initialization_sha256": reference["initialization_sha256"],
        "global_classifier_init_sha256": reference["head_sha"],
        "reference_group": "R5_Global_Only_512",
        "reference_pretrained_revision": reference["initialization_revision"],
        "test_access": False,
    }


def save_experiment_config(experiment: str) -> dict:
    reference = load_r5_reference()
    optimization = copy.deepcopy(EXPERIMENTS[experiment]["optimization"])
    run_dir = run_dir_for(experiment)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_payload = None
    config_payload = {
        "experiment": "DINOv3_Generalization_Optimization",
        "group": run_dir.name,
        "h_experiment": experiment,
        "optimization": optimization,
        "image_size": IMAGE_SIZE,
        "backbone": "DINOv3 ConvNeXt-Tiny",
        "initialization_repository": reference["initialization_repository"],
        "initialization_revision": reference["initialization_revision"],
        "initialization_sha256": reference["initialization_sha256"],
        "global_classifier_init_sha256": reference["head_sha"],
        "profile": "P2: freeze Stem/Stage1/Stage2; train Stage3/Stage4/final norm/head",
        "selected_runtime": reference["runtime"],
        "experiment_config": reference["config"],
        "freeze_config": reference["freeze"],
        "reference_experiment": "R5_Global_Only_512",
        "reference_config": str(R5_DIR / "config.json"),
        "test_access": False,
        "dataset": dataset_payload,
    }
    if (run_dir / "config.json").is_file():
        existing = read_json(run_dir / "config.json")
        if existing.get("optimization") != optimization:
            raise RuntimeError(
                f"{experiment} existing config has a different optimization block")
    else:
        save_json_atomic(run_dir / "config.json", config_payload)
    return config_payload


def prepare_data(experiment: str):
    train_data, valid_data, data_config = build_h_datasets()
    if len(valid_data) != 859:
        raise RuntimeError("H-series must use the official 859-Study Validation split")
    if train_data.dataset.split != "train" or valid_data.dataset.split != "valid":
        raise RuntimeError("H-series may only use official Train/Validation")
    return train_data, valid_data, data_config


def _valid_loader_config(runtime: dict) -> dict:
    return {
        "image_budget": int(runtime["image_budget"]),
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
        "prefetch_factor": int(runtime["prefetch_factor"]),
        "seed": SEED,
        "channels_last": bool(runtime["channels_last"]),
    }


def valid_study_image_counts(valid_data) -> np.ndarray:
    base = valid_data.dataset.base
    return np.asarray([len(base.studies[index]["image_paths"])
                       for index in range(len(valid_data))], dtype=np.int64)


def _probe_l2sp_gradients(model, reference, regularized_names,
                          batch, device, runtime) -> dict:
    """One extra forward/backward used once per epoch for gradient diagnostics."""
    parameters = dict(model.named_parameters())
    regularized = [parameters[name] for name in regularized_names]
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
        per_study, eligible = study_bce_per_study(study_logits, labels, known)
        bce = per_study[eligible].mean()
    reg = l2sp_penalty(model, reference)
    grad_bce = torch.autograd.grad(bce, regularized, retain_graph=True,
                                   allow_unused=True)
    grad_reg = torch.autograd.grad(reg, regularized, retain_graph=False,
                                   allow_unused=True)
    bce_norm = math.sqrt(sum(float(grad.pow(2).sum()) for grad in grad_bce
                             if grad is not None))
    reg_norm = math.sqrt(sum(float(grad.pow(2).sum()) for grad in grad_reg
                             if grad is not None))
    model.zero_grad(set_to_none=True)
    return {
        "probe_bce": float(bce.detach().float().cpu()),
        "probe_l2sp_penalty": float(reg.detach().float().cpu()),
        "bce_grad_norm": bce_norm,
        "l2sp_unit_grad_norm": reg_norm,
        "grad_ratio": bce_norm / (reg_norm + 1e-30),
    }


def train_experiment(experiment: str, force: bool = False,
                     smoke: bool = False) -> dict:
    if experiment not in ("H1", "H2", "H3", "H3B", "H2B", "H2C", "H4"):
        raise ValueError("train_experiment handles H1/H2/H3/H3B/H2B/H2C/H4 only")
    optimization = copy.deepcopy(EXPERIMENTS[experiment]["optimization"])
    mode = optimization["mode"]
    reference = load_r5_reference()
    runtime = reference["runtime"]
    config = copy.deepcopy(reference["config"])
    freeze = copy.deepcopy(reference["freeze"])
    run_dir = run_dir_for(experiment)
    if smoke:
        run_dir = run_dir / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not smoke:
        save_experiment_config(experiment)
        if (run_dir / "training_status.json").is_file() and not force:
            status = read_json(run_dir / "training_status.json")
            if status.get("complete"):
                return status
    train_data, valid_data, data_config = prepare_data(experiment)
    if smoke:
        config_payload = {"optimization": optimization, "smoke": True}
    else:
        config_payload = read_json(run_dir / "config.json")
        if config_payload.get("optimization") != optimization:
            raise RuntimeError("H-series config optimization block changed")
    config_payload["dataset"] = {
        "train_studies": len(train_data),
        "train_images": int(sum(train_data.image_counts)),
        "validation_studies": len(valid_data),
        "validation_images": int(sum(valid_data.image_counts)),
        "data_config": str(DATA_CONFIG_PATH),
        "label_names": LABEL_NAMES,
    }
    save_json_atomic(run_dir / "config.json", config_payload)

    device = torch.device("cuda")
    head_dropout = float(optimization.get("head_dropout", 0.0))
    builder = make_h_builder(runtime, reference["head_sha"], run_dir / "audit",
                             head_dropout=head_dropout)
    model, verification, train_modules, frozen_modules, trainable_names = builder(
        PROFILE, device, runtime, SEED)
    if verification["weights_sha256"] != reference["initialization_sha256"]:
        raise RuntimeError("H-series model did not load the pinned original weights")
    optimizer = feature_profile_optimizer(model, config)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    run_context = _experiment_context(experiment, optimization)
    run_context["trainable_parameters"] = trainable_count
    run_context["frozen_parameters"] = frozen_count

    probe_metadata = None
    if mode == "ap_sampling":
        probe_metadata = study_view_metadata(train_data)
        sampler = APWeightedStudyBatchSampler(
            train_data, probe_metadata["ap_only"], int(runtime["image_budget"]),
            seed=SEED,
            target_ap_fraction=float(optimization["target_ap_fraction"]),
            category_array=probe_metadata["categories"])
        train_loader, sampler, loader_generator = make_h_loader(
            train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
            seed=SEED, num_workers=int(runtime["num_workers"]),
            pin_memory=bool(runtime["pin_memory"]),
            prefetch_factor=int(runtime["prefetch_factor"]),
            batch_sampler=sampler)
        save_json_atomic(run_dir / "ap_view_metadata.json", {
            "counts": probe_metadata["counts"],
            "n_studies": probe_metadata["n_studies"],
            "target_ap_fraction": float(optimization["target_ap_fraction"]),
            "ap_weight_reference": float(optimization["ap_weight"]),
            "definition": (
                "AP-only Study = every image has a non-empty ViewPosition and all "
                "valid ViewPosition values equal 'AP'; missing/mixed views are "
                "reported as their own categories and are not counted as AP-only."),
            "category_definition": {
                "ap_only": "all images AP",
                "pa_only": "all images PA",
                "lateral_only": "all images LATERAL/LL/PA LLD",
                "mixed_or_other": "multiple view types within the Study",
                "missing_view": "at least one image has empty ViewPosition",
            },
        })
    else:
        train_loader, sampler, loader_generator = make_loader(
            train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
            seed=SEED, num_workers=int(runtime["num_workers"]),
            pin_memory=bool(runtime["pin_memory"]),
            prefetch_factor=int(runtime["prefetch_factor"]),
            collate_fn=collate_masked_studies)

    max_epochs = 1 if smoke else int(freeze["training"]["max_epochs"])
    accumulation = int(runtime["grad_accumulation"])
    epoch_updates = []
    for epoch_index in range(max_epochs):
        sampler.set_epoch(epoch_index)
        epoch_updates.append(math.ceil(len(sampler) / accumulation))
    total_updates = sum(epoch_updates)
    training = dict(config["training"])
    training["warmup_updates"] = sum(epoch_updates[:int(training["warmup_epochs"])])

    l2sp_enabled = mode == "l2sp"
    regularized_names = []
    reference_parameters = {}
    l2sp_lambda = None
    if l2sp_enabled:
        if "lambda_sp" in optimization:
            # H3-B: one pre-specified weak L2-SP strength; identical parameter
            # set/normalization as H3, only the scalar lambda changes.
            l2sp_lambda = float(optimization["lambda_sp"])
            regularized_names = l2sp_parameter_names(model)
            h3_reference_path = (
                EXPERIMENT_ROOT / "H3_L2SP_Regularization" / "l2sp_calibration.json")
            if h3_reference_path.is_file():
                h3_names = list(read_json(h3_reference_path)[
                    "regularized_parameter_names"])
                if set(h3_names) != set(regularized_names):
                    raise RuntimeError(
                        "H3-B parameter set differs from the audited H3 set")
            lambda_source = str(optimization.get("lambda_source", "explicit config"))
            calibration_source = "explicit H3-B config (0.1 x H3 lambda)"
        else:
            calibration_path = (EXPERIMENT_ROOT / "H3_L2SP_Regularization"
                                / "l2sp_calibration.json")
            if not calibration_path.is_file():
                raise RuntimeError(
                    "H3 requires the Train-only L2-SP calibration file; run calibrate first")
            calibration = read_json(calibration_path)
            l2sp_lambda = float(calibration["lambda_sp"])
            regularized_names = list(calibration["regularized_parameter_names"])
            lambda_source = "H3 Train-only gradient calibration"
            calibration_source = str(calibration_path)
        if l2sp_lambda <= 0:
            raise RuntimeError("L2-SP lambda_sp must be positive")
        missing = set(regularized_names) - set(dict(model.named_parameters()))
        if missing:
            raise RuntimeError(f"L2-SP parameter names are missing: {missing}")
        reference_parameters = l2sp_reference(model, regularized_names)
        optimization["lambda_sp"] = l2sp_lambda
        optimization["regularized_parameter_count"] = int(
            sum(dict(model.named_parameters())[name].numel()
                for name in regularized_names))
        save_json_atomic(run_dir / "l2sp_calibration_reference.json", {
            "lambda_sp": l2sp_lambda,
            "lambda_source": lambda_source,
            "regularized_parameter_names": regularized_names,
            "regularized_parameter_count": optimization["regularized_parameter_count"],
            "calibration_source": calibration_source,
            "note": "lambda_sp is fixed before training; no validation search.",
        })
        run_context["lambda_sp"] = l2sp_lambda
        run_context["regularized_parameter_count"] = optimization[
            "regularized_parameter_count"]

    history, per_class_history = [], []
    per_epoch_class_history = []
    best_auroc, best_epoch = -math.inf, 0
    no_improve_count, global_step, start_epoch = 0, 0, 0
    last_path = run_dir / "last.pth"
    best_path = run_dir / "best.pth"
    if last_path.is_file() and not force:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if (saved.get("experiment") != run_context["experiment"]
                or saved.get("run_context") != run_context
                or saved["profile"] != PROFILE
                or saved["initial_weights_sha256"] != verification["weights_sha256"]):
            raise RuntimeError("H-series resume checkpoint identity mismatch")
        if saved["runtime"] != runtime or saved["trainable_parameter_names"] != trainable_names:
            raise RuntimeError("H-series resume runtime or freeze manifest changed")
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
        if not mode == "ap_sampling":
            loader_generator.set_state(saved["loader_generator_state"])
        restore_rng(saved["rng_state"])
        print(f"[{experiment}] resuming after epoch {start_epoch}", flush=True)

    status_path = run_dir / "training_status.json"
    stop_reason = None
    try:
        for epoch_index in range(start_epoch, max_epochs):
            epoch_number = epoch_index + 1
            sampler.set_epoch(epoch_index)
            iterator = iter(train_loader)
            set_profile_training_mode(model, frozen_modules)
            train_started = time.perf_counter()
            data_wait_seconds = 0.0
            bce_sum_epoch = 0.0
            eligible_studies_epoch = 0
            train_images = train_studies = 0
            updates = 0
            train_class_accumulator = class_bce_batch_accumulator()
            train_class_single = (class_bce_batch_accumulator()
                                  if mode == "single_study_weight" else None)
            train_class_multi = (class_bce_batch_accumulator()
                                 if mode == "single_study_weight" else None)
            h2 = {"single_sum": 0.0, "single_count": 0, "multi_sum": 0.0,
                  "multi_count": 0, "weighted_sum": 0.0, "weight_total": 0.0,
                  "unweighted_sum": 0.0, "eligible": 0}
            h3 = {"reg_sum": 0.0, "reg_count": 0, "objective_sum": 0.0,
                  "updates": 0}
            torch.cuda.reset_peak_memory_stats(device)
            probe = None
            if l2sp_enabled:
                first_batch = next(iterator, None)
                if first_batch is None:
                    raise RuntimeError("Empty training loader")
                probe = _probe_l2sp_gradients(
                    model, reference_parameters, regularized_names, first_batch,
                    device, runtime)
                iterator = itertools.chain([first_batch], iterator)
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
                weight_total_window = 0.0
                if mode == "single_study_weight":
                    single_weight = float(optimization["single_weight"])
                    for batch in window:
                        counts = batch["image_counts"].to(device)
                        known = batch["known"].to(device)
                        eligible = known.any(dim=1)
                        weights = torch.where(counts == 1, single_weight, 1.0)
                        weight_total_window += float(weights[eligible].sum().cpu())
                    if weight_total_window <= 0:
                        raise RuntimeError("H2 normalization weight sum is not positive")
                optimizer.zero_grad(set_to_none=True)
                lrs = update_learning_rates(optimizer, global_step, total_updates,
                                            training)
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
                            image_logits.float(), study_index,
                            batch["num_studies"], image_counts)
                        per_study, eligible = study_bce_per_study(
                            study_logits, labels, known)
                        if mode == "single_study_weight":
                            weights = torch.where(
                                image_counts == 1, single_weight, 1.0)
                            weighted = per_study[eligible] * weights[eligible]
                            loss = weighted.sum() / weight_total_window
                        else:
                            bce_sum = per_study[eligible].sum()
                            loss = bce_sum / max(eligible_window, 1)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite {experiment} loss at epoch {epoch_number}")
                    loss.backward()
                    raw_sum = float(per_study[eligible].sum().detach().float().cpu())
                    bce_sum_epoch += raw_sum
                    eligible_studies_epoch += int(eligible.sum())
                    train_images += int(images.shape[0])
                    train_studies += int(batch["num_studies"])
                    accumulate_class_bce(train_class_accumulator, study_logits,
                                         labels, known)
                    if mode == "single_study_weight":
                        single_mask = eligible & (image_counts == 1)
                        multi_mask = eligible & (image_counts >= 2)
                        if bool(single_mask.any()):
                            h2["single_sum"] += float(
                                per_study[single_mask].sum().detach().float().cpu())
                            h2["single_count"] += int(single_mask.sum())
                            accumulate_class_bce(
                                train_class_single, study_logits[single_mask],
                                labels[single_mask], known[single_mask])
                        if bool(multi_mask.any()):
                            h2["multi_sum"] += float(
                                per_study[multi_mask].sum().detach().float().cpu())
                            h2["multi_count"] += int(multi_mask.sum())
                            accumulate_class_bce(
                                train_class_multi, study_logits[multi_mask],
                                labels[multi_mask], known[multi_mask])
                        h2["unweighted_sum"] += raw_sum
                        h2["eligible"] += int(eligible.sum())
                        weights = torch.where(image_counts == 1, single_weight, 1.0)
                        h2["weighted_sum"] += float(
                            (per_study[eligible] * weights[eligible]).sum()
                            .detach().float().cpu())
                        h2["weight_total"] += float(weights[eligible].sum().cpu())
                    if l2sp_enabled:
                        with torch.no_grad():
                            h3["reg_sum"] += float(
                                l2sp_penalty(model, reference_parameters)
                                .float().cpu())
                        h3["reg_count"] += 1
                if l2sp_enabled:
                    reg = l2sp_penalty(model, reference_parameters)
                    reg_term = l2sp_lambda * reg
                    reg_term.backward()
                    h3["objective_sum"] += float(
                        reg_term.detach().float().cpu())
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, float(training["gradient_clip_norm"]),
                    error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                updates += 1
                h3["updates"] += 1
                del window
                if smoke and updates >= 4:
                    break
            train_seconds = time.perf_counter() - train_started
            validation = validate(
                model, valid_data, device, **_valid_loader_config(runtime),
                batch_forward=r5_batch_forward, collate_fn=collate_masked_studies)
            score = float(validation["macro_auroc"])
            improved = math.isfinite(score) and score > best_auroc
            if improved:
                best_auroc, best_epoch = score, epoch_number
                no_improve_count = 0
            else:
                no_improve_count += 1
            train_bce = bce_sum_epoch / max(eligible_studies_epoch, 1)
            valid_class = class_bce_breakdown(
                validation["logits"], validation["targets"], validation["known"])
            train_class = finalize_class_bce(train_class_accumulator)
            per_epoch_rows = merge_class_bce(train_class, valid_class)
            valid_single_rows = valid_multi_rows = None
            if mode == "single_study_weight":
                valid_counts = valid_study_image_counts(valid_data)
                single_valid_mask = valid_counts == 1
                multi_valid_mask = valid_counts >= 2
                valid_single_rows = class_bce_breakdown(
                    validation["logits"][single_valid_mask],
                    validation["targets"][single_valid_mask],
                    validation["known"][single_valid_mask])
                valid_multi_rows = class_bce_breakdown(
                    validation["logits"][multi_valid_mask],
                    validation["targets"][multi_valid_mask],
                    validation["known"][multi_valid_mask])
                train_single_rows = finalize_class_bce(train_class_single)
                train_multi_rows = finalize_class_bce(train_class_multi)
            for row in per_epoch_rows:
                row["epoch"] = epoch_number
                if mode == "single_study_weight":
                    name = row["disease"]
                    if name == "Overall BCE":
                        row["train_single_bce"] = float(np.mean(
                            [train_single_rows[key]["bce"] for key in LABEL_NAMES]))
                        row["train_multi_bce"] = float(np.mean(
                            [train_multi_rows[key]["bce"] for key in LABEL_NAMES]))
                        row["valid_single_bce"] = float(np.mean(
                            [valid_single_rows[key]["bce"] for key in LABEL_NAMES]))
                        row["valid_multi_bce"] = float(np.mean(
                            [valid_multi_rows[key]["bce"] for key in LABEL_NAMES]))
                    else:
                        row["train_single_bce"] = train_single_rows[name]["bce"]
                        row["train_multi_bce"] = train_multi_rows[name]["bce"]
                        row["valid_single_bce"] = valid_single_rows[name]["bce"]
                        row["valid_multi_bce"] = valid_multi_rows[name]["bce"]
            per_class_rows = [
                {"epoch": epoch_number, **item, "valid_bce": valid_class[item["class_name"]]["bce"],
                 "valid_positive_bce": valid_class[item["class_name"]]["positive_bce"],
                 "valid_negative_bce": valid_class[item["class_name"]]["negative_bce"]}
                for item in validation["per_class"]]
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
                "trainable_parameters": trainable_count,
                "frozen_parameters": frozen_count,
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
            row["train_bce_raw"] = train_bce
            if mode == "single_study_weight":
                row.update({
                    "single_study_count": int(h2["single_count"]),
                    "multi_study_count": int(h2["multi_count"]),
                    "single_raw_bce": (h2["single_sum"] / h2["single_count"]
                                       if h2["single_count"] else float("nan")),
                    "multi_raw_bce": (h2["multi_sum"] / h2["multi_count"]
                                      if h2["multi_count"] else float("nan")),
                    "weighted_objective_bce": (h2["weighted_sum"] / h2["weight_total"]
                                               if h2["weight_total"] else float("nan")),
                    "unweighted_batch_bce": (h2["unweighted_sum"] / h2["eligible"]
                                             if h2["eligible"] else float("nan")),
                    "single_weight": float(optimization["single_weight"]),
                })
            if mode == "head_dropout":
                row["head_dropout"] = head_dropout
                row["objective_loss"] = train_bce
            if l2sp_enabled:
                mean_reg = h3["reg_sum"] / max(h3["reg_count"], 1)
                drift = stage_relative_drift(model, reference_parameters)
                row.update({
                    "l2sp_penalty": mean_reg,
                    "l2sp_term": l2sp_lambda * mean_reg,
                    "objective_loss": train_bce + l2sp_lambda * mean_reg,
                    "stage3_relative_drift": drift["stage3"]["relative_drift"],
                    "stage4_relative_drift": drift["stage4"]["relative_drift"],
                    "bce_grad_norm": probe["bce_grad_norm"] if probe else float("nan"),
                    "l2sp_unit_grad_norm": (probe["l2sp_unit_grad_norm"]
                                            if probe else float("nan")),
                    "gradient_ratio": probe["grad_ratio"] if probe else float("nan"),
                    "lambda_sp": l2sp_lambda,
                })
            if mode == "ap_sampling":
                stats = sampler.statistics()
                row.update({f"h1_{key}": value for key, value in stats.items()
                            if key not in ("epoch", "per_class_positive_draw_fraction",
                                           "per_class_original_positive_fraction",
                                           "repeat_histogram",
                                           "category_draw_fractions")})
                row["h1_ap_only_draw_fraction"] = stats["ap_only_draw_fraction"]
                row["h1_repeat_once"] = stats["repeat_histogram"]["once"]
                row["h1_repeat_twice"] = stats["repeat_histogram"]["twice"]
                row["h1_repeat_three_plus"] = stats["repeat_histogram"]["three_plus"]
                row["h1_repeat_never"] = stats["repeat_histogram"]["never"]
                for key, value in stats["category_draw_fractions"].items():
                    row[f"h1_category_{key}_fraction"] = value
                for name in LABEL_NAMES:
                    safe = name.replace(" ", "_")
                    row[f"h1_positive_draw_fraction_{safe}"] = (
                        stats["per_class_positive_draw_fraction"][name])
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
            per_class_history.extend(per_class_rows)
            per_epoch_class_history.extend(per_epoch_rows)
            write_csv(run_dir / "training_history.csv", history)
            write_csv(run_dir / "training_log.csv", history)
            write_csv(run_dir / "per_epoch_class_loss.csv",
                      per_epoch_class_history)
            write_csv(run_dir / "per_class_metrics.csv", per_class_history)
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            prediction_frame(validation).to_csv(
                predictions_dir / f"val_epoch_{epoch_number:02d}.csv",
                index=False, encoding="utf-8")
            if improved:
                prediction_frame(validation).to_csv(
                    predictions_dir / "val_best.csv", index=False, encoding="utf-8")
                write_csv(run_dir / "best_per_class_metrics.csv",
                          validation["per_class"])
            if should_stop:
                stop_reason = "three_condition_early_stop"
            elif smoke:
                stop_reason = "smoke_stop"
            elif epoch_number >= int(freeze["training"]["max_epochs"]):
                stop_reason = "max_epochs_reached"
            complete = stop_reason is not None
            payload = checkpoint_payload(
                PROFILE, model, optimizer, epoch_number, history, per_class_history,
                best_auroc, best_epoch, no_improve_count, global_step,
                verification["weights_sha256"], runtime, sampler, loader_generator,
                trainable_names, complete, stop_reason,
                experiment_name=run_context["experiment"], run_context=run_context)
            payload["optimization"] = optimization
            save_json_atomic(run_dir / "checkpoint_metadata.json", {
                "epoch": epoch_number,
                "optimization": optimization,
                "trainable_parameter_names": trainable_names,
                "runtime": runtime,
            })
            from dino_experiment import save_torch_atomic as _save_torch

            _save_torch(last_path, payload)
            if improved:
                _save_torch(best_path, payload)
            if should_stop:
                save_json_atomic(run_dir / "early_stop.json", {
                    "stop_epoch": epoch_number,
                    "stop_reason": stop_reason,
                    "train_down_3": train_down_3,
                    "valid_up_3": valid_up_3,
                    "auc_no_improve_count": no_improve_count,
                    "best_epoch": best_epoch,
                    "best_macro_auroc": best_auroc,
                })
            status = {
                "experiment": experiment,
                "label": EXPERIMENTS[experiment]["label"],
                "completed_epochs": epoch_number,
                "max_epochs": int(freeze["training"]["max_epochs"]),
                "best_epoch": best_epoch,
                "best_macro_auroc": best_auroc,
                "last_checkpoint": str(last_path),
                "best_checkpoint": str(best_path),
                "complete": complete,
                "stop_reason": stop_reason,
                "auc_no_improve_count": no_improve_count,
                "optimization": optimization,
                "test_access": False,
            }
            save_json_atomic(status_path, status)
            print(
                f"[{experiment}] epoch {epoch_number}/{max_epochs} "
                f"train_bce={train_bce:.6f} valid_bce={validation['valid_loss']:.6f} "
                f"macro_auroc={score:.6f} updates={updates} "
                f"train_down_3={train_down_3} valid_up_3={valid_up_3} "
                f"no_improve={no_improve_count} stop={stop_reason} "
                f"best_epoch={best_epoch} best_auroc={best_auroc:.6f}", flush=True)
            if complete:
                break
    except Exception as exc:
        save_json_atomic(run_dir / "training_failure.json", {
            "experiment": experiment,
            "epoch_in_progress": len(history) + 1,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "last_completed_epoch": len(history),
        })
        raise
    finally:
        shutdown_loader(train_loader)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    final_status = read_json(status_path)
    result = {
        "experiment": experiment,
        "completed_epochs": final_status["completed_epochs"],
        "stop_reason": final_status["stop_reason"],
        "best_epoch": final_status["best_epoch"],
        "best_macro_auroc": final_status["best_macro_auroc"],
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "output_dir": str(run_dir),
    }
    save_json_atomic(run_dir / "training_result.json", result)
    _write_final_training_report(experiment, run_dir, history, final_status)
    if experiment in ("H3B", "H2B", "H2C", "H4"):
        import shutil

        shutil.copy2(best_path, run_dir / "best_checkpoint.pth")
        shutil.copy2(last_path, run_dir / "last_checkpoint.pth")
    return result


def _write_final_training_report(experiment: str, run_dir: Path, history: list[dict],
                                 status: dict) -> None:
    best = max(history, key=lambda row: float(row["macro_auroc"]))
    latest = history[-1]
    lines = [
        f"# {experiment} training report ({EXPERIMENTS[experiment]['label']})", "",
        f"- Experiment: {EXPERIMENTS[experiment]['description']}",
        f"- Optimization: `{json.dumps(EXPERIMENTS[experiment]['optimization'], ensure_ascii=False)}`",
        f"- Status: {status['completed_epochs']}/{status['max_epochs']} epochs; "
        f"stop reason: `{status['stop_reason']}`.",
        f"- Best epoch: {best['epoch']}; Validation Macro-AUROC: {best['macro_auroc']:.6f}; "
        f"Validation BCE: {best['valid_loss']:.6f}.",
        f"- Last epoch: {latest['epoch']}; raw Train BCE: {latest['train_bce']:.6f}; "
        f"Validation BCE: {latest['valid_loss']:.6f}; Macro-AUROC: {latest['macro_auroc']:.6f}.",
        "- Raw BCE (unweighted Study mean) is used for every reported comparison; "
        "H2/H3 optimized objectives are logged separately.",
        "- Test split was not read; model selection uses official Validation only.",
        "",
        "| Epoch | Raw Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Updates |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_bce']:.6f} | {row['valid_loss']:.6f} | "
            f"{row['macro_auroc']:.6f} | {row['macro_auprc']:.6f} | "
            f"{row['optimizer_updates']} |")
    lines.append("")
    (run_dir / "final_report.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    (run_dir / "training_report.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


# ---------------------------------------------------------------------------
# H3 calibration (Train only)
# ---------------------------------------------------------------------------

def calibrate_l2sp(target_ratio: float = 0.05, calibration_updates: int = 100) -> dict:
    experiment = "H3"
    run_dir = run_dir_for(experiment)
    run_dir.mkdir(parents=True, exist_ok=True)
    reference = load_r5_reference()
    runtime = reference["runtime"]
    config = copy.deepcopy(reference["config"])
    freeze = copy.deepcopy(reference["freeze"])
    train_data, _, _ = prepare_data(experiment)
    device = torch.device("cuda")
    builder = make_h_builder(runtime, reference["head_sha"], run_dir / "calibration")
    model, verification, *_ = builder(PROFILE, device, runtime, SEED)
    optimizer = feature_profile_optimizer(model, config)
    names = l2sp_parameter_names(model)
    reference_parameters = l2sp_reference(model, names)
    initial_penalty = float(l2sp_penalty(model, reference_parameters).float().cpu())
    if initial_penalty > 1e-12:
        raise RuntimeError("L2-SP penalty must be zero at the original pretrained state")
    train_loader, _, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        prefetch_factor=int(runtime["prefetch_factor"]),
        collate_fn=collate_masked_studies)
    training = dict(config["training"])
    training["warmup_updates"] = 2
    total_updates = calibration_updates * 2
    iterator = iter(train_loader)
    accumulation = int(runtime["grad_accumulation"])
    updates = 0
    model.train()
    while updates < calibration_updates:
        window = []
        for _ in range(accumulation):
            try:
                window.append(next(iterator))
            except StopIteration:
                iterator = iter(train_loader)
                window.append(next(iterator))
        eligible_window = sum(int(batch["known"].any(dim=1).sum()) for batch in window)
        optimizer.zero_grad(set_to_none=True)
        update_learning_rates(optimizer, updates, total_updates, training)
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
                    image_logits.float(), study_index, batch["num_studies"],
                    image_counts)
                per_study, eligible = study_bce_per_study(study_logits, labels, known)
                loss = per_study[eligible].sum() / max(eligible_window, 1)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                        if p.requires_grad], 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        updates += 1
    shutdown_loader(train_loader)
    # Gradient-scale probe after a non-zero parameter shift.
    train_loader, _, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=False,
        seed=SEED, num_workers=0, pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    batch = next(iter(train_loader))
    shutdown_loader(train_loader)
    model.eval()
    probe = _probe_l2sp_gradients(model, reference_parameters, names, batch,
                                  device, runtime)
    lambda_sp = target_ratio * probe["bce_grad_norm"] / max(
        probe["l2sp_unit_grad_norm"], 1e-30)
    drift = stage_relative_drift(model, reference_parameters)
    payload = {
        "experiment": experiment,
        "purpose": "Train-only gradient-scale calibration for the single pre-specified "
                   "L2-SP strength; no Validation/Test tuning.",
        "calibration_updates": int(calibration_updates),
        "target_gradient_ratio": float(target_ratio),
        "initial_l2sp_penalty": initial_penalty,
        "after_calibration_stage3_relative_drift": drift["stage3"],
        "after_calibration_stage4_relative_drift": drift["stage4"],
        "probe": probe,
        "lambda_sp": float(lambda_sp),
        "regularized_parameter_names": names,
        "regularized_parameter_count": int(sum(
            dict(model.named_parameters())[name].numel() for name in names)),
        "regularized_namespaces": [
            "backbone.model.downsample_layers.2.",
            "backbone.model.stages.2.",
            "backbone.model.downsample_layers.3.",
            "backbone.model.stages.3.",
        ],
        "excluded_parameters": ["classifier.*", "frozen backbone parameters"],
        "test_access": False,
    }
    write_json(run_dir / "l2sp_calibration.json", payload)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[H3] calibrated lambda_sp={lambda_sp:.6g} "
          f"(bce_grad={probe['bce_grad_norm']:.6g}, "
          f"l2sp_unit_grad={probe['l2sp_unit_grad_norm']:.6g})", flush=True)
    return payload
