import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, collate_studies
from datasets.hybrid import build_hybrid_transform
from engine.trainer import _draw_curve, validate
from hybrid.backbone_factory import build_hybrid_backbone
from hybrid.common import (PROJECT_DIR, autocast_context, build_adamw,
                           build_update_scheduler, count_parameters, current_lrs,
                           device_and_amp, read_yaml, resolve_data_config, StandardStudyBCE,
                           seed_everything, seed_worker, sha256_file,
                           sha256_state_dict, write_csv, write_json)
from hybrid.pretrain import encoder_checkpoint_path
from hybrid.common import update_counts_for_epoch
from model.hybrid import HybridClassifier


def _save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_curves(output_dir, history, per_class_rows):
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    _draw_curve(output_dir / "loss_curve.png", [
        ("Train Loss", epochs, [float(row["train_loss"]) for row in history], (39, 110, 220)),
        ("Valid Loss", epochs, [float(row["val_loss"]) for row in history], (220, 75, 55)),
    ], "Study-mean BCE", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(output_dir / "macro_auroc_curve.png", [
        ("Validation Macro AUROC", epochs,
         [float(row["macro_auroc"]) for row in history], (39, 145, 88)),
    ], "Macro AUROC", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(output_dir / "images_per_second_curve.png", [
        ("End-to-end images/s", epochs,
         [float(row["images_per_s"]) for row in history], (112, 75, 180)),
    ], "Images / second", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(output_dir / "validation_inference_speed.png", [
        ("Validation Studies/s", epochs,
         [float(row["validation_studies_per_s"]) for row in history], (39, 110, 220)),
        ("Validation images/s", epochs,
         [float(row["validation_images_per_s"]) for row in history], (220, 75, 55)),
    ], "Validation samples / second", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(output_dir / "peak_memory_curve.png", [
        ("Peak allocated GiB", epochs,
         [float(row["peak_gpu_allocated_gib"]) for row in history], (39, 110, 220)),
        ("Peak reserved GiB", epochs,
         [float(row["peak_gpu_reserved_gib"]) for row in history], (220, 75, 55)),
    ], "GPU memory (GiB)", markers=(2, 5, 7, 10, 15, 20))
    lr_series = []
    for key, label, color in (
        ("lr_convnext", "ConvNeXtV2 LR", (39, 110, 220)),
        ("lr_vit", "ViT LR", (44, 160, 44)),
        ("lr_head", "Head LR", (220, 75, 55)),
    ):
        lr_series.append((label, epochs, [float(row[key]) for row in history], color))
    _draw_curve(output_dir / "learning_rate_curve.png", lr_series,
                "Learning Rate (log scale)", logarithmic=True, markers=(2, 5, 7, 10, 15, 20))
    class_names = sorted({row["class_name"] for row in per_class_rows})
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207),
    ]
    series = []
    for index, label in enumerate(class_names):
        rows = sorted((row for row in per_class_rows if row["class_name"] == label),
                      key=lambda row: int(row["epoch"]))
        series.append((label, [int(row["epoch"]) for row in rows],
                       [float(row["auroc"]) for row in rows], palette[index % len(palette)]))
    _draw_curve(output_dir / "per_class_auroc_curve.png", series,
                "Per-class Validation AUROC", markers=(2, 5, 7, 10, 15, 20))


def _validation_predictions(validation, class_names):
    rows = []
    for row_index, (subject_id, study_id) in enumerate(
            zip(validation["subject_ids"], validation["study_ids"])):
        row = {"Subject_id": subject_id, "Study_id": study_id}
        for class_index, class_name in enumerate(class_names):
            row[f"target_{class_name}"] = float(validation["targets"][row_index, class_index])
            row[f"prob_{class_name}"] = float(validation["probabilities"][row_index, class_index])
        rows.append(row)
    return rows


def _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_metric,
                        best_epoch, stale_epochs, config, train_generator, val_generator):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale_epochs),
        "config": config,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "train_loader_generator_state": train_generator.get_state(),
        "val_loader_generator_state": val_generator.get_state(),
    }


def _restore_rng(state):
    if "rng" in state:
        torch.set_rng_state(state["rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng"):
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng"]])
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])


def _train_epoch(model, loader, criterion, optimizer, scheduler, config,
                 device, amp_enabled, amp_dtype):
    model.train()
    model.set_grad_checkpointing(bool(config["gradient_checkpointing"]))
    accumulation = int(config["grad_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    window = []
    loss_total = torch.zeros((), device=device, dtype=torch.float32)
    valid_total = 0
    gradient_norm_total = torch.zeros((), device=device, dtype=torch.float32)
    images_total = 0
    updates = 0
    lr_start = current_lrs(optimizer)
    lr_last = dict(lr_start)
    epoch_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def update_window(batches):
        nonlocal loss_total, valid_total, gradient_norm_total, images_total, updates, lr_last
        valid_count_window = sum(int(batch["labels"].shape[0]) for batch in batches)
        if valid_count_window <= 0:
            return
        for batch in batches:
            images = batch["images"].to(device, non_blocking=True)
            study_index = batch["study_index"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            known = batch["known"]
            with autocast_context(device, amp_enabled, amp_dtype):
                logits = model(images, study_index, batch["num_studies"])
                loss = criterion(logits, labels, known)
            eligible = int(labels.shape[0])
            if eligible:
                (loss * (eligible / valid_count_window)).backward()
                loss_total = loss_total + loss.detach().float() * eligible
                valid_total += eligible
            images_total += int(images.shape[0])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                              float(config["gradient_clip_norm"]),
                                              error_if_nonfinite=True)
        gradient_norm_total = gradient_norm_total + norm.detach().float()
        lr_last = current_lrs(optimizer)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

    for batch in loader:
        window.append(batch)
        if len(window) == accumulation:
            update_window(window)
            window = []
    if window:
        update_window(window)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - epoch_started
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return {
        "loss": float((loss_total / max(valid_total, 1)).detach()),
        "gradient_norm": float((gradient_norm_total / max(updates, 1)).detach()),
        "optimizer_updates": updates,
        "lr_start": lr_start,
        "lr_last_used": lr_last,
        "elapsed_s": elapsed,
        "images": images_total,
        "images_per_s": images_total / elapsed if elapsed else 0.0,
        "peak_gpu_allocated_gib": peak_allocated,
        "peak_gpu_reserved_gib": peak_reserved,
    }


def run_classifier(experiment, experiments_config_path="configs/hybrid_experiments.yaml",
                   resume=False):
    if experiment not in {"h0", "h1", "h2"}:
        raise ValueError("experiment must be h0, h1, or h2")
    experiments = read_yaml(experiments_config_path)
    data_config = resolve_data_config(experiments["data_config"])
    train_config = read_yaml(experiments["train_config"])
    model_config = read_yaml(experiments["model_config"])
    if (int(train_config["epochs"]) != 20 or int(train_config["warmup_epochs"]) != 2 or
            int(train_config["cosine_end_epoch"]) != 20 or train_config["early_stopping_enabled"]):
        raise ValueError("Hybrid supervised training is fixed to 20 epochs with a 2+18 schedule and no early stopping")
    if train_config["loss"] != "bce" or train_config["weighting"] != "none":
        raise ValueError("H0/H1/H2 must use unweighted Standard BCE")
    if train_config["amp_dtype"] != "bfloat16":
        raise ValueError("Hybrid supervised training requires BF16 on the verified GPU")
    fixed_values = {
        "seed": 42,
        "learning_rate_convnext": 1e-5,
        "learning_rate_vit": 1e-5,
        "learning_rate_head": 2e-4,
        "min_lr_convnext": 5e-7,
        "min_lr_vit": 5e-7,
        "min_lr_head": 1e-5,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "scheduler_step_unit": "optimizer_update",
    }
    for field, expected in fixed_values.items():
        actual = train_config[field]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Hybrid supervised setting {field} must be {expected}")
        elif actual != expected:
            raise ValueError(f"Hybrid supervised setting {field} must be {expected}")
    if (model_config["convnext_model_name"] != "convnextv2_tiny.fcmae_ft_in1k" or
            model_config["vit_model_name"] != "vit_small_patch16_224.augreg_in1k" or
            int(model_config["image_size"]) != 320 or model_config["study_fusion"] != "logit_mean"):
        raise ValueError("Hybrid supervised architecture must match the shared H0/H1/H2 contract")
    if (model_config["feature_fusion"] != "concat" or
            model_config["classification_head"] != "linear" or
            model_config["convnext_pooling"] != "gap" or
            model_config["vit_pooling"] != "cls_token"):
        raise ValueError("Hybrid feature fusion must remain Stage4 GAP + ViT CLS + Linear(1152,10)")
    if train_config.get("image_budget") is None or train_config.get("grad_accumulation") is None:
        raise RuntimeError("GPU benchmark must choose the supervised batch configuration before training")
    output_dir = Path(experiments["outputs"][experiment])
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "best.pth", output_dir / "last.pth"
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing {experiment} run: {output_dir}")

    seed = int(train_config["seed"])
    seed_everything(seed)
    device, amp_enabled, amp_dtype = device_and_amp(
        train_config["amp"], train_config["amp_dtype"])
    initial_path = Path(experiments["initial_weights"])
    if not initial_path.is_absolute():
        initial_path = PROJECT_DIR / initial_path
    initial_metadata_path = initial_path.with_suffix(".json")
    if not initial_path.is_file() or not initial_metadata_path.is_file():
        raise FileNotFoundError("Shared ImageNet initial checkpoint and metadata must exist")
    initial_hash = sha256_file(initial_path)
    initial_metadata = json.loads(initial_metadata_path.read_text(encoding="utf-8"))
    backbone = build_hybrid_backbone(initial_path, map_location="cpu",
                                     initial_checkpoint_sha256=initial_hash)
    encoder_source = None
    if experiment in {"h1", "h2"}:
        pretrain_key = "h1_pretrain" if experiment == "h1" else "h2_pretrain"
        encoder_source = encoder_checkpoint_path(experiments["outputs"][pretrain_key])
        if not encoder_source.is_file():
            raise FileNotFoundError(f"Completed self-supervised encoder checkpoint is missing: {encoder_source}")
        encoder_payload = torch.load(encoder_source, map_location="cpu", weights_only=False)
        backbone.load_state_dict(encoder_payload["backbone"], strict=True)
    torch.manual_seed(seed)
    model = HybridClassifier(backbone, num_labels=int(model_config["num_labels"])).to(device)
    model.set_grad_checkpointing(bool(train_config["gradient_checkpointing"]))

    dataset_options = {
        "label_column": data_config["label_column"],
        "label_semantics": data_config["label_semantics"],
        "image_id_column": data_config["image_id_column"],
        "subject_column": data_config["subject_column"],
        "study_column": data_config["study_column"],
    }
    train_data = StudyDataset(
        data_config["train_csv"], data_config["image_root"],
        transform=build_hybrid_transform(data_config["image_size"], train=True),
        **dataset_options)
    val_data = StudyDataset(
        data_config["val_csv"], data_config["image_root"],
        transform=build_hybrid_transform(data_config["image_size"], train=False),
        **dataset_options)
    if train_data.label_tensors()[1].logical_not().any() or val_data.label_tensors()[1].logical_not().any():
        raise ValueError("Closed-world classification requires a complete ten-label target for every Study")
    train_subjects = {subject for subject, _ in train_data.study_keys}
    val_subjects = {subject for subject, _ in val_data.study_keys}
    if train_subjects & val_subjects:
        raise ValueError(f"Train/Validation Subject overlap detected: {len(train_subjects & val_subjects)}")

    sampler = StudyBatchSampler(train_data, int(train_config["image_budget"]),
                                shuffle=True, seed=seed)
    val_sampler = StudyBatchSampler(val_data, int(train_config["image_budget"]), shuffle=False)
    loader_options = {
        "num_workers": int(train_config["num_workers"]),
        "pin_memory": bool(train_config["pin_memory"]),
        "collate_fn": collate_studies,
        "persistent_workers": bool(train_config["num_workers"]),
        "worker_init_fn": seed_worker if train_config["num_workers"] else None,
    }
    train_generator = torch.Generator().manual_seed(int(train_config["data_loader_worker_seed"]))
    val_generator = torch.Generator().manual_seed(int(train_config["data_loader_worker_seed"]) + 1)
    train_loader = DataLoader(train_data, batch_sampler=sampler,
                              generator=train_generator, **loader_options)
    val_loader = DataLoader(val_data, batch_sampler=val_sampler,
                            generator=val_generator, **loader_options)
    val_image_count = int(sum(val_data.image_counts))

    groups = [
        {"params": model.backbone.convnext.parameters(), "lr": float(train_config["learning_rate_convnext"]), "name": "convnext"},
        {"params": model.backbone.vit.parameters(), "lr": float(train_config["learning_rate_vit"]), "name": "vit"},
        {"params": model.classifier.parameters(), "lr": float(train_config["learning_rate_head"]), "name": "head"},
    ]
    optimizer = build_adamw(groups, train_config["weight_decay"],
                            fused=train_config["optimizer_fused"])
    accumulation = int(train_config["grad_accumulation"])
    updates_per_epoch = []
    for epoch_index in range(int(train_config["epochs"])):
        sampler.set_epoch(epoch_index)
        updates_per_epoch.append(update_counts_for_epoch(len(sampler), accumulation))
    scheduler, scheduler_info = build_update_scheduler(
        optimizer, updates_per_epoch, train_config["warmup_epochs"],
        train_config["cosine_end_epoch"], min_factor=0.05)
    criterion = StandardStudyBCE().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    audit_path = PROJECT_DIR / "outputs/hybrid/shared/manifests/data_audit.json"
    smoke_path = PROJECT_DIR / "outputs/hybrid/smoke_test/hybrid_smoke_test.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Official split audit must complete before training")
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise RuntimeError("Hybrid real-sample Smoke Test must pass before supervised training")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    config = {
        "experiment": experiment,
        "run_dir": str(output_dir),
        "data": {key: str(data_config[key]) for key in ("image_root", "train_csv", "val_csv")},
        "data_label_semantics": data_config["label_semantics"],
        "train_manifest_sha256": audit["train_manifest_sha256"],
        "valid_manifest_sha256": audit["valid_manifest_sha256"],
        "test_split_read": False,
        "model": model_config,
        "training": train_config,
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial_hash,
        "initial_weight_metadata": initial_metadata,
        "self_supervised_encoder_checkpoint": str(encoder_source) if encoder_source else None,
        "self_supervised_encoder_sha256": sha256_file(encoder_source) if encoder_source else None,
        "timm_version": __import__("timm").__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "scheduler": scheduler_info,
        "parameter_counts": {
            "convnext_v2": count_parameters(model.backbone.convnext),
            "vit": count_parameters(model.backbone.vit),
            "classification_head": count_parameters(model.classifier),
            "total": count_parameters(model),
            "trainable": count_parameters(model, trainable_only=True),
        },
        "input_shape": [1, 3, data_config["image_size"], data_config["image_size"]],
        "feature_fusion_shape": [1, 1152],
        "output_shape": [1, 10],
        "profiler_flops_per_image": smoke.get("classifier_profiler_flops_per_image"),
        "loss": "Standard BCEWithLogitsLoss; equal valid-label mean within each Study, then equal Study mean",
        "study_fusion": "logit_mean",
    }
    write_json(output_dir / "config.json", config)
    logger = __import__("hybrid.common", fromlist=["create_logger"]).create_logger(
        output_dir / "logs/train.log", f"hybrid_{experiment}")
    logger.info("experiment=%s device=%s gpu=%s image_budget=%d grad_accumulation=%d "
                "checkpointing=%s initial_sha256=%s",
                experiment, device, torch.cuda.get_device_name(device),
                train_config["image_budget"], accumulation,
                train_config["gradient_checkpointing"], initial_hash)

    history_path = output_dir / "train_log.csv"
    classes_path = output_dir / "per_class_metrics.csv"
    history = _read_csv(history_path) if resume else []
    per_class_rows = _read_csv(classes_path) if resume else []
    start_epoch, best_metric, best_epoch, stale_epochs = 0, -math.inf, 0, 0
    if resume:
        if not last_path.is_file() or not Path(resume).is_file() or not Path(resume).samefile(last_path):
            raise ValueError("Supervised resume is restricted to this experiment's own last.pth")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config") != config:
            raise ValueError("Resume classifier configuration differs from the saved run configuration")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state)
        train_generator.set_state(state["train_loader_generator_state"].cpu())
        val_generator.set_state(state["val_loader_generator_state"].cpu())
        start_epoch = int(state["epoch"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        logger.info("resumed last checkpoint at epoch %d", start_epoch)

    for epoch_index in range(start_epoch, int(train_config["epochs"])):
        epoch = epoch_index + 1
        sampler.set_epoch(epoch_index)
        train_stats = _train_epoch(model, train_loader, criterion, optimizer, scheduler,
                                   train_config, device, amp_enabled, amp_dtype)
        val_started = time.perf_counter()
        validation = validate(model, val_loader, criterion, device, amp_enabled, amp_dtype)
        validation_seconds = time.perf_counter() - val_started
        scheduler_lrs = current_lrs(optimizer)
        valid_auprcs = [item["auprc"] for item in validation["per_label"]
                        if math.isfinite(float(item["auprc"]))]
        macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")
        score = float(validation["macro_auroc"])
        improved = math.isfinite(score) and score > best_metric
        if improved:
            best_metric, best_epoch, stale_epochs = score, epoch, 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": float(validation["loss"]),
            "train_val_gap": float(validation["loss"] - train_stats["loss"]),
            "macro_auroc": score,
            "macro_auprc": macro_auprc,
            "macro_f1": float(validation["macro_f1"]),
            "lr_convnext": train_stats["lr_start"]["convnext"],
            "lr_vit": train_stats["lr_start"]["vit"],
            "lr_head": train_stats["lr_start"]["head"],
            "lr_convnext_next": scheduler_lrs["convnext"],
            "lr_vit_next": scheduler_lrs["vit"],
            "lr_head_next": scheduler_lrs["head"],
            "gradient_norm": train_stats["gradient_norm"],
            "optimizer_updates": train_stats["optimizer_updates"],
            "train_images": train_stats["images"],
            "train_s": train_stats["elapsed_s"],
            "validation_s": validation_seconds,
            "validation_studies_per_s": len(val_data) / validation_seconds if validation_seconds else 0.0,
            "validation_images_per_s": val_image_count / validation_seconds if validation_seconds else 0.0,
            "images_per_s": train_stats["images_per_s"],
            "peak_gpu_allocated_gib": train_stats["peak_gpu_allocated_gib"],
            "peak_gpu_reserved_gib": train_stats["peak_gpu_reserved_gib"],
            "best_epoch": best_epoch,
            "best_macro_auroc": best_metric,
        }
        history.append(row)
        for class_index, item in enumerate(validation["per_label"]):
            per_class_rows.append({"epoch": epoch,
                                   "class_name": data_config["class_names"][class_index],
                                   **item})
        write_csv(history_path, history)
        write_csv(classes_path, per_class_rows)
        predictions = _validation_predictions(validation, data_config["class_names"])
        write_csv(output_dir / "predictions" / f"val_epoch_{epoch:02d}.csv", predictions)
        _write_curves(output_dir, history, per_class_rows)

        state = _checkpoint_payload(model, optimizer, scheduler, scaler, epoch,
                                    best_metric, best_epoch, stale_epochs, config,
                                    train_generator, val_generator)
        _save_torch(last_path, state)
        if improved:
            _save_torch(best_path, state)
            write_csv(output_dir / "predictions/val_best.csv", predictions)
        if epoch == int(train_config["epochs"]):
            _save_torch(output_dir / f"epoch{epoch}.pth", state)

        logger.info("epoch=%d/%d train_loss=%.6f valid_loss=%.6f macro_auroc=%.6f "
                    "macro_auprc=%.6f macro_f1=%.6f lr=%s updates=%d images=%d "
                    "images_s=%.2f peak_allocated_gib=%.3f peak_reserved_gib=%.3f "
                    "best_epoch=%d best_macro_auroc=%.6f",
                    epoch, train_config["epochs"], row["train_loss"], row["val_loss"],
                    row["macro_auroc"], row["macro_auprc"], row["macro_f1"],
                    {key: scheduler_lrs[key] for key in ("convnext", "vit", "head")},
                    row["optimizer_updates"], row["train_images"], row["images_per_s"],
                    row["peak_gpu_allocated_gib"], row["peak_gpu_reserved_gib"],
                    best_epoch, best_metric)
        _write_report(output_dir, experiment, config, history, per_class_rows,
                      complete=(epoch == int(train_config["epochs"])))

    return {"experiment": experiment, "best_epoch": best_epoch,
            "best_macro_auroc": best_metric, "completed_epochs": len(history),
            "output_dir": str(output_dir)}


def _write_report(output_dir, experiment, config, history, per_class_rows, complete):
    best_row = max(history, key=lambda row: float(row["macro_auroc"]))
    status = "completed 20 epochs" if complete else f"in progress through epoch {history[-1]['epoch']}"
    best_size = (output_dir / "best.pth").stat().st_size if (output_dir / "best.pth").is_file() else 0
    last_size = (output_dir / "last.pth").stat().st_size if (output_dir / "last.pth").is_file() else 0
    lines = [
        f"# Hybrid {experiment.upper()} classification report", "",
        f"- Status: {status}",
        f"- Best Epoch: {best_row['epoch']}",
        f"- Best Validation Macro-AUROC: {float(best_row['macro_auroc']):.6f}",
        f"- Best Train Loss / Valid Loss: {float(best_row['train_loss']):.6f} / {float(best_row['val_loss']):.6f}",
        f"- Best Macro-AUPRC / Macro-F1: {float(best_row['macro_auprc']):.6f} / {float(best_row['macro_f1']):.6f}",
        f"- Best checkpoint: `{output_dir / 'best.pth'}`",
        f"- Last checkpoint: `{output_dir / 'last.pth'}`",
        "- Test split was not read.",
        "",
        "## Configuration and provenance", "",
        f"- ConvNeXtV2: `{config['model']['convnext_model_name']}`; ViT: `{config['model']['vit_model_name']}`",
        f"- ImageNet initialization SHA-256: `{config['initial_checkpoint_sha256']}`",
        f"- Train/Validation manifest SHA-256: `{config['train_manifest_sha256']}` / `{config['valid_manifest_sha256']}`",
        f"- Device: `{config['device_name']}`; PyTorch `{config['torch_version']}`; timm `{config['timm_version']}`",
        f"- Parameters: CNN {config['parameter_counts']['convnext_v2']:,}; ViT {config['parameter_counts']['vit']:,}; "
        f"head {config['parameter_counts']['classification_head']:,}; total {config['parameter_counts']['total']:,}",
        "- Study outputs are the arithmetic mean of image-level raw logits; the loss is unweighted Standard BCE.",
        "- Input is 320×320 ResizePad and one shared light augmentation, then each branch uses its own ImageNet normalization.",
        "",
        "## Per-epoch classification metrics", "",
        f"- Classifier FLOPs per image (torch.profiler estimate): `{config.get('profiler_flops_per_image')}`",
        f"- Best / Last checkpoint size: {best_size:,} / {last_size:,} bytes",
        "| Epoch | Train Loss | Valid Loss | Macro-AUROC | Macro-AUPRC | Macro-F1 | CNN/ViT/Head LR | Train images/s | Val Studies/s | Val Images/s | Peak Allocated/Reserved GiB |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {float(row['train_loss']):.6f} | {float(row['val_loss']):.6f} | "
            f"{float(row['macro_auroc']):.6f} | {float(row['macro_auprc']):.6f} | "
            f"{float(row['macro_f1']):.6f} | {float(row['lr_convnext']):.3e}/"
            f"{float(row['lr_vit']):.3e}/{float(row['lr_head']):.3e} | "
            f"{float(row['images_per_s']):.2f} | {float(row['validation_studies_per_s']):.2f} | "
            f"{float(row['validation_images_per_s']):.2f} | "
            f"{float(row['peak_gpu_allocated_gib']):.3f}/{float(row['peak_gpu_reserved_gib']):.3f} |"
        )
    best_classes = [row for row in per_class_rows if int(row["epoch"]) == int(best_row["epoch"])]
    lines.extend(["", "## Best-epoch per-class metrics", "",
                  "| Class | AUROC | AUPRC | F1@0.5 | Positive Studies | Negative Studies | TP | FP | TN | FN |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for item in best_classes:
        lines.append(f"| {item['class_name']} | {float(item['auroc']):.6f} | "
                     f"{float(item['auprc']):.6f} | {float(item['f1']):.6f} | "
                     f"{int(item['n_positive'])} | {int(item['n_negative'])} | "
                     f"{int(item['tp'])} | {int(item['fp'])} | {int(item['tn'])} | {int(item['fn'])} |")
    (output_dir / "hybrid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("h0", "h1", "h2"), required=True)
    parser.add_argument("--experiments-config", default="configs/hybrid_experiments.yaml")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    result = run_classifier(args.experiment, args.experiments_config, resume=args.resume or False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
