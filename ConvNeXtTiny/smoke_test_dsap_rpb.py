"""Run synthetic shape/gradient checks and one real-Study training/inference smoke pass."""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from engine.losses import MaskedBCEWithLogitsLoss
from engine.trainer import train_one_epoch
from model import ChestXrayClassifier
from model.classifier import MultiLabelHead
from model.study_pooling import pool_by_study
from train import build_lr_scheduler
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


def _check_gradient(parameter, name):
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError(f"{name} has no finite gradient")
    if float(parameter.grad.abs().sum()) <= 0:
        raise RuntimeError(f"{name} has a zero gradient")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default=str(PROJECT_DIR / "configs/train_dsap_rpb.yaml"))
    parser.add_argument("--model-config", default=str(PROJECT_DIR / "configs/model_dsap_rpb.yaml"))
    args = parser.parse_args()
    torch.manual_seed(42)
    data_path = resolve_config_path(PROJECT_DIR / "configs/data.yaml")
    train_path = resolve_config_path(args.train_config)
    model_path = resolve_config_path(args.model_config)
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    train_cfg = read_yaml(train_path)
    model_cfg = read_yaml(model_path)
    output_dir = (PROJECT_DIR / train_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "smoke_test_summary.json"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChestXrayClassifier(**{key: value for key, value in model_cfg.items()
                                      if key != "name"}).to(device)
        image_size = int(data_cfg["image_size"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synthetic = torch.randn(8, 3, image_size, image_size, device=device)
        study_index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=device)
        amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
        amp_dtype = torch.bfloat16 if train_cfg["amp_dtype"] == "bfloat16" else torch.float16
        if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            amp_dtype = torch.float16

        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                                    enabled=amp_enabled):
            image_logits, diagnostics = model(synthetic, return_attention=True)
            study_logits = model(synthetic, study_index, num_studies=3)
            separate_image_logits = model(synthetic)
        feature_shape = diagnostics["feature_shape"]
        height, width = feature_shape[-2:]
        if tuple(image_logits.shape) != (8, 10):
            raise RuntimeError(f"Invalid image logits shape: {tuple(image_logits.shape)}")
        if tuple(study_logits.shape) != (3, 10):
            raise RuntimeError(f"Invalid Study logits shape: {tuple(study_logits.shape)}")
        if tuple(diagnostics["disease_attention"].shape) != (8, 10, height, width):
            raise RuntimeError("Disease Attention Map shape does not match the actual feature map")
        if tuple(diagnostics["spatial_attention"].shape) != (8, 4, height * width, height * width):
            raise RuntimeError("Spatial Self-Attention shape is incorrect")
        if not torch.allclose(
            study_logits,
            pool_by_study(separate_image_logits, study_index, num_studies=3),
            atol=1e-5, rtol=1e-5,
        ):
            raise RuntimeError("Study logit_mean does not match the average of image logits")
        if not torch.isfinite(diagnostics["spatial_attention"]).all():
            raise FloatingPointError("Spatial attention contains NaN or Inf")
        if not torch.isfinite(diagnostics["disease_attention"]).all():
            raise FloatingPointError("Disease attention contains NaN or Inf")
        if not torch.allclose(
            diagnostics["disease_attention"].sum(dim=(-2, -1)),
            torch.ones(8, 10, device=device), atol=1e-4, rtol=1e-4,
        ):
            raise RuntimeError("Disease attention weights do not sum to one over spatial tokens")
        initial_alpha = diagnostics["alpha"]
        if not torch.allclose(initial_alpha, torch.full_like(initial_alpha, 0.1), atol=1e-5):
            raise RuntimeError("Disease fusion alpha did not start at 0.1")

        criterion = MaskedBCEWithLogitsLoss().to(device)
        targets = torch.randint(0, 2, (8, 10), device=device).float()
        known = torch.ones_like(targets, dtype=torch.bool)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            synthetic_logits = model(synthetic)
            synthetic_loss = criterion(synthetic_logits.float(), targets, known)
        synthetic_loss.backward()
        _check_gradient(next(model.backbone.parameters()), "ConvNeXt backbone")
        _check_gradient(model.classifier.disease_queries, "Disease Query")
        _check_gradient(model.classifier.spatial_attention.relative_position_bias.bias_table,
                        "Relative Position Bias")
        _check_gradient(model.classifier.spatial_attention.q_projection.weight, "Spatial Q")
        _check_gradient(model.classifier.spatial_attention.k_projection.weight, "Spatial K")
        _check_gradient(model.classifier.spatial_attention.v_projection.weight, "Spatial V")
        _check_gradient(model.classifier.alpha_logits, "alpha")
        _check_gradient(model.classifier.classifier_weight, "Disease classifiers")
        initial_alpha_values = [float(value) for value in diagnostics["alpha"].detach().cpu()]
        synthetic_peak_memory = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                                 if device.type == "cuda" else 0.0)
        model.zero_grad(set_to_none=True)
        del synthetic, synthetic_logits, synthetic_loss, image_logits, study_logits
        del separate_image_logits, diagnostics, initial_alpha
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if data_cfg.get("label_semantics") != "closed_world":
            raise RuntimeError("The configured data semantics are not the confirmed closed-world rule")
        columns = dict(image_id_column=data_cfg["image_id_column"],
                       subject_column=data_cfg["subject_column"],
                       study_column=data_cfg["study_column"])
        train_data = StudyDataset(
            data_cfg["train_csv"], data_cfg["image_root"],
            build_transform(image_size, train=True), data_cfg["label_column"],
            data_cfg["label_semantics"], **columns,
        )
        val_data = StudyDataset(
            data_cfg["val_csv"], data_cfg["image_root"],
            build_transform(image_size), data_cfg["label_column"],
            data_cfg["label_semantics"], **columns,
        )
        for split, dataset in (("train", train_data), ("validation", val_data)):
            _, valid = dataset.label_tensors()
            if not bool(valid.all()):
                raise RuntimeError(f"{split} contains incomplete labels under closed_world semantics")
        train_loader = DataLoader(
            train_data,
            batch_sampler=StudyBatchSampler(train_data, train_cfg["image_budget"], shuffle=False,
                                             seed=train_cfg["seed"]),
            num_workers=0, collate_fn=collate_studies,
        )
        val_loader = DataLoader(
            val_data,
            batch_sampler=StudyBatchSampler(val_data, train_cfg["image_budget"], shuffle=False),
            num_workers=0, collate_fn=collate_studies,
        )
        train_batch, val_batch = next(iter(train_loader)), next(iter(val_loader))
        backbone_count = sum(parameter.numel() for parameter in model.backbone.parameters())
        gap_head_count = sum(parameter.numel() for parameter in
                             MultiLabelHead(model.backbone.channels["stage4"], 10, normalize=True).parameters())
        parameter_counts = {
            "original_gap_total": backbone_count + gap_head_count,
            "modified_total": sum(parameter.numel() for parameter in model.parameters()),
        }
        parameter_counts["new_parameters"] = (
            parameter_counts["modified_total"] - parameter_counts["original_gap_total"]
        )
        groups = [
            {"params": model.backbone.parameters(), "lr": train_cfg["learning_rate_backbone"]},
            {"params": model.classifier.parameters(), "lr": train_cfg["learning_rate_head"]},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])
        optimized_parameter_ids = {id(parameter) for group in optimizer.param_groups
                                   for parameter in group["params"]}
        trainable_parameter_ids = {id(parameter) for parameter in model.parameters()
                                   if parameter.requires_grad}
        if optimized_parameter_ids != trainable_parameter_ids:
            missing = len(trainable_parameter_ids - optimized_parameter_ids)
            extra = len(optimized_parameter_ids - trainable_parameter_ids)
            raise RuntimeError(f"Optimizer parameter coverage mismatch: missing={missing}, extra={extra}")
        accumulation = int(train_cfg["grad_accumulation"])
        schedule_sampler = StudyBatchSampler(
            train_data, train_cfg["image_budget"], shuffle=True, seed=train_cfg["seed"]
        )
        updates_per_epoch = []
        for epoch_index in range(int(train_cfg["epochs"])):
            schedule_sampler.set_epoch(epoch_index)
            updates_per_epoch.append(math.ceil(len(schedule_sampler) / accumulation))
        schedule_sampler.set_epoch(0)
        scheduler, scheduler_metadata = build_lr_scheduler(
            optimizer, train_cfg, updates_per_epoch
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and amp_dtype == torch.float16
        )
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        lr_before_update = [float(group["lr"]) for group in optimizer.param_groups]
        train_stats = train_one_epoch(
            model, [train_batch], criterion, optimizer, device, scaler, amp_enabled,
            amp_dtype, accumulation, train_cfg["gradient_clip_norm"],
            scheduler=scheduler, scheduler_step_unit="optimizer_update",
        )
        if train_stats["optimizer_updates"] != 1:
            raise RuntimeError("Real-data smoke pass did not perform one optimizer update")
        lr_after_update = [float(group["lr"]) for group in optimizer.param_groups]
        if lr_before_update == lr_after_update:
            raise RuntimeError("Update-based scheduler did not advance after the optimizer update")
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                                    enabled=amp_enabled):
            val_images = val_batch["images"].to(device, non_blocking=True)
            val_index = val_batch["study_index"].to(device, non_blocking=True)
            val_logits = model(val_images, val_index, val_batch["num_studies"])
            val_labels = val_batch["labels"].to(device, non_blocking=True)
            val_known = val_batch["known"].to(device, non_blocking=True)
            val_loss = criterion(val_logits.float(), val_labels, val_known)
        if not torch.isfinite(val_loss):
            raise FloatingPointError("Real-data validation loss is non-finite")
        real_peak_memory = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                            if device.type == "cuda" else 0.0)
        peak_memory = max(synthetic_peak_memory, real_peak_memory)

        with tempfile.TemporaryDirectory(prefix="dsap_rpb_smoke_", dir=output_dir) as temp_name:
            temp_dir = Path(temp_name)
            checkpoint_path = temp_dir / "smoke_checkpoint.pth"
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler,
                epoch=1, best_metric=0.5, config={"model_config": model_cfg},
                labels=data_cfg["class_names"], stale_epochs=0, best_epoch=1,
            )
            reloaded = ChestXrayClassifier(**{key: value for key, value in model_cfg.items()
                                              if key != "name" and key != "pretrained"},
                                           pretrained=False).to(device)
            optimizer_reloaded = torch.optim.AdamW([
                {"params": reloaded.backbone.parameters(), "lr": train_cfg["learning_rate_backbone"]},
                {"params": reloaded.classifier.parameters(), "lr": train_cfg["learning_rate_head"]},
            ], weight_decay=train_cfg["weight_decay"])
            scheduler_reloaded, _ = build_lr_scheduler(
                optimizer_reloaded, train_cfg, updates_per_epoch
            )
            scaler_reloaded = torch.amp.GradScaler(
                "cuda", enabled=amp_enabled and amp_dtype == torch.float16
            )
            state = load_checkpoint(
                checkpoint_path, reloaded, optimizer_reloaded, scheduler_reloaded,
                scaler_reloaded, map_location=device,
            )
            if state["best_epoch"] != 1:
                raise RuntimeError("Checkpoint did not preserve best_epoch metadata")

            first_study = val_data.studies[0]
            smoke_csv = temp_dir / "valid.csv"
            pd.DataFrame({
                data_cfg["image_id_column"]: first_study["image_ids"],
                data_cfg["subject_column"]: [first_study["subject_id"]] * len(first_study["image_ids"]),
                data_cfg["study_column"]: [first_study["study_id"]] * len(first_study["image_ids"]),
            }).to_csv(smoke_csv, index=False, encoding="utf-8")
            prediction_path = temp_dir / "validation_smoke_predictions.csv"
            prediction_env = os.environ.copy()
            prediction_env["PYTHONIOENCODING"] = "utf-8"
            prediction = subprocess.run([
                sys.executable, str(PROJECT_DIR / "predict.py"),
                "--csv", str(smoke_csv), "--checkpoint", str(checkpoint_path),
                "--output", str(prediction_path),
            ], cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
                encoding="utf-8", env=prediction_env)
            prediction_frame = pd.read_csv(prediction_path, dtype={"Subject_id": "string",
                                                                    "Study_id": "string"})
            expected_columns = ["Subject_id", "Study_id", "Predict_class", "Probability"]
            if prediction_frame.columns.tolist() != expected_columns:
                raise RuntimeError("Prediction smoke output does not match official column order")
            if len(prediction_frame) != 10 or set(prediction_frame["Predict_class"]) != set(range(1, 11)):
                raise RuntimeError("Prediction smoke output does not contain exactly 10 label rows")
            if not prediction_frame["Probability"].between(0.0, 1.0).all():
                raise RuntimeError("Prediction smoke generated probabilities outside [0,1]")

        result = {
            "passed": True,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "amp_enabled": amp_enabled,
            "amp_dtype": str(amp_dtype),
            "input_size": image_size,
            "backbone_stage4_shape": list(feature_shape),
            "position_encoding_shape": [1, height * width, model_cfg["dsap_rpb_d_model"]],
            "spatial_attention_shape": [8, model_cfg["dsap_rpb_heads"], height * width, height * width],
            "disease_attention_shape": [8, 10, height, width],
            "study_logit_mean_passed": True,
            "initial_alpha": initial_alpha_values,
            "synthetic_gradient_checks": ["backbone", "disease_queries", "relative_position_bias",
                                          "spatial_qkv", "alpha", "disease_classifier"],
            "real_train_loss": float(train_stats["loss"]),
            "real_val_loss": float(val_loss.detach().float().cpu()),
            "real_optimizer_updates": train_stats["optimizer_updates"],
            "backbone_frozen_for_first_epoch_smoke": True,
            "optimizer_parameter_coverage_passed": True,
            "scheduler_advanced_after_update": True,
            "learning_rate_before_update": lr_before_update,
            "learning_rate_after_update": lr_after_update,
            "scheduler_metadata": scheduler_metadata,
            "checkpoint_reload_passed": True,
            "validation_predict_script_passed": True,
            "validation_prediction_rows": int(len(prediction_frame)),
            "parameter_counts": parameter_counts,
            "synthetic_peak_gpu_gb": synthetic_peak_memory,
            "real_train_peak_gpu_gb": real_peak_memory,
            "smoke_peak_gpu_gb": peak_memory,
            "validation_predict_stdout": (prediction.stdout or "").strip(),
            "test_split_accessed": False,
        }
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        summary_path.write_text(json.dumps({"passed": False, "error": repr(exc)},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
