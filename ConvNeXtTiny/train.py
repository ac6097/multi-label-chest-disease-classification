import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from engine.losses import AsymmetricLoss, MaskedBCEWithLogitsLoss, estimate_pos_weight
from engine.trainer import fit
from model import ChestXrayClassifier
from model.classifier import MultiLabelHead
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths
from utils.logger import create_logger
from utils.seed import seed_everything


def _optimizer_update_schedule_spec(train_config, updates_per_epoch):
    epochs = int(train_config["epochs"])
    warmup_epochs = int(train_config["warmup_epochs"])
    cosine_end_epoch = int(train_config.get("cosine_end_epoch", epochs))
    if isinstance(updates_per_epoch, int):
        updates_by_epoch = [int(updates_per_epoch)] * epochs
    else:
        updates_by_epoch = [int(value) for value in updates_per_epoch]
    if len(updates_by_epoch) != epochs or any(value < 1 for value in updates_by_epoch):
        raise ValueError("Provide one positive optimizer-update count for every training epoch")
    if not warmup_epochs < cosine_end_epoch <= epochs:
        raise ValueError("cosine_end_epoch must be after warmup and no later than epochs")
    peak_lrs = [float(train_config["learning_rate_backbone"]),
                float(train_config["learning_rate_head"])]
    min_lrs = [float(train_config.get("min_lr_backbone", 0.0)),
               float(train_config.get("min_lr_head", 0.0))]
    if any(peak <= 0 for peak in peak_lrs):
        raise ValueError("Peak learning rates must be positive")
    if any(minimum < 0 or minimum > peak for minimum, peak in zip(min_lrs, peak_lrs)):
        raise ValueError("Minimum learning rates must be between zero and their peak rates")
    min_factors = [minimum / peak for minimum, peak in zip(min_lrs, peak_lrs)]
    if not math.isclose(min_factors[0], min_factors[1], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Backbone and Head minimum LR ratios must match to preserve their LR ratio")
    total_steps = sum(updates_by_epoch)
    warmup_steps = sum(updates_by_epoch[:warmup_epochs])
    cosine_end_step = sum(updates_by_epoch[:cosine_end_epoch])
    cosine_steps = cosine_end_step - warmup_steps
    return {
        "unit": "optimizer_update",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "updates_per_epoch": updates_by_epoch,
        "cosine_end_epoch": cosine_end_epoch,
        "cosine_end_step": cosine_end_step,
        "cosine_steps": cosine_steps,
        "min_lr_by_group": min_lrs,
        "min_lr_factor": min_factors[0],
        "peak_lr_by_group": peak_lrs,
    }


def optimizer_update_lr_factor(step, spec):
    if step < spec["warmup_steps"]:
        return (step + 1) / spec["warmup_steps"]
    cosine_index = step - spec["warmup_steps"]
    if cosine_index < spec["cosine_steps"]:
        progress = cosine_index / max(spec["cosine_steps"] - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return spec["min_lr_factor"] + (1.0 - spec["min_lr_factor"]) * cosine
    return spec["min_lr_factor"]


def build_lr_schedule_table(train_config, updates_per_epoch):
    """Return the LR actually used by the first and last optimizer updates of each epoch."""
    spec = _optimizer_update_schedule_spec(train_config, updates_per_epoch)
    rows = []
    cumulative_steps = [0]
    for count in spec["updates_per_epoch"]:
        cumulative_steps.append(cumulative_steps[-1] + count)
    for epoch in range(1, int(train_config["epochs"]) + 1):
        first_step = cumulative_steps[epoch - 1]
        last_step = cumulative_steps[epoch] - 1
        next_step = cumulative_steps[epoch]
        if epoch <= int(train_config["warmup_epochs"]):
            phase = "warmup"
        elif epoch <= spec["cosine_end_epoch"]:
            phase = "cosine"
        else:
            phase = "low_lr_finetune"
        rows.append({
            "epoch": epoch,
            "optimizer_updates": spec["updates_per_epoch"][epoch - 1],
            "phase": phase,
            "backbone_lr_start": spec["peak_lr_by_group"][0] * optimizer_update_lr_factor(first_step, spec),
            "head_lr_start": spec["peak_lr_by_group"][1] * optimizer_update_lr_factor(first_step, spec),
            "backbone_lr_last_used": spec["peak_lr_by_group"][0] * optimizer_update_lr_factor(last_step, spec),
            "head_lr_last_used": spec["peak_lr_by_group"][1] * optimizer_update_lr_factor(last_step, spec),
            "backbone_lr_next": spec["peak_lr_by_group"][0] * optimizer_update_lr_factor(next_step, spec),
            "head_lr_next": spec["peak_lr_by_group"][1] * optimizer_update_lr_factor(next_step, spec),
        })
    return spec, rows


def build_lr_scheduler(optimizer, train_config, updates_per_epoch):
    """Build the configured warmup/cosine schedule in epoch or optimizer-update units."""
    epochs = int(train_config["epochs"])
    warmup_epochs = int(train_config["warmup_epochs"])
    unit = train_config.get("scheduler_step_unit", "epoch")
    if unit == "optimizer_update":
        spec = _optimizer_update_schedule_spec(train_config, updates_per_epoch)
        if len(optimizer.param_groups) != len(spec["peak_lr_by_group"]):
            raise ValueError("The update-level schedule expects exactly Backbone and Head LR groups")

        def schedule(step):
            return optimizer_update_lr_factor(step, spec)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), spec
    if unit == "epoch":
        def schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs + 1) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        update_counts = ([int(value) for value in updates_per_epoch]
                         if not isinstance(updates_per_epoch, int)
                         else int(updates_per_epoch))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), {
            "unit": unit, "total_steps": epochs, "warmup_steps": warmup_epochs,
            "updates_per_epoch": update_counts,
        }
    raise ValueError("scheduler_step_unit must be epoch or optimizer_update")


def _git_provenance():
    repository_root = PROJECT_DIR.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, capture_output=True,
        text=True, check=True, encoding="utf-8"
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository_root, capture_output=True,
        text=True, check=True, encoding="utf-8"
    ).stdout.strip())
    return revision, dirty


def _imagenet_weight_metadata(pretrained):
    if not pretrained:
        return None, None
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not weight_path.is_file():
        raise FileNotFoundError(f"Loaded torchvision weight file was not found: {weight_path}")
    digest = hashlib.sha256()
    with weight_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return str(weights), digest.hexdigest()


def _validate_dsap_resume(output_dir, checkpoint_name, resume_path, model_cfg, train_cfg, data_cfg):
    checkpoint_path = Path(resume_path)
    expected_path = output_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if not expected_path.is_file() or not checkpoint_path.samefile(expected_path):
        raise ValueError(
            f"DSAP-RPB can resume only from this experiment's last checkpoint: {expected_path}"
        )
    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved = metadata.get("config") or {}
    if saved.get("model_config") != model_cfg:
        raise ValueError("Resume checkpoint model configuration does not match the current DSAP-RPB model")
    fixed_train_fields = (
        "seed", "data_loader_worker_seed", "epochs", "image_budget", "grad_accumulation",
        "learning_rate_backbone", "learning_rate_head", "weight_decay",
        "warmup_epochs", "patience", "freeze_backbone_epochs",
        "gradient_clip_norm", "loss", "amp", "amp_dtype", "scheduler_step_unit",
    )
    for field in fixed_train_fields:
        if saved.get(field) != train_cfg.get(field):
            raise ValueError(f"Resume checkpoint {field} does not match the current training config")
    saved_data = saved.get("data") or {}
    for field in ("image_root", "train_csv", "val_csv"):
        saved_path = Path(saved_data[field]) if saved_data.get(field) else None
        current_path = Path(data_cfg[field]) if data_cfg.get(field) else None
        if saved_path is None or current_path is None or not saved_path.samefile(current_path):
            raise ValueError(f"Resume checkpoint {field} does not match the current data path")
    if metadata.get("epoch", 0) < 1:
        raise ValueError("Resume checkpoint metadata has no completed epoch")
    return metadata


def _validate_spatial_resume(output_dir, checkpoint_name, resume_path, model_cfg, train_cfg, data_cfg):
    checkpoint_path = Path(resume_path)
    expected_path = output_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if not expected_path.is_file() or not checkpoint_path.samefile(expected_path):
        raise ValueError(f"Spatial ablations can resume only from this run's last checkpoint: {expected_path}")
    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved = metadata.get("config") or {}
    if saved.get("model_config") != model_cfg:
        raise ValueError("Resume checkpoint model configuration does not match this spatial ablation")
    fixed_fields = (
        "seed", "epochs", "image_budget", "grad_accumulation",
        "learning_rate_backbone", "learning_rate_head", "min_lr_backbone", "min_lr_head",
        "weight_decay", "warmup_epochs", "cosine_end_epoch", "early_stopping_enabled",
        "freeze_backbone_epochs", "gradient_clip_norm", "loss", "weighting", "amp",
        "amp_dtype", "scheduler_step_unit",
    )
    for field in fixed_fields:
        if saved.get(field) != train_cfg.get(field):
            raise ValueError(f"Resume checkpoint {field} does not match the current spatial config")
    saved_data = saved.get("data") or {}
    for field in ("image_root", "train_csv", "val_csv"):
        saved_path = Path(saved_data[field]) if saved_data.get(field) else None
        current_path = Path(data_cfg[field]) if data_cfg.get(field) else None
        if saved_path is None or current_path is None or not saved_path.samefile(current_path):
            raise ValueError(f"Resume checkpoint {field} does not match the current data path")
    if metadata.get("epoch", 0) < 1:
        raise ValueError("Resume checkpoint metadata has no completed epoch")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train the ConvNeXt-Tiny Study-level route")
    parser.add_argument("--data-config", default=str(PROJECT_DIR / "configs/data.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_DIR / "configs/train.yaml"))
    parser.add_argument("--model-config", default=str(PROJECT_DIR / "configs/model.yaml"))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    data_path, train_path, model_path = map(resolve_config_path,
        (args.data_config, args.train_config, args.model_config))
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    train_cfg, model_cfg = read_yaml(train_path), read_yaml(model_path)
    if args.resume:
        train_cfg["resume"] = str(resolve_config_path(args.resume))
    output_dir = Path(train_cfg.get("output_dir", "."))
    train_cfg["output_dir"] = str(output_dir if output_dir.is_absolute()
                                  else (PROJECT_DIR / output_dir).resolve())
    output_dir = Path(train_cfg["output_dir"])

    smoke_test = None
    if model_cfg.get("head") == "dsap_rpb":
        if not model_cfg.get("pretrained", False):
            raise ValueError("DSAP-RPB requires torchvision ImageNet pretrained backbone weights")
        best_checkpoint_name = train_cfg.get("best_checkpoint_name", "best.pth")
        last_checkpoint_name = train_cfg.get("last_checkpoint_name", "last.pth")
        if train_cfg.get("resume"):
            resume_metadata = _validate_dsap_resume(
                output_dir, last_checkpoint_name, train_cfg["resume"], model_cfg,
                train_cfg, data_cfg,
            )
        else:
            resume_metadata = None
            checkpoint_names = [best_checkpoint_name, last_checkpoint_name]
            checkpoint_names.extend(
                f"epoch{int(epoch)}.pth" for epoch in train_cfg.get("milestone_epochs", [])
            )
            for checkpoint_name in checkpoint_names:
                if (output_dir / checkpoint_name).exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing experiment checkpoint: {output_dir / checkpoint_name}"
                    )
        if train_cfg.get("requires_smoke_test"):
            smoke_path = output_dir / "smoke_test_summary.json"
            if not smoke_path.is_file():
                raise FileNotFoundError(f"Required DSAP-RPB smoke test is missing: {smoke_path}")
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            if smoke.get("passed") is not True:
                raise RuntimeError("The DSAP-RPB smoke test did not pass")
            smoke_test = smoke
    elif model_cfg.get("head") == "spatial":
        if not model_cfg.get("pretrained", False):
            raise ValueError("Spatial ablations require torchvision ImageNet pretrained backbone weights")
        best_checkpoint_name = train_cfg.get("best_checkpoint_name", "best.pth")
        last_checkpoint_name = train_cfg.get("last_checkpoint_name", "last.pth")
        if train_cfg.get("resume"):
            resume_metadata = _validate_spatial_resume(
                output_dir, last_checkpoint_name, train_cfg["resume"], model_cfg,
                train_cfg, data_cfg,
            )
        else:
            checkpoint_names = [best_checkpoint_name, last_checkpoint_name]
            checkpoint_names.extend(
                f"epoch{int(epoch)}.pth" for epoch in train_cfg.get("milestone_epochs", [])
            )
            for checkpoint_name in checkpoint_names:
                if (output_dir / checkpoint_name).exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing spatial checkpoint: {output_dir / checkpoint_name}"
                    )
        if train_cfg.get("requires_smoke_test"):
            smoke_path = output_dir / "smoke_test_summary.json"
            if not smoke_path.is_file():
                raise FileNotFoundError(f"Required spatial smoke test is missing: {smoke_path}")
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            if smoke.get("passed") is not True:
                raise RuntimeError("The spatial smoke test did not pass")
            smoke_test = smoke

    semantics = data_cfg.get("label_semantics", "unresolved")
    if semantics != "closed_world":
        raise RuntimeError(
            "Training requires label_semantics: closed_world because the confirmed train/validation "
            "rule is that classes omitted from Predict_class are negative. Update configs/data.yaml."
        )
    for key in ("image_root", "train_csv", "val_csv"):
        if not data_cfg.get(key) or not Path(data_cfg[key]).exists():
            raise FileNotFoundError(f"Configured {key} does not exist: {data_cfg.get(key)}")
    class_ids = data_cfg.get("class_ids", list(range(1, 11)))
    class_names = data_cfg.get("class_names", [])
    if class_ids != list(range(1, 11)) or len(class_names) != 10:
        raise ValueError("configs/data.yaml must define the official ordered 10 class IDs and names")

    seed_everything(int(train_cfg.get("seed", 42)))
    dataset_columns = dict(image_id_column=data_cfg["image_id_column"],
                          subject_column=data_cfg["subject_column"],
                          study_column=data_cfg["study_column"])
    train_data = StudyDataset(data_cfg["train_csv"], data_cfg["image_root"],
        transform=build_transform(data_cfg["image_size"], train=True),
        label_column=data_cfg["label_column"], label_semantics=semantics, **dataset_columns)
    val_data = StudyDataset(data_cfg["val_csv"], data_cfg["image_root"],
        transform=build_transform(data_cfg["image_size"]),
        label_column=data_cfg["label_column"], label_semantics=semantics, **dataset_columns)
    targets, known = train_data.label_tensors()
    _, val_known = val_data.label_tensors()
    for split_name, known_mask in (("train", known), ("validation", val_known)):
        incomplete = int((~known_mask.all(dim=1)).sum())
        if incomplete:
            raise ValueError(
                f"{split_name} contains {incomplete} Study(s) without a complete 10-class label; "
                "closed-world mode only resolves omitted IDs in a non-empty Predict_class value."
            )
    train_subjects = {subject_id for subject_id, _ in train_data.study_keys}
    val_subjects = {subject_id for subject_id, _ in val_data.study_keys}
    overlap = train_subjects & val_subjects
    if overlap:
        raise ValueError(f"Subject leakage across train/validation splits: {len(overlap)} Subjects")
    train_sampler = StudyBatchSampler(train_data, train_cfg["image_budget"], shuffle=True,
                                       seed=train_cfg.get("seed", 42))
    val_sampler = StudyBatchSampler(val_data, train_cfg["image_budget"], shuffle=False)
    loader_args = dict(num_workers=int(data_cfg["num_workers"]),
                       pin_memory=torch.cuda.is_available(), collate_fn=collate_studies)
    train_loader_generator = None
    val_loader_generator = None
    if model_cfg.get("head") == "spatial":
        worker_seed = int(train_cfg.get("data_loader_worker_seed", train_cfg.get("seed", 42)))
        train_loader_generator = torch.Generator().manual_seed(worker_seed)
        val_loader_generator = torch.Generator().manual_seed(worker_seed + 1)
    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler, generator=train_loader_generator, **loader_args)
    val_loader = DataLoader(
        val_data, batch_sampler=val_sampler, generator=val_loader_generator, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {key: value for key, value in model_cfg.items() if key != "name"}
    model = ChestXrayClassifier(**model_kwargs).to(device)
    groups = [
        {"params": model.backbone.parameters(), "lr": train_cfg["learning_rate_backbone"]},
        {"params": (model.classifier.parameters() if model.classifier is not None
                    else model.vector_classifier.parameters()), "lr": train_cfg["learning_rate_head"]},
    ]
    if model.study_attention is not None:
        groups.append({"params": model.study_attention.parameters(),
                       "lr": train_cfg["learning_rate_head"]})
    if model.cbam.__class__.__name__ == "CBAM":
        groups.append({"params": model.cbam.parameters(), "lr": train_cfg["learning_rate_head"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])

    accumulation = int(train_cfg["grad_accumulation"])
    optimizer_updates_by_epoch = []
    for epoch_index in range(int(train_cfg["epochs"])):
        train_sampler.set_epoch(epoch_index)
        optimizer_updates_by_epoch.append(
            math.ceil(len(train_sampler) / max(accumulation, 1))
        )
    train_sampler.set_epoch(0)
    scheduler, scheduler_metadata = build_lr_scheduler(
        optimizer, train_cfg, optimizer_updates_by_epoch
    )
    loss_name = train_cfg.get("loss", "bce").lower()
    weight_counts = None
    if loss_name == "bce":
        criterion = MaskedBCEWithLogitsLoss()
    elif loss_name == "weighted_bce":
        strategy = train_cfg.get("weighting", "sqrt")
        weights = estimate_pos_weight(targets, known, strategy=strategy,
                                      cap=train_cfg.get("pos_weight_cap", 5.0))
        weight_counts = {
            "positive": ((targets == 1) & known).sum(0).tolist(),
            "negative": ((targets == 0) & known).sum(0).tolist(),
            "pos_weight": weights.tolist(),
            "strategy": strategy,
        }
        criterion = MaskedBCEWithLogitsLoss(weights)
    elif loss_name == "asl":
        criterion = AsymmetricLoss(train_cfg.get("gamma_pos", 0.0),
                                   train_cfg.get("gamma_neg", 4.0),
                                   train_cfg.get("asl_clip", 0.05))
    else:
        raise ValueError("loss must be bce, weighted_bce, or asl")
    criterion = criterion.to(device)
    logger = create_logger(Path(train_cfg["output_dir"]) / "outputs/logs/train.log")
    logger.info("device=%s pretrained=%s fusion=%s loss=%s", device,
                model_cfg.get("pretrained", False), model_cfg.get("study_fusion"), loss_name)
    if weight_counts is not None:
        logger.info("training Study class counts and pos_weight=%s", weight_counts)
    if model_cfg.get("head") == "dsap_rpb" and train_cfg.get("resume"):
        logger.info("validated same-run resume checkpoint=%s epoch=%s best_epoch=%s best_macro_auroc=%s",
                    train_cfg["resume"], resume_metadata.get("epoch"),
                    resume_metadata.get("best_epoch"), resume_metadata.get("best_macro_auroc"))
    commit, dirty = _git_provenance()
    weight_source, weight_sha256 = _imagenet_weight_metadata(model_cfg.get("pretrained", False))
    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    baseline_head = MultiLabelHead(model.backbone.channels["stage4"], 10, normalize=True)
    baseline_parameters = sum(parameter.numel() for parameter in baseline_head.parameters())
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_counts = {
        "backbone": backbone_parameters,
        "original_gap_head": baseline_parameters,
        "original_gap_total": backbone_parameters + baseline_parameters,
        "modified_total": model_parameters,
        "new_parameters": model_parameters - backbone_parameters - baseline_parameters,
    }
    if parameter_counts["new_parameters"] < 0:
        raise RuntimeError("DSAP-RPB model parameter count is below the original GAP baseline")
    reference_gap_baseline = None
    baseline_metadata_path = PROJECT_DIR / "checkpoint/best/best_by_macro_auc.json"
    if baseline_metadata_path.is_file():
        baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
        baseline_config = baseline_metadata.get("config", {})
        baseline_model = baseline_config.get("model_config", {})
        if (baseline_model.get("head") == "gap"
                and baseline_model.get("study_fusion") == "logit_mean"
                and baseline_config.get("loss") == "bce"):
            reference_gap_baseline = {
                "epoch": baseline_metadata.get("epoch"),
                "best_macro_auroc": baseline_metadata.get("best_macro_auroc"),
                "backbone_learning_rate": baseline_config.get("learning_rate_backbone"),
                "head_learning_rate": baseline_config.get("learning_rate_head"),
            }
    reference_dsap_rpb_previous_run = None
    previous_dsap_dir = PROJECT_DIR / "outputs/experiments/ConvNeXt_DSAP_RPB"
    previous_best_path = previous_dsap_dir / "best.json"
    previous_last_path = previous_dsap_dir / "last.json"
    if (model_cfg.get("head") == "dsap_rpb"
            and previous_best_path.is_file() and previous_last_path.is_file()):
        previous_best = json.loads(previous_best_path.read_text(encoding="utf-8"))
        previous_last = json.loads(previous_last_path.read_text(encoding="utf-8"))
        reference_dsap_rpb_previous_run = {
            "best_epoch": previous_best.get("epoch"),
            "best_macro_auroc": previous_best.get("best_macro_auroc"),
            "last_epoch": previous_last.get("epoch"),
            "learning_rate_backbone": (previous_best.get("config") or {}).get("learning_rate_backbone"),
            "learning_rate_head": (previous_best.get("config") or {}).get("learning_rate_head"),
            "cosine_end_epoch": (previous_best.get("config") or {}).get("cosine_end_epoch", 40),
        }
    run_config = {
        **train_cfg,
        "data": {**data_cfg, "train_csv": str(data_cfg["train_csv"]),
                 "val_csv": str(data_cfg["val_csv"]), "image_root": str(data_cfg["image_root"])},
        "model_config": model_cfg,
        "class_ids": class_ids,
        "class_names": class_names,
        "label_semantics": semantics,
        "weight_counts": weight_counts,
        "pretrained_weight_source": weight_source,
        "pretrained_weight_sha256": weight_sha256,
        "initialization_policy": ("torchvision ImageNet ConvNeXt-Tiny feature weights with a fresh "
                                  f"{model_cfg.get('head')} classification head; resume is restricted to "
                                  "this run's own last checkpoint"),
        "resume_policy": ("continued from this experiment's last checkpoint" if train_cfg.get("resume")
                          else "fresh ImageNet initialization; no fine-tuned checkpoint loaded"),
        "backbone_initialization": {
            "pretrained": bool(model_cfg.get("pretrained", False)),
            "source": weight_source,
            "sha256": weight_sha256,
            "loaded_feature_parameter_count": backbone_parameters,
            "new_head_randomly_initialized": True,
            "fine_tuned_checkpoint_used": False,
        },
        "source_git_commit": commit,
        "source_worktree_dirty": dirty,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameter_counts": parameter_counts,
        "scheduler_metadata": scheduler_metadata,
        "smoke_test": smoke_test,
        "reference_gap_baseline": reference_gap_baseline,
        "reference_dsap_rpb_previous_run": reference_dsap_rpb_previous_run,
        "study_fusion": model_cfg["study_fusion"],
    }
    fit(model, train_loader, val_loader, criterion, optimizer, scheduler, run_config,
        device, logger, class_names=class_names)


if __name__ == "__main__":
    main()
