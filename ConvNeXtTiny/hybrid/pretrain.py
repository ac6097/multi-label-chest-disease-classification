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

from datasets.hybrid import OfficialTrainImageDataset, build_hybrid_transform
from engine.trainer import _draw_curve
from hybrid.backbone_factory import build_hybrid_backbone
from hybrid.common import (PROJECT_DIR, autocast_context, build_adamw,
                           build_update_scheduler, count_parameters, current_lrs, device_and_amp,
                           read_yaml, resolve_data_config, seed_everything,
                           seed_worker, sha256_file, write_csv, write_json)
from hybrid.visuals import save_latent_reconstruction, save_pixel_reconstruction
from model.hybrid_pretraining import (LatentReconstructionPretrainer,
                                      PixelReconstructionPretrainer,
                                      sample_spatial_mask_for_ids)


def encoder_checkpoint_path(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    return output_dir / "encoder_best.pth"


def _read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _restore_rng(state):
    if state.get("rng") is not None:
        torch.set_rng_state(state["rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng"):
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng"]])
    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])
    if state.get("numpy_rng") is not None:
        np.random.set_state(state["numpy_rng"])


def _save_encoder(path, model, method, epoch, config):
    payload = {
        "method": method,
        "epoch": int(epoch),
        "backbone": model.backbone.state_dict(),
        "config": config,
        "decoder_excluded": True,
        "teacher_student_used": False,
    }
    _atomic_torch_save(path, payload)
    return sha256_file(path)


def _grad_norm(parameters):
    squared_sum = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is not None:
            found = True
            squared_sum += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(squared_sum) if found else 0.0


def _parameter_change(groups, before):
    result = {}
    for name, parameters in groups.items():
        squared_sum = 0.0
        for parameter, initial_value in zip(parameters, before[name]):
            difference = parameter.detach().float().cpu() - initial_value
            squared_sum += float(difference.square().sum())
        result[name] = math.sqrt(squared_sum)
    return result


def _iter_group_parameters(model, method):
    if method == "h2_latent":
        return {
            "cnn_stage3": list(model.backbone.convnext.stages[2].parameters()),
            "cnn_stage4": list(model.backbone.convnext.stages[3].parameters()),
            "vit_pre_block6": [parameter for block in model.backbone.vit.blocks[:6]
                                for parameter in block.parameters()],
            "vit_post_block6": [parameter for block in model.backbone.vit.blocks[6:]
                                 for parameter in block.parameters()],
            "cnn_decoder": list(model.cnn_decoder.parameters()),
            "vit_decoder": list(model.vit_decoder.parameters()),
            "mask_tokens": [model.cnn_mask_token, model.vit_mask_token],
        }
    return {
        "convnext_encoder": list(model.backbone.convnext.parameters()),
        "vit_encoder": list(model.backbone.vit.parameters()),
        "cnn_decoder": list(model.cnn_decoder.parameters()),
        "vit_decoder": list(model.vit_decoder.parameters()),
        "mask_token": [model.vit_mask_token],
    }


def _write_curves(output_dir, history):
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    markers = (2, 5, 10, 15, 20)
    _draw_curve(output_dir / "reconstruction_loss_curve.png", [
        ("Total", epochs, [float(row["total_loss"]) for row in history], (39, 110, 220)),
        ("ConvNeXt", epochs, [float(row["cnn_loss"]) for row in history], (44, 160, 44)),
        ("ViT", epochs, [float(row["vit_loss"]) for row in history], (220, 75, 55)),
    ], "Masked reconstruction loss", markers=markers)
    _draw_curve(output_dir / "pretrain_images_per_second.png", [
        ("End-to-end images/s", epochs, [float(row["images_per_s"]) for row in history], (112, 75, 180)),
    ], "Images / second", markers=markers)
    _draw_curve(output_dir / "pretrain_peak_memory.png", [
        ("Peak allocated GiB", epochs, [float(row["peak_gpu_allocated_gib"]) for row in history], (39, 110, 220)),
        ("Peak reserved GiB", epochs, [float(row["peak_gpu_reserved_gib"]) for row in history], (220, 75, 55)),
    ], "GPU memory (GiB)", markers=markers)
    if "cnn_spatial_variance" in history[0]:
        _draw_curve(output_dir / "latent_variance_curve.png", [
            ("CNN Stage3 spatial variance", epochs,
             [float(row["cnn_spatial_variance"]) for row in history], (39, 110, 220)),
            ("ViT block6 spatial variance", epochs,
             [float(row["vit_spatial_variance"]) for row in history], (220, 75, 55)),
        ], "Feature variance", markers=markers)


def _write_summary(output_dir, method, history, config, run_config, complete):
    if not history:
        return
    best = min(history, key=lambda row: float(row["total_loss"]))
    status = "completed 20 epochs" if complete else "in progress through epoch {}".format(history[-1]["epoch"])
    checkpoint_sizes = {
        name: (output_dir / name).stat().st_size if (output_dir / name).is_file() else 0
        for name in ("best.pth", "last.pth", "encoder_best.pth", "encoder_last.pth")
    }
    lines = [
        f"# Hybrid {method.upper()} self-supervised pretraining", "",
        f"- Status: {status}",
        f"- Epochs completed: {history[-1]['epoch']} / {config['epochs']}",
        f"- Best epoch by training reconstruction loss: {best['epoch']}",
        f"- Best total / CNN / ViT reconstruction loss: {float(best['total_loss']):.6f} / "
        f"{float(best['cnn_loss']):.6f} / {float(best['vit_loss']):.6f}",
        f"- Encoder export for supervised fine-tuning: `{output_dir / 'encoder_best.pth'}`",
        f"- Best/Last full checkpoint bytes: {checkpoint_sizes['best.pth']:,}/{checkpoint_sizes['last.pth']:,}; "
        f"best/last encoder export bytes: {checkpoint_sizes['encoder_best.pth']:,}/{checkpoint_sizes['encoder_last.pth']:,}",
        f"- Initial ImageNet checkpoint SHA-256: `{run_config['initial_checkpoint_sha256']}`",
        f"- Train image manifest SHA-256: `{run_config['train_manifest_sha256']}`",
        f"- Pretraining parameters: CNN {run_config['parameter_counts']['convnext_v2']:,}; "
        f"ViT {run_config['parameter_counts']['vit']:,}; "
        f"CNN/ViT decoders {run_config['parameter_counts']['cnn_decoder']:,}/"
        f"{run_config['parameter_counts']['vit_decoder']:,}; "
        f"total {run_config['parameter_counts']['total']:,}.",
        f"- Data source: official train images only; disease label columns were not read; valid/test were not used.",
        "- Target effective batch: 32 images; each epoch traverses every official train image exactly once.",
        "- Decoder state is retained in resumable best/last pretraining checkpoints and excluded from encoder export.",
        "- Encoder/decoder gradient and parameter-change diagnostics are recorded on the first optimizer update of each epoch; non-finite/zero gradients or collapsed features/outputs stop the run with a failure checkpoint.",
    ]
    if method == "h1_pixel":
        lines.extend([
            "- Mask: exactly 60 of 100 random 32×32 pixel regions per image; masks are keyed by Subject/Study/image ID so H1/H2 share positions even when microbatch sizes differ; the same 10×10 mask is mapped to the ViT 20×20 patch grid.",
            "- Reconstruction target: the shared augmented RGB tensor in [0,1]; MSE is computed only on masked pixels and averaged equally across branches.",
            "- Implementation: ordinary dense PyTorch ConvNeXtV2 receives zero-filled masked pixels. The official sparse FCMAE encoder was not used; dense convolution and visible-neighbor context remain, so this is not a claim of sparse-FCMAE equivalence.",
            "- Visuals show original target, shared masked input, and both branch reconstructions.",
        ])
    else:
        lines.extend([
            "- Mask: exactly 60 of 100 random 32×32 regions; masks are keyed by Subject/Study/image ID so H1/H2 share positions even when microbatch sizes differ; each maps to 2×2 positions on the Stage3 and ViT 20×20 grids.",
            "- CNN target is a detached Stage3 feature map; masked Stage3 output is replaced before Stage4, then reconstructed.",
            "- ViT target is a detached output of Block 6; CLS is excluded from masking and reconstruction; masked patch tokens continue through Blocks 7–12.",
            "- Stage3 and ViT Block 1–6 features are computed before the mask and may contain contextual information from all input regions. This is intermediate-feature reconstruction, not input-level MAE.",
            "- No Teacher–Student/EMA, distillation, or contrastive objective was used.",
            "- Diagnostics record branch/segment gradients, parameter changes, feature spatial variance, and decoder output variance.",
            "- Visuals are latent feature/error maps, not reconstructed X-rays.",
        ])
    lines.extend([
        "", "## Epoch log", "",
        "| Epoch | Total reconstruction loss | CNN loss | ViT loss | Mask ratio | Images/s | Peak allocated/reserved GiB | Encoder gradient norms |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in history:
        gradient_summary = (f"CNN={float(row['cnn_grad_norm']):.4g}, ViT={float(row['vit_grad_norm']):.4g}, "
                            f"decoders={float(row['cnn_decoder_grad_norm']):.4g}/{float(row['vit_decoder_grad_norm']):.4g}"
                            if "cnn_grad_norm" in row else
                            f"S3={float(row['cnn_stage3_grad_norm']):.4g}, S4={float(row['cnn_stage4_grad_norm']):.4g}, "
                            f"ViT pre/post={float(row['vit_pre6_grad_norm']):.4g}/{float(row['vit_post6_grad_norm']):.4g}, "
                            f"decoders={float(row['cnn_decoder_grad_norm']):.4g}/{float(row['vit_decoder_grad_norm']):.4g}")
        lines.append(
            f"| {row['epoch']} | {float(row['total_loss']):.6f} | {float(row['cnn_loss']):.6f} | "
            f"{float(row['vit_loss']):.6f} | {float(row['mask_ratio']):.3f} | "
            f"{float(row['images_per_s']):.2f} | {float(row['peak_gpu_allocated_gib']):.3f}/"
            f"{float(row['peak_gpu_reserved_gib']):.3f} | {gradient_summary} |"
        )
    (output_dir / "pretrain_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _groups_for_optimizer(model, config):
    return [
        {"params": model.backbone.convnext.parameters(), "lr": float(config["learning_rate_encoder"]), "name": "convnext"},
        {"params": model.backbone.vit.parameters(), "lr": float(config["learning_rate_encoder"]), "name": "vit"},
        {"params": list(model.cnn_decoder.parameters()) + list(model.vit_decoder.parameters()) +
                    ([model.cnn_mask_token, model.vit_mask_token]
                     if hasattr(model, "cnn_mask_token") else [model.vit_mask_token]),
         "lr": float(config["learning_rate_decoder"]), "name": "decoder"},
    ]


def _make_model(method, backbone, checkpointing):
    if method == "h1_pixel":
        model = PixelReconstructionPretrainer(backbone, checkpoint_blocks=checkpointing)
    elif method == "h2_latent":
        model = LatentReconstructionPretrainer(backbone, checkpoint_blocks=checkpointing)
    else:
        raise ValueError("method must be h1_pixel or h2_latent")
    return model


def run_pretraining(method, experiments_config_path="configs/hybrid_experiments.yaml",
                    resume=False):
    if method not in {"h1_pixel", "h2_latent"}:
        raise ValueError("method must be h1_pixel or h2_latent")
    experiments = read_yaml(experiments_config_path)
    data_config = resolve_data_config(experiments["data_config"])
    config_key = "h1_pretrain_config" if method == "h1_pixel" else "h2_pretrain_config"
    config = read_yaml(experiments[config_key])
    if (int(config["epochs"]) != 20 or int(config["warmup_epochs"]) != 2 or
            int(config["cosine_end_epoch"]) != 20 or config["early_stopping_enabled"]):
        raise ValueError("H1/H2 pretraining is fixed to 20 epochs with a 2+18 schedule and no early stopping")
    if float(config["mask_ratio"]) != 0.60 or int(config["mask_unit_pixels"]) != 32:
        raise ValueError("H1/H2 mask ratio and 32x32 mask unit are fixed at 60% and 32 pixels")
    if config["amp_dtype"] != "bfloat16":
        raise ValueError("H1/H2 pretraining requires BF16 on the verified GPU")
    fixed_values = {
        "seed": 42,
        "learning_rate_encoder": 1e-5,
        "learning_rate_decoder": 2e-4,
        "min_lr_encoder": 5e-7,
        "min_lr_decoder": 1e-5,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "scheduler_step_unit": "optimizer_update",
        "target_effective_images": 32,
    }
    for field, expected in fixed_values.items():
        actual = config[field]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"H1/H2 pretraining setting {field} must be {expected}")
        elif actual != expected:
            raise ValueError(f"H1/H2 pretraining setting {field} must be {expected}")
    if config.get("image_budget") is None or config.get("grad_accumulation") is None:
        raise RuntimeError("GPU benchmark must choose the self-supervised batch configuration before pretraining")
    if config["image_budget"] * config["grad_accumulation"] != int(config["target_effective_images"]):
        raise ValueError("H1/H2 must use the same exact effective image batch selected by the benchmark")

    output_key = "h1_pretrain" if method == "h1_pixel" else "h2_pretrain"
    output_dir = Path(experiments["outputs"][output_key])
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "best.pth", output_dir / "last.pth"
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing {method} run: {output_dir}")

    seed = int(config["seed"])
    seed_everything(seed)
    device, amp_enabled, amp_dtype = device_and_amp(config["amp"], config["amp_dtype"])
    initial_path = Path(experiments["initial_weights"])
    if not initial_path.is_absolute():
        initial_path = PROJECT_DIR / initial_path
    if not initial_path.is_file():
        raise FileNotFoundError(f"Shared ImageNet initial checkpoint is missing: {initial_path}")
    initial_hash = sha256_file(initial_path)
    backbone = build_hybrid_backbone(initial_path, map_location="cpu",
                                     initial_checkpoint_sha256=initial_hash)
    model = _make_model(method, backbone, bool(config["gradient_checkpointing"])).to(device)
    model.backbone.set_grad_checkpointing(bool(config["gradient_checkpointing"]))

    audit_path = PROJECT_DIR / "outputs/hybrid/shared/manifests/data_audit.json"
    smoke_path = PROJECT_DIR / "outputs/hybrid/smoke_test/hybrid_smoke_test.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Official data audit must complete before self-supervised pretraining")
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise RuntimeError("Hybrid real-sample Smoke Test must pass before self-supervised pretraining")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    dataset = OfficialTrainImageDataset(
        data_config["train_csv"], data_config["image_root"],
        transform=build_hybrid_transform(data_config["image_size"], train=True),
        image_id_column=data_config["image_id_column"],
        subject_column=data_config["subject_column"],
        study_column=data_config["study_column"],
    )
    loader_generator = torch.Generator().manual_seed(int(config["data_loader_worker_seed"]))
    loader = DataLoader(
        dataset, batch_size=int(config["image_budget"]), shuffle=True,
        num_workers=int(config["num_workers"]), pin_memory=bool(config["pin_memory"]),
        drop_last=False, generator=loader_generator,
        worker_init_fn=seed_worker if config["num_workers"] else None,
        persistent_workers=bool(config["num_workers"]),
    )
    accumulation = int(config["grad_accumulation"])
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    optimizer = build_adamw(_groups_for_optimizer(model, config), config["weight_decay"],
                            fused=config["optimizer_fused"])
    scheduler, scheduler_info = build_update_scheduler(
        optimizer, [updates_per_epoch] * int(config["epochs"]),
        config["warmup_epochs"], config["cosine_end_epoch"], min_factor=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    run_config = {
        "method": method,
        "data_source": "official train split only; label columns not read",
        "train_csv": str(data_config["train_csv"]),
        "image_root": str(data_config["image_root"]),
        "train_manifest_sha256": audit["train_manifest_sha256"],
        "train_image_rows": audit["train"]["image_rows"],
        "train_studies": audit["train"]["studies"],
        "train_subjects": audit["train"]["subjects"],
        "test_split_read": False,
        "valid_split_read": False,
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial_hash,
        "config": config,
        "scheduler": scheduler_info,
        "torch_version": torch.__version__,
        "timm_version": __import__("timm").__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "parameter_counts": {
            "convnext_v2": count_parameters(model.backbone.convnext),
            "vit": count_parameters(model.backbone.vit),
            "cnn_decoder": count_parameters(model.cnn_decoder),
            "vit_decoder": count_parameters(model.vit_decoder),
            "mask_tokens": sum(parameter.numel() for name, parameter in model.named_parameters()
                                if "mask_token" in name),
            "total": count_parameters(model),
            "trainable": count_parameters(model, trainable_only=True),
        },
        "effective_batch_images": config["image_budget"] * config["grad_accumulation"],
        "mask_ratio": config["mask_ratio"],
    }
    if method == "h1_pixel":
        run_config["mask_implementation"] = (
            "Dense PyTorch zero-fill pixel masking; the original masked pixels are absent from encoder input, "
            "but this is not sparse-convolution FCMAE and visible-neighbor context remains.")
    else:
        run_config["teacher_student"] = False
        run_config["intermediate_mask_information_limit"] = (
            "Stage3 and ViT block-6 representations are computed from complete inputs before masking, "
            "so their visible positions may already contain masked-region context.")
    write_json(output_dir / "config.json", run_config)
    logger = __import__("hybrid.common", fromlist=["create_logger"]).create_logger(
        output_dir / "logs/pretrain.log", f"hybrid_{method}")

    history_path = output_dir / "pretrain_log.csv"
    history = _read_csv(history_path) if resume else []
    start_epoch, best_metric, best_epoch = 0, math.inf, 0
    if resume:
        if not last_path.is_file() or not Path(resume).is_file() or not Path(resume).samefile(last_path):
            raise ValueError("Self-supervised resume is restricted to this run's own last.pth")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config", {}).get("config") != config:
            raise ValueError("Resume pretraining configuration differs from the saved run configuration")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state)
        loader_generator.set_state(state["loader_generator_state"].cpu())
        start_epoch = int(state["epoch"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])

    for epoch_index in range(start_epoch, int(config["epochs"])):
        epoch = epoch_index + 1
        model.train()
        model.backbone.set_grad_checkpointing(bool(config["gradient_checkpointing"]))
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        totals = {key: torch.zeros((), device=device, dtype=torch.float32)
                  for key in ("loss", "cnn_loss", "vit_loss", "mask_ratio")}
        optional_totals = {}
        images_seen, updates = 0, 0
        gradient_accumulator = {}
        update_change_accumulator = {}
        window = []
        visual_snapshot = None
        param_groups = _iter_group_parameters(model, method)

        def update_window(batches, update_index):
            nonlocal images_seen, updates, visual_snapshot
            window_size = sum(int(batch["image"].shape[0]) for batch in batches)
            if window_size < 1:
                return
            snapshots = ({name: [parameter.detach().float().cpu().clone() for parameter in params]
                          for name, params in param_groups.items()}
                         if update_index == 0 else None)
            outputs_for_visuals = None
            for batch_index, batch in enumerate(batches):
                images = batch["image"].to(device, non_blocking=True)
                mask = sample_spatial_mask_for_ids(
                    batch["subject_id"], batch["study_id"], batch["image_id"],
                    epoch, seed, device)
                with autocast_context(device, amp_enabled, amp_dtype):
                    outputs = model(images, mask)
                if method == "h2_latent" and update_index == 0 and batch_index == 0:
                    collapse_values = {
                        key: float(outputs[key].detach())
                        for key in ("cnn_spatial_variance", "vit_spatial_variance",
                                    "cnn_prediction_spatial_std", "vit_prediction_spatial_std")
                    }
                    if (update_index == 0 and batch_index == 0 and
                            (collapse_values["cnn_spatial_variance"] < 1e-8 or
                            collapse_values["vit_spatial_variance"] < 1e-8 or
                            collapse_values["cnn_prediction_spatial_std"] < 1e-12 or
                            collapse_values["vit_prediction_spatial_std"] < 1e-12)):
                        diagnostic = {"epoch": epoch, "method": method,
                                      "mask_ratio": float(config["mask_ratio"]),
                                      "latent_diagnostics": collapse_values,
                                      "reconstruction_loss": float(outputs["loss"].detach())}
                        write_json(output_dir / "diagnostic_failure.json", diagnostic)
                        _atomic_torch_save(output_dir / "diagnostic_failure.pth", {
                            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "config": run_config,
                            "diagnostic": diagnostic,
                        })
                        raise RuntimeError(f"H2 representation collapse diagnostic failed: {collapse_values}")
                elif method == "h1_pixel" and update_index == 0 and batch_index == 0:
                    pixel_std = {key: float(outputs[key].detach())
                                 for key in ("cnn_prediction_std", "vit_prediction_std")}
                    if any(not math.isfinite(value) or value < 1e-12
                           for value in pixel_std.values()):
                        diagnostic = {"epoch": epoch, "method": method,
                                      "mask_ratio": float(config["mask_ratio"]),
                                      "decoder_output_std": pixel_std,
                                      "reconstruction_loss": float(outputs["loss"].detach())}
                        write_json(output_dir / "diagnostic_failure.json", diagnostic)
                        _atomic_torch_save(output_dir / "diagnostic_failure.pth", {
                            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "config": run_config,
                            "diagnostic": diagnostic,
                        })
                        raise RuntimeError(f"H1 pixel decoder output is constant or non-finite: {pixel_std}")
                image_count = int(images.shape[0])
                (outputs["loss"] * (image_count / window_size)).backward()
                for key in totals:
                    value = outputs[key]
                    totals[key] += value.detach().float() * image_count
                if method == "h2_latent":
                    for key in ("cnn_spatial_variance", "vit_spatial_variance",
                                "cnn_prediction_spatial_std", "vit_prediction_spatial_std"):
                        optional_totals[key] = optional_totals.get(key, torch.zeros((), device=device)) + outputs[key].detach().float() * image_count
                else:
                    for key in ("cnn_prediction_std", "vit_prediction_std"):
                        optional_totals[key] = optional_totals.get(key, torch.zeros((), device=device)) + outputs[key].detach().float() * image_count
                if update_index == 0 and batch_index == 0 and epoch in {1, 5, 10, 15, 20}:
                    if method == "h1_pixel":
                        outputs_for_visuals = (
                            images[0].detach().cpu(), outputs["masked_images"][0].detach().cpu(),
                            outputs["cnn_pixels"][0].detach().cpu(), outputs["vit_pixels"][0].detach().cpu(),
                        )
                    else:
                        outputs_for_visuals = (
                            outputs["cnn_target"][0].detach().cpu(),
                            outputs["cnn_prediction"][0].detach().cpu(),
                            outputs["cnn_mask"][0].detach().cpu(),
                        )
                del outputs
            grad_norms = ({name: _grad_norm(parameters) for name, parameters in param_groups.items()}
                          if update_index == 0 else {})
            if update_index == 0 and any(not math.isfinite(value) or value <= 1e-12
                                         for value in grad_norms.values()):
                diagnostic = {
                    "epoch": epoch,
                    "optimizer_update": update_index + 1,
                    "gradient_norms": grad_norms,
                    "method": method,
                    "mask_ratio": float(config["mask_ratio"]),
                }
                write_json(output_dir / "diagnostic_failure.json", diagnostic)
                _atomic_torch_save(output_dir / "diagnostic_failure.pth", {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "config": run_config,
                    "diagnostic": diagnostic,
                })
                raise RuntimeError(f"Encoder gradient missing or non-finite; stopped with diagnostics: {grad_norms}")
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                          float(config["gradient_clip_norm"]),
                                          error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            changes = (_parameter_change(param_groups, snapshots)
                       if update_index == 0 else {})
            if update_index == 0 and any(not math.isfinite(value) or value <= 0
                                         for value in changes.values()):
                diagnostic = {"epoch": epoch, "optimizer_update": update_index + 1,
                              "gradient_norms": grad_norms, "parameter_changes": changes,
                              "method": method}
                write_json(output_dir / "diagnostic_failure.json", diagnostic)
                _atomic_torch_save(output_dir / "diagnostic_failure.pth", {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "config": run_config,
                    "diagnostic": diagnostic,
                })
                raise RuntimeError(f"Encoder parameter did not update; diagnostics saved: {changes}")
            if update_index == 0:
                gradient_accumulator.update(grad_norms)
                update_change_accumulator.update(changes)
            images_seen += window_size
            updates += 1
            if outputs_for_visuals is not None:
                visual_snapshot = outputs_for_visuals

        try:
            for batch in loader:
                window.append(batch)
                if len(window) == int(config["grad_accumulation"]):
                    update_window(window, updates)
                    window = []
            if window:
                update_window(window, updates)
        except RuntimeError:
            raise

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_started
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        row = {
            "epoch": epoch,
            "total_loss": float((totals["loss"] / max(images_seen, 1)).detach()),
            "cnn_loss": float((totals["cnn_loss"] / max(images_seen, 1)).detach()),
            "vit_loss": float((totals["vit_loss"] / max(images_seen, 1)).detach()),
            "mask_ratio": float((totals["mask_ratio"] / max(images_seen, 1)).detach()),
            "images": images_seen,
            "optimizer_updates": updates,
            "elapsed_s": elapsed,
            "images_per_s": images_seen / elapsed if elapsed else 0.0,
            "peak_gpu_allocated_gib": peak_allocated,
            "peak_gpu_reserved_gib": peak_reserved,
            "encoder_grad_norms": json.dumps(gradient_accumulator),
            "parameter_changes": json.dumps({name: value
                                               for name, value in update_change_accumulator.items()}),
        }
        if method == "h2_latent":
            row.update({key: float((value / max(images_seen, 1)).detach())
                        for key, value in optional_totals.items()})
            row.update({
                "cnn_stage3_grad_norm": gradient_accumulator["cnn_stage3"],
                "cnn_stage4_grad_norm": gradient_accumulator["cnn_stage4"],
                "vit_pre6_grad_norm": gradient_accumulator["vit_pre_block6"],
                "vit_post6_grad_norm": gradient_accumulator["vit_post_block6"],
                "cnn_decoder_grad_norm": gradient_accumulator["cnn_decoder"],
                "vit_decoder_grad_norm": gradient_accumulator["vit_decoder"],
            })
        else:
            row.update({key: float((value / max(images_seen, 1)).detach())
                        for key, value in optional_totals.items()})
            row["cnn_grad_norm"] = gradient_accumulator["convnext_encoder"]
            row["vit_grad_norm"] = gradient_accumulator["vit_encoder"]
            row["cnn_decoder_grad_norm"] = gradient_accumulator["cnn_decoder"]
            row["vit_decoder_grad_norm"] = gradient_accumulator["vit_decoder"]
        history.append(row)
        write_csv(history_path, history)

        diagnostic_values = ({
            key: row[key] for key in (
                "cnn_spatial_variance", "vit_spatial_variance",
                "cnn_prediction_spatial_std", "vit_prediction_spatial_std")
        } if method == "h2_latent" else {
            key: row[key] for key in ("cnn_prediction_std", "vit_prediction_std")
        })
        collapsed = (
            diagnostic_values["cnn_spatial_variance"] < 1e-8 or
            diagnostic_values["vit_spatial_variance"] < 1e-8 or
            diagnostic_values["cnn_prediction_spatial_std"] < 1e-12 or
            diagnostic_values["vit_prediction_spatial_std"] < 1e-12
            if method == "h2_latent" else
            diagnostic_values["cnn_prediction_std"] < 1e-12 or
            diagnostic_values["vit_prediction_std"] < 1e-12
        )
        if collapsed:
            diagnostic = {"epoch": epoch, "method": method,
                          "latent_or_pixel_diagnostics": diagnostic_values,
                          "reconstruction_loss": row["total_loss"]}
            write_json(output_dir / "diagnostic_failure.json", diagnostic)
            _atomic_torch_save(output_dir / "diagnostic_failure.pth", {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "config": run_config,
                "diagnostic": diagnostic,
            })
            raise RuntimeError(f"Self-supervised collapse/constant-output check failed: {diagnostic_values}")

        improved = float(row["total_loss"]) < best_metric
        if improved:
            best_metric, best_epoch = float(row["total_loss"]), epoch
            encoder_hash = _save_encoder(output_dir / "encoder_best.pth", model,
                                         method, epoch, run_config)
        else:
            encoder_hash = None
        _save_encoder(output_dir / "encoder_last.pth", model, method, epoch, run_config)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": run_config,
            "history": history,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "loader_generator_state": loader_generator.get_state(),
        }
        _atomic_torch_save(last_path, state)
        if improved:
            _atomic_torch_save(best_path, state)

        if visual_snapshot is not None:
            visual_dir = output_dir / "visualizations"
            if method == "h1_pixel":
                save_pixel_reconstruction(visual_dir / f"pixel_reconstruction_epoch_{epoch:02d}.png",
                                          *visual_snapshot)
            else:
                save_latent_reconstruction(visual_dir / f"latent_features_epoch_{epoch:02d}.png",
                                           *visual_snapshot)
        _write_curves(output_dir, history)
        _write_summary(output_dir, method, history, config, run_config,
                       complete=(epoch == int(config["epochs"])))
        logger.info("epoch=%d/%d loss=%.6f cnn=%.6f vit=%.6f mask=%.3f images=%d "
                    "updates=%d images_s=%.2f peak_allocated=%.3fGiB peak_reserved=%.3fGiB "
                    "encoder_grad_norms=%s parameter_changes=%s",
                    epoch, config["epochs"], row["total_loss"], row["cnn_loss"],
                    row["vit_loss"], row["mask_ratio"], images_seen, updates,
                    row["images_per_s"], peak_allocated, peak_reserved,
                    row["encoder_grad_norms"], row["parameter_changes"])
        if encoder_hash:
            logger.info("best encoder checkpoint SHA-256=%s", encoder_hash)

    return {"method": method, "completed_epochs": len(history),
            "best_epoch": best_epoch, "best_train_reconstruction_loss": best_metric,
            "output_dir": str(output_dir),
            "encoder_checkpoint": str(output_dir / "encoder_best.pth")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("h1_pixel", "h2_latent"), required=True)
    parser.add_argument("--experiments-config", default="configs/hybrid_experiments.yaml")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    result = run_pretraining(args.method, args.experiments_config, resume=args.resume or False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
