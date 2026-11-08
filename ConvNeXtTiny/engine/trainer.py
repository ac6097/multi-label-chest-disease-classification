import math
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from .validator import validate
from utils.checkpoint import load_checkpoint, save_checkpoint


def _learning_rates(optimizer):
    return [float(group["lr"]) for group in optimizer.param_groups]


def _draw_curve(path, series, y_label, logarithmic=False, markers=()):
    width, height = 1200, 720
    left, top, right, bottom = 100, 85, 1135, 625
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((left, 24), Path(path).stem.replace("_", " "), fill="black", font=title_font)
    transformed = []
    for label, epochs, values, color in series:
        points = []
        for epoch, value in zip(epochs, values):
            if not math.isfinite(float(value)) or (logarithmic and float(value) <= 0):
                continue
            y = math.log10(float(value)) if logarithmic else float(value)
            points.append((int(epoch), y))
        if points:
            transformed.append((label, points, color))
    all_values = [value for _, points, _ in transformed for _, value in points]
    if not all_values:
        return
    y_min, y_max = min(all_values), max(all_values)
    if y_max - y_min < 1e-12:
        padding = max(abs(y_min) * 0.05, 0.05)
    else:
        padding = (y_max - y_min) * 0.08
    y_min, y_max = y_min - padding, y_max + padding
    all_epochs = [epoch for _, points, _ in transformed for epoch, _ in points]
    x_min, x_max = min(all_epochs), max(all_epochs)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    draw.text((left, top - 30), y_label, fill="black", font=font)
    draw.text(((left + right) // 2, bottom + 40), "Epoch", fill="black", font=font)
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        display = f"{10 ** value:.1e}" if logarithmic else f"{value:.3g}"
        draw.text((left - 75, y - 8), display, fill="black", font=font)
    for marker in markers:
        if x_min <= marker <= x_max:
            x = left + int((marker - x_min) / (x_max - x_min) * (right - left))
            for y in range(top + 24, bottom, 18):
                draw.line((x, y, x, min(y + 8, bottom)), fill=(150, 150, 150), width=1)
            draw.text((min(x + 3, right - 38), top + 2), f"E{marker}", fill=(90, 90, 90), font=font)
    for legend_index, (label, points, color) in enumerate(transformed):
        xy = [
            (left + int((epoch - x_min) / (x_max - x_min) * (right - left)),
             bottom - int((value - y_min) / (y_max - y_min) * (bottom - top)))
            for epoch, value in points
        ]
        if len(xy) > 1:
            draw.line(xy, fill=color, width=4)
        for x, y in xy:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        legend_y = top + 10 + 28 * legend_index
        draw.line((right - 290, legend_y + 9, right - 255, legend_y + 9), fill=color, width=4)
        draw.text((right - 245, legend_y), label, fill=color, font=font)
    image.save(path)


def _write_experiment_artifacts(output_dir, config, history_rows, per_class_rows,
                                model, best_epoch, stopped_early, training_complete=False):
    output_dir = Path(output_dir)
    smoke_test = config.get("smoke_test") or {}
    pd.DataFrame(history_rows).to_csv(output_dir / "train_log.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / "per_class_metrics.csv", index=False)
    if not history_rows:
        return
    epochs = [row["epoch"] for row in history_rows]
    markers = config.get("plot_epoch_markers", (2, 7, 20, 40))
    _draw_curve(output_dir / "loss_curve.png", [
        ("Train Loss", epochs, [row["train_loss"] for row in history_rows], (39, 110, 220)),
        ("Val Loss", epochs, [row["val_loss"] for row in history_rows], (220, 75, 55)),
    ], "Study-mean BCE Loss", markers=markers)
    _draw_curve(output_dir / "auroc_curve.png", [
        ("Validation Macro AUROC", epochs,
         [row["macro_auroc"] for row in history_rows], (39, 145, 88)),
    ], "Macro AUROC", markers=markers)
    _draw_curve(output_dir / "lr_curve.png", [
        ("Backbone LR", epochs, [row["backbone_lr_start"] for row in history_rows], (39, 110, 220)),
        ("Head LR", epochs, [row["head_lr_start"] for row in history_rows], (220, 75, 55)),
    ], "Learning Rate (log scale)", logarithmic=True, markers=markers)
    class_names = config.get("class_names") or sorted({row["class_name"] for row in per_class_rows})
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207),
    ]
    class_series = []
    for index, class_name in enumerate(class_names):
        rows = sorted((row for row in per_class_rows if row["class_name"] == class_name),
                      key=lambda row: int(row["epoch"]))
        class_series.append((class_name, [row["epoch"] for row in rows],
                             [row["auroc"] for row in rows], palette[index % len(palette)]))
    _draw_curve(output_dir / "per_class_auroc_curve.png", class_series,
                "Per-class Validation AUROC", markers=markers)

    best_row = next((row for row in history_rows if row["epoch"] == best_epoch), None)
    best_classes = [row for row in per_class_rows if row["epoch"] == best_epoch]
    alpha = None
    if getattr(model, "head_name", None) == "dsap_rpb":
        alpha = [float(value) for value in model.classifier.alpha.detach().cpu()]
    model_config = config.get("model_config") or {}
    head_name = model_config.get("head", getattr(model, "head_name", "unknown"))
    if head_name == "spatial":
        feature_mode = model_config.get("feature_mode", "stage4")
        pooling_mode = model_config.get("pooling_mode", "gap")
        if feature_mode == "stage3_stage4":
            feature_path = "Stage3/Stage4 1×1 投影 → Stage4 双线性尺寸对齐 → 拼接 → 1×1 卷积融合 → GroupNorm/GELU"
        else:
            feature_path = "Stage4 空间特征"
        pooling_path = "1×1 + 2×2 固定 Adaptive Average Pooling 拼接" if pooling_mode == "spp" else "GAP"
        network_path = (f"ConvNeXt-Tiny ImageNet Backbone → {feature_path} → {pooling_path} → "
                        "LayerNorm → Linear(10)；Study logits 使用 Logit Mean。")
    elif head_name == "dsap_rpb":
        network_path = ("ConvNeXt-Tiny ImageNet Backbone Stage4 → DSAP-RPB 空间注意力/疾病查询 → "
                        "Study Logit Mean。")
    else:
        network_path = (f"ConvNeXt-Tiny ImageNet Backbone → {head_name} classifier → "
                        f"Study {model_config.get('study_fusion', 'logit_mean')}。")
    lines = [
        f"# ConvNeXt {head_name} 实验摘要",
        "",
        "## 实验状态",
        "",
        f"- 完成轮次：{history_rows[-1]['epoch']} / {config['epochs']}",
        f"- 停止原因：{'Early Stopping' if stopped_early else ('达到训练轮数上限或正常结束' if training_complete else '训练进行中；本文件为当前进度快照')}",
        f"- Best Epoch：{best_epoch}",
        f"- Best Validation Macro-AUROC：{best_row['macro_auroc']:.6f}" if best_row else "- Best 指标不可用",
        f"- Best Epoch Train Loss：{best_row['train_loss']:.6f}" if best_row else "",
        f"- Best Epoch Val Loss：{best_row['val_loss']:.6f}" if best_row else "",
        f"- Best Epoch Train-Val Gap：{best_row['train_val_gap']:.6f}" if best_row else "",
        f"- Best Epoch Macro-F1：{best_row['macro_f1']:.6f}" if best_row else "",
        "- Train-Val Loss Gap 定义为 Val Loss − Train Loss。Train 使用增强、Validation 不使用增强，因此 Loss 绝对值并非完全同分布，应结合趋势和 AUROC 判断泛化。",
        "",
        "## 网络路径",
        "",
        network_path,
        "",
        "## 最佳轮次逐类验证指标",
        "",
        "| 疾病 | AUROC | AUPRC | Precision | Recall | F1@0.5 | TP | FP | TN | FN | 阳性数 | 阴性数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_classes:
        lines.append(
            f"| {row['class_name']} | {row['auroc']:.6f} | {row['auprc']:.6f} | "
            f"{row.get('precision', float('nan')):.6f} | {row.get('recall', float('nan')):.6f} | "
            f"{row['f1']:.6f} | {row.get('tp', 0)} | {row.get('fp', 0)} | "
            f"{row.get('tn', 0)} | {row.get('fn', 0)} | {row['n_positive']} | {row['n_negative']} |"
        )
    if alpha is not None:
        lines.extend([
            "",
            "## 可学习融合系数",
            "",
            "- Best checkpoint alpha：" + ", ".join(
                f"{name}={value:.6f}" for name, value in zip(config.get("class_names", []), alpha)
            ),
        ])
    lines.extend([
        "",
        "## 可复现性与产物",
        "",
        f"- Seed：{config.get('seed')}",
        f"- Optimizer：AdamW，Weight Decay={config.get('weight_decay')}",
        f"- Scheduler：{config.get('scheduler_step_unit')} 级 Linear Warmup + Cosine，warmup={config.get('warmup_epochs')} epochs",
        f"- Cosine 结束 Epoch：{config.get('cosine_end_epoch', config.get('epochs'))}；最低 LR：Backbone={config.get('min_lr_backbone', 0.0)}，Head={config.get('min_lr_head', 0.0)}",
        f"- Early Stopping：{'enabled' if config.get('early_stopping_enabled', True) else 'disabled'}；最大训练轮数={config.get('epochs')}",
        f"- Backbone LR：{config.get('learning_rate_backbone')}；Head LR：{config.get('learning_rate_head')}",
        f"- 初始化：{config.get('initialization_policy')}",
        f"- ImageNet 权重来源：{config.get('pretrained_weight_source')}",
        f"- ImageNet 权重 SHA-256：{config.get('pretrained_weight_sha256')}",
        f"- Stage4 smoke 特征形状：{smoke_test.get('backbone_stage4_shape')}",
        f"- 参数计数：GAP 参照={config.get('parameter_counts', {}).get('original_gap_total'):,}，"
        f"当前模型={config.get('parameter_counts', {}).get('modified_total'):,}，"
        f"相对 GAP 参照新增={config.get('parameter_counts', {}).get('new_parameters'):,}",
        f"- 训练峰值显存：{max(row['peak_gpu_gb'] for row in history_rows):.3f} GiB",
        f"- Smoke Test 峰值显存：{smoke_test.get('smoke_peak_gpu_gb', float('nan')):.3f} GiB",
        f"- Best checkpoint：{output_dir / config.get('best_checkpoint_name', 'best.pth')}",
        f"- Last checkpoint：{output_dir / config.get('last_checkpoint_name', 'last.pth')}",
        "- Epoch 20 checkpoint：" + (str(output_dir / "epoch20.pth")
                                     if 20 in config.get("milestone_epochs", []) else "未启用"),
        "- 其他产物：config.json、train_log.csv、per_class_metrics.csv、loss_curve.png、auroc_curve.png、per_class_auroc_curve.png、lr_curve.png。",
    ])
    if head_name == "dsap_rpb":
        lines.extend([
            "- Relative Position Bias 表参数量为 num_heads × (2Hmax−1) × (2Wmax−1)；注意力矩阵复杂度为 O(B × heads × N²)，N=H×W。",
            "- Attention Map 是模型权重的可视化，不代表经过医学验证的病灶定位。",
        ])
    reference = config.get("reference_gap_baseline")
    if reference:
        lines.extend([
            "",
            "## 历史 GAP 参照",
            "",
            f"历史 ConvNeXt GAP best Macro-AUROC={reference.get('best_macro_auroc'):.6f}，Best Epoch={reference.get('epoch')}。",
            "该参照配置与当前实验不同；差值不能单独归因于某一个结构或训练因素。",
        ])
    previous_dsap = config.get("reference_dsap_rpb_previous_run")
    if previous_dsap and head_name == "dsap_rpb":
        current_best = best_row["macro_auroc"] if best_row else float("nan")
        previous_best = previous_dsap.get("best_macro_auroc", float("nan"))
        lines.extend([
            "",
            "## 上一轮 DSAP-RPB 调度实验对照",
            "",
            f"- 上一轮：best Epoch {previous_dsap.get('best_epoch')}，Macro-AUROC={previous_best:.6f}，"
            f"last Epoch={previous_dsap.get('last_epoch')}。",
            f"- 本轮：best Epoch {best_epoch}，Macro-AUROC={current_best:.6f}，完成 Epoch={history_rows[-1]['epoch']}。",
            f"- 本轮相对差值：{current_best - previous_best:+.6f} Macro-AUROC。",
            "- 上一轮在 14 Epoch 处中止；本轮目标为完整 40 Epoch。因此总训练步数不同，不能把差异全部归因于 cosine 周期或低学习率阶段。",
        ])
    phase_rows = []
    warmup_end = int(config.get("warmup_epochs", 0))
    cosine_end = int(config.get("cosine_end_epoch", config["epochs"]))
    max_epoch = int(config["epochs"])
    phase_specs = []
    if warmup_end > 0:
        phase_specs.append(("Warmup", 1, warmup_end))
    if cosine_end > warmup_end:
        phase_specs.append(("Cosine", warmup_end + 1, cosine_end))
    if max_epoch > cosine_end:
        phase_specs.append(("Minimum LR", cosine_end + 1, max_epoch))
    for phase_name, first_epoch, last_epoch in phase_specs:
        rows = [row for row in history_rows if first_epoch <= int(row["epoch"]) <= last_epoch]
        if rows:
            best_phase = max(rows, key=lambda row: row["macro_auroc"])
            phase_rows.append((phase_name, rows, best_phase))
    if phase_rows:
        lines.extend(["", "## 分阶段趋势", "", "| 阶段 | Epoch 数 | Train Loss 首→末 | Val Loss 首→末 | AUROC 首→末 | 阶段最佳 AUROC |", "|---|---:|---:|---:|---:|---:|"])
        for phase_name, rows, best_phase in phase_rows:
            lines.append(
                f"| {phase_name} | {len(rows)} | {rows[0]['train_loss']:.5f} → {rows[-1]['train_loss']:.5f} | "
                f"{rows[0]['val_loss']:.5f} → {rows[-1]['val_loss']:.5f} | "
                f"{rows[0]['macro_auroc']:.5f} → {rows[-1]['macro_auroc']:.5f} | "
                f"{best_phase['macro_auroc']:.6f} (Epoch {best_phase['epoch']}) |"
            )
    (output_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _forward_studies(model, batch, device, amp_enabled, amp_dtype):
    images = batch["images"].to(device, non_blocking=True)
    study_index = batch["study_index"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(images, study_index, batch["num_studies"])
    return logits.float()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_enabled,
                    amp_dtype, accumulation, clip_norm, freeze_batchnorm_stats=False,
                    scheduler=None, scheduler_step_unit="epoch"):
    model.train()
    if not next(model.backbone.parameters()).requires_grad:
        model.backbone.eval()
    elif freeze_batchnorm_stats:
        for module in model.backbone.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_valid_studies, total_norm, steps, updates = 0.0, 0, 0.0, 0, 0
    skipped_updates = 0
    lr_start = _learning_rates(optimizer)
    lr_last_used = list(lr_start)
    batch_sampler = getattr(loader, "batch_sampler", None)
    planned_batches = batch_sampler._batches() if hasattr(batch_sampler, "_batches") else None
    if planned_batches is not None and hasattr(loader.dataset, "studies"):
        study_is_valid = [bool(item["valid"].any()) for item in loader.dataset.studies]
        accumulation_counts = []
        for start in range(0, len(planned_batches), accumulation):
            window = planned_batches[start:start + accumulation]
            accumulation_counts.append(sum(study_is_valid[index]
                                           for batch_indices in window for index in batch_indices))
    else:
        accumulation_counts = None
    for index, batch in enumerate(loader):
        labels = batch["labels"].to(device, non_blocking=True)
        known = batch["known"].to(device, non_blocking=True)
        logits = _forward_studies(model, batch, device, amp_enabled, amp_dtype)
        loss = criterion(logits, labels, known)
        eligible_studies = int(known.any(dim=1).sum().item())
        if accumulation_counts is not None:
            window_valid_studies = max(accumulation_counts[index // accumulation], 1)
            scaled_loss = loss * (eligible_studies / window_valid_studies)
        else:
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            scaled_loss = loss / window_size
        scaler.scale(scaled_loss).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            total_norm += float(norm)
            lr_last_used = _learning_rates(optimizer)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_succeeded = not scaler.is_enabled() or scaler.get_scale() >= scale_before
            if update_succeeded:
                updates += 1
                if scheduler is not None and scheduler_step_unit == "optimizer_update":
                    scheduler.step()
            else:
                skipped_updates += 1
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * eligible_studies
        total_valid_studies += eligible_studies
        steps += 1
    return {"loss": total_loss / max(total_valid_studies, 1),
            "gradient_norm": total_norm / max(updates + skipped_updates, 1),
            "optimizer_updates": updates, "skipped_updates": skipped_updates,
            "lr_start": lr_start, "lr_last_used": lr_last_used}


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device,
        logger, class_names=None):
    amp_dtype = torch.bfloat16 if config["amp_dtype"] == "bfloat16" else torch.float16
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    if amp_dtype == torch.bfloat16 and amp_enabled and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 is unsupported on this GPU; using FP16 autocast")
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    output_dir = Path(config["output_dir"])
    best, stale, start_epoch = float("-inf"), 0, 0
    best_epoch = None
    best_name = config.get("best_checkpoint_name", "checkpoint/best/best_by_macro_auc.pth")
    last_name = config.get("last_checkpoint_name", "checkpoint/last/last.pth")
    best_path, last_path = output_dir / best_name, output_dir / last_name
    experiment_artifacts = bool(config.get("save_experiment_artifacts", False))
    history_rows, per_class_rows = [], []
    stopped_early = False
    if experiment_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
    resume = config.get("resume")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(state.get("epoch", 0))
        stored_best = state.get("best_metric")
        best = float("-inf") if stored_best is None else float(stored_best)
        stale = int(state.get("stale_epochs", 0))
        best_epoch = state.get("best_epoch")
        logger.info("resumed checkpoint=%s epoch=%d best_macro_auroc=%s", resume, start_epoch, best)
        if experiment_artifacts:
            history_path = output_dir / "train_log.csv"
            per_class_path = output_dir / "per_class_metrics.csv"
            if history_path.is_file():
                history_rows = pd.read_csv(history_path).to_dict("records")
            if per_class_path.is_file():
                per_class_rows = pd.read_csv(per_class_path).to_dict("records")

    for epoch in range(start_epoch, int(config["epochs"])):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        frozen = epoch < int(config.get("freeze_backbone_epochs", 0))
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            model.backbone.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        scheduler_step_unit = config.get("scheduler_step_unit", "epoch")
        if scheduler_step_unit not in {"epoch", "optimizer_update"}:
            raise ValueError("scheduler_step_unit must be epoch or optimizer_update")
        train_started = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
            amp_enabled, amp_dtype, int(config["grad_accumulation"]),
            float(config["gradient_clip_norm"]),
            bool(config.get("freeze_batchnorm_stats", False)),
            scheduler if scheduler_step_unit == "optimizer_update" else None,
            scheduler_step_unit)
        train_elapsed = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation = validate(model, val_loader, criterion, device, amp_enabled, amp_dtype)
        validation_elapsed = time.perf_counter() - validation_started
        if scheduler and scheduler_step_unit == "epoch":
            scheduler.step()
        peak_memory = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                       if device.type == "cuda" else 0.0)
        next_lrs = _learning_rates(optimizer)
        logger.info("epoch=%d/%d train_loss=%.5f val_loss=%.5f train_val_gap=%.5f "
                    "macro_auroc=%s macro_f1=%s gradient_norm=%.4f "
                    "lr_start=%s lr_last_used=%s lr_next=%s optimizer_updates=%d "
                    "skipped_updates=%d peak_gpu_gb=%.3f train_s=%.1f val_s=%.1f",
                    epoch + 1, config["epochs"], train_stats["loss"], validation["loss"],
                    validation["loss"] - train_stats["loss"],
                    validation["macro_auroc"], validation["macro_f1"],
                    train_stats["gradient_norm"], train_stats["lr_start"],
                    train_stats["lr_last_used"], next_lrs, train_stats["optimizer_updates"],
                    train_stats["skipped_updates"], peak_memory,
                    train_elapsed, validation_elapsed)
        for index, item in enumerate(validation["per_label"]):
            name = class_names[index] if class_names else index + 1
            logger.info("label[%s] valid=%d positive=%d negative=%d auroc=%s auprc=%s f1=%s",
                        name, item["n_valid"], item["n_positive"], item["n_negative"],
                        item["auroc"], item["auprc"], item["f1"])

        score = validation["macro_auroc"]
        improved = math.isfinite(score) and score > best
        if improved:
            best, stale = score, 0
            best_epoch = epoch + 1
        else:
            stale += 1
        save_args = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                         epoch=epoch + 1, best_metric=best, config=config,
                         labels=class_names, stale_epochs=stale, best_epoch=best_epoch)
        save_checkpoint(last_path, **save_args)
        if improved:
            if config.get("save_full_state_for_best", False):
                save_checkpoint(best_path, **save_args)
            else:
                # Legacy experiments keep their deployable best model weights-only.
                save_checkpoint(best_path, model=model, epoch=epoch + 1, best_metric=best,
                                config=config, labels=class_names, stale_epochs=stale,
                                best_epoch=best_epoch)
            rows = []
            for row_index, (subject_id, study_id) in enumerate(
                    zip(validation["subject_ids"], validation["study_ids"])):
                row = {"Subject_id": subject_id, "Study_id": study_id}
                for label_index, name in enumerate(class_names or range(1, 11)):
                    row[f"true_{name}"] = (float(validation["targets"][row_index, label_index])
                                            if validation["known"][row_index, label_index] else np.nan)
                    row[f"known_{name}"] = bool(validation["known"][row_index, label_index])
                    row[f"prob_{name}"] = float(validation["probabilities"][row_index, label_index])
                rows.append(row)
            predictions_path = output_dir / "outputs/predictions/val_best.csv"
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(predictions_path, index=False)
        if epoch + 1 in set(int(value) for value in config.get("milestone_epochs", [])):
            milestone_path = output_dir / f"epoch{epoch + 1}.pth"
            save_checkpoint(milestone_path, **save_args)
        if experiment_artifacts:
            history_rows.append({
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "val_loss": validation["loss"],
                "train_val_gap": validation["loss"] - train_stats["loss"],
                "macro_auroc": validation["macro_auroc"],
                "macro_f1": validation["macro_f1"],
                "backbone_lr_start": train_stats["lr_start"][0],
                "head_lr_start": train_stats["lr_start"][1],
                "backbone_lr_last_used": train_stats["lr_last_used"][0],
                "head_lr_last_used": train_stats["lr_last_used"][1],
                "backbone_lr_next": next_lrs[0],
                "head_lr_next": next_lrs[1],
                "gradient_norm": train_stats["gradient_norm"],
                "optimizer_updates": train_stats["optimizer_updates"],
                "skipped_updates": train_stats["skipped_updates"],
                "peak_gpu_gb": peak_memory,
                "train_s": train_elapsed,
                "val_s": validation_elapsed,
                "best_epoch": best_epoch,
                "best_macro_auroc": best,
                "patience_counter": stale,
            })
            for label_index, item in enumerate(validation["per_label"]):
                per_class_rows.append({
                    "epoch": epoch + 1,
                    "class_id": label_index + 1,
                    "class_name": class_names[label_index] if class_names else str(label_index + 1),
                    **item,
                })
            _write_experiment_artifacts(
                output_dir, config, history_rows, per_class_rows, model, best_epoch, False
            )
        if config.get("early_stopping_enabled", True) and stale >= int(config["patience"]):
            logger.info("early stopping after %d epochs without Macro-AUROC improvement", stale)
            stopped_early = True
            break
    if experiment_artifacts and history_rows:
        if best_path.exists():
            best_state = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best_state["model"])
        _write_experiment_artifacts(
            output_dir, config, history_rows, per_class_rows, model, best_epoch, stopped_early,
            training_complete=True
        )
