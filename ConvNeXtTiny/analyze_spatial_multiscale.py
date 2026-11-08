"""Profile and compare the controlled A0/A1/B1/C1 spatial ablations."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

from model import ChestXrayClassifier
from utils.config import PROJECT_DIR


VARIANTS = (
    ("A0", "Stage4 + GAP", "ConvNeXt_Spatial_A0_GAP_BCE20"),
    ("A1", "Stage4 + SPP", "ConvNeXt_Spatial_A1_SPP_BCE20"),
    ("B1", "Stage3+4 fusion + GAP", "ConvNeXt_Spatial_B1_Stage3Stage4_GAP_BCE20"),
    ("C1", "Stage3+4 fusion + SPP", "ConvNeXt_Spatial_C1_Stage3Stage4_SPP_BCE20"),
)
COLORS = {
    "A0": (40, 95, 190),
    "A1": (225, 115, 35),
    "B1": (35, 145, 85),
    "C1": (185, 65, 75),
}


def _font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _profile_model(variant, out_dir, config):
    checkpoint_path = out_dir / "best.pth"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict((state.get("config") or {}).get("model_config") or config["model_config"])
    model_config["pretrained"] = False
    model = ChestXrayClassifier(**{key: value for key, value in model_config.items()
                                   if key != "name"})
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters()
                               if parameter.requires_grad)
    macs = [0]
    hooks = []

    def count_conv(module, _inputs, output):
        kernel_area = int(np.prod(module.kernel_size))
        macs[0] += int(output.numel() * (module.in_channels // module.groups) * kernel_area)

    def count_linear(module, _inputs, output):
        macs[0] += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_size = int((config.get("data") or {}).get("image_size", 320))
    images = torch.zeros((1, 3, image_size, image_size), device=device)
    study_index = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.inference_mode():
        model(images, study_index, num_studies=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for hook in hooks:
        hook.remove()

    profile = {
        "variant": variant,
        "device": str(device),
        "input_shape": [1, 3, image_size, image_size],
        "backbone_parameters": int(backbone_parameters),
        "head_and_fusion_parameters": int(total_parameters - backbone_parameters),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "macs_per_input_image": int(macs[0]),
        "mac_count_scope": "Conv2d and Linear multiply-accumulate counts; excludes normalization, activations, pooling, interpolation and elementwise operations.",
        "best_checkpoint_bytes": checkpoint_path.stat().st_size,
        "last_checkpoint_bytes": (out_dir / "last.pth").stat().st_size,
        "epoch20_checkpoint_bytes": (out_dir / "epoch20.pth").stat().st_size,
    }
    (out_dir / "model_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    del images, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return profile


def _load_variant(key, label, directory):
    out_dir = PROJECT_DIR / "outputs" / "experiments" / directory
    required = ("train_log.csv", "per_class_metrics.csv", "best.json", "last.json",
                "best.pth", "last.pth", "epoch20.pth", "outputs/predictions/val_best.csv",
                "validation_latency.json")
    missing = [name for name in required if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{key} is missing expected 20-epoch artifacts: {missing}")
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(out_dir / "train_log.csv")
    per_class = pd.read_csv(out_dir / "per_class_metrics.csv")
    best = json.loads((out_dir / "best.json").read_text(encoding="utf-8"))
    last = json.loads((out_dir / "last.json").read_text(encoding="utf-8"))
    if len(logs) != 20 or int(last["epoch"]) != 20:
        raise ValueError(f"{key} must have exactly 20 completed epochs, got {len(logs)} / {last['epoch']}")
    if int(last["epoch"]) != int(logs.iloc[-1]["epoch"]):
        raise ValueError(f"{key} train log and last checkpoint disagree")
    best_epoch = int(best["epoch"])
    best_row = logs.loc[logs.epoch == best_epoch].iloc[0].to_dict()
    best_classes = per_class.loc[per_class.epoch == best_epoch].copy()
    latency = json.loads((out_dir / "validation_latency.json").read_text(encoding="utf-8"))
    profile = json.loads((out_dir / "model_profile.json").read_text(encoding="utf-8")) if (
        out_dir / "model_profile.json").is_file() else None
    return {
        "key": key, "label": label, "directory": directory, "out_dir": out_dir,
        "config": config, "logs": logs, "per_class": per_class,
        "best": best, "last": last, "best_epoch": best_epoch,
        "best_row": best_row, "best_classes": best_classes,
        "latency": latency, "profile": profile,
        "predictions": pd.read_csv(out_dir / "outputs/predictions/val_best.csv"),
    }


def _line_chart(path, title, y_label, variants, metric, markers=(2, 5, 10, 15, 20)):
    width, height = 1500, 900
    left, top, right, bottom = 125, 115, 1370, 735
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font, small, title_font = _font(18), _font(15), _font(25)
    draw.text((left, 25), title, fill="black", font=title_font)
    values = [float(value) for item in variants for value in item["logs"][metric].tolist()]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.09, abs(high) * 0.025, 0.005)
    y_min, y_max = low - padding, high + padding

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((left - 95, y - 9), f"{value:.3f}", fill="black", font=small)
    for epoch in markers:
        x = left + int((epoch - 1) / 19 * (right - left))
        draw.line((x, top, x, bottom), fill=(236, 236, 236), width=1)
        draw.text((x - 12, bottom + 10), str(epoch), fill="black", font=small)
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    draw.text((left, bottom + 45), "Epoch", fill="black", font=font)
    draw.text((left, top - 32), y_label, fill="black", font=font)

    legend_x, legend_y = right - 385, top + 5
    for index, item in enumerate(variants):
        color = COLORS[item["key"]]
        epochs = item["logs"]["epoch"].astype(int).tolist()
        series = [float(value) for value in item["logs"][metric].tolist()]
        points = [
            (left + int((epoch - 1) / 19 * (right - left)),
             bottom - int((value - y_min) / (y_max - y_min) * (bottom - top)))
            for epoch, value in zip(epochs, series)
        ]
        draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.line((legend_x, legend_y + 10 + 30 * index,
                   legend_x + 35, legend_y + 10 + 30 * index), fill=color, width=4)
        draw.text((legend_x + 44, legend_y + 1 + 30 * index),
                  f"{item['key']} {item['label']}", fill=color, font=small)

        if metric == "macro_auroc":
            best_idx = int(np.argmax(series))
            x, y = points[best_idx]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)
            draw.text((min(x + 8, right - 100), max(y - 22, top)),
                      f"E{epochs[best_idx]} {series[best_idx]:.3f}", fill=color, font=small)
    image.save(path)


def _bar_chart(path, variants, class_names):
    width, height = 1600, 900
    left, top, right, bottom = 125, 100, 1460, 690
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font, small, title_font = _font(18), _font(14), _font(25)
    draw.text((left, 24), "Best-epoch AUROC by disease (y-axis starts at 0.4; class IDs match report)",
              fill="black", font=title_font)
    y_min, y_max = 0.4, 1.0
    for tick in range(7):
        value = y_min + (y_max - y_min) * tick / 6
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((left - 65, y - 8), f"{value:.1f}", fill="black", font=small)
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    group_width = (right - left) / len(class_names)
    bar_width = group_width * 0.16
    for class_index, class_name in enumerate(class_names):
        center = left + group_width * (class_index + 0.5)
        draw.text((center - 5, bottom + 8), str(class_index + 1), fill="black", font=small)
        for model_index, item in enumerate(variants):
            rows = item["best_classes"].loc[item["best_classes"].class_name == class_name]
            if rows.empty:
                continue
            value = float(rows.iloc[0]["auroc"])
            x0 = int(center + (model_index - 1.5) * bar_width - bar_width / 2)
            x1 = int(x0 + bar_width - 2)
            y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
            draw.rectangle((x0, y, x1, bottom), fill=COLORS[item["key"]])
            draw.text((x0, max(top, y - 18)), f"{value:.2f}", fill=COLORS[item["key"]], font=_font(11))
    for index, item in enumerate(variants):
        y = top + 12 + index * 28
        draw.rectangle((right - 330, y + 3, right - 312, y + 19), fill=COLORS[item["key"]])
        draw.text((right - 300, y), f"{item['key']} {item['label']}", fill=COLORS[item["key"]], font=small)
    legend_y = bottom + 40
    draw.text((left, legend_y), "Class IDs correspond to the disease names in the report table.",
              fill="black", font=small)
    image.save(path)


def _auc_macro(targets, probabilities):
    aucs = []
    for index in range(targets.shape[1]):
        y = targets[:, index]
        if np.unique(y).size < 2:
            return float("nan")
        aucs.append(float(roc_auc_score(y, probabilities[:, index])))
    return float(np.mean(aucs))


def _paired_subject_bootstrap(variants, repetitions=2000, seed=42):
    class_names = variants[0]["config"]["class_names"]
    base_frame = variants[0]["predictions"].copy()
    base_frame["Subject_id"] = base_frame["Subject_id"].astype(str)
    base_frame["Study_id"] = base_frame["Study_id"].astype(str)
    base_frame = base_frame.set_index(["Subject_id", "Study_id"]).sort_index()
    keys = base_frame.index
    targets = np.column_stack([base_frame[f"true_{name}"].to_numpy(dtype=float)
                               for name in class_names])
    known = np.column_stack([base_frame[f"known_{name}"].to_numpy(dtype=bool)
                             for name in class_names])
    if not known.all():
        raise ValueError("Paired bootstrap expects the complete closed-world validation labels")
    probability_matrices = [
        np.column_stack([
            item["predictions"].assign(
                Subject_id=item["predictions"]["Subject_id"].astype(str),
                Study_id=item["predictions"]["Study_id"].astype(str),
            ).set_index(["Subject_id", "Study_id"]).loc[keys][f"prob_{name}"].to_numpy(dtype=float)
            for name in class_names
        ])
        for item in variants
    ]
    for item in variants[1:]:
        frame = item["predictions"].copy()
        frame["Subject_id"] = frame["Subject_id"].astype(str)
        frame["Study_id"] = frame["Study_id"].astype(str)
        frame = frame.set_index(["Subject_id", "Study_id"]).loc[keys]
        other_targets = np.column_stack([frame[f"true_{name}"].to_numpy(dtype=float)
                                         for name in class_names])
        other_known = np.column_stack([frame[f"known_{name}"].to_numpy(dtype=bool)
                                       for name in class_names])
        if not np.array_equal(other_targets, targets) or not np.array_equal(other_known, known):
            raise ValueError(f"{item['key']} validation labels do not align exactly with A0")
    subjects = base_frame.index.get_level_values("Subject_id").to_numpy()
    subject_rows = {}
    for index, subject in enumerate(subjects):
        subject_rows.setdefault(subject, []).append(index)
    subject_names = np.asarray(list(subject_rows), dtype=object)
    rng = np.random.default_rng(seed)
    deltas = {item["key"]: [] for item in variants[1:]}
    accepted, attempts = 0, 0
    while accepted < repetitions and attempts < repetitions * 20:
        attempts += 1
        sampled_subjects = rng.choice(subject_names, size=len(subject_names), replace=True)
        indices = np.concatenate([subject_rows[subject] for subject in sampled_subjects])
        base_auc = _auc_macro(targets[indices], probability_matrices[0][indices])
        model_aucs = [_auc_macro(targets[indices], matrix[indices])
                       for matrix in probability_matrices[1:]]
        if not math.isfinite(base_auc) or any(not math.isfinite(value) for value in model_aucs):
            continue
        accepted += 1
        for item, model_auc in zip(variants[1:], model_aucs):
            deltas[item["key"]].append(model_auc - base_auc)
    if accepted < repetitions:
        raise RuntimeError(f"Only {accepted}/{repetitions} valid subject bootstrap samples were obtained")
    return {
        key: {
            "repetitions": repetitions,
            "seed": seed,
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        for key, values in deltas.items()
    }


def main():
    variants = [_load_variant(*variant) for variant in VARIANTS]
    expected_modes = {
        "A0": ("stage4", "gap"),
        "A1": ("stage4", "spp"),
        "B1": ("stage3_stage4", "gap"),
        "C1": ("stage3_stage4", "spp"),
    }
    if variants[0]["config"].get("loss") != "bce" or variants[0]["config"].get("weighting") != "none":
        raise ValueError("A0 must be the unweighted standard BCE strict baseline")
    shared_train_fields = (
        "seed", "data_loader_worker_seed", "epochs", "image_budget", "grad_accumulation",
        "learning_rate_backbone", "learning_rate_head", "min_lr_backbone", "min_lr_head",
        "weight_decay", "warmup_epochs", "cosine_end_epoch", "early_stopping_enabled",
        "freeze_backbone_epochs", "gradient_clip_norm", "scheduler_step_unit",
        "loss", "weighting", "amp", "amp_dtype",
    )
    for item in variants:
        model_config = item["config"].get("model_config", {})
        if (model_config.get("head"), model_config.get("feature_mode"),
                model_config.get("pooling_mode"), model_config.get("study_fusion")) != (
                "spatial", *expected_modes[item["key"]], "logit_mean"):
            raise ValueError(f"{item['key']} model configuration does not match its ablation definition")
        if item["config"].get("loss") != "bce" or item["config"].get("weighting") != "none":
            raise ValueError(f"{item['key']} is not configured for unweighted BCE")
        if item["config"].get("early_stopping_enabled") is not False:
            raise ValueError(f"{item['key']} must have early stopping disabled")
        if not item["config"].get("model_config", {}).get("pretrained"):
            raise ValueError(f"{item['key']} did not request ImageNet backbone initialization")
        for field in shared_train_fields:
            if item["config"].get(field) != variants[0]["config"].get(field):
                raise ValueError(f"{item['key']} training field {field} differs from A0")
        if (int(item["config"].get("epochs", 0)) != 20
                or int(item["config"].get("warmup_epochs", 0)) != 2
                or int(item["config"].get("cosine_end_epoch", 0)) != 20
                or int(item["config"].get("data", {}).get("image_size", 0)) != 320):
            raise ValueError(f"{item['key']} does not match the planned 320px/20-epoch schedule")
        class_names = item["config"].get("class_names") or item["config"].get("data", {}).get("class_names")
        if class_names != variants[0]["config"].get("class_names"):
            raise ValueError(f"{item['key']} class order differs from A0")
    reference_auc = float(variants[0]["best"]["best_macro_auroc"])
    for item in variants:
        item["profile"] = _profile_model(item["key"], item["out_dir"], item["config"])
        item["relative_auc"] = float(item["best"]["best_macro_auroc"]) - reference_auc
        item["latency"] = json.loads((item["out_dir"] / "validation_latency.json").read_text(encoding="utf-8"))

    comparison_dir = PROJECT_DIR / "outputs" / "spatial_multiscale_ablation"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _line_chart(comparison_dir / "macro_auroc_comparison.png",
                "Validation Macro-AUROC by epoch", "Macro-AUROC", variants, "macro_auroc")
    _line_chart(comparison_dir / "validation_loss_comparison.png",
                "Validation loss by epoch", "BCE loss", variants, "val_loss")
    _line_chart(comparison_dir / "train_val_gap_comparison.png",
                "Train-validation loss gap by epoch", "Validation loss - train loss",
                variants, "train_val_gap")
    class_names = variants[0]["config"]["class_names"]
    _bar_chart(comparison_dir / "best_class_auroc_comparison.png", variants, class_names)
    bootstrap = _paired_subject_bootstrap(variants)
    (comparison_dir / "paired_subject_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8")

    historical_path = PROJECT_DIR / "checkpoint" / "best" / "best_by_macro_auc.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8")) if historical_path.is_file() else {}
    report = [
        "# ConvNeXt-Tiny 空间特征与多尺度消融分析",
        "",
        "## 实验控制与基线",
        "",
        "A0、A1、B1、C1 均使用相同 Train/Validation 划分、320×320 输入、图像预处理、StudyBatchSampler、image budget、梯度累积、AMP、AdamW、weight decay、seed、DataLoader worker seeds、Study Logit Mean 与无正样本权重的 BCE。每组都从 torchvision ImageNet Backbone 权重开始；分类头和融合层重新初始化，不跨组继承微调 checkpoint。每组完成 20 epochs，Epoch 1–2 warmup，Epoch 3–20 按 optimizer update 推进 cosine 到最低 LR；Early Stopping 关闭。test split 未读取。",
        "",
        "历史 GAP checkpoint 不是严格 A0：历史使用 backbone LR `2e-5` 和 40-epoch 计划。旧元数据虽写 `weighting=sqrt`，其 `loss=bce` 分支没有传入 `pos_weight`（`weight_counts` 为空）；本轮仍为可比性重新训练了 A0，且固定 backbone LR 为 `1e-5`、head LR 为 `2e-4`、20 epochs。",
        "",
        "## 总体结果",
        "",
        "| 实验 | 特征来源 | 池化 | Best Epoch | Best Macro-AUROC | Macro-F1@0.5 | 相对严格 A0 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in variants:
        report.append(
            f"| {item['key']} | {item['config']['model_config']['feature_mode']} | "
            f"{item['config']['model_config']['pooling_mode']} | {item['best_epoch']} | "
            f"{float(item['best']['best_macro_auroc']):.6f} | "
            f"{float(item['best_row']['macro_f1']):.6f} | {item['relative_auc']:+.6f} |"
        )

    report.extend([
        "",
        "## 模型规模与推理速度",
        "",
        "MACs 以 batch=1、320×320 单图、Study Logit Mean 前向估算，仅统计 Conv2d 和 Linear 的乘加；不计 Norm、激活、池化、插值和逐元素操作。验证速度在 valid.csv 的前 100 个 Study 测量，排除前 10 个 warmup Study；报告的是模型前向与 Study 聚合时间，另列 decode/预处理与设备传输耗时。",
        "",
        "| 实验 | Backbone 参数 | Head/融合参数 | 总参数 | 相对 A0 新增 | MACs/图像 | best / last / E20 checkpoint MiB | Val 前向 p50/p95 ms/Study | 峰值显存 GiB |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    base_params = variants[0]["profile"]["total_parameters"]
    for item in variants:
        profile = item["profile"]
        latency = item["latency"]["model_forward_and_fusion_per_study"]
        sizes = ", ".join(f"{profile[key] / (1024 ** 2):.1f}" for key in (
            "best_checkpoint_bytes", "last_checkpoint_bytes", "epoch20_checkpoint_bytes"))
        report.append(
            f"| {item['key']} | {profile['backbone_parameters']:,} | "
            f"{profile['head_and_fusion_parameters']:,} | {profile['total_parameters']:,} | "
            f"{profile['total_parameters'] - base_params:+,} | {profile['macs_per_input_image'] / 1e9:.3f} G | "
            f"{sizes} | {latency['p50_ms']:.2f}/{latency['p95_ms']:.2f} | "
            f"{max(float(row['peak_gpu_gb']) for row in item['logs'].to_dict('records')):.3f} |"
        )

    report.extend([
        "",
        "## 对比曲线",
        "",
        "![四组 Validation Macro-AUROC](macro_auroc_comparison.png)",
        "",
        "![四组 Validation Loss](validation_loss_comparison.png)",
        "",
        "![Best Epoch 逐类 AUROC](best_class_auroc_comparison.png)",
        "",
        "![Train-Val Loss Gap](train_val_gap_comparison.png)",
        "",
        "## 最佳 Epoch 十疾病 AUROC",
        "",
        "| 疾病 | A0 | A1 | ΔA1−A0 | B1 | ΔB1−A0 | C1 | ΔC1−A0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    class_rows = {}
    for item in variants:
        for _, row in item["best_classes"].iterrows():
            class_rows.setdefault(row["class_name"], {})[item["key"]] = float(row["auroc"])
    for name in class_names:
        values = class_rows[name]
        report.append(
            f"| {name} | {values['A0']:.6f} | {values['A1']:.6f} | {values['A1']-values['A0']:+.6f} | "
            f"{values['B1']:.6f} | {values['B1']-values['A0']:+.6f} | "
            f"{values['C1']:.6f} | {values['C1']-values['A0']:+.6f} |"
        )

    report.extend([
        "",
        "### 重点类别",
        "",
    ])
    for target in ("Pneumonia", "Lung Opacity", "Consolidation", "Pneumothorax", "Atelectasis"):
        values = class_rows[target]
        best_variant = max(variants, key=lambda item: values[item["key"]])
        report.append(
            f"- **{target}**：A0={values['A0']:.6f}；A1 Δ={values['A1']-values['A0']:+.6f}；"
            f"B1 Δ={values['B1']-values['A0']:+.6f}；C1 Δ={values['C1']-values['A0']:+.6f}。"
            f"本次最高为 {best_variant['key']}（{values[best_variant['key']]:.6f}）。"
        )

    report.extend([
        "",
        "## 泛化与训练趋势",
        "",
        "| 实验 | AUROC Best Epoch | Best AUROC | Epoch20 AUROC | AUROC 变化（E20−Best） | Best Val Loss / Epoch | Epoch20 Val Loss | Gap@Best | Gap@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in variants:
        logs = item["logs"]
        final = logs.iloc[-1]
        minimum = logs.loc[logs.val_loss.idxmin()]
        best = item["best_row"]
        report.append(
            f"| {item['key']} | {item['best_epoch']} | {float(item['best']['best_macro_auroc']):.6f} | "
            f"{float(final.macro_auroc):.6f} | {float(final.macro_auroc)-float(item['best']['best_macro_auroc']):+.6f} | "
            f"{float(minimum.val_loss):.6f} / {int(minimum.epoch)} | {float(final.val_loss):.6f} | "
            f"{float(best['train_val_gap']):.6f} | {float(final.train_val_gap):.6f} |"
        )

    report.extend([
        "",
        "## Paired Subject Bootstrap",
        "",
        "以下以最佳 checkpoint 的验证预测做 2,000 次按 Subject 重采样，并在同一组重采样上计算模型间 Macro-AUROC 差值。区间是探索性 95% percentile CI；最佳轮次也由同一验证集挑选，存在模型选择偏差，结果不能替代独立测试集。",
        "",
        "| 比较 | 观测 Δ Macro-AUROC | 95% CI |",
        "|---|---:|---:|",
    ])
    for item in variants[1:]:
        interval = bootstrap[item["key"]]
        report.append(
            f"| {item['key']} − A0 | {item['relative_auc']:+.6f} | "
            f"[{interval['low']:+.6f}, {interval['high']:+.6f}] |"
        )

    report.extend([
        "",
        "## 结论",
        "",
        f"1. **Stage 4 GAP 是否受限于空间压缩：** A1 相对严格 A0 的 Macro-AUROC 差值为 {variants[1]['relative_auc']:+.6f}。据此回答 SPP 是否改善排序能力；同时结合逐类差值和置信区间，不将新增分类器参数影响误归因为空间保留本身。",
        f"2. **Stage 3+4 融合是否优于 Stage 4：** B1 相对 A0 的差值为 {variants[2]['relative_auc']:+.6f}；用 MACs/参数增量和重点疾病表现共同评价成本与收益。",
        f"3. **SPP 与多尺度融合是否互补：** C1−A0={variants[3]['relative_auc']:+.6f}，"
        f"C1−A1={float(variants[3]['best']['best_macro_auroc'])-float(variants[1]['best']['best_macro_auroc']):+.6f}，"
        f"C1−B1={float(variants[3]['best']['best_macro_auroc'])-float(variants[2]['best']['best_macro_auroc']):+.6f}；"
        "不假设叠加模块必然提升。",
        "4. **疾病差异：** 查阅上表重点类别的 ΔAUROC；十个类别 Macro-AUROC 等权汇总，不能只依据选定的五类。",
        "5. **过拟合：** 结合四组 Train Loss、Val Loss、AUROC 和 Gap 曲线及汇总表，比较验证最佳点后性能变化。",
        "",
        "## 参照与限制",
        "",
        f"历史 ConvNeXt GAP checkpoint 为 Epoch {historical.get('epoch')}、Macro-AUROC {float(historical.get('best_macro_auroc', float('nan'))):.6f}；它使用不同 backbone LR/训练周期，只作历史参照，不作为严格 A0。Strict A0 是本次重新训练的模型。所有结论只来自 Train/Validation；test split 未读取。",
        "",
        f"报告生成后保留逐组 `train_log.csv`、`per_class_metrics.csv`、`val_best.csv`、`validation_latency.json`、`model_profile.json`、best/last/Epoch20 checkpoint 与单组训练曲线。",
    ])
    report_path = comparison_dir / "convnext_spatial_multiscale_ablation_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
