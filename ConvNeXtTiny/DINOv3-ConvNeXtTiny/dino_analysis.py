"""Paired validation analysis and compact report figures for D0 / D0-C1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score


LABEL_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
SHORT_NAMES = ["Enlarged CM", "Pneumothorax", "Consolidation", "Pneumonia",
               "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity",
               "Pleural Effusion", "No Finding"]


def _columns(prefix):
    return ([f"target_{name.replace(' ', '_')}" for name in LABEL_NAMES],
            [f"prob_{name.replace(' ', '_')}" for name in LABEL_NAMES])


def _auc_vector(frame, suffix=""):
    target_columns, probability_columns = _columns(None)
    aucs = []
    for target_column, probability_column in zip(target_columns, probability_columns):
        y = frame[f"{target_column}{suffix}"].to_numpy(dtype=np.int64)
        p = frame[f"{probability_column}{suffix}"].to_numpy(dtype=np.float64)
        aucs.append(float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"))
    return aucs


def _metric_text(value, signed=False):
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def _bootstrap_subject_differences(merged, replicates: int, seed: int):
    target_columns, probability_columns = _columns(None)
    subjects = merged["Subject_id"].astype(str).to_numpy()
    subject_index = {}
    for index, subject in enumerate(subjects):
        subject_index.setdefault(subject, []).append(index)
    unique_subjects = np.asarray(sorted(subject_index), dtype=object)
    rng = np.random.default_rng(int(seed))
    macro_differences = []
    per_class_differences = [[] for _ in LABEL_NAMES]
    for _ in range(int(replicates)):
        draw = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([subject_index[subject] for subject in draw])
        class_differences = []
        for class_index, (target_column, probability_column) in enumerate(
                zip(target_columns, probability_columns)):
            y = merged[f"{target_column}_d0"].to_numpy(dtype=np.int64)[indices]
            if np.unique(y).size != 2:
                continue
            p0 = merged[f"{probability_column}_d0"].to_numpy(dtype=np.float64)[indices]
            p1 = merged[f"{probability_column}_d0c1"].to_numpy(dtype=np.float64)[indices]
            difference = float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
            per_class_differences[class_index].append(difference)
            class_differences.append(difference)
        if class_differences:
            macro_differences.append(float(np.mean(class_differences)))
    if len(macro_differences) != int(replicates):
        raise RuntimeError(f"Only {len(macro_differences)} valid macro bootstrap replicates")
    return {
        "seed": int(seed),
        "requested_replicates": int(replicates),
        "valid_macro_replicates": len(macro_differences),
        "macro_auroc_delta_ci95": np.quantile(macro_differences, [0.025, 0.975]).tolist(),
        "macro_auroc_delta_replicates": macro_differences,
        "per_class_delta_ci95": {
            name: (np.quantile(values, [0.025, 0.975]).tolist()
                   if values else [None, None])
            for name, values in zip(LABEL_NAMES, per_class_differences)
        },
        "per_class_valid_replicates": {
            name: len(values) for name, values in zip(LABEL_NAMES, per_class_differences)
        },
    }


def _font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def _line_plot(path: Path, title: str, series: list[tuple[str, list[float]]],
               y_min: float | None = None, y_max: float | None = None,
               y_label: str = "Value"):
    width, height = 1100, 660
    left, right, top, bottom = 90, 1040, 70, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    finite = [float(v) for _, values in series for v in values if np.isfinite(v)]
    if not finite:
        raise ValueError(f"Cannot plot empty series for {title}")
    low = min(finite) if y_min is None else y_min
    high = max(finite) if y_max is None else y_max
    if high <= low:
        high = low + 1.0
    pad = (high - low) * 0.05
    low, high = low - pad, high + pad
    draw.text((left, 22), title, fill="black", font=font)
    draw.line((left, top, left, bottom, right, bottom), fill="#333333", width=2)
    for tick in range(6):
        y = bottom - tick * (bottom - top) / 5
        value = low + tick * (high - low) / 5
        draw.line((left, int(y), right, int(y)), fill="#e6e6e6", width=1)
        draw.text((12, int(y) - 8), f"{value:.3f}", fill="#333333", font=font)
    maximum_epochs = max(len(values) for _, values in series)
    colors = ["#1769aa", "#6baed6", "#d1495b", "#f28e2b"]
    for index, (name, values) in enumerate(series):
        points = []
        for epoch, value in enumerate(values, start=1):
            if not np.isfinite(value):
                continue
            x = left + (epoch - 1) * (right - left) / max(1, maximum_epochs - 1)
            y = bottom - (float(value) - low) * (bottom - top) / (high - low)
            points.append((int(x), int(y)))
        if len(points) > 1:
            draw.line(points, fill=colors[index % len(colors)], width=4)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4),
                         fill=colors[index % len(colors)])
        legend_x = left + index * 230
        draw.line((legend_x, 615, legend_x + 35, 615), fill=colors[index % len(colors)], width=4)
        draw.text((legend_x + 45, 605), name, fill="black", font=font)
    draw.text((left, 570), "Epoch", fill="black", font=font)
    draw.text((15, 42), y_label, fill="black", font=font)
    image.save(path)


def _bar_plot(path: Path, title: str, categories: list[str], series: list[tuple[str, list[float]]],
              y_min: float = 0.0, y_max: float = 1.0):
    width, height = 1200, 720
    left, right, top, bottom = 80, 1160, 65, 590
    image = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(image), _font()
    draw.text((left, 20), title, fill="black", font=font)
    draw.line((left, top, left, bottom, right, bottom), fill="#333333", width=2)
    colors = ["#1769aa", "#d1495b"]
    span = right - left
    category_width = span / len(categories)
    bar_width = min(28, category_width / (len(series) + 1))
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        y = bottom - (value - y_min) * (bottom - top) / (y_max - y_min)
        draw.line((left, int(y), right, int(y)), fill="#e7e7e7", width=1)
        draw.text((8, int(y) - 8), f"{value:.2f}", fill="#333333", font=font)
    for category_index, category in enumerate(categories):
        center = left + (category_index + 0.5) * category_width
        for series_index, (name, values) in enumerate(series):
            value = float(values[category_index])
            if not np.isfinite(value):
                continue
            value = min(y_max, max(y_min, value))
            x = center + (series_index - (len(series) - 1) / 2) * (bar_width + 3)
            y = bottom - (value - y_min) * (bottom - top) / (y_max - y_min)
            draw.rectangle((int(x), int(y), int(x + bar_width), bottom),
                           fill=colors[series_index % len(colors)])
        draw.text((int(center - category_width * 0.43), bottom + 8), category,
                  fill="black", font=font)
    for i, (name, _) in enumerate(series):
        x = left + i * 250
        draw.rectangle((x, 665, x + 24, 685), fill=colors[i % len(colors)])
        draw.text((x + 32, 662), name, fill="black", font=font)
    image.save(path)


def _horizontal_bars(path: Path, title: str, values: dict[str, float]):
    width, height = 1000, 360
    left, right, top = 250, 940, 80
    image = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(image), _font()
    draw.text((left, 20), title, fill="black", font=font)
    maximum = max(values.values()) * 1.15 if values else 1.0
    colors = ["#1769aa", "#d1495b"]
    for index, (name, value) in enumerate(values.items()):
        y = top + index * 105
        bar_width = int((right - left) * value / max(maximum, 1e-9))
        draw.text((20, y + 12), name, fill="black", font=font)
        draw.rectangle((left, y, left + bar_width, y + 45), fill=colors[index % len(colors)])
        draw.text((left + bar_width + 12, y + 12), f"{value:.2f} images/s",
                  fill="black", font=font)
    image.save(path)


def compare_runs(config):
    root = Path(config["outputs"]["root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    d0_dir, c1_dir = root / "d0", root / "d0c1"
    for variant_dir in (d0_dir, c1_dir):
        status_path = variant_dir / "training_status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing completed training status: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not status.get("complete") or int(status.get("completed_epochs", 0)) != 20:
            raise RuntimeError(f"{variant_dir.name} has not completed all 20 epochs")
    d0 = pd.read_csv(d0_dir / "predictions" / "val_best.csv", dtype={
        "Subject_id": "string", "Study_id": "string"})
    c1 = pd.read_csv(c1_dir / "predictions" / "val_best.csv", dtype={
        "Subject_id": "string", "Study_id": "string"})
    keys = ["Subject_id", "Study_id"]
    if d0.duplicated(keys).any() or c1.duplicated(keys).any():
        raise RuntimeError("Best validation predictions contain duplicate Study keys")
    if set(map(tuple, d0[keys].astype(str).to_numpy())) != set(map(tuple, c1[keys].astype(str).to_numpy())):
        raise RuntimeError("D0 and D0-C1 validation Study sets differ")
    target_columns, probability_columns = _columns(None)
    merged = d0.merge(c1, on=keys, suffixes=("_d0", "_d0c1"),
                      validate="one_to_one", sort=True)
    if len(merged) != 859:
        raise RuntimeError(f"Expected the shared official validation set to contain 859 Studies, got {len(merged)}")
    for column in target_columns:
        if not np.array_equal(merged[f"{column}_d0"], merged[f"{column}_d0c1"]):
            raise RuntimeError(f"D0 and D0-C1 validation target mismatch: {column}")
    auc_d0, auc_c1 = _auc_vector(merged, "_d0"), _auc_vector(merged, "_d0c1")
    macro_d0 = float(np.nanmean(auc_d0))
    macro_c1 = float(np.nanmean(auc_c1))
    delta = macro_c1 - macro_d0
    d0_history = pd.read_csv(d0_dir / "train_log.csv")
    c1_history = pd.read_csv(c1_dir / "train_log.csv")
    d0_status = json.loads((d0_dir / "training_status.json").read_text(encoding="utf-8"))
    c1_status = json.loads((c1_dir / "training_status.json").read_text(encoding="utf-8"))
    bootstrap = _bootstrap_subject_differences(
        merged, config["comparison"]["subject_bootstrap_replicates"],
        config["comparison"]["subject_bootstrap_seed"])
    bootstrap["macro_auroc_delta_point_estimate"] = delta
    bootstrap["per_class_delta_point_estimate"] = {
        name: (float(c1_auc - d0_auc) if np.isfinite(c1_auc) and np.isfinite(d0_auc) else None)
        for name, d0_auc, c1_auc in zip(LABEL_NAMES, auc_d0, auc_c1)
    }
    bootstrap["validation_studies"] = len(merged)
    bootstrap["validation_subjects"] = int(merged["Subject_id"].nunique())
    bootstrap_path = root / "paired_subject_bootstrap.json"
    bootstrap_path.write_text(json.dumps(bootstrap, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    delta_ci = bootstrap["macro_auroc_delta_ci95"]
    if delta_ci[0] > 0:
        recommended_variant = "d0c1"
        recommendation_basis = "paired Subject-bootstrap 95% CI for D0-C1 minus D0 is above zero"
    elif delta_ci[1] < 0:
        recommended_variant = "d0"
        recommendation_basis = "paired Subject-bootstrap 95% CI for D0-C1 minus D0 is below zero"
    else:
        recommended_variant = "d0c1" if macro_c1 > macro_d0 else "d0"
        recommendation_basis = "paired difference is inconclusive; selected the higher observed validation Macro-AUROC point estimate"
    class_delta = bootstrap["per_class_delta_point_estimate"]
    summary = {
        "validation_studies": len(merged),
        "validation_subjects": int(merged["Subject_id"].nunique()),
        "d0_best_epoch": int(d0_status["best_epoch"]),
        "d0_best_macro_auroc": macro_d0,
        "d0c1_best_epoch": int(c1_status["best_epoch"]),
        "d0c1_best_macro_auroc": macro_c1,
        "d0c1_minus_d0_macro_auroc": delta,
        "macro_auroc_delta_ci95": bootstrap["macro_auroc_delta_ci95"],
        "d0_pneumonia_auroc": float(auc_d0[3]) if np.isfinite(auc_d0[3]) else None,
        "d0c1_pneumonia_auroc": float(auc_c1[3]) if np.isfinite(auc_c1[3]) else None,
        "d0_pneumonia_delta": class_delta[LABEL_NAMES[3]],
        "d0_lung_opacity_auroc": float(auc_d0[7]) if np.isfinite(auc_d0[7]) else None,
        "d0c1_lung_opacity_auroc": float(auc_c1[7]) if np.isfinite(auc_c1[7]) else None,
        "d0_lung_opacity_delta": class_delta[LABEL_NAMES[7]],
        "d0_exceeds_080": macro_d0 > 0.80,
        "d0c1_exceeds_080": macro_c1 > 0.80,
        "v1_historical_macro_auroc": config["comparison"]["historical_v1_macro_auroc"],
        "d0_delta_vs_v1": macro_d0 - config["comparison"]["historical_v1_macro_auroc"],
        "d0c1_delta_vs_v1": macro_c1 - config["comparison"]["historical_v1_macro_auroc"],
        "recommended_next_base": recommended_variant,
        "recommendation_basis": recommendation_basis,
    }
    _plot_comparison(root, d0_history, c1_history, auc_d0, auc_c1,
                     summary, config)
    per_class = []
    for index, class_name in enumerate(LABEL_NAMES):
        per_class.append({
            "class_id": index + 1,
            "class_name": class_name,
            "d0_auroc": float(auc_d0[index]) if np.isfinite(auc_d0[index]) else None,
            "d0c1_auroc": float(auc_c1[index]) if np.isfinite(auc_c1[index]) else None,
            "d0c1_minus_d0": class_delta[class_name],
            "delta_ci95_low": bootstrap["per_class_delta_ci95"][class_name][0],
            "delta_ci95_high": bootstrap["per_class_delta_ci95"][class_name][1],
            "valid_bootstrap_replicates": bootstrap["per_class_valid_replicates"][class_name],
        })
    pd.DataFrame(per_class).to_csv(root / "per_class_auroc_comparison.csv",
                                   index=False, encoding="utf-8")
    (root / "d0_d0c1_comparison.md").write_text(
        _comparison_markdown(summary, per_class, d0_history, c1_history, config),
        encoding="utf-8")
    (root / "d0_d0c1_comparison.json").write_text(
        json.dumps({"summary": summary, "per_class": per_class,
                    "paired_subject_bootstrap": {k: v for k, v in bootstrap.items()
                                                  if k != "macro_auroc_delta_replicates"}},
                   indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return summary


def _plot_comparison(root, d0_history, c1_history, auc_d0, auc_c1,
                     summary, config):
    d0_history = d0_history.sort_values("epoch")
    c1_history = c1_history.sort_values("epoch")
    _line_plot(root / "macro_auroc_curve.png", "Validation Study Macro-AUROC",
               [("D0", d0_history["macro_auroc"].tolist()),
                ("D0-C1", c1_history["macro_auroc"].tolist())],
               y_min=0.5, y_max=1.0, y_label="Macro-AUROC")
    _line_plot(root / "train_validation_loss_curve.png", "Train / Validation Loss",
               [("D0 Train BCE", d0_history["train_bce"].tolist()),
                ("D0 Valid BCE", d0_history["valid_loss"].tolist()),
                ("D0-C1 Train Total", c1_history["train_total"].tolist()),
                ("D0-C1 Valid BCE", c1_history["valid_loss"].tolist())],
               y_label="Loss")
    _bar_plot(root / "per_class_auroc_comparison.png", "Best-checkpoint per-class AUROC",
              SHORT_NAMES, [("D0", auc_d0), ("D0-C1", auc_c1)], 0.0, 1.0)
    runtime_path = root / "dino_selected_runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    _horizontal_bars(root / "images_per_second_comparison.png",
                     "Repeated benchmark mean end-to-end throughput",
                     {"D0": float(runtime["measured_d0_images_per_s"]),
                      "D0-C1": float(runtime["measured_d0c1_images_per_s"])})


def _comparison_markdown(summary, per_class, d0_history, c1_history, config):
    lines = [
        "# DINOv3 D0 与 D0-C1 对照分析", "",
        f"- 配对验证集：{summary['validation_studies']} Studies / {summary['validation_subjects']} Subjects；D0 与 D0-C1 Study keys 和标签完全一致。",
        f"- D0 最佳轮次：Epoch {summary['d0_best_epoch']}，Macro-AUROC {summary['d0_best_macro_auroc']:.6f}。",
        f"- D0-C1 最佳轮次：Epoch {summary['d0c1_best_epoch']}，Macro-AUROC {summary['d0c1_best_macro_auroc']:.6f}。",
        f"- Macro-AUROC 差值（D0-C1 − D0）：{summary['d0c1_minus_d0_macro_auroc']:+.6f}，按 Subject 配对 Bootstrap 95% CI `{summary['macro_auroc_delta_ci95']}`。",
        f"- D0 超过 0.80：{summary['d0_exceeds_080']}；D0-C1 超过 0.80：{summary['d0c1_exceeds_080']}。",
        f"- V1 历史参照 Macro-AUROC：{summary['v1_historical_macro_auroc']:.6f}；D0 差值 {summary['d0_delta_vs_v1']:+.6f}；D0-C1 差值 {summary['d0c1_delta_vs_v1']:+.6f}。",
        f"- Pneumonia AUROC：D0 {_metric_text(summary['d0_pneumonia_auroc'])}；D0-C1 {_metric_text(summary['d0c1_pneumonia_auroc'])}；差值 {_metric_text(summary['d0_pneumonia_delta'], signed=True)}。",
        f"- Lung Opacity AUROC：D0 {_metric_text(summary['d0_lung_opacity_auroc'])}；D0-C1 {_metric_text(summary['d0c1_lung_opacity_auroc'])}；差值 {_metric_text(summary['d0_lung_opacity_delta'], signed=True)}。",
        f"- 下一轮基础模型建议：`{summary['recommended_next_base'].upper()}`；依据：{summary['recommendation_basis']}。",
        "- V1 是历史参照，不是与本实验严格匹配的预训练消融。",
        "- D0-C1 只在训练期间使用 Projection MLP；推理结构与 D0 相同。",
        "- Test split 未读取。",
        "", "## 逐类 AUROC", "",
        "| Class | D0 | D0-C1 | 差值 | Subject Bootstrap 95% CI |", "|---|---:|---:|---:|---:|",
    ]
    for row in per_class:
        lines.append(f"| {row['class_name']} | {_metric_text(row['d0_auroc'])} | {_metric_text(row['d0c1_auroc'])} | {_metric_text(row['d0c1_minus_d0'], signed=True)} | [{row['delta_ci95_low']}, {row['delta_ci95_high']}] |")
    lines.extend(["", "## 后期验证表现", ""])
    for name, history in (("D0", d0_history), ("D0-C1", c1_history)):
        early = history[history["epoch"] <= 6]
        late = history[history["epoch"] >= 7]
        early_peak = float(early["macro_auroc"].max())
        late_peak = float(late["macro_auroc"].max())
        final = float(history.sort_values("epoch").iloc[-1]["macro_auroc"])
        lines.append(f"- {name}: Epoch 1–6 peak {early_peak:.6f}; Epoch 7–20 peak {late_peak:.6f}; final {final:.6f}; final-minus-early-peak {final - early_peak:+.6f}.")
    lines.extend(["", "## 统计口径", "",
                  f"- {config['comparison']['subject_bootstrap_replicates']} 次 Subject-cluster paired bootstrap，seed {config['comparison']['subject_bootstrap_seed']}；同一抽样 Subject 同时用于 D0 和 D0-C1。",
                  "- 每轮 validation Macro-AUROC / AUPRC 对可计算的 10 类做算术平均；F1 阈值固定 0.5。",
                  "- Benchmark 速度来自共享 runtime 的长测重复均值，详见 `dino_gpu_benchmark.md`。", ""])
    return "\n".join(lines)
