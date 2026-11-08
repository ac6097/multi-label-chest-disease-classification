import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

from engine.trainer import _draw_curve
from hybrid.common import PROJECT_DIR, read_yaml, write_csv, write_json


MODELS = ("h0", "h1", "h2")
COLORS = {"h0": (39, 110, 220), "h1": (44, 160, 44), "h2": (220, 75, 55)}
LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]


def _read_run(run_dir):
    run_dir = Path(run_dir)
    log = pd.read_csv(run_dir / "train_log.csv")
    per_class = pd.read_csv(run_dir / "per_class_metrics.csv")
    predictions = pd.read_csv(run_dir / "predictions/val_best.csv", dtype={
        "Subject_id": "string", "Study_id": "string",
    })
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return {"log": log, "per_class": per_class, "predictions": predictions, "config": config}


def _probability_matrix(frame):
    return frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=np.float64)


def _target_matrix(frame):
    return frame[[f"target_{label}" for label in LABELS]].to_numpy(dtype=np.int64)


def _macro_auc(targets, probabilities):
    aucs = []
    for index in range(targets.shape[1]):
        if np.unique(targets[:, index]).size < 2:
            return np.nan
        aucs.append(roc_auc_score(targets[:, index], probabilities[:, index]))
    return float(np.mean(aucs))


def paired_subject_bootstrap(frames, pairs=("h1-h0", "h2-h0", "h2-h1"),
                             replicates=2000, seed=42):
    base = frames["h0"]
    keys = list(zip(base["Subject_id"].astype(str), base["Study_id"].astype(str)))
    for name in MODELS[1:]:
        frame = frames[name]
        other = list(zip(frame["Subject_id"].astype(str), frame["Study_id"].astype(str)))
        if other != keys:
            raise ValueError(f"Validation prediction Studies are not identically ordered for {name}")
    subjects = np.asarray(sorted(base["Subject_id"].astype(str).unique()))
    subject_rows = {subject: np.flatnonzero(base["Subject_id"].astype(str).to_numpy() == subject)
                    for subject in subjects}
    targets = _target_matrix(base)
    probabilities = {name: _probability_matrix(frames[name]) for name in MODELS}
    rng = np.random.default_rng(seed)
    deltas = {pair: [] for pair in pairs}
    rejected = 0
    for _ in range(int(replicates)):
        selected_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        rows = np.concatenate([subject_rows[subject] for subject in selected_subjects])
        sampled_targets = targets[rows]
        if any(np.unique(sampled_targets[:, column]).size < 2 for column in range(sampled_targets.shape[1])):
            rejected += 1
            continue
        aucs = {name: _macro_auc(sampled_targets, probabilities[name][rows]) for name in MODELS}
        for pair in pairs:
            left, right = pair.split("-")
            deltas[pair].append(aucs[left] - aucs[right])
    summaries = {}
    for pair in pairs:
        values = np.asarray(deltas[pair], dtype=np.float64)
        if values.size == 0:
            raise RuntimeError(f"No valid paired bootstrap replicates for {pair}")
        summaries[pair] = {
            "delta_macro_auroc": _macro_auc(targets, probabilities[pair.split("-")[0]]) -
                                 _macro_auc(targets, probabilities[pair.split("-")[1]]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "valid_replicates": int(values.size),
            "requested_replicates": int(replicates),
        }
    return {"method": "paired bootstrap resampling of Subjects with replacement",
            "seed": int(seed), "rejected_replicates": rejected,
            "subject_count": int(len(subjects)), "study_count": int(len(base)),
            "comparisons": summaries}


def _font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_auc_bars(path, best_class_metrics):
    width, height = 1660, 900
    left, right = 390, 1585
    top, bottom = 100, 835
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font, small = _font(20), _font(16)
    draw.text((left, 28), "Best-epoch per-class AUROC", fill="black", font=_font(28))
    for tick in range(6):
        value = 0.5 + tick * 0.1
        x = left + int((value - 0.5) / 0.5 * (right - left))
        draw.line((x, top, x, bottom), fill=(220, 220, 220), width=1)
        draw.text((x - 20, bottom + 12), f"{value:.1f}", fill="black", font=small)
    legend_x = left + 20
    for model in MODELS:
        draw.rectangle((legend_x, 68, legend_x + 20, 84), fill=COLORS[model])
        draw.text((legend_x + 27, 65), model.upper(), fill="black", font=small)
        legend_x += 130
    group_height = (bottom - top) / len(LABELS)
    for index, label in enumerate(LABELS):
        group_top = top + index * group_height
        draw.text((8, int(group_top + 10)), label, fill="black", font=small)
        for offset, model in enumerate(MODELS):
            auc = best_class_metrics[model][label]
            y = int(group_top + 7 + offset * 18)
            x0 = left + int((0.5 - 0.5) / 0.5 * (right - left))
            x1 = left + int((auc - 0.5) / 0.5 * (right - left)) if math.isfinite(auc) else x0
            draw.rectangle((x0, y, max(x1, x0 + 1), y + 13), fill=COLORS[model])
            draw.text((min(max(x1 + 6, left + 4), right - 48), y - 2),
                      f"{auc:.3f}" if math.isfinite(auc) else "NA", fill="black", font=small)
    image.save(path)


def _save_class_reports(data, reports_dir):
    for model in MODELS:
        run_dir = Path(data[model]["config"]["run_dir"])
        report = run_dir / "hybrid_report.md"
        if model in {"h1", "h2"}:
            pretrain_key = f"{model}_pretrain_dir"
            pretrain_report_path = Path(data[model]["config"][pretrain_key]) / "pretrain_report.md"
            content = (f"# Hybrid {model.upper()} report\n\n## Self-supervised stage\n\n"
                       + pretrain_report_path.read_text(encoding="utf-8")
                       + "\n\n## Classification fine-tuning\n\n"
                       + report.read_text(encoding="utf-8"))
        else:
            content = report.read_text(encoding="utf-8")
        (reports_dir / f"hybrid_{model}_report.md").write_text(content, encoding="utf-8")


def compare_results(experiments_config_path="configs/hybrid_experiments.yaml",
                    bootstrap_replicates=2000):
    experiments = read_yaml(experiments_config_path)
    data = {}
    for model in MODELS:
        run_dir = Path(experiments["outputs"][model])
        if not run_dir.is_absolute():
            run_dir = PROJECT_DIR / run_dir
        run = _read_run(run_dir)
        run["config"]["run_dir"] = str(run_dir)
        if len(run["log"]) != 20 or int(run["log"].iloc[-1]["epoch"]) != 20:
            raise RuntimeError(f"{model.upper()} classifier has not completed 20 epochs")
        if model in {"h1", "h2"}:
            pretrain_dir = Path(experiments["outputs"][f"{model}_pretrain"])
            if not pretrain_dir.is_absolute():
                pretrain_dir = PROJECT_DIR / pretrain_dir
            pretrain_log = pd.read_csv(pretrain_dir / "pretrain_log.csv")
            if len(pretrain_log) != 20 or int(pretrain_log.iloc[-1]["epoch"]) != 20:
                raise RuntimeError(f"{model.upper()} self-supervised pretraining has not completed 20 epochs")
            run["config"][f"{model}_pretrain_dir"] = str(pretrain_dir)
        data[model] = run

    bootstrap = paired_subject_bootstrap(
        {model: data[model]["predictions"] for model in MODELS},
        replicates=bootstrap_replicates, seed=42)
    reports_dir = PROJECT_DIR / "outputs/hybrid/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _save_class_reports(data, reports_dir)
    write_json(reports_dir / "paired_subject_bootstrap.json", bootstrap)

    best_rows, class_metrics, summary_rows = {}, {}, []
    for model in MODELS:
        run = data[model]
        log = run["log"]
        best_index = int(log["macro_auroc"].astype(float).idxmax())
        best_row = log.iloc[best_index]
        best_epoch = int(best_row["epoch"])
        best_rows[model] = best_row
        best_class_rows = run["per_class"][run["per_class"]["epoch"].astype(int) == best_epoch]
        class_metrics[model] = {
            str(row["class_name"]): float(row["auroc"])
            for _, row in best_class_rows.iterrows()
        }
        config = run["config"]
        parameter_count = int(config["parameter_counts"]["total"])
        self_supervised = {"h0": "None", "h1": "Pixel reconstruction", "h2": "Latent feature reconstruction"}[model]
        summary_rows.append({
            "model": model.upper(),
            "self_supervised_method": self_supervised,
            "best_epoch": best_epoch,
            "macro_auroc": float(best_row["macro_auroc"]),
            "pneumonia_auroc": class_metrics[model]["Pneumonia"],
            "lung_opacity_auroc": class_metrics[model]["Lung Opacity"],
            "parameters": parameter_count,
            "images_per_s_at_best_epoch": float(best_row["images_per_s"]),
        })

    _draw_curve(reports_dir / "hybrid_macro_auroc.png", [
        (model.upper(), data[model]["log"]["epoch"].astype(int).tolist(),
         data[model]["log"]["macro_auroc"].astype(float).tolist(), COLORS[model])
        for model in MODELS
    ], "Validation Macro-AUROC", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(reports_dir / "hybrid_validation_loss.png", [
        (model.upper(), data[model]["log"]["epoch"].astype(int).tolist(),
         data[model]["log"]["val_loss"].astype(float).tolist(), COLORS[model])
        for model in MODELS
    ], "Validation Study BCE", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(reports_dir / "hybrid_train_val_gap.png", [
        (model.upper(), data[model]["log"]["epoch"].astype(int).tolist(),
         data[model]["log"]["train_val_gap"].astype(float).tolist(), COLORS[model])
        for model in MODELS
    ], "Validation Loss − Train Loss", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(reports_dir / "hybrid_images_per_second.png", [
        (model.upper(), data[model]["log"]["epoch"].astype(int).tolist(),
         data[model]["log"]["images_per_s"].astype(float).tolist(), COLORS[model])
        for model in MODELS
    ], "Train images/s", markers=(2, 5, 7, 10, 15, 20))
    _draw_curve(reports_dir / "hybrid_peak_memory.png", [
        series
        for model in MODELS
        for series in (
            (f"{model.upper()} allocated", data[model]["log"]["epoch"].astype(int).tolist(),
             data[model]["log"]["peak_gpu_allocated_gib"].astype(float).tolist(), COLORS[model]),
            (f"{model.upper()} reserved", data[model]["log"]["epoch"].astype(int).tolist(),
             data[model]["log"]["peak_gpu_reserved_gib"].astype(float).tolist(),
             tuple(max(0, component - 50) for component in COLORS[model])),
        )
    ], "Peak GPU memory (GiB)", markers=(2, 5, 7, 10, 15, 20))
    _draw_auc_bars(reports_dir / "hybrid_best_per_class_auroc.png", class_metrics)
    write_csv(reports_dir / "hybrid_summary.csv", summary_rows)

    rows = []
    for model, run in data.items():
        per_class = run["per_class"]
        for _, row in per_class.iterrows():
            if int(row["epoch"]) == int(best_rows[model]["epoch"]):
                rows.append({"model": model.upper(), "epoch": int(row["epoch"]), **row.to_dict()})
    write_csv(reports_dir / "hybrid_best_per_class_metrics.csv", rows)

    h0_baseline = 0.791286
    h0_baseline_2 = 0.793451
    selected = max(summary_rows, key=lambda row: row["macro_auroc"])
    lines = [
        "# Hybrid H0/H1/H2 comparison", "",
        "## Best-checkpoint summary", "",
        "| Model | Additional self-supervision | Best Epoch | Macro-AUROC | Pneumonia AUROC | Lung Opacity AUROC | Parameters | Train images/s at best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['model']} | {row['self_supervised_method']} | {row['best_epoch']} | "
                     f"{row['macro_auroc']:.6f} | {row['pneumonia_auroc']:.6f} | "
                     f"{row['lung_opacity_auroc']:.6f} | {row['parameters']:,} | "
                     f"{row['images_per_s_at_best_epoch']:.2f} |")
    lines.extend([
        "", "## Epoch 7 to Epoch 20 generalization trend", "",
        "| Model | AUROC Epoch 7 | AUROC Epoch 20 | AUROC change 7→20 | Train-Val gap Epoch 7 | Train-Val gap Epoch 20 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for model in MODELS:
        log = data[model]["log"].set_index("epoch")
        e7, e20 = log.loc[7], log.loc[20]
        lines.append(
            f"| {model.upper()} | {float(e7['macro_auroc']):.6f} | {float(e20['macro_auroc']):.6f} | "
            f"{float(e20['macro_auroc']) - float(e7['macro_auroc']):+.6f} | "
            f"{float(e7['train_val_gap']):+.6f} | {float(e20['train_val_gap']):+.6f} |"
        )
    lines.extend([
        "", f"Highest observed validation Macro-AUROC: **{selected['model']}** "
        f"at epoch {selected['best_epoch']} ({selected['macro_auroc']:.6f}).",
        "All three classifiers completed the configured 20 epochs; selection used Validation Study Macro-AUROC.",
        "",
        "## Paired Subject bootstrap", "",
        f"Resampled {bootstrap['subject_count']} Subjects with replacement over {bootstrap['study_count']} Validation Studies; "
        f"seed={bootstrap['seed']}, requested replicates={bootstrap['requested_replicates'] if 'requested_replicates' in bootstrap else bootstrap['comparisons']['h1-h0']['requested_replicates']}. "
        f"Invalid replicates rejected for a missing class polarity: {bootstrap['rejected_replicates']}.",
        "| Difference | Δ Macro-AUROC | Paired 95% CI | Valid replicates |",
        "|---|---:|---:|---:|",
    ])
    for pair, stats in bootstrap["comparisons"].items():
        lines.append(f"| {pair.upper()} | {stats['delta_macro_auroc']:+.6f} | "
                     f"[{stats['ci95_low']:+.6f}, {stats['ci95_high']:+.6f}] | "
                     f"{stats['valid_replicates']} |")
    lines.extend([
        "", "## Interpretation", "",
        f"- H1−H0 isolates the added official-train pixel reconstruction stage under a shared downstream classifier architecture; observed Δ AUROC is {bootstrap['comparisons']['h1-h0']['delta_macro_auroc']:+.6f}.",
        f"- H2−H0 compares the latent Stage3/Block6 reconstruction stage with no additional pretraining; observed Δ AUROC is {bootstrap['comparisons']['h2-h0']['delta_macro_auroc']:+.6f}.",
        f"- H2−H1 compares the two self-supervised targets; observed Δ AUROC is {bootstrap['comparisons']['h2-h1']['delta_macro_auroc']:+.6f}.",
        f"- Pneumonia AUROC: H0={class_metrics['h0']['Pneumonia']:.6f}, H1={class_metrics['h1']['Pneumonia']:.6f}, H2={class_metrics['h2']['Pneumonia']:.6f}.",
        f"- Lung Opacity AUROC: H0={class_metrics['h0']['Lung Opacity']:.6f}, H1={class_metrics['h1']['Lung Opacity']:.6f}, H2={class_metrics['h2']['Lung Opacity']:.6f}.",
        "- Compare epoch 5–7 peaks and subsequent declines in the curves. The absolute train/validation losses use the same BCE but train augmentation differs from validation.",
        "- Historical ConvNeXt-Tiny scores 0.791286 and 0.793451 are context only; those runs used different architecture/training conditions and are not matched ablation baselines.",
        "- H0/H1/H2 share the same ImageNet initial checkpoint, classifier architecture, downstream split, loss and selected supervised budget. H1/H2 each used the same number of official Train images and epochs; self-supervised optimizer updates can differ because their measured microbatch configurations differ.",
        "- No Test CSV, Test image, or submission metric was read or generated.",
        "",
        "## Artifacts", "",
        "- `hybrid_gpu_benchmark.md` and `.csv`: real-data training throughput selection.",
        "- `hybrid_h0_report.md`, `hybrid_h1_report.md`, `hybrid_h2_report.md`: per-run training and pretraining details.",
        "- `hybrid_macro_auroc.png`, `hybrid_validation_loss.png`, `hybrid_train_val_gap.png`, `hybrid_best_per_class_auroc.png`, `hybrid_images_per_second.png`, `hybrid_peak_memory.png`.",
        "- `paired_subject_bootstrap.json`: paired confidence intervals.",
    ])
    (reports_dir / "hybrid_h0_h1_h2_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison report saved: {reports_dir / 'hybrid_h0_h1_h2_comparison.md'}")
    return {"summary": summary_rows, "bootstrap": bootstrap,
            "report": str(reports_dir / "hybrid_h0_h1_h2_comparison.md")}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-config", default="configs/hybrid_experiments.yaml")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = compare_results(args.experiments_config, args.bootstrap_replicates)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
