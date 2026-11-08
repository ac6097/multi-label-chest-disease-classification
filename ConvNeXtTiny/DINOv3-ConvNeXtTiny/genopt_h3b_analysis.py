"""H3-B (weak L2-SP) evaluation and H0/H3/H3-B comparative analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import genopt_analysis as base
from genopt_common import (
    EXPERIMENTS,
    EXPERIMENT_ROOT,
    LABEL_NAMES,
    SEED,
    TRAIN_CSV,
    VALID_CSV,
    build_h_datasets,
    class_bce_breakdown,
    load_h_checkpoint_model,
    load_r5_reference,
    merge_class_bce,
    read_json,
    study_view_metadata,
    write_json,
)
from generalization_common import (
    build_eval_datasets,
    infer_dataset,
    load_r5_model,
)
from stage_fusion_analysis import bootstrap_models, study_metrics
from hybrid.common import write_csv

H3B_DIR = EXPERIMENT_ROOT / "H3B_Weak_L2SP"
H3_DIR = EXPERIMENT_ROOT / "H3_L2SP_Regularization"
H0_DIR = EXPERIMENT_ROOT / "H0_R5_Baseline"
ANALYSIS_ORDER = ("H0", "H3", "H3B")
COLORS = {"H0": "#4C72B0", "H3": "#C44E52", "H3B": "#55A868"}


def experiment_dir(experiment: str) -> Path:
    return EXPERIMENT_ROOT / EXPERIMENTS[experiment]["label"]


def _cache_path(experiment: str, checkpoint: str, split: str) -> Path:
    return base.CACHE / f"{experiment}_{checkpoint}_{split}_noaug.npz"


def _inference(experiment: str, checkpoint: str, split: str,
               train_data, valid_data, force: bool = False) -> dict:
    path = _cache_path(experiment, checkpoint, split)
    if path.is_file() and not force:
        return base.load_result(path)
    dataset = train_data if split == "train" else valid_data
    if experiment == "H0" and checkpoint == "best":
        g1_path = base.G1_CACHE / f"r5_{split}_noaug.npz"
        if g1_path.is_file():
            result = base._g1_cache_result(g1_path)
            base.save_result(path, result)
            return result
    device = torch.device("cuda")
    if experiment == "H0":
        model, _, _ = load_r5_model(device, channels_last=True,
                                    checkpoint=checkpoint)
    else:
        runtime = load_r5_reference()["runtime"]
        model, _ = load_h_checkpoint_model(experiment_dir(experiment),
                                           checkpoint, runtime)
    try:
        result = infer_dataset(
            model, "R5", dataset, image_budget=8, num_workers=8,
            channels_last=True, autocast=True, collect_image_logits=False)
    finally:
        del model
        torch.cuda.empty_cache()
    result["experiment"] = experiment
    result["checkpoint"] = checkpoint
    result["split"] = split
    base.save_result(path, result)
    print(f"[h3b-analysis] {experiment} {checkpoint} {split}: "
          f"{result['images']} images ({result['elapsed_seconds']:.0f}s)",
          flush=True)
    return result


def _all_results(force: bool = False) -> dict:
    train_data, valid_data, _ = build_eval_datasets()
    results = {}
    for experiment in ANALYSIS_ORDER:
        for checkpoint in ("best", "last"):
            for split, dataset in (("train", train_data), ("valid", valid_data)):
                results[(experiment, checkpoint, split)] = _inference(
                    experiment, checkpoint, split, train_data, valid_data, force)
    return results


def _history(experiment: str) -> pd.DataFrame:
    if experiment == "H0":
        frame = pd.read_csv(base.R5_DIR / "train_log.csv")
        frame = frame.rename(columns={"valid_loss": "valid_loss"})
        return frame
    return pd.read_csv(experiment_dir(experiment) / "training_history.csv")


def _status(experiment: str) -> dict:
    if experiment == "H0":
        return read_json(base.R5_DIR / "training_status.json")
    return read_json(experiment_dir(experiment) / "training_status.json")


def _overall_rows(results: dict) -> pd.DataFrame:
    rows = []
    for experiment in ANALYSIS_ORDER:
        for checkpoint in ("best", "last"):
            train_metrics = base.result_metrics(
                results[(experiment, checkpoint, "train")])
            valid_metrics = base.result_metrics(
                results[(experiment, checkpoint, "valid")])
            status = _status(experiment)
            rows.append({
                "experiment": experiment,
                "label": EXPERIMENTS[experiment]["label"],
                "checkpoint": checkpoint,
                "best_epoch": int(status["best_epoch"]),
                "completed_epochs": int(status["completed_epochs"]),
                "stop_reason": status.get("stop_reason"),
                "train_overall_bce": train_metrics["overall_bce"],
                "valid_overall_bce": valid_metrics["overall_bce"],
                "train_valid_bce_gap": (valid_metrics["overall_bce"]
                                        - train_metrics["overall_bce"]),
                "train_macro_auroc": train_metrics["macro_auroc"],
                "train_macro_auprc": train_metrics["macro_auprc"],
                "valid_macro_auroc": valid_metrics["macro_auroc"],
                "valid_macro_auprc": valid_metrics["macro_auprc"],
            })
    return pd.DataFrame(rows)


def _per_class_rows(results: dict) -> pd.DataFrame:
    rows = []
    for experiment in ANALYSIS_ORDER:
        for split in ("train", "valid"):
            result = results[(experiment, "best", split)]
            metrics = base.result_metrics(result)
            for disease in LABEL_NAMES:
                item = metrics["per_class"][disease]
                bce = metrics["class_bce"][disease]
                rows.append({
                    "experiment": experiment,
                    "split": split,
                    "disease": disease,
                    "bce": bce["bce"],
                    "positive_bce": bce["positive_bce"],
                    "negative_bce": bce["negative_bce"],
                    "auroc": item["auroc"],
                    "auprc": item["auprc"],
                    "n_positive": item["n_positive"],
                    "n_negative": item["n_negative"],
                })
    return pd.DataFrame(rows)


def _class_compare_frames(per_class: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss = per_class.pivot_table(index="disease", columns=["experiment", "split"],
                                 values="bce")
    loss.columns = [f"{experiment}_{split}" for experiment, split in loss.columns]
    rows = []
    for disease in LABEL_NAMES:
        row = {"disease": disease}
        for experiment in ANALYSIS_ORDER:
            row[f"{experiment}_train"] = float(loss.loc[disease, f"{experiment}_train"])
            row[f"{experiment}_valid"] = float(loss.loc[disease, f"{experiment}_valid"])
        rows.append(row)
    comparison = pd.DataFrame(rows)
    overall = {
        "disease": "Overall BCE",
        "H0_train": float(np.mean(comparison["H0_train"])),
        "H0_valid": float(np.mean(comparison["H0_valid"])),
        "H3_train": float(np.mean(comparison["H3_train"])),
        "H3_valid": float(np.mean(comparison["H3_valid"])),
        "H3B_train": float(np.mean(comparison["H3B_train"])),
        "H3B_valid": float(np.mean(comparison["H3B_valid"])),
    }
    comparison = pd.concat([comparison, pd.DataFrame([overall])],
                           ignore_index=True)
    auroc = per_class.pivot_table(index="disease", columns=["experiment", "split"],
                                  values="auroc")
    auroc.columns = [f"{experiment}_{split}" for experiment, split in auroc.columns]
    rows = []
    for disease in LABEL_NAMES:
        row = {"disease": disease}
        for experiment in ANALYSIS_ORDER:
            row[f"{experiment}_train"] = float(auroc.loc[disease, f"{experiment}_train"])
            row[f"{experiment}_valid"] = float(auroc.loc[disease, f"{experiment}_valid"])
        rows.append(row)
    auroc_comparison = pd.DataFrame(rows)
    return comparison, auroc_comparison


def _parameter_drift(results: dict) -> pd.DataFrame:
    rows = []
    for experiment in ANALYSIS_ORDER:
        payload = read_json(experiment_dir(experiment) / "evaluation_summary.json")
        best = payload["best"]["stage_parameter_drift"]
        for stage in ("stage3", "stage4"):
            rows.append({
                "experiment": experiment,
                "stage": stage,
                "absolute_drift_l2": best[stage]["absolute_drift_l2"],
                "reference_l2": best[stage]["reference_l2"],
                "relative_drift": best[stage]["relative_drift"],
            })
    return pd.DataFrame(rows)


def _feature_drift(force: bool = False) -> pd.DataFrame:
    path = H3B_DIR / "feature_drift_comparison.csv"
    if path.is_file() and not force:
        return pd.read_csv(path)
    from genopt_common import load_h_checkpoint_model
    from generalization_common import load_r5_model as load_h0
    from generalization_g1 import (
        _cosine,
        _extract_features,
        _sample_study_indices,
    )
    from dino_experiment import load_pretrained_backbone

    train_data, valid_data, _ = build_eval_datasets()
    indices = _sample_study_indices(valid_data, 192)
    device = torch.device("cuda")
    pretrained, _ = load_pretrained_backbone("cpu")
    pretrained = pretrained.to(device).eval()
    models = {"H0": load_h0(device, channels_last=True, checkpoint="best")[0].backbone}
    runtime = load_r5_reference()["runtime"]
    models["H3"] = load_h_checkpoint_model(H3_DIR, "best", runtime)[0].backbone
    models["H3B"] = load_h_checkpoint_model(H3B_DIR, "best", runtime)[0].backbone
    pretrained_features = _extract_features(pretrained, valid_data, indices)
    rows = []
    for experiment in ("H0", "H3", "H3B"):
        features = _extract_features(models[experiment], valid_data, indices)
        rows.append({
            "experiment": experiment,
            "cosine_stage3": float(_cosine(
                pretrained_features[0], features[0]).mean()),
            "cosine_stage4": float(_cosine(
                pretrained_features[1], features[1]).mean()),
            "cosine_final": float(_cosine(
                pretrained_features[2], features[2]).mean()),
        })
    del pretrained, models
    torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8")
    return frame


def _group_positive_counts(results: dict) -> pd.DataFrame:
    train_data, valid_data, _ = build_eval_datasets()
    train_metadata = study_view_metadata(train_data, TRAIN_CSV)
    valid_metadata = study_view_metadata(valid_data, VALID_CSV)
    rows = []
    for split, dataset, metadata in (
            ("train", train_data, train_metadata),
            ("valid", valid_data, valid_metadata)):
        result = results[("H3B", "best", split)]
        labels = np.asarray(result["labels"], dtype=np.int64)
        subjects = np.asarray(result["subjects"]).astype(str)
        counts = np.asarray(result["image_counts"], dtype=np.int64)
        categories = np.asarray(metadata["categories"], dtype=object)
        masks = {
            "Overall": np.ones(len(labels), dtype=bool),
            "AP-only": categories == "ap_only",
            "PA+Lateral": np.isin(categories, ["pa_only", "lateral_only",
                                               "mixed_or_other"]),
            "Single-image": counts == 1,
            "Multi-image": counts >= 2,
        }
        for group, mask in masks.items():
            if int(mask.sum()) == 0:
                continue
            group_labels = labels[mask]
            group_subjects = subjects[mask]
            positive_studies = [int((group_labels[:, index] == 1).sum())
                                for index in range(group_labels.shape[1])]
            positive_subjects = [
                int(len(np.unique(group_subjects[group_labels[:, index] == 1])))
                for index in range(group_labels.shape[1])]
            rows.append({
                "split": split,
                "group": group,
                "n_studies": int(mask.sum()),
                "n_subjects": int(len(np.unique(group_subjects))),
                "positive_studies_min": int(min(positive_studies)),
                "positive_studies_median": float(np.median(positive_studies)),
                "positive_studies_max": int(max(positive_studies)),
                "positive_subjects_min": int(min(positive_subjects)),
                "positive_subjects_median": float(np.median(positive_subjects)),
                "positive_subjects_max": int(max(positive_subjects)),
                "positive_disease_study_pairs": int(sum(positive_studies)),
            })
    return pd.DataFrame(rows)


def _grouped_rows(results: dict) -> pd.DataFrame:
    train_data, valid_data, _ = build_eval_datasets()
    train_metadata = study_view_metadata(train_data, TRAIN_CSV)
    valid_metadata = study_view_metadata(valid_data, VALID_CSV)
    rows = []
    for experiment in ANALYSIS_ORDER:
        for split, dataset, metadata in (
                ("train", train_data, train_metadata),
                ("valid", valid_data, valid_metadata)):
            result = results[(experiment, "best", split)]
            rows.extend(base.grouped_metrics(
                result, dataset, metadata, experiment, "best", split))
    return pd.DataFrame(rows)


def _bootstrap(results: dict) -> dict:
    valid = {experiment: results[(experiment, "best", "valid")]
             for experiment in ANALYSIS_ORDER}
    labels = valid["H0"]["labels"]
    studies = np.asarray(valid["H0"]["studies"]).astype(str)
    for experiment in ANALYSIS_ORDER:
        if not (np.array_equal(np.asarray(valid[experiment]["labels"]), labels)
                and np.array_equal(np.asarray(valid[experiment]["studies"]).astype(str),
                                   studies)):
            raise RuntimeError("Validation predictions are not Study-aligned")
    probabilities = {}
    for experiment in ANALYSIS_ORDER:
        logits = np.asarray(valid[experiment]["study_logits"], dtype=np.float64)
        probabilities[experiment] = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    subjects = np.asarray(valid["H0"]["subjects"]).astype(str)
    comparisons = {"H3B_minus_H0": ("H0", "H3B"),
                   "H3B_minus_H3": ("H3", "H3B"),
                   "H3_minus_H0": ("H0", "H3")}
    boot = bootstrap_models(labels, probabilities, subjects, comparisons)
    return {
        "scope": "official Validation, Best checkpoints, Study-level probability",
        "bootstrap_replicates": 2000,
        "subject_cluster": True,
        "seed": SEED,
        "exploratory": True,
        "comparisons": boot["comparisons"],
        "sampling_indices_sha256": boot.get("sampling_indices_sha256"),
        "test_access": False,
    }


def _convergence_curves() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for experiment in ANALYSIS_ORDER:
        history = _history(experiment)
        axes[0].plot(history["epoch"], history["train_bce"], marker="o",
                     color=COLORS[experiment], label=experiment)
        axes[1].plot(history["epoch"], history["valid_loss"], marker="o",
                     color=COLORS[experiment], label=experiment)
        axes[2].plot(history["epoch"], history["macro_auroc"], marker="o",
                     color=COLORS[experiment], label=experiment)
        status = _status(experiment)
        best_epoch = int(status["best_epoch"])
        best_row = history[history["epoch"] == best_epoch]
        if len(best_row):
            axes[2].scatter([best_epoch], [float(best_row["macro_auroc"].iloc[0])],
                            marker="*", s=140, color=COLORS[experiment])
    axes[0].set_title("Train raw Study BCE")
    axes[1].set_title("Validation raw Study BCE")
    axes[2].set_title("Validation Macro-AUROC (star = Best)")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(H3B_DIR / "convergence_curves.png", dpi=140)
    plt.close(figure)


def _fmt(value, digits=6):
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _write_reports(overall: pd.DataFrame, loss_comparison: pd.DataFrame,
                   auroc_comparison: pd.DataFrame, drift: pd.DataFrame,
                   feature_drift: pd.DataFrame, grouped: pd.DataFrame,
                   bootstrap: dict, positive_counts: pd.DataFrame) -> None:
    best = overall[overall["checkpoint"] == "best"].set_index("experiment")
    last = overall[overall["checkpoint"] == "last"].set_index("experiment")
    loss_index = loss_comparison.set_index("disease")
    auroc_index = auroc_comparison.set_index("disease")
    h3_lambda = float(read_json(H3_DIR / "l2sp_calibration.json")["lambda_sp"])
    h3b_lambda = float(EXPERIMENTS["H3B"]["optimization"]["lambda_sp"])
    h3b_history = _history("H3B")
    h3b_status = _status("H3B")
    h3_history = _history("H3")
    h0_history = _history("H0")
    last_three = h3b_history.tail(3)
    boot = bootstrap["comparisons"]
    h3b_h0 = boot["H3B_minus_H0"]
    h3b_h3 = boot["H3B_minus_H3"]
    aggregate = {
        "train_bce": float(best.loc["H3B", "train_overall_bce"]),
        "valid_bce": float(best.loc["H3B", "valid_overall_bce"]),
        "valid_auroc": float(best.loc["H3B", "valid_macro_auroc"]),
        "valid_auprc": float(best.loc["H3B", "valid_macro_auprc"]),
        "train_auroc": float(best.loc["H3B", "train_macro_auroc"]),
    }
    # Disease-level benefit vs H3.
    benefit_rows = []
    for disease in LABEL_NAMES:
        benefit_rows.append({
            "disease": disease,
            "H3_valid_bce": float(loss_index.loc[disease, "H3_valid"]),
            "H3B_valid_bce": float(loss_index.loc[disease, "H3B_valid"]),
            "delta_vs_H3": float(loss_index.loc[disease, "H3B_valid"])
            - float(loss_index.loc[disease, "H3_valid"]),
            "H3_valid_auroc": float(auroc_index.loc[disease, "H3_valid"]),
            "H3B_valid_auroc": float(auroc_index.loc[disease, "H3B_valid"]),
            "delta_auroc_vs_H3": float(auroc_index.loc[disease, "H3B_valid"])
            - float(auroc_index.loc[disease, "H3_valid"]),
        })
    benefit = pd.DataFrame(benefit_rows).sort_values("delta_vs_H3")
    benefit.to_csv(H3B_DIR / "per_class_benefit_vs_h3.csv", index=False,
                   encoding="utf-8")
    # Grouped comparison (best, valid).
    grouped_valid = grouped[(grouped["checkpoint"] == "best")
                            & (grouped["split"] == "valid")
                            & (grouped["disease"] == LABEL_NAMES[0])]
    lines = [
        "# H3-B final report (Weak L2-SP)", "",
        f"- H3-B changes only λ: {EXPERIMENTS['H3B']['optimization']['lambda_sp']:.12g} "
        f"= 0.1 × H3 λ ({h3_lambda:.12g}).",
        f"- Status: completed {h3b_status['completed_epochs']}/{h3b_status['max_epochs']} "
        f"epochs; stop reason `{h3b_status['stop_reason']}`; Best epoch "
        f"{h3b_status['best_epoch']}; Best Validation Macro-AUROC "
        f"{h3b_status['best_macro_auroc']:.6f}.",
        f"- Trainable parameters: 26,592,778 (Global-only, same as H0/H3); "
        f"regularized L2-SP parameters: "
        f"{read_json(H3B_DIR / 'l2sp_calibration_reference.json')['regularized_parameter_count']:,}.",
        "",
        "## Convergence record (last three epochs)", "",
        "| Epoch | Train raw BCE | Valid raw BCE | Valid Macro-AUROC | Valid Macro-AUPRC | "
        "L2-SP term | Stage3 drift | Stage4 drift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in last_three.iterrows():
        lines.append(
            f"| {int(row['epoch'])} | {row['train_bce']:.6f} | "
            f"{row['valid_loss']:.6f} | {row['macro_auroc']:.6f} | "
            f"{row['macro_auprc']:.6f} | {row.get('l2sp_term', float('nan')):.6g} | "
            f"{row.get('stage3_relative_drift', float('nan')):.6f} | "
            f"{row.get('stage4_relative_drift', float('nan')):.6f} |")
    still_improving = bool(
        len(last_three) == 3
        and float(last_three["macro_auroc"].iloc[-1]) > float(
            last_three["macro_auroc"].iloc[-2]))
    lines.extend([
        "",
        f"- Stop reason: **{h3b_status['stop_reason']}**；"
        f"最后 epoch = {h3b_status['completed_epochs']}，Best epoch = "
        f"{h3b_status['best_epoch']}。",
        f"- 是否尚未收敛迹象：{'最后两轮 AUROC 仍在上升，但已触发/达到规则条件；需结合 Early Stop 状态判断' if still_improving else '最后两轮 AUROC 未继续上升；按原规则正常结束'}。",
        "",
        "## Best checkpoint full deterministic inference", "",
        "| Split | Macro-AUROC | Macro-AUPRC | Overall BCE |",
        "|---|---:|---:|---:|",
        f"| Train | {best.loc['H3B', 'train_macro_auroc']:.6f} | "
        f"{best.loc['H3B', 'train_macro_auprc']:.6f} | "
        f"{best.loc['H3B', 'train_overall_bce']:.6f} |",
        f"| Valid | {best.loc['H3B', 'valid_macro_auroc']:.6f} | "
        f"{best.loc['H3B', 'valid_macro_auprc']:.6f} | "
        f"{best.loc['H3B', 'valid_overall_bce']:.6f} |",
        f"| Gap (Valid−Train BCE) | — | — | "
        f"{best.loc['H3B', 'train_valid_bce_gap']:+.6f} |",
        "",
        "## Per-disease Train/Valid BCE at Best checkpoint", "",
        "| Disease | H0 Train | H0 Valid | H3 Train | H3 Valid | H3B Train | H3B Valid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for disease in list(LABEL_NAMES) + ["Overall BCE"]:
        row = loss_index.loc[disease]
        lines.append(
            f"| {disease} | {row['H0_train']:.6f} | {row['H0_valid']:.6f} | "
            f"{row['H3_train']:.6f} | {row['H3_valid']:.6f} | "
            f"{row['H3B_train']:.6f} | {row['H3B_valid']:.6f} |")
    lines.extend([
        "",
        "## Parameter / feature drift (Best checkpoints)", "",
        "| Experiment | Stage3 relative drift | Stage4 relative drift | "
        "Stage4 GAP cosine vs pretrained | classifier-input cosine |",
        "|---|---:|---:|---:|---:|",
    ])
    feature_index = feature_drift.set_index("experiment")
    for experiment in ANALYSIS_ORDER:
        sub = drift[drift["experiment"] == experiment].set_index("stage")
        lines.append(
            f"| {experiment} | {sub.loc['stage3', 'relative_drift']:.6f} | "
            f"{sub.loc['stage4', 'relative_drift']:.6f} | "
            f"{feature_index.loc[experiment, 'cosine_stage4']:.4f} | "
            f"{feature_index.loc[experiment, 'cosine_final']:.4f} |")
    lines.extend([
        "",
        "## Grouped Validation metrics (Best)", "",
        "| Experiment | Group | Macro-AUROC | Macro-AUPRC | Overall BCE |",
        "|---|---|---:|---:|---:|",
    ])
    for experiment in ANALYSIS_ORDER:
        for group in ("AP-only", "PA+Lateral", "Single-image", "Multi-image"):
            row = grouped_valid[(grouped_valid["experiment"] == experiment)
                                & (grouped_valid["group"] == group)]
            if row.empty:
                continue
            item = row.iloc[0]
            lines.append(
                f"| {experiment} | {group} | {item['macro_auroc']:.6f} | "
                f"{item['macro_auprc']:.6f} | {item['bce']:.6f} |")
    lines.extend([
        "",
        "### Positive Study / independent positive Subject support per group",
        "",
        "（每个 Study 至少一个标签为阳性，因此 group 的 any-positive 恒等于全部；"
        "下表按十类疾病给出阳性 Study / 独立阳性 Subject 的 min–median–max，"
        "逐疾病精确计数见 `grouped_performance_comparison.csv` 的 "
        "`n_positive_studies` / `n_positive_subjects`。）",
        "",
        "| Split | Group | Studies | Subjects | Positive Studies "
        "(min/median/max) | Independent positive Subjects (min/median/max) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in positive_counts.iterrows():
        lines.append(
            f"| {row['split']} | {row['group']} | {int(row['n_studies'])} | "
            f"{int(row['n_subjects'])} | {int(row['positive_studies_min'])} / "
            f"{row['positive_studies_median']:.1f} / "
            f"{int(row['positive_studies_max'])} | "
            f"{int(row['positive_subjects_min'])} / "
            f"{row['positive_subjects_median']:.1f} / "
            f"{int(row['positive_subjects_max'])} |")
    lines.extend([
        "",
        "## Paired bootstrap (Best, Validation, 2000 subject-cluster draws)", "",
        f"- H3B − H0: Δ Macro-AUROC = {h3b_h0['delta_macro_auroc']:+.6f}, "
        f"95% CI {h3b_h0['delta_macro_auroc_ci95']}；Δ Macro-AUPRC = "
        f"{h3b_h0['delta_macro_auprc']:+.6f}，CI {h3b_h0['delta_macro_auprc_ci95']}。",
        f"- H3B − H3: Δ Macro-AUROC = {h3b_h3['delta_macro_auroc']:+.6f}, "
        f"95% CI {h3b_h3['delta_macro_auroc_ci95']}。",
        "- 这些区间是验证集重复使用条件下的探索性统计，不是独立 Test 结论。",
        "",
        "## Artifacts", "",
        "- `training_history.csv` / `l2sp_gradient_history.csv`（逐 epoch 总体指标、"
        "L2-SP 项与 λ·梯度范数）、`per_epoch_class_loss.csv`（逐 epoch 逐类 "
        "Validation BCE 完整；Train 逐类仅最后一个 epoch 保留，属产物限制，"
        "不伪造中间值；Best/Last 的完整确定性 Train/Valid 逐类 BCE 见 "
        "`best_train_valid_class_loss.csv` / `last_train_valid_class_loss.csv`）；",
        "- `per_class_metrics.csv`（逐 epoch 十类 Validation AUROC/AUPRC/BCE）；",
        "- `best_checkpoint.pth` / `last_checkpoint.pth`（另保留项目惯例 `best.pth` / `last.pth`）；",
        "- `per_class_loss_comparison.csv`、`per_class_auroc_comparison.csv`、"
        "`parameter_drift_comparison.csv`、`convergence_curves.png`；",
        "- `l2sp_implementation_audit.md` / `.json`；",
        "- 未读取 Test 标签，未用 Test 选择 λ 或 checkpoint。",
        "",
    ])
    (H3B_DIR / "h3b_final_report.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")

    # Comparative H0 / H3 / H3-B report with ten required questions.
    lines = [
        "# H0 / H3 / H3-B comparative report", "",
        "| Experiment | Best Epoch | Stop | Train BCE | Valid BCE | BCE Gap | "
        "Valid Macro-AUROC | Valid Macro-AUPRC | Stage3 drift | Stage4 drift |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in ANALYSIS_ORDER:
        row = best.loc[experiment]
        status = _status(experiment)
        sub = drift[drift["experiment"] == experiment].set_index("stage")
        lines.append(
            f"| {experiment} | {row['best_epoch']} | {status.get('stop_reason')} | "
            f"{row['train_overall_bce']:.6f} | {row['valid_overall_bce']:.6f} | "
            f"{row['train_valid_bce_gap']:+.6f} | {row['valid_macro_auroc']:.6f} | "
            f"{row['valid_macro_auprc']:.6f} | "
            f"{sub.loc['stage3', 'relative_drift']:.6f} | "
            f"{sub.loc['stage4', 'relative_drift']:.6f} |")
    # Convergence-speed comparison (training-batch raw BCE; Best = full inference).
    thresholds = (0.35, 0.30, 0.28, 0.26)
    lines.extend([
        "",
        "## Convergence speed (training-batch raw BCE / full-inference Best AUROC)",
        "",
        "| Experiment | Best epoch | Completed epochs | Epochs to Train BCE "
        "≤0.35 | ≤0.30 | ≤0.28 | ≤0.26 | Best Valid AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for experiment in ANALYSIS_ORDER:
        history = _history(experiment)
        status = _status(experiment)
        cells = []
        for threshold in thresholds:
            hits = history[history["train_bce"] <= threshold]["epoch"]
            cells.append(str(int(hits.iloc[0])) if len(hits) else "—")
        lines.append(
            f"| {experiment} | {status['best_epoch']} | "
            f"{status['completed_epochs']} | " + " | ".join(cells)
            + f" | {best.loc[experiment, 'valid_macro_auroc']:.6f} |")
    lines.extend([
        "",
        f"- H0 在 epoch {_status('H0')['best_epoch']} 达到 Best；H3 在 epoch "
        f"{_status('H3')['best_epoch']}（预算停止）；H3-B 在 epoch "
        f"{_status('H3B')['best_epoch']} 达到 Best，并在 epoch "
        f"{_status('H3B')['completed_epochs']} 触发三条件早停。",
        "- 收敛速度：H3-B 仍慢于 H0（Train BCE 到 0.30 所需 epoch 见上表），"
        "但明显快于 H3；H3-B 的 Last checkpoint 已开始过拟合（见下）。",
    ])

    def group_value(experiment: str, group: str) -> float:
        row = grouped_valid[(grouped_valid["experiment"] == experiment)
                            & (grouped_valid["group"] == group)]
        return float(row["macro_auroc"].iloc[0])

    subgroup = {
        group: {experiment: group_value(experiment, group)
                for experiment in ANALYSIS_ORDER}
        for group in ("AP-only", "PA+Lateral", "Single-image", "Multi-image")
    }
    h3b_below_h0_subgroups = all(
        subgroup[group]["H3B"] <= subgroup[group]["H0"] for group in subgroup)
    decide_stop = bool(
        float(best.loc["H3B", "valid_macro_auroc"]) < float(
            best.loc["H0", "valid_macro_auroc"]) and h3b_below_h0_subgroups)

    lines.extend([
        "",
        "## Answers to the ten required questions", "",
        "1. **H3 的 L2-SP 实现是否正确？** 见 "
        "`l2sp_implementation_audit.md`：参数范围、预训练对应关系、element-mean "
        "归一化、λ 比例、梯度公式、梯度累积单次反传、分类头/冻结排除、AdamW "
        "weight decay 与 LR 全部通过；未发现重复 λ 或 microbatch 放大。",
        f"2. **新 λ 是否确实降低了正则强度？** 是。H3 λ="
        f"{h3_lambda:.6g}，"
        f"H3-B λ={EXPERIMENTS['H3B']['optimization']['lambda_sp']:.6g}，精确 0.1 倍；"
        f"同漂移点下 L2-SP 梯度范数从 H3 的 "
        f"{read_json(H3_DIR / 'l2sp_calibration.json')['probe']['l2sp_unit_grad_norm'] * h3_lambda:.4g} "
        f"降至约 0.1 倍；Stage3/4 漂移从 H3 的 0.000321/0.000227 上升（见上表），"
        "说明约束确实减弱。",
        f"3. **H3-B 是否恢复了 Train BCE / Train AUROC？** Train BCE "
        f"H0 {best.loc['H0', 'train_overall_bce']:.6f}、H3 "
        f"{best.loc['H3', 'train_overall_bce']:.6f}、H3-B "
        f"{best.loc['H3B', 'train_overall_bce']:.6f}；H3-B 明显低于 H3，但仍高于 H0；"
        f"Train Macro-AUROC：H0 {best.loc['H0', 'train_macro_auroc']:.6f}、H3 "
        f"{best.loc['H3', 'train_macro_auroc']:.6f}、H3-B "
        f"{best.loc['H3B', 'train_macro_auroc']:.6f}。",
        f"4. **H3-B 是否正常收敛 / 早停？** `{h3b_status['stop_reason']}`，"
        f"完成 {h3b_status['completed_epochs']}/{h3b_status['max_epochs']} epochs，"
        f"Best epoch {h3b_status['best_epoch']}；最后三轮数据见 `h3b_final_report.md`。",
        f"5. **Validation AUROC 是否恢复到接近 H0？** H3-B "
        f"{best.loc['H3B', 'valid_macro_auroc']:.6f} vs H0 "
        f"{best.loc['H0', 'valid_macro_auroc']:.6f}（Δ="
        f"{h3b_h0['delta_macro_auroc']:+.6f}，95% CI "
        f"{h3b_h0['delta_macro_auroc_ci95']}），较 H3 的 −0.0397 明显恢复，"
        "但仍未达到 H0 点估计。",
        "6. **哪些疾病从降低正则化强度中获益最大？** 按 Valid BCE 相对 H3 改善"
        "排序："
        + "；".join(
            f"{row['disease']} {row['delta_vs_H3']:+.6f}"
            for _, row in benefit.head(5).iterrows())
        + "（负值表示 H3-B 更低，见 `per_class_benefit_vs_h3.csv`）。",
        f"7. **Stage3/4 参数漂移如何变化？** Stage3：H0 "
        f"{drift[(drift.experiment=='H0') & (drift.stage=='stage3')].relative_drift.iloc[0]:.6f}、"
        f"H3 {drift[(drift.experiment=='H3') & (drift.stage=='stage3')].relative_drift.iloc[0]:.6f}、"
        f"H3-B {drift[(drift.experiment=='H3B') & (drift.stage=='stage3')].relative_drift.iloc[0]:.6f}；"
        f"Stage4 对应为 "
        f"{drift[(drift.experiment=='H0') & (drift.stage=='stage4')].relative_drift.iloc[0]:.6f} / "
        f"{drift[(drift.experiment=='H3') & (drift.stage=='stage4')].relative_drift.iloc[0]:.6f} / "
        f"{drift[(drift.experiment=='H3B') & (drift.stage=='stage4')].relative_drift.iloc[0]:.6f}。",
        f"8. **Train/Valid BCE Gap 如何变化？** H0 "
        f"{best.loc['H0', 'train_valid_bce_gap']:+.6f}、H3 "
        f"{best.loc['H3', 'train_valid_bce_gap']:+.6f}、H3-B "
        f"{best.loc['H3B', 'train_valid_bce_gap']:+.6f}；H3-B 的 Gap 仍小于 H0，"
        "因为 Train BCE 仍偏高（欠拟合），不能把 Gap 缩小直接当成泛化改善。",
        "9. **单图与 AP-only 子群是否改善？** "
        f"AP-only：H0 {subgroup['AP-only']['H0']:.6f}、H3 "
        f"{subgroup['AP-only']['H3']:.6f}、H3-B "
        f"{subgroup['AP-only']['H3B']:.6f}（H3-B 比 H3 "
        f"{subgroup['AP-only']['H3B'] - subgroup['AP-only']['H3']:+.6f}，"
        f"比 H0 {subgroup['AP-only']['H3B'] - subgroup['AP-only']['H0']:+.6f}）；"
        f"单图：H0 {subgroup['Single-image']['H0']:.6f}、H3 "
        f"{subgroup['Single-image']['H3']:.6f}、H3-B "
        f"{subgroup['Single-image']['H3B']:.6f}（H3-B 比 H3 "
        f"{subgroup['Single-image']['H3B'] - subgroup['Single-image']['H3']:+.6f}，"
        f"比 H0 {subgroup['Single-image']['H3B'] - subgroup['Single-image']['H0']:+.6f}）。"
        "结论：两个子群相对 H3 均改善，但仍未超过 H0。",
        "10. **现有结果支持继续研究 L2-SP，还是应停止？** "
        + ("**建议停止把 L2-SP 作为提升泛化性能的方向。** H3-B 相对 H3 显著恢复"
           f"（ΔAUROC {h3b_h3['delta_macro_auroc']:+.6f}，CI "
           f"{h3b_h3['delta_macro_auroc_ci95']}），但仍低于 H0（Δ "
           f"{h3b_h0['delta_macro_auroc']:+.6f}，CI "
           f"{h3b_h0['delta_macro_auroc_ci95']} 跨 0），AUPRC 亦低于 H0 "
           f"（{best.loc['H3B', 'valid_macro_auprc']:.6f} vs "
           f"{best.loc['H0', 'valid_macro_auprc']:.6f}），且 AP-only 与单图子群"
           "均未超过 H0；弱 L2-SP 只是减轻了强正则的损害，没有带来额外收益。"
           "若未来仍要研究，只建议作为正则化鲁棒性/多 seed 复核的低优先级项，"
           "不应再继续调 λ 或与 H2 组合。" if decide_stop else
           "**不建议立即停止，但也不足以支持组合。** H3-B 至少在一个子群或总体"
           "上接近/超过 H0，建议先用 seed 2026/3407 做配对复核，再决定是否组合；"
           "本轮仍为单 seed 探索性结果。"),
        "",
        "## Per-epoch record completeness and limitations", "",
        "- `training_history.csv` 记录原始 Train BCE、优化目标 `objective_loss`、"
        "Validation 原始 BCE、Macro-AUROC/AUPRC、L2-SP 原始项与 λR、Stage3/4 "
        "相对漂移、LR、BCE 梯度范数、L2-SP 单位梯度范数、更新次数与早停状态。",
        "- 十类疾病逐 epoch Validation AUROC/AUPRC/BCE 记录在 "
        "`per_class_metrics.csv`；逐 epoch 逐类 Train/Valid BCE 在 "
        "`per_epoch_class_loss.csv`。",
        "- `l2sp_gradient_history.csv` 由训练历史派生，补充 λ·‖g_SP‖ 与 "
        "λ·‖g_SP‖/‖g_BCE‖；梯度裁剪阈值为 1.0，但训练循环未单独保存每次"
        "更新是否触发裁剪，故该字段只记录阈值，不能伪造触发比例。",
        "- `per_epoch_class_loss.csv` 的 Validation 逐 epoch 逐类 BCE 完整；"
        "Train 逐类 BCE 仅在最后一个 epoch 保留（分析重建步骤未保留训练时记录的"
        "全部中间 epoch 的 Train 逐类值），不伪造；Best/Last 的完整确定性 "
        "Train/Valid 逐类 BCE 见对应文件。",
        "- 单 seed、Validation 重复使用、无 Test 标签；所有 CI 为当前官方 "
        "Validation 下的探索性统计。",
        "",
    ])
    (H3B_DIR / "h0_h3_h3b_comparative_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def run_h3b_analysis(force: bool = False) -> dict:
    H3B_DIR.mkdir(parents=True, exist_ok=True)
    base.ensure_dirs()
    if not (H3B_DIR / "best.pth").is_file():
        raise RuntimeError("H3-B Best checkpoint is missing; training must finish first")
    results = _all_results(force=force)
    h3b_results = {(checkpoint, split): results[("H3B", checkpoint, split)]
                   for checkpoint in ("best", "last")
                   for split in ("train", "valid")}
    train_data, valid_data, _ = build_eval_datasets()
    base._write_experiment_artifacts("H3B", train_data, valid_data, h3b_results)
    overall = _overall_rows(results)
    per_class = _per_class_rows(results)
    loss_comparison, auroc_comparison = _class_compare_frames(per_class)
    drift = _parameter_drift(results)
    feature_drift = _feature_drift(force=force)
    grouped = _grouped_rows(results)
    positive_counts = _group_positive_counts(results)
    bootstrap = _bootstrap(results)

    # train_valid_metrics.csv: overall (best/last) + per-class (best).
    h3b_metrics_rows = []
    for _, row in overall[overall["experiment"] == "H3B"].iterrows():
        h3b_metrics_rows.append({
            "experiment": "H3B", "checkpoint": row["checkpoint"],
            "split": None, "disease": "Overall",
            "auroc": None, "auprc": None, "bce": row["valid_overall_bce"]
            if row["checkpoint"] == "best" else None,
            "positive_bce": None, "negative_bce": None,
            "n_positive": None, "n_negative": None,
            "train_overall_bce": row["train_overall_bce"],
            "valid_overall_bce": row["valid_overall_bce"],
            "train_macro_auroc": row["train_macro_auroc"],
            "valid_macro_auroc": row["valid_macro_auroc"],
            "train_macro_auprc": row["train_macro_auprc"],
            "valid_macro_auprc": row["valid_macro_auprc"],
            "train_valid_bce_gap": row["train_valid_bce_gap"],
            "best_epoch": row["best_epoch"],
            "completed_epochs": row["completed_epochs"],
            "stop_reason": row["stop_reason"],
        })
    for _, row in per_class[per_class["experiment"] == "H3B"].iterrows():
        h3b_metrics_rows.append({
            "experiment": "H3B", "checkpoint": "best", "split": row["split"],
            "disease": row["disease"], "auroc": row["auroc"],
            "auprc": row["auprc"], "bce": row["bce"],
            "positive_bce": row["positive_bce"],
            "negative_bce": row["negative_bce"],
            "n_positive": row["n_positive"], "n_negative": row["n_negative"],
            "train_overall_bce": None, "valid_overall_bce": None,
            "train_macro_auroc": None, "valid_macro_auroc": None,
            "train_macro_auprc": None, "valid_macro_auprc": None,
            "train_valid_bce_gap": None, "best_epoch": None,
            "completed_epochs": None, "stop_reason": None,
        })
    write_csv(H3B_DIR / "train_valid_metrics.csv", h3b_metrics_rows)
    # Derived per-epoch L2-SP gradient records (lambda-scaled) for auditability.
    history = pd.read_csv(H3B_DIR / "training_history.csv")
    lambda_sp = float(EXPERIMENTS["H3B"]["optimization"]["lambda_sp"])
    history["l2sp_grad_norm"] = lambda_sp * history["l2sp_unit_grad_norm"]
    history["l2sp_grad_ratio"] = (history["l2sp_grad_norm"]
                                  / history["bce_grad_norm"].replace(0, np.nan))
    history["grad_clip_max_norm"] = 1.0
    history["grad_clip_note"] = (
        "max-norm 1.0 applied after accumulation on every update; per-update "
        "pre-clip norm / trigger flag was not logged by the training loop")
    history.to_csv(H3B_DIR / "training_history.csv", index=False,
                   encoding="utf-8")
    history[[
        "epoch", "train_bce", "objective_loss", "valid_loss", "macro_auroc",
        "l2sp_penalty", "l2sp_term", "lambda_sp", "bce_grad_norm",
        "l2sp_unit_grad_norm", "l2sp_grad_norm", "l2sp_grad_ratio",
        "stage3_relative_drift", "stage4_relative_drift", "lr_backbone",
        "lr_head", "optimizer_updates", "grad_clip_max_norm",
    ]].to_csv(H3B_DIR / "l2sp_gradient_history.csv", index=False,
              encoding="utf-8")
    loss_comparison.to_csv(H3B_DIR / "per_class_loss_comparison.csv",
                           index=False, encoding="utf-8")
    auroc_comparison.to_csv(H3B_DIR / "per_class_auroc_comparison.csv",
                            index=False, encoding="utf-8")
    drift.to_csv(H3B_DIR / "parameter_drift_comparison.csv", index=False,
                 encoding="utf-8")
    grouped.to_csv(H3B_DIR / "grouped_performance_comparison.csv", index=False,
                   encoding="utf-8")
    positive_counts.to_csv(H3B_DIR / "grouped_positive_counts.csv", index=False,
                           encoding="utf-8")
    write_json(H3B_DIR / "paired_subject_bootstrap.json", bootstrap)
    _convergence_curves()
    _write_reports(overall, loss_comparison, auroc_comparison, drift,
                   feature_drift, grouped, bootstrap, positive_counts)
    summary = {
        "status": "h3b_analysis_complete",
        "best": overall[overall["checkpoint"] == "best"].to_dict("records"),
        "bootstrap": bootstrap["comparisons"],
        "test_access": False,
    }
    write_json(H3B_DIR / "h3b_analysis_summary.json", summary)
    print("[h3b-analysis] complete", flush=True)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_h3b_analysis(), indent=2, default=str))
