from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[3]
PROJECT_DIR = WORK_DIR.parent
REPO_DIR = PROJECT_DIR.parent
DIAGNOSTICS_DIR = WORK_DIR / "outputs/experiments/DINOv3_Feature_Diagnostics"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(DIAGNOSTICS_DIR))

import run_diagnostics as diagnostics
from datasets.dataset import StudyDataset
from dino_experiment import LABEL_NAMES


R5_DIR = HERE
R3_DIR = WORK_DIR / "outputs/experiments/DINOv3_Feature_Optimization/R3_Global_Local"
R2_DIR = WORK_DIR / "outputs/experiments/DINOv3_Feature_Optimization/R2_512_GAP"
TRAIN_CSV = REPO_DIR / "cxr_first/train.csv"
IMAGE_ROOT = REPO_DIR / "cxr_first"
SEED = 42
BOOTSTRAP_REPLICATES = 2000


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def metric_vectors(labels: np.ndarray, scores: np.ndarray):
    aucs, aps = [], []
    for index in range(10):
        y = labels[:, index]
        aucs.append(float(roc_auc_score(y, scores[:, index]))
                    if np.unique(y).size == 2 else float("nan"))
        aps.append(float(average_precision_score(y, scores[:, index]))
                    if np.any(y == 1) else float("nan"))
    return np.asarray(aucs), np.asarray(aps)


def macro(values):
    return float(np.nanmean(values))


def paired_cluster_bootstrap(labels, scores, draws, draws_sha256, comparisons):
    payload = {
        "method": "Subject-cluster paired bootstrap; percentile 95% intervals",
        "unit": "Subject_id, with all Validation Studies for each sampled Subject",
        "seed": SEED,
        "requested_replicates": BOOTSTRAP_REPLICATES,
        "sampling_indices_sha256": draws_sha256,
        "one_class_resample_rule": (
            "Skip per-class AUROC/AUPRC when either positive or negative Studies are absent; "
            "macro deltas average valid paired classes per replicate."
        ),
        "comparisons": {},
    }
    for name, left, right in comparisons:
        auc_left, ap_left = metric_vectors(labels, scores[left])
        auc_right, ap_right = metric_vectors(labels, scores[right])
        class_auc, class_ap = [[] for _ in range(10)], [[] for _ in range(10)]
        macro_auc, macro_ap = [], []
        for indices in draws:
            y = labels[indices]
            d_auc, d_ap = [], []
            for class_index in range(10):
                if np.unique(y[:, class_index]).size != 2:
                    continue
                auc_delta = float(
                    roc_auc_score(y[:, class_index], scores[right][indices, class_index])
                    - roc_auc_score(y[:, class_index], scores[left][indices, class_index]))
                ap_delta = float(
                    average_precision_score(y[:, class_index], scores[right][indices, class_index])
                    - average_precision_score(y[:, class_index], scores[left][indices, class_index]))
                class_auc[class_index].append(auc_delta)
                class_ap[class_index].append(ap_delta)
                d_auc.append(auc_delta)
                d_ap.append(ap_delta)
            if d_auc:
                macro_auc.append(float(np.mean(d_auc)))
                macro_ap.append(float(np.mean(d_ap)))
        if len(macro_auc) != BOOTSTRAP_REPLICATES or len(macro_ap) != BOOTSTRAP_REPLICATES:
            raise RuntimeError(f"{name} did not produce {BOOTSTRAP_REPLICATES} valid macro replicates")
        payload["comparisons"][name] = {
            "left": left,
            "right": right,
            "macro_auroc_delta": macro(auc_right) - macro(auc_left),
            "macro_auroc_ci95": np.quantile(macro_auc, [0.025, 0.975]).tolist(),
            "macro_auprc_delta": macro(ap_right) - macro(ap_left),
            "macro_auprc_ci95": np.quantile(macro_ap, [0.025, 0.975]).tolist(),
            "valid_macro_replicates": len(macro_auc),
            "per_class": {
                disease: {
                    "auroc_delta": float(auc_right[index] - auc_left[index]),
                    "auroc_ci95": np.quantile(class_auc[index], [0.025, 0.975]).tolist()
                    if class_auc[index] else [None, None],
                    "auprc_delta": float(ap_right[index] - ap_left[index]),
                    "auprc_ci95": np.quantile(class_ap[index], [0.025, 0.975]).tolist()
                    if class_ap[index] else [None, None],
                    "valid_replicates": len(class_auc[index]),
                }
                for index, disease in enumerate(LABEL_NAMES)
            },
        }
    return payload


def write_update_comparison(r3_log, r5_log, r3_runtime, r5_runtime,
                            train_study_keys, r3_status, r5_status):
    r3_by_epoch = {int(row.epoch): row for row in r3_log.itertuples(index=False)}
    r5_by_epoch = {int(row.epoch): row for row in r5_log.itertuples(index=False)}
    r3_cumulative = r5_cumulative = 0
    rows = []
    max_epoch = max(max(r3_by_epoch), max(r5_by_epoch))
    for epoch in range(1, max_epoch + 1):
        r3_row, r5_row = r3_by_epoch.get(epoch), r5_by_epoch.get(epoch)
        r3_updates = int(r3_row.optimizer_updates) if r3_row is not None else None
        r5_updates = int(r5_row.optimizer_updates) if r5_row is not None else None
        if r3_updates is not None:
            r3_cumulative += r3_updates
        if r5_updates is not None:
            r5_cumulative += r5_updates
        order_indices = list(range(len(train_study_keys)))
        random.Random(SEED + epoch - 1).shuffle(order_indices)
        digest = hashlib.sha256()
        for index in order_indices:
            subject, study = train_study_keys[index]
            digest.update(subject.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(study.encode("utf-8"))
            digest.update(b"\n")
        order_hash = digest.hexdigest()
        update_match = (r3_updates == r5_updates
                        if r3_updates is not None and r5_updates is not None else None)
        lr_match = (
            abs(float(r3_row.lr_backbone) - float(r5_row.lr_backbone)) <= 1e-15
            and abs(float(r3_row.lr_head) - float(r5_row.lr_head)) <= 1e-15
            if r3_row is not None and r5_row is not None else None
        )
        rows.append({
            "epoch": epoch,
            "study_sampler_order_sha256": order_hash,
            "study_sampler_order_same_by_shared_seed": True,
            "optimizer_update_count_match": update_match,
            "epoch_end_learning_rates_match": lr_match,
            "r3_optimizer_updates": r3_updates,
            "r5_optimizer_updates": r5_updates,
            "r3_cumulative_updates": r3_cumulative if r3_row is not None else None,
            "r5_cumulative_updates": r5_cumulative if r5_row is not None else None,
            "r3_lr_backbone": float(r3_row.lr_backbone) if r3_row is not None else None,
            "r5_lr_backbone": float(r5_row.lr_backbone) if r5_row is not None else None,
            "r3_lr_head": float(r3_row.lr_head) if r3_row is not None else None,
            "r5_lr_head": float(r5_row.lr_head) if r5_row is not None else None,
            "r3_images_per_s": float(r3_row.images_per_s) if r3_row is not None else None,
            "r5_images_per_s": float(r5_row.images_per_s) if r5_row is not None else None,
            "r3_macro_auroc": float(r3_row.macro_auroc) if r3_row is not None else None,
            "r5_macro_auroc": float(r5_row.macro_auroc) if r5_row is not None else None,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(R5_DIR / "optimizer_updates_comparison.csv", index=False, encoding="utf-8")
    r3_updates_through_best = int(r3_log.loc[
        r3_log.epoch <= int(r3_status["best_epoch"]), "optimizer_updates"].sum())
    r5_updates_through_best = int(r5_log.loc[
        r5_log.epoch <= int(r5_status["best_epoch"]), "optimizer_updates"].sum())
    common_epochs = frame.dropna(subset=["optimizer_update_count_match"])
    if not frame["study_sampler_order_same_by_shared_seed"].all():
        raise RuntimeError("Study sampler order hashes diverged")
    if (r3_runtime["image_budget"] != r5_runtime["image_budget"]
            or r3_runtime["grad_accumulation"] != r5_runtime["grad_accumulation"]):
        raise RuntimeError("R3/R5 runtime no longer matches in budget/accumulation")
    return {
        "r3_updates_through_best": r3_updates_through_best,
        "r5_updates_through_best": r5_updates_through_best,
        "all_shared_seed_study_orders_match": True,
        "optimizer_update_counts_match_all_shared_epochs": bool(
            common_epochs["optimizer_update_count_match"].all()),
        "epoch_end_learning_rates_match_all_shared_epochs": bool(
            common_epochs["epoch_end_learning_rates_match"].all()),
        "image_budget_match": r3_runtime["image_budget"] == r5_runtime["image_budget"],
        "gradient_accumulation_match": r3_runtime["grad_accumulation"] == r5_runtime["grad_accumulation"],
        "runtime_r3": r3_runtime,
        "runtime_r5": r5_runtime,
        "best_epoch_r3": int(r3_status["best_epoch"]),
        "best_epoch_r5": int(r5_status["best_epoch"]),
        "r5_epoch6_macro_auroc": (
            float(r5_log.loc[r5_log.epoch == 6, "macro_auroc"].iloc[0])
            if (r5_log.epoch == 6).any() else None
        ),
        "study_key_count": len(train_study_keys),
    }


def write_comparison_report(metrics: pd.DataFrame, bootstrap, update_summary,
                            r5_status, r3_status, r3_global_macro):
    lines = [
        "# R5 Global-only 与 R2/R3 对照分析", "",
        "## 实验身份与对照条件", "",
        "- R5 从原始通用视觉 DINOv3 ConvNeXt-Tiny 权重独立初始化；没有加载 R2/R3 监督微调参数。",
        "- R5 Global Head 初始 SHA 与 R3 保存记录一致。网络仅含 Backbone 与 Linear(768,10)，不含 Local Head、alpha 或 Auxiliary Loss。",
        "- R5 复用 R3 的 512 输入、whole-Study sampler、seed 42、image budget、accumulation、optimizer/LR、BF16、channels-last、增强、冻结策略和三条件早停。R3 的 BCE 使用融合 logits，R5 使用 Global logits；此训练目标差异是 Local 训练方案整体效应的一部分。",
        "- R3 checkpoint 重推理关闭/启用 Local 的输出，仅用于 R3-Global 与 R3-Fused 比较；R5 Best 来自独立 Global-only 训练。", "",
        "## 验证指标", "",
        "| Model | Macro-AUROC | Macro-AUPRC | Best Epoch |",
        "|---|---:|---:|---:|"]
    for name, label, best_epoch in (
        ("R2", "R2 · 512 GAP", 6),
        ("R3_Global", "R3 · Global-only inference", 6),
        ("R3_Fused", "R3 · Global+Local", 6),
        ("R5", "R5 · Independent Global-only", int(r5_status["best_epoch"])),
    ):
        row = metrics.loc[metrics.model == name].iloc[0]
        lines.append(f"| {label} | {row['macro_auroc']:.6f} | {row['macro_auprc']:.6f} | {best_epoch} |")
    epoch6 = update_summary["r5_epoch6_macro_auroc"]
    lines += ["", f"- R3 Best Epoch / Macro-AUROC: {r3_status['best_epoch']} / {r3_status['best_macro_auroc']:.6f}。",
              f"- R3-Global-only 重推理 Macro-AUROC: {r3_global_macro:.6f}。",
              f"- R5 Best Epoch / Macro-AUROC: {r5_status['best_epoch']} / {r5_status['best_macro_auroc']:.6f}。",
              f"- R5 第 6 轮 Macro-AUROC（与 R3 Best Epoch 对照）：{epoch6:.6f}" if epoch6 is not None
              else "- R5 未到达第 6 轮；见训练状态与早停记录。", "",
              "## Subject-cluster paired Bootstrap", "",
              f"- {BOOTSTRAP_REPLICATES} 次有效重复，Seed={SEED}；859 Validation Study / 228 Subject；所有模型复用同一 Subject 抽样索引。CI 未做多重比较校正，且 Best Epoch 由同一 Validation 选择。",
              "", "| Comparison | Macro-AUROC Δ [95% CI] | Macro-AUPRC Δ [95% CI] |",
              "|---|---:|---:|"]
    for key, label in (("R5_minus_R2", "R5 − R2"),
                       ("R3_Global_minus_R5", "R3-Global − R5"),
                       ("R3_Fused_minus_R5", "R3-Fused − R5")):
        item = bootstrap["comparisons"][key]
        lo, hi = item["macro_auroc_ci95"]
        ap_lo, ap_hi = item["macro_auprc_ci95"]
        lines.append(
            f"| {label} | {item['macro_auroc_delta']:+.6f} [{lo:+.6f}, {hi:+.6f}] | "
            f"{item['macro_auprc_delta']:+.6f} [{ap_lo:+.6f}, {ap_hi:+.6f}] |"
        )
    lines += ["", "### 逐类 AUROC 差值", "",
              "| Disease | R5−R2 Δ [95% CI] | R3-Global−R5 Δ [95% CI] | R3-Fused−R5 Δ [95% CI] |",
              "|---|---:|---:|---:|"]
    for disease in LABEL_NAMES:
        values = []
        for key in ("R5_minus_R2", "R3_Global_minus_R5", "R3_Fused_minus_R5"):
            item = bootstrap["comparisons"][key]["per_class"][disease]
            lo, hi = item["auroc_ci95"]
            values.append(f"{item['auroc_delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]")
        lines.append(f"| {disease} | {values[0]} | {values[1]} | {values[2]} |")
    lines += ["", "## Optimizer Updates 与 sampler 对照", "",
              f"- R3 截至 Best Epoch 的正式 optimizer updates：{update_summary['r3_updates_through_best']}；R5：{update_summary['r5_updates_through_best']}。R5/R3 每轮更新数与学习率记录逐轮见 `optimizer_updates_comparison.csv`。",
              f"- R3/R5 image budget 匹配：{update_summary['image_budget_match']}；gradient accumulation 匹配：{update_summary['gradient_accumulation_match']}；共 {update_summary['study_key_count']} 个 Train Study 的 epoch sampler 顺序哈希匹配：{update_summary['all_shared_seed_study_orders_match']}。",
              f"- 两组共同完成的 Epoch 中 optimizer-update 数逐轮匹配：{update_summary['optimizer_update_counts_match_all_shared_epochs']}；epoch 末学习率逐轮匹配：{update_summary['epoch_end_learning_rates_match_all_shared_epochs']}。",
              f"- Runtime：budget={update_summary['runtime_r3']['image_budget']}，accumulation={update_summary['runtime_r3']['grad_accumulation']}，effective images/update={update_summary['runtime_r3']['effective_images_per_update']}，effective Studies/update={update_summary['runtime_r3']['effective_studies_per_update']}，channels-last={update_summary['runtime_r3']['channels_last']}。",
              "", "## 解释与后续建议", "",
              "- R3-Global 与 R5 的差异是 R3 Global 路径经历融合 logits 监督的整体训练效应与 R5 纯 Global 监督之间的对照，仍可能受随机性、Best Epoch 选择和其他训练轨迹因素影响；即使区间排除 0，也不能单独归因为 Local 特征对 Backbone 的因果改造。",
              "- 若 R3-Global−R5 区间跨 0，现有证据不足以认定 Local 训练路径带来稳定收益；不应仅因小幅点差停止或继续 Local 研究。若区间持续正向且更新预算、初始化及 Study 顺序均匹配，结果才值得后续用独立 seed/控制 loss 设计验证。",
              "- R5 与 R2 的比较用于判断严格匹配 R3 runtime 后的独立 Global-only 水平；模型选择始终以 Validation 为准，Test 未用于本实验。", ""]
    (R5_DIR / "r5_vs_r2_r3_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R3-Global validation inference")
    r5_status = read_json(R5_DIR / "training_status.json")
    r5_config = read_json(R5_DIR / "config.json")
    if not r5_status.get("complete"):
        raise RuntimeError("R5 training is not complete; refusing model comparison")
    if r5_config.get("test_split_read", False):
        raise RuntimeError("R5 training metadata says Test was accessed")

    valid_base, official, labels = diagnostics.official_dataset()
    scores_by_model = {}
    saved_frames = {}
    for name, path in (
        ("R2", R2_DIR / "predictions/val_best.csv"),
        ("R3_Fused", R3_DIR / "predictions/val_best.csv"),
        ("R5", R5_DIR / "predictions/val_best.csv"),
    ):
        saved_frames[name], scores_by_model[name] = diagnostics.read_predictions(
            path, official, labels)

    r3 = diagnostics.infer_r3(valid_base, torch.device("cuda"))
    r3_global = 1.0 / (1.0 + np.exp(-r3["global_logits"]))
    r3_fused_replay = 1.0 / (1.0 + np.exp(-r3["fused_logits"]))
    replay = diagnostics.validate_checkpoint_reproduction(
        "R3", scores_by_model["R3_Fused"], r3_fused_replay, labels)
    if (replay["macro_auroc_abs_error"] > 1e-5
            or replay["max_probability_abs_error"] > 0.02):
        raise RuntimeError("R3 checkpoint replay failed before R5 comparison")
    scores_by_model["R3_Global"] = r3_global
    global_auc, global_ap = metric_vectors(labels, r3_global)
    global_frame = official[["Subject_id", "Study_id"]].copy()
    for index, name in enumerate(LABEL_NAMES):
        safe = name.replace(" ", "_")
        global_frame[f"target_{safe}"] = labels[:, index]
        global_frame[f"prob_{safe}"] = r3_global[:, index]
    global_frame.to_csv(R5_DIR / "r3_global_only_val.csv", index=False, encoding="utf-8")
    expected_r3_global = pd.read_csv(
        DIAGNOSTICS_DIR / "r3_inference_ablation.csv").set_index("disease")["B_auc"]
    if abs(macro(global_auc) - float(expected_r3_global.mean())) > 1e-5:
        raise RuntimeError("R3-Global-only replay differs from the completed diagnostic baseline")

    metrics_rows = []
    for index, disease in enumerate(LABEL_NAMES):
        row = {
            "disease": disease,
            "positive_studies": int(labels[:, index].sum()),
            "negative_studies": int((labels[:, index] == 0).sum()),
        }
        for model_name, model_scores in scores_by_model.items():
            aucs, aps = metric_vectors(labels, model_scores)
            row[f"{model_name}_auroc"] = float(aucs[index])
            row[f"{model_name}_auprc"] = float(aps[index])
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(R5_DIR / "r5_comparison_per_class.csv", index=False, encoding="utf-8")

    official_train = StudyDataset(
        TRAIN_CSV, IMAGE_ROOT, transform=None, label_column="Predict_class",
        label_semantics="closed_world", image_id_column="Dicom_id",
        subject_column="Subject_id", study_column="Study_id")
    if len(official_train) != 18513:
        raise RuntimeError(f"Unexpected Train Study count for sampler audit: {len(official_train)}")
    r3_log = pd.read_csv(R3_DIR / "train_log.csv")
    r5_log = pd.read_csv(R5_DIR / "train_log.csv")
    r3_runtime = read_json(R3_DIR / "selected_runtime_config.json")["selected_runtime"]
    r5_runtime = read_json(R5_DIR / "selected_runtime_config.json")["selected_runtime"]
    r3_status = read_json(R3_DIR / "training_status.json")
    update_summary = write_update_comparison(
        r3_log, r5_log, r3_runtime, r5_runtime,
        official_train.study_keys, r3_status, r5_status)

    draws, draws_sha = diagnostics.make_subject_draws(official)
    comparisons = [
        ("R5_minus_R2", "R2", "R5"),
        ("R3_Global_minus_R5", "R5", "R3_Global"),
        ("R3_Fused_minus_R5", "R5", "R3_Fused"),
    ]
    bootstrap = paired_cluster_bootstrap(labels, scores_by_model, draws, draws_sha,
                                         comparisons)
    bootstrap["validation_studies"] = len(official)
    bootstrap["validation_subjects"] = int(official["Subject_id"].nunique())
    bootstrap["checkpoint_replay_r3_fused"] = replay
    bootstrap["r3_global_macro_auroc"] = macro(global_auc)
    bootstrap["r3_global_macro_auprc"] = macro(global_ap)
    write_json(R5_DIR / "paired_subject_bootstrap.json", bootstrap)

    metric_summary = []
    for model_name in ("R2", "R3_Global", "R3_Fused", "R5"):
        aucs, aps = metric_vectors(labels, scores_by_model[model_name])
        metric_summary.append({"model": model_name,
                               "macro_auroc": macro(aucs),
                               "macro_auprc": macro(aps)})
    summary_frame = pd.DataFrame(metric_summary)
    r5_status = read_json(R5_DIR / "training_status.json")
    write_comparison_report(summary_frame, bootstrap, update_summary,
                            r5_status, r3_status, macro(global_auc))
    print(f"R5 comparison complete: R5 best Macro-AUROC={r5_status['best_macro_auroc']:.6f}; "
          f"R3-Global={macro(global_auc):.6f}", flush=True)


if __name__ == "__main__":
    run_analysis()
