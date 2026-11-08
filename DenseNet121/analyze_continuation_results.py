"""Summarize paired BCE continuations from their logs and saved Study predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


CLASS_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]
KEYS = ["Subject_id", "Study_id"]
LOG_SUMMARY = re.compile(
    r"epoch=(\d+)/(\d+) train_loss=([^ ]+) val_loss=([^ ]+) macro_auroc=([^ ]+) "
    r"macro_f1=([^ ]+) gradient_norm=([^ ]+) lr=(\[[^]]+\]) peak_gpu_gb=([^ ]+) "
    r"train_s=([^ ]+) val_s=([^ ]+)"
)
LOG_CLASS = re.compile(
    r"label\[(.+)\] valid=(\d+) positive=(\d+) negative=(\d+) "
    r"auroc=([^ ]+) auprc=([^ ]+) f1=([^ ]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    values = []
    for class_index in range(targets.shape[1]):
        labels = targets[:, class_index].astype(int)
        if labels.min() == labels.max():
            continue
        values.append(float(roc_auc_score(labels, probabilities[:, class_index])))
    return float(np.mean(values)) if values else float("nan")


def prediction_metrics(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, list[dict]]:
    per_class = []
    aucs, f1s = [], []
    for index, name in enumerate(CLASS_NAMES):
        labels = targets[:, index].astype(int)
        scores = probabilities[:, index]
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        auc = float(roc_auc_score(labels, scores)) if positives and negatives else float("nan")
        ap = float(average_precision_score(labels, scores)) if positives else float("nan")
        f1 = float(f1_score(labels, scores >= 0.5, zero_division=0))
        predicted = scores >= 0.5
        tp = int(((labels == 1) & predicted).sum())
        fp = int(((labels == 0) & predicted).sum())
        fn = int(((labels == 1) & ~predicted).sum())
        if np.isfinite(auc):
            aucs.append(auc)
        f1s.append(f1)
        per_class.append({
            "class_id": index + 1, "class_name": name,
            "n_positive": positives, "n_negative": negatives,
            "auroc": auc, "auprc": ap, "f1_at_0_5": f1, "threshold": 0.5,
            "true_positive_at_0_5": tp, "false_positive_at_0_5": fp,
            "false_negative_at_0_5": fn,
        })
    return (float(np.mean(aucs)) if aucs else float("nan"),
            float(np.mean(f1s)) if f1s else float("nan"), per_class)


def load_saved_prediction(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Subject_id": "string", "Study_id": "string"})
    frame["Subject_id"] = frame["Subject_id"].astype(str)
    frame["Study_id"] = frame["Study_id"].astype(str)
    targets, probabilities = [], []
    for name in CLASS_NAMES:
        if not frame[f"known_{name}"].astype(str).str.lower().eq("true").all():
            raise ValueError(f"Incomplete labels in {path} for {name}")
        targets.append(frame[f"true_{name}"].to_numpy(float))
        probabilities.append(frame[f"prob_{name}"].to_numpy(float))
    target_matrix = np.column_stack(targets)
    probability_matrix = np.column_stack(probabilities)
    if not np.isfinite(probability_matrix).all():
        raise ValueError(f"Non-finite probability in {path}")
    return pd.DataFrame({
        "Subject_id": frame["Subject_id"], "Study_id": frame["Study_id"],
        **{f"y_{i + 1}": target_matrix[:, i] for i in range(10)},
        **{f"{method}_{i + 1}": probability_matrix[:, i] for i in range(10)},
    })


def read_training_log(path: Path, method: str) -> tuple[list[dict], list[dict]]:
    epochs, classes = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        summary_match = LOG_SUMMARY.search(line)
        if summary_match:
            epoch, total_epochs, train_loss, val_loss, macro_auc_value, macro_f1, gradient_norm, lr, peak_gpu, train_s, val_s = summary_match.groups()
            epochs.append({
                "method": method, "epoch": int(epoch), "max_epochs": int(total_epochs),
                "train_loss": float(train_loss), "val_loss": float(val_loss),
                "macro_auroc": float(macro_auc_value), "macro_f1_at_0_5": float(macro_f1),
                "gradient_norm": float(gradient_norm), "learning_rates": lr,
                "peak_gpu_gb": float(peak_gpu), "train_seconds": float(train_s),
                "validation_seconds": float(val_s),
            })
        class_match = LOG_CLASS.search(line)
        if class_match:
            name, valid, positives, negatives, auc, ap, f1 = class_match.groups()
            classes.append({
                "method": method, "epoch": len(epochs), "class_name": name,
                "n_valid": int(valid), "n_positive": int(positives), "n_negative": int(negatives),
                "auroc": float(auc), "auprc": float(ap), "f1_at_0_5": float(f1), "threshold": 0.5,
            })
    return epochs, classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs/experiments/paired_analysis"))
    args = parser.parse_args()
    output_dir = resolve_config_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_DIR / "configs/data.yaml"
    data = resolve_data_paths(read_yaml(data_path), data_path)
    validation = pd.read_csv(data["val_csv"], dtype={"Subject_id": "string", "Study_id": "string"})
    study_metadata = (validation.groupby(KEYS, sort=False).size().rename("n_images").reset_index())
    study_metadata["Subject_id"] = study_metadata["Subject_id"].astype(str)
    study_metadata["Study_id"] = study_metadata["Study_id"].astype(str)

    diagnostics = PROJECT_DIR / "outputs/experiments/densenet_epoch6_diagnostics"
    base = pd.read_csv(diagnostics / "epoch6_study_predictions.csv",
                       dtype={"Subject_id": "string", "Study_id": "string"})
    base["Subject_id"] = base["Subject_id"].astype(str)
    base["Study_id"] = base["Study_id"].astype(str)
    base = base[KEYS + [f"y_{i}" for i in range(1, 11)] +
                [f"logit_mean_{i}" for i in range(1, 11)]].rename(
                    columns={f"logit_mean_{i}": f"base_{i}" for i in range(1, 11)})

    experiment_specs = [
        ("standard_bce", PROJECT_DIR / "outputs/experiments/paired_standard_bce"),
        ("sqrt_weighted_bce", PROJECT_DIR / "outputs/experiments/paired_sqrt_weighted_bce"),
    ]
    frames = {"base": base}
    summaries, class_metrics, epoch_metrics, epoch_class_metrics = [], [], [], []
    manifest = {"validation_studies": len(base), "validation_csv": str(data["val_csv"]),
                "bootstrap_repetitions": int(args.bootstrap_repetitions), "runs": {}}

    base_y = base[[f"y_{i}" for i in range(1, 11)]].to_numpy(float)
    base_p = base[[f"base_{i}" for i in range(1, 11)]].to_numpy(float)
    base_auc, base_f1, base_per_class = prediction_metrics(base_y, base_p)
    summaries.append({"method": "epoch6_base", "best_epoch": 6, "epochs_run": 0,
                      "macro_auroc": base_auc, "macro_f1_at_0_5": base_f1,
                      "checkpoint": "DenseNet121/checkpoint/best/best_by_macro_auc.pth",
                      "checkpoint_sha256": sha256(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth")})
    for item in base_per_class:
        class_metrics.append({"method": "epoch6_base", **item})

    for method, run_dir in experiment_specs:
        best_json_path = run_dir / "checkpoint/best/best_by_macro_auc.json"
        best_json = json.loads(best_json_path.read_text(encoding="utf-8"))
        best_epoch = int(best_json["epoch"])
        best_checkpoint = run_dir / "checkpoint/best/best_by_macro_auc.pth"
        prediction_path = run_dir / "outputs/predictions/val_best.csv"
        log_path = run_dir / "outputs/logs/train.log"
        frame = load_saved_prediction(prediction_path, method)
        frame = frame.merge(study_metadata, on=KEYS, validate="one_to_one", how="inner")
        if set(map(tuple, frame[KEYS].itertuples(index=False, name=None))) != set(
                map(tuple, base[KEYS].itertuples(index=False, name=None))):
            raise ValueError(f"Study keys differ from the validation baseline for {method}")
        frame = frame.set_index(KEYS).loc[base.set_index(KEYS).index].reset_index()
        frames[method] = frame
        targets = frame[[f"y_{i}" for i in range(1, 11)]].to_numpy(float)
        probabilities = frame[[f"{method}_{i}" for i in range(1, 11)]].to_numpy(float)
        macro_auc_value, macro_f1_value, per_class = prediction_metrics(targets, probabilities)
        epochs, logged_classes = read_training_log(log_path, method)
        epoch_metrics.extend(epochs)
        epoch_class_metrics.extend(logged_classes)
        summaries.append({
            "method": method, "best_epoch": best_epoch, "epochs_run": len(epochs),
            "macro_auroc": macro_auc_value, "macro_f1_at_0_5": macro_f1_value,
            "best_metric_in_checkpoint": best_json["best_macro_auroc"],
            "checkpoint": str(best_checkpoint), "checkpoint_sha256": sha256(best_checkpoint),
            "checkpoint_bytes": best_checkpoint.stat().st_size,
            "init_checkpoint": best_json["config"]["initial_checkpoint"],
            "init_checkpoint_sha256": best_json["config"]["initial_checkpoint_sha256"],
            "loss": best_json["config"]["loss"],
            "weight_counts": best_json["config"].get("weight_counts"),
            "training_log": str(log_path),
        })
        for item in per_class:
            class_metrics.append({"method": method, **item})
        for group_type, groups in (
            ("image_count", frame["n_images"].astype(str)),
            ("single_vs_multi", frame["n_images"].map(
                lambda count: "single_image" if count == 1 else "multi_image")),
        ):
            for group in sorted(groups.unique(), key=str):
                mask = groups.to_numpy() == group
                y_group = targets[mask]
                p_group = probabilities[mask]
                group_auc = macro_auc(y_group, p_group)
                group_f1 = float(np.mean([
                    f1_score(y_group[:, index], p_group[:, index] >= 0.5, zero_division=0)
                    for index in range(10)
                ]))
                summaries.append({"method": method, "group_type": group_type, "group": group,
                                  "n_studies": int(mask.sum()), "macro_auroc": group_auc,
                                  "macro_f1_at_0_5": group_f1})
        manifest["runs"][method] = {
            "best_checkpoint": str(best_checkpoint), "best_checkpoint_sha256": sha256(best_checkpoint),
            "best_epoch": best_epoch, "epochs_run": len(epochs),
            "initial_checkpoint": best_json["config"]["initial_checkpoint"],
            "initial_checkpoint_sha256": best_json["config"]["initial_checkpoint_sha256"],
            "loss": best_json["config"]["loss"], "weight_counts": best_json["config"].get("weight_counts"),
            "config": best_json["config"], "log_sha256": sha256(log_path),
        }

    # Paired bootstrap resamples whole Subjects and reuses the same sampled indices for all runs.
    subject_ids = base["Subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subject_ids)
    subject_indices = {subject: np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    rng = np.random.default_rng(42)
    bootstrapped = {"epoch6_base": [], "standard_bce": [], "sqrt_weighted_bce": [],
                    "weighted_minus_standard": [], "weighted_minus_base": [], "standard_minus_base": []}
    for _ in range(max(int(args.bootstrap_repetitions), 0)):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in sampled])
        y = base_y[indices]
        auc_base = macro_auc(y, base_p[indices])
        auc_standard = macro_auc(y, frames["standard_bce"][[f"standard_bce_{i}" for i in range(1, 11)]].to_numpy(float)[indices])
        auc_weighted = macro_auc(y, frames["sqrt_weighted_bce"][[f"sqrt_weighted_bce_{i}" for i in range(1, 11)]].to_numpy(float)[indices])
        bootstrapped["epoch6_base"].append(auc_base)
        bootstrapped["standard_bce"].append(auc_standard)
        bootstrapped["sqrt_weighted_bce"].append(auc_weighted)
        bootstrapped["weighted_minus_standard"].append(auc_weighted - auc_standard)
        bootstrapped["weighted_minus_base"].append(auc_weighted - auc_base)
        bootstrapped["standard_minus_base"].append(auc_standard - auc_base)
    bootstrap_rows = []
    point_estimates = {
        "epoch6_base": base_auc,
        "standard_bce": float(next(item["macro_auroc"] for item in summaries if item.get("method") == "standard_bce")),
        "sqrt_weighted_bce": float(next(item["macro_auroc"] for item in summaries if item.get("method") == "sqrt_weighted_bce")),
        "weighted_minus_standard": float(next(item["macro_auroc"] for item in summaries if item.get("method") == "sqrt_weighted_bce") -
                                          next(item["macro_auroc"] for item in summaries if item.get("method") == "standard_bce")),
        "weighted_minus_base": float(next(item["macro_auroc"] for item in summaries if item.get("method") == "sqrt_weighted_bce") - base_auc),
        "standard_minus_base": float(next(item["macro_auroc"] for item in summaries if item.get("method") == "standard_bce") - base_auc),
    }
    for comparison, values in bootstrapped.items():
        if values:
            bootstrap_rows.append({
                "comparison": comparison, "point_estimate": point_estimates[comparison],
                "bootstrap_mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "repetitions": len(values), "cluster": "Subject_id",
            })

    epoch_metrics_df = pd.DataFrame(epoch_metrics)
    epoch_metrics_df.to_csv(output_dir / "continuation_epoch_metrics.csv", index=False)
    pd.DataFrame(epoch_class_metrics).to_csv(output_dir / "continuation_epoch_class_metrics.csv", index=False)
    pd.DataFrame(class_metrics).to_csv(output_dir / "continuation_class_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "continuation_summary_and_groups.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "continuation_subject_bootstrap.csv", index=False)
    figure_dir = PROJECT_DIR.parent / "docs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, label in (("standard_bce", "Standard BCE"), ("sqrt_weighted_bce", "Sqrt Weighted BCE")):
        run_epochs = epoch_metrics_df[epoch_metrics_df["method"] == method].sort_values("epoch")
        axes[0].plot(run_epochs["epoch"], run_epochs["macro_auroc"], marker="o", label=label)
        first_val_loss = float(run_epochs.iloc[0]["val_loss"])
        axes[1].plot(run_epochs["epoch"], run_epochs["val_loss"] / first_val_loss,
                     marker="o", label=label)
    axes[0].axhline(base_auc, color="black", linestyle="--", linewidth=1,
                    label="Epoch 6 base")
    axes[0].set_title("Validation Macro-AUROC")
    axes[0].set_xlabel("Continuation epoch")
    axes[0].set_ylabel("Study Macro-AUROC")
    axes[0].set_ylim(0.74, 0.81)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Validation loss, normalized within each run")
    axes[1].set_xlabel("Continuation epoch")
    axes[1].set_ylabel("Validation loss / epoch 1 loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "densenet_paired_continuation_curves.png", dpi=160)
    plt.close(fig)
    try:
        manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=PROJECT_DIR.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_head"] = None
    manifest["train_py_sha256"] = sha256(PROJECT_DIR / "train.py")
    manifest["paired_configs"] = {
        "standard_bce": str(PROJECT_DIR / "configs/experiments/paired_standard_bce.yaml"),
        "sqrt_weighted_bce": str(PROJECT_DIR / "configs/experiments/paired_sqrt_weighted_bce.yaml"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    summary_frame = pd.DataFrame(summaries)
    headline = summary_frame[summary_frame["group_type"].isna()]
    print(json.dumps({
        "summary": headline.to_dict(orient="records"),
        "subject_bootstrap": bootstrap_rows,
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
