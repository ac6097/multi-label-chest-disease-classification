"""Paired Subject-cluster comparison for the planned DenseNet attention matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


MODEL_PREDICTIONS = {
    "B0": PROJECT_DIR / "outputs/predictions/val_best.csv",
    "B1": PROJECT_DIR / "outputs/experiments/dn121_low_lr_gap/outputs/predictions/val_best.csv",
    "A1_LCA": PROJECT_DIR / "outputs/experiments/dn121_low_lr_lca/outputs/predictions/val_best.csv",
    "A2_CBAM": PROJECT_DIR / "outputs/experiments/dn121_low_lr_cbam/outputs/predictions/val_best.csv",
}
MODEL_CHECKPOINTS = {
    "B0": PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth",
    "B1": PROJECT_DIR / "outputs/experiments/dn121_low_lr_gap/checkpoint/best/best_by_macro_auc.pth",
    "A1_LCA": PROJECT_DIR / "outputs/experiments/dn121_low_lr_lca/checkpoint/best/best_by_macro_auc.pth",
    "A2_CBAM": PROJECT_DIR / "outputs/experiments/dn121_low_lr_cbam/checkpoint/best/best_by_macro_auc.pth",
}
PAIRS = [
    ("B1_minus_B0", "B0", "B1"),
    ("A1_LCA_minus_B1", "B1", "A1_LCA"),
    ("A2_CBAM_minus_B1", "B1", "A2_CBAM"),
    ("A1_LCA_minus_A2_CBAM", "A2_CBAM", "A1_LCA"),
]


def _load_predictions(path: Path, class_names: list[str]) -> pd.DataFrame:
    keys = ["Subject_id", "Study_id"]
    frame = pd.read_csv(path, dtype={key: "string" for key in keys})
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate Study keys in {path}")
    known = [f"known_{name}" for name in class_names]
    targets = [f"true_{name}" for name in class_names]
    probabilities = [f"prob_{name}" for name in class_names]
    if not frame[known].astype(str).apply(
        lambda column: column.str.lower().eq("true").all()
    ).all():
        raise ValueError(f"Validation labels are incomplete in {path}")
    values = frame[probabilities].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Invalid probabilities in {path}")
    frame = frame.sort_values(keys).reset_index(drop=True)
    frame.attrs["targets"] = frame[targets].to_numpy(dtype=np.int8)
    frame.attrs["probabilities"] = frame[probabilities].to_numpy(dtype=np.float64)
    return frame


def _class_metrics(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    auroc, auprc = [], []
    for index in range(targets.shape[1]):
        labels = targets[:, index].astype(int)
        if labels.min() == labels.max():
            raise ValueError(f"Bootstrap sample has one class only at label index {index}")
        auroc.append(float(roc_auc_score(labels, probabilities[:, index])))
        auprc.append(float(average_precision_score(labels, probabilities[:, index])))
    return np.asarray(auroc), np.asarray(auprc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "outputs/experiments/dn121_low_lr_comparison"),
    )
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive")

    data_config_path = PROJECT_DIR / "configs/data.yaml"
    data_cfg = resolve_data_paths(read_yaml(data_config_path), data_config_path)
    class_names = list(data_cfg["class_names"])
    frames = {
        name: _load_predictions(path, class_names)
        for name, path in MODEL_PREDICTIONS.items()
    }
    reference = frames["B0"]
    keys = ["Subject_id", "Study_id"]
    base_targets = reference.attrs["targets"]
    subject_values = reference["Subject_id"].to_numpy()
    unique_subjects = pd.unique(subject_values)
    groups = {
        subject: np.flatnonzero(subject_values == subject)
        for subject in unique_subjects
    }
    if not np.isfinite(base_targets).all():
        raise ValueError("B0 target matrix contains non-finite values")

    model_metrics = {}
    class_metric_rows = []
    for name, frame in frames.items():
        if not frame[keys].equals(reference[keys]):
            raise ValueError(f"{name} validation Study keys differ from B0")
        targets = frame.attrs["targets"]
        if not np.array_equal(targets, base_targets):
            raise ValueError(f"{name} validation targets differ from B0")
        probabilities = frame.attrs["probabilities"]
        aucs, aps = _class_metrics(targets, probabilities)
        model_metrics[name] = {
            "predictions": str(MODEL_PREDICTIONS[name]),
            "checkpoint": str(MODEL_CHECKPOINTS[name]),
            "n_studies": int(len(frame)),
            "n_subjects": int(len(unique_subjects)),
            "macro_auroc": float(np.mean(aucs)),
            "per_class": [
                {
                    "class_name": class_names[index],
                    "n_positive": int(base_targets[:, index].sum()),
                    "n_negative": int(len(base_targets) - base_targets[:, index].sum()),
                    "auroc": float(aucs[index]),
                    "auprc": float(aps[index]),
                }
                for index in range(len(class_names))
            ],
        }
        class_metric_rows.extend(
            {
                "model": name,
                "class_name": class_names[index],
                "n_positive": int(base_targets[:, index].sum()),
                "n_negative": int(len(base_targets) - base_targets[:, index].sum()),
                "auroc": float(aucs[index]),
                "auprc": float(aps[index]),
            }
            for index in range(len(class_names))
        )

    probability_matrices = {
        name: frame.attrs["probabilities"]
        for name, frame in frames.items()
    }
    rng = np.random.default_rng(args.seed)
    bootstraps = {pair[0]: {"macro": [], "classes": []} for pair in PAIRS}
    for _ in range(args.bootstrap_repetitions):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        row_indices = np.concatenate([groups[subject] for subject in sampled_subjects])
        try:
            aucs_by_model = {
                name: _class_metrics(
                    base_targets[row_indices],
                    probability_matrices[name][row_indices],
                )[0]
                for name in frames
            }
        except ValueError:
            continue
        for pair_name, reference_name, candidate_name in PAIRS:
            deltas = aucs_by_model[candidate_name] - aucs_by_model[reference_name]
            bootstraps[pair_name]["classes"].append(deltas)
            bootstraps[pair_name]["macro"].append(float(np.mean(deltas)))

    comparison_results = []
    for pair_name, reference_name, candidate_name in PAIRS:
        deltas = np.asarray(bootstraps[pair_name]["classes"], dtype=np.float64)
        macro = np.asarray(bootstraps[pair_name]["macro"], dtype=np.float64)
        if len(macro) == 0:
            raise RuntimeError(f"No valid bootstrap samples for {pair_name}")
        ref_metrics = model_metrics[reference_name]
        cand_metrics = model_metrics[candidate_name]
        comparison_results.append(
            {
                "comparison": pair_name,
                "reference": reference_name,
                "candidate": candidate_name,
                "reference_macro_auroc": ref_metrics["macro_auroc"],
                "candidate_macro_auroc": cand_metrics["macro_auroc"],
                "macro_auroc_delta": cand_metrics["macro_auroc"] - ref_metrics["macro_auroc"],
                "macro_auroc_delta_subject_bootstrap_95ci": [
                    float(np.percentile(macro, 2.5)),
                    float(np.percentile(macro, 97.5)),
                ],
                "bootstrap_repetitions_requested": args.bootstrap_repetitions,
                "bootstrap_repetitions_valid": int(len(macro)),
                "per_class": [
                    {
                        "class_name": class_names[index],
                        "reference_auroc": ref_metrics["per_class"][index]["auroc"],
                        "candidate_auroc": cand_metrics["per_class"][index]["auroc"],
                        "delta_auroc": (
                            cand_metrics["per_class"][index]["auroc"]
                            - ref_metrics["per_class"][index]["auroc"]
                        ),
                        "reference_auprc": ref_metrics["per_class"][index]["auprc"],
                        "candidate_auprc": cand_metrics["per_class"][index]["auprc"],
                        "delta_auprc": (
                            cand_metrics["per_class"][index]["auprc"]
                            - ref_metrics["per_class"][index]["auprc"]
                        ),
                        "delta_auroc_subject_bootstrap_95ci": [
                            float(np.percentile(deltas[:, index], 2.5)),
                            float(np.percentile(deltas[:, index], 97.5)),
                        ],
                    }
                    for index in range(len(class_names))
                ],
            }
        )

    output_dir = resolve_config_path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "validation_csv": str(data_cfg["val_csv"]),
        "n_studies": int(len(reference)),
        "n_subjects": int(len(unique_subjects)),
        "seed": args.seed,
        "model_metrics": model_metrics,
        "comparisons": comparison_results,
    }
    (output_dir / "subject_bootstrap_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(class_metric_rows).to_csv(
        output_dir / "per_class_metrics.csv", index=False, encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "n_studies": report["n_studies"],
        "n_subjects": report["n_subjects"],
        "comparisons": [
            {
                "comparison": item["comparison"],
                "macro_auroc_delta": item["macro_auroc_delta"],
                "ci95": item["macro_auroc_delta_subject_bootstrap_95ci"],
                "valid_repetitions": item["bootstrap_repetitions_valid"],
            }
            for item in comparison_results
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
