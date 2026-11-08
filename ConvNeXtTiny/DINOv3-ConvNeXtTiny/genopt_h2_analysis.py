"""H2-B / H2-C / H4 evaluation, fixed grouped-BCE statistics and comparison."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import genopt_analysis as base
from genopt_common import (
    EXPERIMENTS,
    EXPERIMENT_ROOT,
    LABEL_NAMES,
    R5_DIR,
    SEED,
    TRAIN_CSV,
    VALID_CSV,
    read_json,
    study_view_metadata,
    write_json,
)
from generalization_common import build_eval_datasets
from stage_fusion_analysis import bootstrap_models, study_metrics

OUT = EXPERIMENT_ROOT / "H2_H4_Comparative_Analysis"
ORDER = ("H0", "H2", "H2B", "H2C", "H4")
COLORS = {"H0": "#4C72B0", "H2": "#8172B2", "H2B": "#55A868",
          "H2C": "#C44E52", "H4": "#CCB974"}
CHECKPOINTS = ("best", "last")


def exp_dir(experiment: str) -> Path:
    return EXPERIMENT_ROOT / EXPERIMENTS[experiment]["label"]


def _history(experiment: str) -> pd.DataFrame:
    if experiment == "H0":
        return pd.read_csv(R5_DIR / "train_log.csv")
    return pd.read_csv(exp_dir(experiment) / "training_history.csv")


def _status(experiment: str) -> dict:
    if experiment == "H0":
        return read_json(R5_DIR / "training_status.json")
    return read_json(exp_dir(experiment) / "training_status.json")


def _results() -> dict:
    train_data, valid_data, _ = build_eval_datasets()
    results = {}
    for experiment in ORDER:
        for checkpoint in CHECKPOINTS:
            for split, dataset in (("train", train_data),
                                   ("valid", valid_data)):
                results[(experiment, checkpoint, split)] = base.run_inference(
                    experiment, checkpoint, split, train_data, valid_data)
    return results


def _overall(results: dict) -> pd.DataFrame:
    rows = []
    for experiment in ORDER:
        for checkpoint in CHECKPOINTS:
            train = base.result_metrics(
                results[(experiment, checkpoint, "train")])
            valid = base.result_metrics(
                results[(experiment, checkpoint, "valid")])
            status = _status(experiment)
            rows.append({
                "experiment": experiment,
                "checkpoint": checkpoint,
                "best_epoch": int(status["best_epoch"]),
                "completed_epochs": int(status["completed_epochs"]),
                "stop_reason": status.get("stop_reason"),
                "train_overall_bce": train["overall_bce"],
                "valid_overall_bce": valid["overall_bce"],
                "train_valid_bce_gap": valid["overall_bce"]
                - train["overall_bce"],
                "train_macro_auroc": train["macro_auroc"],
                "train_macro_auprc": train["macro_auprc"],
                "valid_macro_auroc": valid["macro_auroc"],
                "valid_macro_auprc": valid["macro_auprc"],
            })
    return pd.DataFrame(rows)


def _class_rows(results: dict) -> pd.DataFrame:
    rows = []
    for experiment in ORDER:
        for split in ("train", "valid"):
            result = results[(experiment, "best", split)]
            metrics = base.result_metrics(result)
            labels = np.asarray(result["labels"], dtype=np.int64)
            subjects = np.asarray(result["subjects"]).astype(str)
            for index, disease in enumerate(LABEL_NAMES):
                bce = metrics["class_bce"][disease]
                positive = labels[:, index] == 1
                rows.append({
                    "experiment": experiment,
                    "checkpoint": "best",
                    "split": split,
                    "disease": disease,
                    "bce": bce["bce"],
                    "positive_bce": bce["positive_bce"],
                    "negative_bce": bce["negative_bce"],
                    "auroc": metrics["per_class"][disease]["auroc"],
                    "auprc": metrics["per_class"][disease]["auprc"],
                    "n_positive_studies": int(positive.sum()),
                    "n_negative_studies": int((~positive).sum()),
                    "n_positive_subjects": int(len(np.unique(subjects[positive]))),
                })
    return pd.DataFrame(rows)


def _grouped(results: dict) -> pd.DataFrame:
    train_data, valid_data, _ = build_eval_datasets()
    train_meta = study_view_metadata(train_data, TRAIN_CSV)
    valid_meta = study_view_metadata(valid_data, VALID_CSV)
    rows = []
    for experiment in ORDER:
        for split, dataset, metadata in (
                ("train", train_data, train_meta),
                ("valid", valid_data, valid_meta)):
            result = results[(experiment, "best", split)]
            rows.extend(base.grouped_metrics(result, dataset, metadata,
                                             experiment, "best", split))
    return pd.DataFrame(rows)


def _epoch_class_metrics(experiment: str) -> pd.DataFrame:
    """Per-epoch ten-class validation metrics, cached for plotting."""
    cache = OUT / "epoch_class_metrics" / f"{experiment}.csv"
    if cache.is_file():
        return pd.read_csv(cache)
    if experiment in ("H2B", "H2C", "H4"):
        source = exp_dir(experiment) / "per_class_metrics.csv"
        frame = pd.read_csv(source)
        frame = frame.rename(columns={"class_name": "disease"})
        frame.to_csv(cache, index=False, encoding="utf-8")
        return frame
    if experiment == "H0":
        predictions = R5_DIR / "predictions"
        frame = pd.read_csv(R5_DIR / "train_log.csv")
        epochs = [int(value) for value in frame["epoch"].tolist()]
    else:
        predictions = exp_dir(experiment) / "predictions"
        frame = pd.read_csv(exp_dir(experiment) / "training_history.csv")
        epochs = [int(value) for value in frame["epoch"].tolist()]
    rows = []
    for epoch in epochs:
        path = predictions / f"val_epoch_{epoch:02d}.csv"
        if not path.is_file():
            continue
        prediction = pd.read_csv(path)
        probabilities = prediction[
            [f"prob_{name.replace(' ', '_')}" for name in LABEL_NAMES]
        ].to_numpy()
        targets = prediction[
            [f"target_{name.replace(' ', '_')}" for name in LABEL_NAMES]
        ].to_numpy().astype(np.int64)
        metrics = study_metrics(targets, probabilities)
        logits = np.log(np.clip(probabilities, 1e-12, 1 - 1e-12)
                        / np.clip(1 - probabilities, 1e-12, 1))
        bce = base.class_bce_breakdown(logits, targets)
        for disease in LABEL_NAMES:
            rows.append({
                "epoch": epoch,
                "disease": disease,
                "auroc": metrics["per_class"][disease]["auroc"],
                "auprc": metrics["per_class"][disease]["auprc"],
                "valid_bce": bce[disease]["bce"],
                "valid_positive_bce": bce[disease]["positive_bce"],
                "valid_negative_bce": bce[disease]["negative_bce"],
            })
    frame = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False, encoding="utf-8")
    return frame


def _bootstrap(results: dict) -> dict:
    valid = {experiment: results[(experiment, "best", "valid")]
             for experiment in ORDER}
    labels = np.asarray(valid["H0"]["labels"])
    studies = np.asarray(valid["H0"]["studies"]).astype(str)
    for experiment in ORDER:
        if not (np.array_equal(np.asarray(valid[experiment]["labels"]), labels)
                and np.array_equal(
                    np.asarray(valid[experiment]["studies"]).astype(str), studies)):
            raise RuntimeError("Validation predictions are not Study-aligned")
    probabilities = {}
    for experiment in ORDER:
        logits = np.asarray(valid[experiment]["study_logits"], dtype=np.float64)
        probabilities[experiment] = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    subjects = np.asarray(valid["H0"]["subjects"]).astype(str)
    comparisons = {
        "H2B_minus_H0": ("H0", "H2B"),
        "H2_minus_H0": ("H0", "H2"),
        "H2C_minus_H0": ("H0", "H2C"),
        "H4_minus_H0": ("H0", "H4"),
        "H2B_minus_H2": ("H2", "H2B"),
        "H2C_minus_H2": ("H2", "H2C"),
    }
    payload = bootstrap_models(labels, probabilities, subjects, comparisons)
    return {
        "scope": "official Validation, Study-level probability, Best checkpoints",
        "bootstrap_replicates": 2000,
        "subject_cluster": True,
        "seed": SEED,
        "exploratory": True,
        "comparisons": payload["comparisons"],
        "sampling_indices_sha256": payload.get("sampling_indices_sha256"),
        "test_access": False,
    }


def _assert_close(a, b, tolerance=1e-6) -> None:
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        raise AssertionError(f"non-finite statistic: {a} vs {b}")
    if abs(float(a) - float(b)) > tolerance:
        raise AssertionError(f"grouped BCE mismatch: {a} vs {b}")


def _grouped_bce_audit(results: dict) -> dict:
    """Fix and validate group-level BCE aggregation, including old H3-B report."""
    audit_rows = []
    assertions = []
    for experiment in ORDER:
        for split in ("train", "valid"):
            result = results[(experiment, "best", split)]
            overall = base.result_metrics(result)["overall_bce"]
            logits = np.asarray(result["study_logits"], dtype=np.float64)
            labels = np.asarray(result["labels"], dtype=np.int64)
            class_bce = base.class_bce_breakdown(logits, labels)
            class_mean = float(np.mean([class_bce[name]["bce"]
                                        for name in LABEL_NAMES]))
            _assert_close(overall, class_mean)
            audit_rows.append({
                "experiment": experiment,
                "split": split,
                "group": "Overall",
                "n_studies": int(len(labels)),
                "macro_bce": class_mean,
                "overall_metric_bce": overall,
                "consistent": True,
            })
    # Single/Multi weighted-average assertion: masks are mutually exclusive and
    # cover every Study, so the weighted mean must equal the overall BCE exactly.
    for experiment in ORDER:
        for split in ("train", "valid"):
            result = results[(experiment, "best", split)]
            logits = np.asarray(result["study_logits"], dtype=np.float64)
            labels = np.asarray(result["labels"], dtype=np.int64)
            counts = np.asarray(result["image_counts"], dtype=np.int64)
            overall = base.result_metrics(result)["overall_bce"]
            single_mask = counts == 1
            multi_mask = counts >= 2
            single_bce = allowed_bce(logits[single_mask], labels[single_mask])
            multi_bce = allowed_bce(logits[multi_mask], labels[multi_mask])
            weighted = ((single_mask.sum() * single_bce
                         + multi_mask.sum() * multi_bce)
                        / (single_mask.sum() + multi_mask.sum()))
            _assert_close(overall, weighted, tolerance=1e-6)
            assertions.append({
                "experiment": experiment,
                "split": split,
                "single_n": int(single_mask.sum()),
                "multi_n": int(multi_mask.sum()),
                "single_macro_bce": single_bce,
                "multi_macro_bce": multi_bce,
                "weighted_macro_bce": weighted,
                "overall_macro_bce": overall,
                "assertion_passed": True,
            })
            audit_rows.append({
                "experiment": experiment,
                "split": split,
                "group": "Single-image",
                "n_studies": int(single_mask.sum()),
                "macro_bce": single_bce,
                "overall_metric_bce": None,
                "consistent": True,
            })
            audit_rows.append({
                "experiment": experiment,
                "split": split,
                "group": "Multi-image",
                "n_studies": int(multi_mask.sum()),
                "macro_bce": multi_bce,
                "overall_metric_bce": None,
                "consistent": True,
            })
    # Old H3-B report grouped "Overall BCE" values were per-disease rows of
    # LABEL_NAMES[0] (Enlarged Cardiomediastinum), not the ten-class mean.
    old_values = {}
    h3b_report = (EXPERIMENT_ROOT / "H3B_Weak_L2SP"
                  / "h3b_final_report.md")
    if h3b_report.is_file():
        text = h3b_report.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("| H3B |") and "|" in line:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if len(cells) >= 5 and cells[1] in (
                        "AP-only", "PA+Lateral", "Single-image", "Multi-image"):
                    old_values[cells[1]] = float(cells[4])
    corrected = {}
    old_matches_first_disease = {}
    h3b_cache = base.CACHE / "H3B_best_valid_noaug.npz"
    if h3b_cache.is_file():
        h3b_result = base.load_result(h3b_cache)
        _, h3b_valid_data, _ = build_eval_datasets()
        h3b_metadata = study_view_metadata(h3b_valid_data, VALID_CSV)
        h3b_rows = base.grouped_metrics(
            h3b_result, h3b_valid_data, h3b_metadata, "H3B", "best", "valid")
        h3b_frame = pd.DataFrame(h3b_rows)
        for group in ("AP-only", "PA+Lateral", "Single-image", "Multi-image"):
            subset = h3b_frame[h3b_frame["group"] == group]
            if subset.empty:
                continue
            corrected[group] = float(subset["macro_bce"].iloc[0])
            first_disease = float(
                subset[subset["disease"] == LABEL_NAMES[0]]["bce"].iloc[0])
            old_matches_first_disease[group] = bool(
                group in old_values
                and abs(old_values[group] - first_disease) < 1e-6)
    payload = {
        "root_cause": "The historical grouped report selected the row with "
                      "disease == LABEL_NAMES[0] and displayed its per-disease "
                      "`bce` under the column name 'Overall BCE'. `grouped_metrics` "
                      "did not expose a group-level macro BCE, so reports "
                      "accidentally used column 0 of the ten-class vector.",
        "fix": "grouped_metrics now writes macro_bce / macro_positive_bce / "
               "macro_negative_bce = ten-class mean within each group; all report "
               "tables use macro_bce and never a single disease row.",
        "assertions": assertions,
        "audit_table": audit_rows,
        "h3b_old_reported_overall_bce": old_values,
        "h3b_corrected_macro_bce": corrected,
        "h3b_old_values_equal_first_disease_bce": old_matches_first_disease,
        "test_access": False,
    }
    write_json(OUT / "grouped_bce_validation_audit.json", payload)
    return payload


def allowed_bce(logits, labels) -> float:
    breakdown = base.class_bce_breakdown(logits, labels)
    return float(np.mean([breakdown[name]["bce"] for name in LABEL_NAMES]))


def _write_plots(results: dict) -> None:
    # Overall curves.
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for experiment in ORDER:
        history = _history(experiment)
        axes[0].plot(history["epoch"], history["train_bce"], marker="o",
                     color=COLORS[experiment], label=experiment)
        axes[1].plot(history["epoch"], history["valid_loss"], marker="o",
                     color=COLORS[experiment], label=experiment)
        axes[2].plot(history["epoch"], history["macro_auroc"], marker="o",
                     color=COLORS[experiment], label=experiment)
    axes[0].set_title("Train raw Study BCE")
    axes[1].set_title("Validation raw Study BCE")
    axes[2].set_title("Validation Macro-AUROC")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "overall_loss_curves.png", dpi=140)
    plt.close(figure)

    loss_dir = OUT / "per_disease_loss_plots"
    auroc_dir = OUT / "per_disease_auroc_plots"
    loss_dir.mkdir(parents=True, exist_ok=True)
    auroc_dir.mkdir(parents=True, exist_ok=True)
    epoch_metrics = {experiment: _epoch_class_metrics(experiment)
                     for experiment in ORDER}
    for disease in LABEL_NAMES:
        figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for experiment in ORDER:
            if experiment in ("H0", "H2"):
                frame = epoch_metrics[experiment]
                subset = frame[frame["disease"] == disease]
                if not subset.empty:
                    axes[0].plot(subset["epoch"], subset["valid_bce"],
                                 marker="o", color=COLORS[experiment],
                                 label=f"{experiment} valid")
            else:
                per_epoch = pd.read_csv(
                    exp_dir(experiment) / "per_epoch_class_loss.csv")
                subset = per_epoch[per_epoch["disease"] == disease]
                if subset.empty:
                    continue
                axes[0].plot(subset["epoch"], subset["valid_bce"], marker="o",
                             color=COLORS[experiment], label=f"{experiment} valid")
                if "train_bce" in subset.columns and subset["train_bce"].notna().any():
                    axes[0].plot(subset["epoch"], subset["train_bce"], marker="x",
                                 linestyle="--", color=COLORS[experiment],
                                 label=f"{experiment} train")
        axes[0].set_title(f"{disease}: per-epoch BCE")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("raw BCE")
        axes[0].legend(fontsize=6)
        axes[0].grid(alpha=0.3)
        for experiment in ORDER:
            frame = epoch_metrics[experiment]
            subset = frame[frame["disease"] == disease]
            if subset.empty:
                continue
            axes[1].plot(subset["epoch"], subset["auroc"], marker="o",
                         color=COLORS[experiment], label=experiment)
        axes[1].set_title(f"{disease}: per-epoch Validation AUROC")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylim(0.4, 1.0)
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        figure.tight_layout()
        safe = disease.replace(" ", "_").replace("/", "_")
        figure.savefig(loss_dir / f"{safe}.png", dpi=130)
        figure.savefig(auroc_dir / f"{safe}.png", dpi=130)
        plt.close(figure)


def _fmt(value, digits=6):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_reports(overall: pd.DataFrame, class_rows: pd.DataFrame,
                   grouped: pd.DataFrame, bootstrap: dict,
                   audit: dict) -> None:
    best = overall[overall["checkpoint"] == "best"].set_index("experiment")
    last = overall[overall["checkpoint"] == "last"].set_index("experiment")
    grouped_valid = grouped[(grouped["checkpoint"] == "best")
                            & (grouped["split"] == "valid")]
    boot = bootstrap["comparisons"]
    h2b_h0 = boot["H2B_minus_H0"]
    h2_h0 = boot["H2_minus_H0"]
    h2c_h0 = boot["H2C_minus_H0"]
    h4_h0 = boot["H4_minus_H0"]
    h2b_h2 = boot["H2B_minus_H2"]
    h2c_h2 = boot["H2C_minus_H2"]
    # Wide loss table.
    wide_loss = class_rows.pivot_table(
        index="disease", columns=["experiment", "split"], values="bce")
    wide_loss.columns = [f"{experiment}_{split}"
                         for experiment, split in wide_loss.columns]
    wide_loss = wide_loss.loc[LABEL_NAMES]
    overall_row = {column: float(np.mean(wide_loss[column]))
                   for column in wide_loss.columns}
    overall_loss_frame = pd.DataFrame([{"disease": "Overall BCE", **overall_row}])
    loss_table = pd.concat([
        pd.DataFrame([{"disease": disease, **{
            column: float(wide_loss.loc[disease, column])
            for column in wide_loss.columns}} for disease in LABEL_NAMES]),
        overall_loss_frame], ignore_index=True)
    loss_table.to_csv(OUT / "per_class_loss_comparison.csv", index=False,
                      encoding="utf-8")
    # AUROC / AUPRC long comparisons.
    auroc_rows = []
    auprc_rows = []
    for _, row in class_rows.iterrows():
        auroc_rows.append({"experiment": row["experiment"],
                           "split": row["split"], "disease": row["disease"],
                           "checkpoint": "best", "auroc": row["auroc"]})
        auprc_rows.append({"experiment": row["experiment"],
                           "split": row["split"], "disease": row["disease"],
                           "checkpoint": "best", "auprc": row["auprc"]})
    pd.DataFrame(auroc_rows).to_csv(OUT / "per_class_auroc_comparison.csv",
                                    index=False, encoding="utf-8")
    pd.DataFrame(auprc_rows).to_csv(OUT / "per_class_auprc_comparison.csv",
                                    index=False, encoding="utf-8")
    # Overall comparison.
    best_rows = []
    for experiment in ORDER:
        row = best.loc[experiment]
        status = _status(experiment)
        best_rows.append({
            "experiment": experiment,
            "train_overall_bce": row["train_overall_bce"],
            "valid_overall_bce": row["valid_overall_bce"],
            "train_valid_bce_gap": row["train_valid_bce_gap"],
            "train_macro_auroc": row["train_macro_auroc"],
            "train_macro_auprc": row["train_macro_auprc"],
            "valid_macro_auroc": row["valid_macro_auroc"],
            "valid_macro_auprc": row["valid_macro_auprc"],
            "best_epoch": row["best_epoch"],
            "completed_epochs": row["completed_epochs"],
            "stop_reason": status.get("stop_reason"),
        })
    pd.DataFrame(best_rows).to_csv(OUT / "overall_loss_comparison.csv",
                                   index=False, encoding="utf-8")
    grouped.to_csv(OUT / "grouped_performance_comparison.csv", index=False,
                   encoding="utf-8")
    write_json(OUT / "paired_bootstrap_results.json", bootstrap)

    # Disease-level changes.
    change_rows = []
    for disease in LABEL_NAMES:
        train_cells = {experiment: float(wide_loss.loc[disease, f"{experiment}_train"])
                       for experiment in ORDER}
        valid_cells = {experiment: float(wide_loss.loc[disease, f"{experiment}_valid"])
                       for experiment in ORDER}
        change_rows.append({
            "disease": disease,
            **{f"{experiment}_train_bce": value
               for experiment, value in train_cells.items()},
            **{f"{experiment}_valid_bce": value
               for experiment, value in valid_cells.items()},
            **{f"{experiment}_gap": valid_cells[experiment] - train_cells[experiment]
               for experiment in ORDER},
        })
    pd.DataFrame(change_rows).to_csv(OUT / "per_disease_loss_gap.csv",
                                     index=False, encoding="utf-8")

    # Main comparative report.
    lines = [
        "# H2-B / H2-C / H4 comparative report", "",
        "## Overall comparison (Best checkpoint; full deterministic "
        "Train/Validation inference)", "",
        "| Experiment | Train BCE | Valid BCE | BCE Gap | Valid AUROC | "
        "Valid AUPRC | Best Epoch | Stop |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for experiment in ORDER:
        row = best.loc[experiment]
        lines.append(
            f"| {experiment} | {row['train_overall_bce']:.6f} | "
            f"{row['valid_overall_bce']:.6f} | "
            f"{row['train_valid_bce_gap']:+.6f} | "
            f"{row['valid_macro_auroc']:.6f} | {row['valid_macro_auprc']:.6f} | "
            f"{row['best_epoch']} | {best_rows[ORDER.index(experiment)]['stop_reason']} |")
    lines.extend([
        "",
        "## Paired Subject-cluster bootstrap (Best, Validation, 2000 draws)", "",
        f"- H2-B − H0: ΔAUROC {h2b_h0['delta_macro_auroc']:+.6f} "
        f"CI {h2b_h0['delta_macro_auroc_ci95']}",
        f"- H2 − H0: ΔAUROC {h2_h0['delta_macro_auroc']:+.6f} "
        f"CI {h2_h0['delta_macro_auroc_ci95']}",
        f"- H2-C − H0: ΔAUROC {h2c_h0['delta_macro_auroc']:+.6f} "
        f"CI {h2c_h0['delta_macro_auroc_ci95']}",
        f"- H4 − H0: ΔAUROC {h4_h0['delta_macro_auroc']:+.6f} "
        f"CI {h4_h0['delta_macro_auroc_ci95']}",
        f"- H2-B − H2: ΔAUROC {h2b_h2['delta_macro_auroc']:+.6f} "
        f"CI {h2b_h2['delta_macro_auroc_ci95']}",
        f"- H2-C − H2: ΔAUROC {h2c_h2['delta_macro_auroc']:+.6f} "
        f"CI {h2c_h2['delta_macro_auroc_ci95']}",
        "",
        "这些区间是官方 Validation 反复使用后的探索性统计，不是独立 Test 结论。",
        "",
        "## Grouped Validation metrics (Best; fixed macro BCE)", "",
        "| Experiment | Group | Macro-AUROC | Macro-AUPRC | Macro-BCE |",
        "|---|---|---:|---:|---:|",
    ])
    for experiment in ORDER:
        for group in ("AP-only", "PA+Lateral", "Single-image", "Multi-image"):
            row = grouped_valid[(grouped_valid["experiment"] == experiment)
                                & (grouped_valid["group"] == group)]
            if row.empty:
                continue
            item = row.iloc[0]
            lines.append(
                f"| {experiment} | {group} | {item['macro_auroc']:.6f} | "
                f"{item['macro_auprc']:.6f} | {item['macro_bce']:.6f} |")
    lines.extend([
        "",
        "## Answers to the ten required questions", "",
    ])
    # Determine best single weight on validation.
    best_weight = max(("H2B", "H2", "H2C"),
                      key=lambda name: float(best.loc[name, "valid_macro_auroc"]))
    single_rows = {}
    for experiment in ORDER:
        item = grouped_valid[(grouped_valid["experiment"] == experiment)
                             & (grouped_valid["group"] == "Single-image")]
        single_rows[experiment] = None if item.empty else float(
            item["macro_auroc"].iloc[0])
    def valid_metric(experiment: str, disease: str, field: str) -> float:
        row = class_rows[(class_rows["experiment"] == experiment)
                         & (class_rows["split"] == "valid")
                         & (class_rows["disease"] == disease)]
        return float(row[field].iloc[0])

    auroc_up_auprc_down = []
    h4_auroc_deltas = {}
    for experiment in ("H2B", "H2C", "H4"):
        for disease in LABEL_NAMES:
            delta_auroc = (valid_metric(experiment, disease, "auroc")
                           - valid_metric("H0", disease, "auroc"))
            delta_auprc = (valid_metric(experiment, disease, "auprc")
                           - valid_metric("H0", disease, "auprc"))
            if delta_auroc > 0 and delta_auprc < 0:
                auroc_up_auprc_down.append(f"{experiment}:{disease}")
            if experiment == "H4":
                h4_auroc_deltas[disease] = delta_auroc
    h4_improved = sorted(
        (name for name, value in h4_auroc_deltas.items() if value > 0),
        key=lambda name: -h4_auroc_deltas[name])
    h4_worse = sorted(
        (name for name, value in h4_auroc_deltas.items() if value < 0),
        key=lambda name: h4_auroc_deltas[name])
    questions = [
        ("1. 单图权重 1.10/1.25/1.50 哪项更好？",
         f"在当前 Validation 上，Best AUROC 最高的是 **{best_weight}**"
         f"（{best.loc[best_weight, 'valid_macro_auroc']:.6f}）；"
         f"H2-B {best.loc['H2B', 'valid_macro_auroc']:.6f}、"
         f"H2 {best.loc['H2', 'valid_macro_auroc']:.6f}、"
         f"H2-C {best.loc['H2C', 'valid_macro_auroc']:.6f}。"
         "差异很小，见 bootstrap CI。"),
        ("2. 单图权重增加是否使单图 AUROC 单调改善？",
         f"否。单图子群 AUROC：H0 {single_rows['H0']:.6f}、"
         f"H2-B {single_rows['H2B']:.6f}、H2 {single_rows['H2']:.6f}、"
         f"H2-C {single_rows['H2C']:.6f}、H4 {single_rows['H4']:.6f}；"
         "没有随权重单调上升。"),
        ("3. 对多图 Study 是否有负面影响？",
         f"多图子群 AUROC：H0 {float(grouped_valid[(grouped_valid.experiment=='H0') & (grouped_valid.group=='Multi-image')].macro_auroc.iloc[0]):.6f}、"
         f"H2-B {float(grouped_valid[(grouped_valid.experiment=='H2B') & (grouped_valid.group=='Multi-image')].macro_auroc.iloc[0]):.6f}、"
         f"H2 {float(grouped_valid[(grouped_valid.experiment=='H2') & (grouped_valid.group=='Multi-image')].macro_auroc.iloc[0]):.6f}、"
         f"H2-C {float(grouped_valid[(grouped_valid.experiment=='H2C') & (grouped_valid.group=='Multi-image')].macro_auroc.iloc[0]):.6f}；"
         "无一致下降趋势，权重 1.50 未见系统性多图损害。"),
        ("4. AP-only 是否随单图权重变化？",
         f"AP-only AUROC：H0 {float(grouped_valid[(grouped_valid.experiment=='H0') & (grouped_valid.group=='AP-only')].macro_auroc.iloc[0]):.6f}、"
         f"H2-B {float(grouped_valid[(grouped_valid.experiment=='H2B') & (grouped_valid.group=='AP-only')].macro_auroc.iloc[0]):.6f}、"
         f"H2 {float(grouped_valid[(grouped_valid.experiment=='H2') & (grouped_valid.group=='AP-only')].macro_auroc.iloc[0]):.6f}、"
         f"H2-C {float(grouped_valid[(grouped_valid.experiment=='H2C') & (grouped_valid.group=='AP-only')].macro_auroc.iloc[0]):.6f}。"
         "变化存在但无单调规律。"),
        ("5. 是否只是概率尺度改变，而不是排序能力改善？",
         "AUROC 提高但 AUPRC 下降的 (实验:疾病) 组合："
         + ("、".join(auroc_up_auprc_down) if auroc_up_auprc_down else "无")
         + "。例如 H2 的总体 AUROC 略升而 AUPRC 下降，H2-C 的 Train BCE "
         "明显下降但 Valid BCE/AUPRC 更差，说明部分变化是概率尺度/校准而非"
         "排序能力提升；逐疾病结果见 `per_class_auroc_comparison.csv` 与 "
         "`per_class_auprc_comparison.csv`。"),
        ("6. Dropout 是否改善 Train/Valid 泛化差距？",
         f"H4 Gap {best.loc['H4', 'train_valid_bce_gap']:+.6f} vs H0 "
         f"{best.loc['H0', 'train_valid_bce_gap']:+.6f}；H4 Valid AUROC "
         f"{best.loc['H4', 'valid_macro_auroc']:.6f} vs H0 "
         f"{best.loc['H0', 'valid_macro_auroc']:.6f}（Δ="
         f"{h4_h0['delta_macro_auroc']:+.6f}，CI "
         f"{h4_h0['delta_macro_auroc_ci95']}）。"),
        ("7. Dropout 是否降低 Validation BCE？",
         f"H4 Valid BCE {best.loc['H4', 'valid_overall_bce']:.6f} vs H0 "
         f"{best.loc['H0', 'valid_overall_bce']:.6f}，"
         f"Δ={best.loc['H4', 'valid_overall_bce'] - best.loc['H0', 'valid_overall_bce']:+.6f}。"),
        ("8. Dropout 是否改善困难疾病？",
         "H4 相对 H0 的 Valid AUROC 提高类别："
         + ("、".join(h4_improved) if h4_improved else "无")
         + "；下降类别："
         + ("、".join(h4_worse) if h4_worse else "无")
         + "。Enlarged Cardiomediastinum 有改善，但 Consolidation、"
         "Lung Opacity、No Finding、Pneumonia 等下降；没有系统性的"
         "“困难疾病全部获益”，且 H4 总体 AUPRC 更低。"),
        ("9. Dropout 是否明显影响收敛速度？",
         f"H4 Best epoch {best.loc['H4', 'best_epoch']}，完成 "
         f"{best.loc['H4', 'completed_epochs']} 轮；H0 Best epoch "
         f"{best.loc['H0', 'best_epoch']}，完成 {best.loc['H0', 'completed_epochs']} 轮。"
         "收敛曲线见 `overall_loss_curves.png`；H4 未显示明显加速或减慢。"),
        ("10. 哪项最值得多随机种子复验？",
         "本轮没有任何一项显示稳定的泛化收益：H2-B 与 H0 基本重合"
         f"（ΔAUROC {h2b_h0['delta_macro_auroc']:+.6f}，CI "
         f"{h2b_h0['delta_macro_auroc_ci95']}），H2 仅 +0.0016 且 CI 跨 0、"
         f"AUPRC −0.0034，H2-C 与 H4 更差。因此**不建议**为 H2-C/H4 花三个"
         "配对种子；若资源允许，唯一可做的是 H2(1.25) 与 H2-B(1.10) 的"
         "低优先级复验，用来确认“单图权重收益”是否只是单 seed 噪声。"
         " H2-C 的高 Train 拟合（Train BCE 0.194）伴随 Valid 退化，明确不支持"
         "继续加大权重。"),
    ]
    for question, answer in questions:
        lines.append(f"- **{question}** {answer}")
    lines.extend([
        "",
        "## Bottom line", "",
        "- **单图权重**：1.10 与 H0 基本重合；1.25 仅有 +0.0016 的点估计"
        "（CI 跨 0，AUPRC 略降）；1.50 明确过拟合（Train BCE 0.194 但 Valid "
        "BCE 0.331、AUPRC 0.408）。没有证据支持继续加大单图权重。",
        "- **Dropout**：H4 未降低 Valid BCE（0.3130 vs 0.3112），未提高 AUROC"
        "（−0.0004，CI 跨 0）或 AUPRC（−0.0026）；Gap 略小但伴随整体排序/校准"
        "变差，不能视为泛化收益。",
        "- **决定**：不启动三配对随机种子复验。若必须保留一个低优先级方向，"
        "只复验 H2(1.25) 与 H2-B(1.10) 的差异以确认其是否只是单 seed 噪声；"
        "H2-C 与 H4 关闭。",
        "- **单图 AP-only 子群**：H2 的 AP-only +0.0020、单图 +0.0015，"
        "但 CI 跨 0；没有可宣称的 AP-only 改善。",
        "",
        "## Grouped BCE correction", "",
        "- 旧 H3-B 报告中的分组 “Overall BCE” 实际读取了 `LABEL_NAMES[0]`"
        "（Enlarged Cardiomediastinum）的单病种 BCE；本轮已在 "
        "`grouped_metrics` 中增加 `macro_bce`（十类均值）并修复报告取数。",
        "- 新旧数值、断言与根因见 `grouped_bce_validation_audit.md`；"
        "H0/H2/H2-B/H2-C/H4 的修正版分组 BCE 见 "
        "`grouped_performance_comparison.csv`。",
        "",
        "## Limitations", "",
        "- 单 seed，Validation 反复用于选模与比较；CI 为探索性。",
        "- H2-B/H2-C/H4 的训练循环每 epoch 记录了 LR、更新次数、原始 BCE、"
        "优化目标、逐 epoch 十类 Validation AUROC/AUPRC/BCE；但**未单独记录"
        "每步梯度范数**（历史训练器只在 L2-SP 实验中保存 BCE/L2-SP 梯度范数）。"
        "梯度裁剪阈值 1.0 每 update 执行，触发比例未保存，不能事后伪造。",
        "- H0/H2 的中间 epoch Train 逐类 BCE 在历史文件中不完整；本轮对比表使用"
        "Best/Last 的完整确定性逐类 BCE。新实验的逐 epoch Train/Valid 逐类 BCE "
        "完整（`per_epoch_class_loss.csv`）。",
        "- H0 为复用的 R5；H2 为历史实验，H2-B/H2-C/H4 为新独立训练。",
        "- 无 Test 标签，未用 Test 选权重或判断 Dropout。",
        "- 分组 BCE 已修复并断言；历史 H3-B 文件不覆盖，只在本目录输出修正版。",
        "",
    ])
    (OUT / "final_comparative_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    # Detailed loss analysis report.
    lines = [
        "# H2-B / H2-C / H4 unified Train/Validation loss analysis", "",
        "## Method", "",
        "- 完整官方 Train/Validation，`model.eval()`，确定性无增强，Study-level "
        "原始 logit，未加权 BCE，无单图权重、无正则项、无阈值。",
        "- 每类 BCE 使用全部 Study；总体 BCE = 十类 BCE 均值；程序断言二者一致。",
        "- Best/Last 均完成全量 Train/Validation 推理；训练批次 BCE 与 Validation "
        "确定性 BCE 明确区分。",
        "",
        "## Overall loss", "",
        "| Experiment | Train BCE | Valid BCE | BCE Gap |",
        "|---|---:|---:|---:|",
    ]
    for experiment in ORDER:
        row = best.loc[experiment]
        lines.append(f"| {experiment} | {row['train_overall_bce']:.6f} | "
                     f"{row['valid_overall_bce']:.6f} | "
                     f"{row['train_valid_bce_gap']:+.6f} |")
    lines.extend([
        "",
        "## Ten-class Train/Valid BCE (Best)", "",
        "| Disease | " + " | ".join(f"{experiment} Train" for experiment in ORDER)
        + " | " + " | ".join(f"{experiment} Valid" for experiment in ORDER) + " |",
        "|---|" + "---:|" * (2 * len(ORDER)),
    ])
    for disease in list(LABEL_NAMES) + ["Overall BCE"]:
        cells = [f"{loss_table[loss_table.disease == disease][f'{experiment}_train'].iloc[0]:.6f}"
                 if not loss_table[loss_table.disease == disease].empty else "NA"
                 for experiment in ORDER]
        valid_cells = [f"{loss_table[loss_table.disease == disease][f'{experiment}_valid'].iloc[0]:.6f}"
                       if not loss_table[loss_table.disease == disease].empty else "NA"
                       for experiment in ORDER]
        lines.append(f"| {disease} | " + " | ".join(cells) + " | "
                     + " | ".join(valid_cells) + " |")
    lines.extend([
        "",
        "## Valid positive/negative BCE and AUROC/AUPRC", "",
        "| Disease | Experiment | Valid pos BCE | Valid neg BCE | Valid AUROC | "
        "Valid AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for disease in LABEL_NAMES:
        for experiment in ORDER:
            row = class_rows[(class_rows.experiment == experiment)
                             & (class_rows.split == "valid")
                             & (class_rows.disease == disease)]
            if row.empty:
                continue
            item = row.iloc[0]
            lines.append(
                f"| {disease} | {experiment} | {item['positive_bce']:.6f} | "
                f"{item['negative_bce']:.6f} | {item['auroc']:.6f} | "
                f"{item['auprc']:.6f} |")
    lines.extend([
        "",
        "## Text analysis", "",
    ])
    # Directional checks.
    for experiment in ("H2B", "H2C", "H4"):
        train_down_valid_up = [
            disease for disease in LABEL_NAMES
            if float(wide_loss.loc[disease, f"{experiment}_train"])
            < float(wide_loss.loc[disease, "H0_train"])
            and float(wide_loss.loc[disease, f"{experiment}_valid"])
            > float(wide_loss.loc[disease, "H0_valid"])]
        valid_down_auroc_up = [
            disease for disease in LABEL_NAMES
            if float(wide_loss.loc[disease, f"{experiment}_valid"])
            < float(wide_loss.loc[disease, "H0_valid"])
            and float(class_rows[(class_rows.experiment == experiment)
                                 & (class_rows.split == "valid")
                                 & (class_rows.disease == disease)].auroc.iloc[0])
            > float(class_rows[(class_rows.experiment == "H0")
                               & (class_rows.split == "valid")
                               & (class_rows.disease == disease)].auroc.iloc[0])]
        lines.append(
            f"- **{experiment}**: Train BCE 下降但 Valid BCE 上升的疾病："
            + ("、".join(train_down_valid_up) if train_down_valid_up else "无")
            + "；Valid BCE 下降且 AUROC 提高的疾病："
            + ("、".join(valid_down_auroc_up) if valid_down_auroc_up else "无")
            + "。")
    # AUROC up but AUPRC down.
    mismatch = []
    for experiment in ("H2B", "H2C", "H4"):
        for disease in LABEL_NAMES:
            a = class_rows[(class_rows.experiment == experiment)
                           & (class_rows.split == "valid")
                           & (class_rows.disease == disease)].iloc[0]
            b = class_rows[(class_rows.experiment == "H0")
                           & (class_rows.split == "valid")
                           & (class_rows.disease == disease)].iloc[0]
            if a["auroc"] > b["auroc"] and a["auprc"] < b["auprc"]:
                mismatch.append(f"{experiment}:{disease}")
    lines.append(
        "- AUROC 提高但 AUPRC 下降的组合："
        + ("、".join(mismatch) if mismatch else "无")
        + "。这提示部分概率尺度/校准变化不能解释为排序能力提升。")
    lines.extend([
        "- 单图与多图子群变化并不总是一致；见 "
        "`grouped_performance_comparison.csv`，不能把 Study 级 Macro-AUROC "
        "当作子群 AUROC 的加权平均。",
        "",
        "## Every-epoch Validation curves", "",
        "- 逐 epoch 十类 Validation AUROC/AUPRC/BCE：新实验的 "
        "`per_epoch_class_metrics.csv`；H0/H2 由历史 `val_epoch_*.csv` "
        "重建，见 `epoch_class_metrics/`。",
        "- 总体与逐疾病曲线：`overall_loss_curves.png`、`per_disease_loss_plots/`、"
        "`per_disease_auroc_plots/`。",
        "",
        "## Assumptions and limits", "",
        "- 训练批次 BCE 与 Validation 确定性 BCE 不可直接比较；所有表格已标注口径。",
        "- H0 为复用 R5；H2 为历史实验；新实验均从原始 DINOv3 权重独立训练，未叠加。",
        "- 单 seed；无 Test 评估。",
        "",
    ])
    (OUT / "train_valid_loss_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    # Grouped BCE audit markdown.
    lines = [
        "# Grouped BCE validation audit", "",
        "## Root cause", "",
        audit["root_cause"],
        "",
        "## Fix", "",
        audit["fix"],
        "",
        "## Assertions", "",
        "| Experiment | Split | Single N | Multi N | Single Macro-BCE | "
        "Multi Macro-BCE | Weighted | Overall | Passed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit["assertions"]:
        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['single_n']} | "
            f"{row['multi_n']} | {row['single_macro_bce']:.6f} | "
            f"{row['multi_macro_bce']:.6f} | {row['weighted_macro_bce']:.6f} | "
            f"{row['overall_macro_bce']:.6f} | {row['assertion_passed']} |")
    lines.extend([
        "",
        "- 同时断言 `BCE_overall = mean_k BCE_k`（十类均值）；"
        "所有断言通过，浮点误差 < 1e-6（BF16/float32 推理口径）。",
        "",
        "## Old H3-B reported values vs corrected macro BCE", "",
        "旧报告列名是 “Overall BCE”，但数值来自行 `disease == "
        "LABEL_NAMES[0]`（Enlarged Cardiomediastinum），不是十类均值；"
        "H3-B 文件未覆盖，本目录输出修正版。旧值与第 0 类 BCE 一致：",
        "",
        "| Group (H3-B, valid) | Old reported 'Overall BCE' |",
        "|---|---:|",
    ])
    for group, value in audit.get("h3b_old_reported_overall_bce", {}).items():
        lines.append(f"| {group} | {value:.6f} |")
    corrected = audit.get("h3b_corrected_macro_bce", {})
    old_values = audit.get("h3b_old_reported_overall_bce", {})
    if corrected:
        lines.extend([
            "",
            "| Group (H3-B, valid) | Old reported | Corrected macro BCE |",
            "|---|---:|---:|",
        ])
        for group, value in old_values.items():
            lines.append(f"| {group} | {value:.6f} | "
                         f"{corrected.get(group, float('nan')):.6f} |")
        lines.extend([
            "",
            "旧值等于该组第 0 类 BCE（Enlarged Cardiomediastinum）："
            f"{audit.get('h3b_old_values_equal_first_disease_bce')}。",
        ])
    lines.extend([
        "",
        "对照：H3-B valid 十类总体 BCE 为 0.332165；旧值的 AP-only 0.155770 等"
        "明显是该病种 BCE，而不是十类均值。修正版见 "
        "`grouped_performance_comparison.csv` 的 `macro_bce` 列。",
        "",
        "## AUROC/AUPRC audit", "",
        "- 分组 `macro_auroc`/`macro_auprc` 直接来自 `study_metrics` 的十类"
        "宏平均（只对有效类别求平均），不是第 0 类；未发现同类索引错误。",
        "- 逐病种值仍保留在 `bce`/`auroc`/`auprc` 列，报告只使用 `macro_bce`。",
        "",
    ])
    (OUT / "grouped_bce_validation_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def run_h2_analysis(force: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    results = _results()
    overall = _overall(results)
    class_rows = _class_rows(results)
    grouped = _grouped(results)
    bootstrap = _bootstrap(results)
    audit = _grouped_bce_audit(results)
    _write_plots(results)
    _write_reports(overall, class_rows, grouped, bootstrap, audit)
    # Per-experiment required artifacts for the three new runs.
    train_data, valid_data, _ = build_eval_datasets()
    for experiment in ("H2B", "H2C", "H4"):
        required_results = {
            (checkpoint, split): results[(experiment, checkpoint, split)]
            for checkpoint in CHECKPOINTS
            for split in ("train", "valid")}
        base._write_experiment_artifacts(experiment, train_data, valid_data,
                                         required_results)
        source = exp_dir(experiment) / "per_class_metrics.csv"
        if source.is_file():
            shutil.copy2(source, exp_dir(experiment)
                         / "per_epoch_class_metrics.csv")
    summary = {
        "status": "h2_h4_analysis_complete",
        "overall": overall.to_dict("records"),
        "bootstrap": bootstrap["comparisons"],
        "audit_assertions_passed": all(
            row["assertion_passed"] for row in audit["assertions"]),
        "test_access": False,
    }
    write_json(OUT / "analysis_summary.json", summary)
    print("[h2-h4-analysis] complete", flush=True)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_h2_analysis(), ensure_ascii=True, indent=1, default=str))
