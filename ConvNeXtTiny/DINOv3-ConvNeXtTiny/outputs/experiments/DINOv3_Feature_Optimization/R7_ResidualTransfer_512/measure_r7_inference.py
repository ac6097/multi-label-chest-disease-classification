"""Measure standalone R7 per-Study inference latency on official Validation only."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[3]
PROJECT_DIR = WORK_DIR.parent
REPO_DIR = PROJECT_DIR.parent
PACKAGE_DIR = HERE / "submission_export" / "R7_Inference"
DATA_DIR = REPO_DIR / "cxr_first"
VALID_CSV = DATA_DIR / "valid.csv"
LABELS = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(PACKAGE_DIR))

from datasets.dataset import StudyDataset  # noqa: E402
from r7_inference import R7Predictor  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def metric_summary(labels: np.ndarray, probabilities: np.ndarray):
    aucs, aps = [], []
    rows = []
    for index, name in enumerate(LABELS):
        auc = float(roc_auc_score(labels[:, index], probabilities[:, index]))
        ap = float(average_precision_score(labels[:, index], probabilities[:, index]))
        aucs.append(auc)
        aps.append(ap)
        rows.append({
            "Disease": name,
            "AUROC": auc,
            "AUPRC": ap,
            "Positive_Studies": int(labels[:, index].sum()),
            "Negative_Studies": int((labels[:, index] == 0).sum()),
        })
    return float(np.mean(aucs)), float(np.mean(aps)), rows


def summarize_saved_outputs(predictions: pd.DataFrame, latency: pd.DataFrame,
                            total_seconds: float, peak_gpu_gib: float):
    if len(predictions) != 859 or int(predictions["Image_Count"].sum()) != 1422:
        raise RuntimeError("Saved standalone R7 inference outputs have unexpected Validation counts")
    if predictions.duplicated(["Subject_id", "Study_id"]).any():
        raise RuntimeError("Standalone R7 inference outputs contain duplicate Studies")
    labels = predictions[[f"target_{label.replace(' ', '_')}" for label in LABELS]]\
        .to_numpy(dtype=np.uint8)
    probabilities = predictions[[f"prob_R7_Fused_{label.replace(' ', '_')}"
                                 for label in LABELS]].to_numpy(dtype=np.float64)
    macro_auc, macro_ap, per_class = metric_summary(labels, probabilities)

    batch_predictions = pd.read_csv(
        HERE / "r7_validation_predictions.csv",
        dtype={"Subject_id": "string", "Study_id": "string"})
    if batch_predictions.duplicated(["Subject_id", "Study_id"]).any():
        raise RuntimeError("R7 batched Validation predictions contain duplicate Study keys")
    aligned = predictions.merge(
        batch_predictions, on=["Subject_id", "Study_id"], suffixes=("_package", "_batch"),
        validate="one_to_one", sort=False)
    package_labels = aligned[[f"target_{label.replace(' ', '_')}_package"
                               for label in LABELS]].to_numpy(dtype=np.uint8)
    if not np.array_equal(package_labels, labels):
        raise RuntimeError("Standalone and batched R7 Validation labels differ")
    batch_probabilities = aligned[[f"prob_R7_Fused_{label.replace(' ', '_')}_batch"
                                   for label in LABELS]].to_numpy(dtype=np.float64)
    replay_delta = np.abs(probabilities - batch_probabilities)
    batch_auc, batch_ap, _ = metric_summary(labels, batch_probabilities)
    total_seconds = max(float(total_seconds), 1e-9)
    size = read_json(HERE / "r7_model_size_metrics.json")

    replay = {
        "standalone_package_replay": {
            "validation_studies": int(len(predictions)),
            "validation_images": int(predictions["Image_Count"].sum()),
            "macro_auroc": macro_auc,
            "macro_auprc": macro_ap,
            "max_probability_abs_error_vs_batched_validation": float(replay_delta.max()),
            "mean_probability_abs_error_vs_batched_validation": float(replay_delta.mean()),
            "batched_macro_auroc": batch_auc,
            "batched_macro_auprc": batch_ap,
            "max_abs_per_class_auroc_delta": max(
                abs(item["AUROC"] - roc_auc_score(
                    labels[:, index], batch_probabilities[:, index]))
                for index, item in enumerate(per_class)),
        },
        "latency": {
            "measurement": "One Study per inference call; end-to-end includes NAS image reads, ResizePad, tensor conversion, forward, and Study Logit Mean.",
            "total_validation_seconds": total_seconds,
            "images_per_second_end_to_end": int(predictions["Image_Count"].sum()) / total_seconds,
            "studies_per_second_end_to_end": len(predictions) / total_seconds,
            "mean_model_forward_ms_per_study": float(latency["model_forward_ms"].mean()),
            "median_model_forward_ms_per_study": float(latency["model_forward_ms"].median()),
            "p90_model_forward_ms_per_study": float(latency["model_forward_ms"].quantile(0.9)),
            "mean_end_to_end_ms_per_study": float(latency["end_to_end_ms"].mean()),
            "median_end_to_end_ms_per_study": float(latency["end_to_end_ms"].median()),
            "p90_end_to_end_ms_per_study": float(latency["end_to_end_ms"].quantile(0.9)),
            "peak_gpu_allocated_gib": float(peak_gpu_gib),
        },
        "model_size": size,
        "per_class_metrics": per_class,
        "test_data_accessed": False,
    }
    write_json(HERE / "r7_inference_cost.json", replay)
    print(json.dumps(replay, ensure_ascii=False), flush=True)
    return replay


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize-existing":
        predictions = pd.read_csv(
            HERE / "r7_package_validation_predictions.csv",
            dtype={"Subject_id": "string", "Study_id": "string"})
        latency = pd.read_csv(
            HERE / "r7_inference_latency.csv",
            dtype={"Subject_id": "string", "Study_id": "string"})
        if len(predictions) != len(latency):
            raise RuntimeError("Saved R7 package predictions and latency rows differ in length")
        total_seconds = float(latency["end_to_end_ms"].sum() / 1000.0)
        peak_gpu_gib = float(latency["peak_gpu_allocated_gib"].max())
        summarize_saved_outputs(predictions, latency, total_seconds, peak_gpu_gib)
        return

    valid = StudyDataset(
        VALID_CSV, DATA_DIR, transform=None,
        label_column="Predict_class", label_semantics="closed_world",
        image_id_column="Dicom_id", subject_column="Subject_id",
        study_column="Study_id",
    )
    if len(valid) != 859 or int(sum(valid.image_counts)) != 1422:
        raise RuntimeError(
            f"Unexpected official Validation size: {len(valid)} Studies, "
            f"{sum(valid.image_counts)} images")

    predictor = R7Predictor(PACKAGE_DIR, device="cuda")
    prediction_rows, latency_rows = [], []
    image_total = 0
    peak_gpu_gib = 0.0
    started = time.perf_counter()
    for study in valid.studies:
        probabilities = predictor.predict_study_paths(study["image_paths"])
        image_count = len(study["image_paths"])
        image_total += image_count
        peak_gpu_gib = max(peak_gpu_gib, predictor.last_peak_gpu_gib)
        prediction = {
            "Subject_id": study["subject_id"],
            "Study_id": study["study_id"],
            "Image_Count": image_count,
        }
        for index, label in enumerate(LABELS):
            safe_name = label.replace(" ", "_")
            prediction[f"target_{safe_name}"] = int(study["target"][index])
            prediction[f"prob_R7_Fused_{safe_name}"] = float(probabilities[index])
        prediction_rows.append(prediction)
        latency_rows.append({
            "Subject_id": study["subject_id"],
            "Study_id": study["study_id"],
            "Image_Count": image_count,
            "model_forward_ms": predictor.last_inference_ms,
            "end_to_end_ms": predictor.last_end_to_end_ms,
            "peak_gpu_allocated_gib": predictor.last_peak_gpu_gib,
        })

    total_seconds = time.perf_counter() - started
    predictions = pd.DataFrame(prediction_rows)
    latency = pd.DataFrame(latency_rows)
    if predictions.duplicated(["Subject_id", "Study_id"]).any():
        raise RuntimeError("Standalone R7 inference produced duplicate Study keys")
    prediction_path = HERE / "r7_package_validation_predictions.csv"
    latency_path = HERE / "r7_inference_latency.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8")
    latency.to_csv(latency_path, index=False, encoding="utf-8")

    summarize_saved_outputs(predictions, latency, total_seconds, peak_gpu_gib)


if __name__ == "__main__":
    main()
