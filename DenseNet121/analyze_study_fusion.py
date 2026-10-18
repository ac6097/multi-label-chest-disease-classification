"""Compare image-score pooling methods on validation without retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from model import ChestXrayClassifier
from model.study_pooling import pool_by_study
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths
from utils.metrics import multilabel_metrics


FUSIONS = ("logit_mean", "probability_mean", "probability_max")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_metrics(targets, probabilities, known, class_names):
    metrics = multilabel_metrics(targets, probabilities, known)
    macro_auc = metrics["macro_auroc"]
    return {
        "macro_auroc": float(macro_auc) if np.isfinite(macro_auc) else None,
        "per_label": [
            {
                "class_id": i + 1,
                "class_name": class_names[i],
                **{
                    key: (int(value) if key in {"n_valid", "n_positive", "n_negative"}
                          else (float(value) if np.isfinite(value) else None))
                    for key, value in item.items()
                },
            }
            for i, item in enumerate(metrics["per_label"])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_DIR / "configs/data.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", default=None, help="Defaults to configured valid_csv")
    parser.add_argument("--output-prefix",
                        default=str(PROJECT_DIR / "outputs/reports/study_fusion_validation"))
    args = parser.parse_args()

    data_path = resolve_config_path(args.data_config)
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    csv_file = Path(args.csv) if args.csv else data_cfg.get("val_csv")
    if csv_file is None or not Path(csv_file).is_file():
        raise FileNotFoundError(f"Validation CSV not found: {csv_file}")
    if data_cfg.get("label_semantics") != "closed_world":
        raise RuntimeError("Study fusion evaluation requires confirmed closed_world labels")

    checkpoint = resolve_config_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_config = state.get("config") or {}
    saved_data = run_config.get("data") or {}
    saved_model_cfg = dict(run_config.get("model_config") or read_yaml(
        resolve_config_path(PROJECT_DIR / "configs/model.yaml")))
    saved_model_cfg.pop("name", None)
    if saved_model_cfg.get("study_fusion", "logit_mean") not in FUSIONS:
        raise ValueError("Offline score fusion requires a checkpoint trained with logit_mean, "
                         "probability_mean, or probability_max")
    saved_model_cfg["pretrained"] = False

    image_size = int(saved_data.get("image_size", data_cfg["image_size"]))
    dataset = StudyDataset(
        csv_file,
        data_cfg["image_root"],
        transform=build_transform(image_size),
        label_column=data_cfg["label_column"],
        label_semantics=data_cfg["label_semantics"],
        image_id_column=data_cfg["image_id_column"],
        subject_column=data_cfg["subject_column"],
        study_column=data_cfg["study_column"],
    )
    targets, known = dataset.label_tensors()
    if not bool(known.all()):
        raise ValueError("Validation contains a Study without a complete 10-class label")
    train_cfg = run_config
    image_budget = int(train_cfg.get("image_budget", 32))
    loader = DataLoader(
        dataset,
        batch_sampler=StudyBatchSampler(dataset, image_budget, shuffle=False),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_studies,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXrayClassifier(**saved_model_cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    logits_by_fusion = {name: [] for name in FUSIONS}
    image_counts, subject_ids, study_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            image_logits = model(images).float()
            image_probabilities = torch.sigmoid(image_logits)
            study_index = batch["study_index"].to(device, non_blocking=True)
            logits_by_fusion["logit_mean"].append(
                pool_by_study(image_logits, study_index, batch["num_studies"], "mean").cpu())
            logits_by_fusion["probability_mean"].append(torch.logit(
                pool_by_study(image_probabilities, study_index, batch["num_studies"], "mean")
                .clamp(1e-6, 1.0 - 1e-6)).cpu())
            logits_by_fusion["probability_max"].append(torch.logit(
                pool_by_study(image_probabilities, study_index, batch["num_studies"], "max")
                .clamp(1e-6, 1.0 - 1e-6)).cpu())
            image_counts.extend(batch["image_counts"].tolist())
            subject_ids.extend(batch["subject_id"])
            study_ids.extend(batch["study_id"])

    names = data_cfg["class_names"]
    targets_np, known_np = targets.numpy(), known.numpy()
    probabilities = {}
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_csv": str(Path(csv_file).resolve()),
        "device": str(device),
        "input_size": image_size,
        "num_studies": len(dataset),
        "study_image_count": {
            "mean": float(np.mean(image_counts)),
            "max": int(np.max(image_counts)),
            "multi_image_studies": int(np.sum(np.asarray(image_counts) > 1)),
        },
        "interpretation": "All three fusions use the same per-image logits from one checkpoint.",
        "results": {},
    }
    multi_image = np.asarray(image_counts) > 1
    for fusion in FUSIONS:
        logits = torch.cat(logits_by_fusion[fusion]).numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        probabilities[fusion] = probs
        report["results"][fusion] = {
            "all_studies": summarize_metrics(targets_np, probs, known_np, names),
            "single_image_studies": summarize_metrics(
                targets_np[~multi_image], probs[~multi_image], known_np[~multi_image], names),
            "multi_image_studies": summarize_metrics(
                targets_np[multi_image], probs[multi_image], known_np[multi_image], names),
        }

    prediction_frame = pd.DataFrame({"Subject_id": subject_ids, "Study_id": study_ids,
                                     "num_images": image_counts})
    for i, name in enumerate(names):
        prediction_frame[f"true_{i + 1}_{name}"] = targets_np[:, i]
    for fusion, probs in probabilities.items():
        for i in range(10):
            prediction_frame[f"prob_{fusion}_{i + 1}"] = probs[:, i]

    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = (PROJECT_DIR / prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report_path, predictions_path = prefix.with_suffix(".json"), prefix.with_suffix(".csv")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    prediction_frame.to_csv(predictions_path, index=False)
    print(json.dumps({"report": str(report_path), "predictions": str(predictions_path),
                      "macro_auroc": {name: value["all_studies"]["macro_auroc"]
                                      for name, value in report["results"].items()}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
