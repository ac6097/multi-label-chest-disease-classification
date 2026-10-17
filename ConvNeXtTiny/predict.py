import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


def main():
    parser = argparse.ArgumentParser(description="Write ConvNeXt-Tiny Study-level long-format predictions")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--output", default=str(PROJECT_DIR / "outputs/predictions/test_predictions.csv"))
    args = parser.parse_args()
    data_path = resolve_config_path(PROJECT_DIR / "configs/data.yaml")
    model_path = resolve_config_path(PROJECT_DIR / "configs/model.yaml")
    train_path = resolve_config_path(PROJECT_DIR / "configs/train.yaml")
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    model_cfg, train_cfg = read_yaml(model_path), read_yaml(train_path)
    checkpoint_path = resolve_config_path(args.checkpoint)
    csv_file = args.csv or data_cfg.get("test_csv")
    if not csv_file:
        raise ValueError("Set test_csv in configs/data.yaml or pass --csv")
    dataset = StudyDataset(csv_file, data_cfg["image_root"], build_transform(data_cfg["image_size"]),
        label_column=None, label_semantics="unresolved",
        image_id_column=data_cfg["image_id_column"], subject_column=data_cfg["subject_column"],
        study_column=data_cfg["study_column"])
    loader = DataLoader(dataset,
        batch_sampler=StudyBatchSampler(dataset, train_cfg["image_budget"], shuffle=False),
        num_workers=data_cfg["num_workers"], pin_memory=torch.cuda.is_available(),
        collate_fn=collate_studies)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_model_cfg = dict(state.get("config", {}).get("model_config", model_cfg))
    saved_model_cfg["pretrained"] = False  # Do not download weights during evaluation/inference.
    model = ChestXrayClassifier(**{key: value for key, value in saved_model_cfg.items()
                                   if key != "name"}).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    class_ids = data_cfg.get("class_ids", list(range(1, 11)))
    if class_ids != list(range(1, 11)):
        raise ValueError("class_ids must be exactly [1, 2, ..., 10] in official order")
    output_rows = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            study_index = batch["study_index"].to(device, non_blocking=True)
            logits = model(images, study_index, batch["num_studies"])
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("Model generated a non-finite Study probability")
            for index, (subject_id, study_id) in enumerate(zip(batch["subject_id"], batch["study_id"])):
                for class_index, class_id in enumerate(class_ids):
                    output_rows.append({
                        "Subject_id": str(subject_id),
                        "Study_id": str(study_id),
                        "Predict_class": int(class_id),
                        "Probability": float(probabilities[index, class_index]),
                    })
    frame = pd.DataFrame(output_rows, columns=["Subject_id", "Study_id", "Predict_class", "Probability"])
    if not frame["Probability"].between(0.0, 1.0).all():
        raise ValueError("Generated Probability values must be in [0,1]")
    counts = frame.groupby(["Subject_id", "Study_id"], sort=False).size()
    if len(frame) != len(dataset) * 10 or not (counts == 10).all():
        raise ValueError("Output must contain exactly ten rows per test Study")
    expected_classes = set(class_ids)
    if any(set(group["Predict_class"]) != expected_classes
           for _, group in frame.groupby(["Subject_id", "Study_id"], sort=False)):
        raise ValueError("Every Study must contain each class ID 1 through 10 exactly once")
    expected_keys = set(dataset.study_keys)
    actual_keys = set(zip(frame["Subject_id"], frame["Study_id"]))
    if actual_keys != expected_keys:
        raise ValueError("Prediction Study keys do not exactly match the test CSV")
    output = Path(args.output)
    if not output.is_absolute():
        output = (PROJECT_DIR / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, columns=["Subject_id", "Study_id", "Predict_class", "Probability"])
    print(f"Wrote {len(frame)} rows for {len(dataset)} Studies to {output}")


if __name__ == "__main__":
    main()
