"""Measure Study-level model latency after a checkpoint exists; does not train."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from datasets import StudyDataset, build_transform, collate_studies
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


def percentile_summary(values):
    return {"mean_ms": float(np.mean(values) * 1000),
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "max_ms": float(np.max(values) * 1000)}


def main():
    parser = argparse.ArgumentParser(description="Measure configured Study-level inference latency")
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--csv", default=None, help="Defaults to test.csv; labels are not loaded")
    parser.add_argument("--warmup-studies", type=int, default=10)
    parser.add_argument("--max-studies", type=int, default=100)
    parser.add_argument("--output", default=str(PROJECT_DIR / "outputs/metrics/latency.json"))
    args = parser.parse_args()

    data_path = resolve_config_path(PROJECT_DIR / "configs/data.yaml")
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    csv_file = args.csv or data_cfg["test_csv"]
    dataset = StudyDataset(csv_file, data_cfg["image_root"],
        build_transform(data_cfg["image_size"]), label_column=None,
        image_id_column=data_cfg["image_id_column"], subject_column=data_cfg["subject_column"],
        study_column=data_cfg["study_column"])
    checkpoint = resolve_config_path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = dict(state.get("config", {}).get("model_config", read_yaml(
        resolve_config_path(PROJECT_DIR / "configs/model.yaml"))))
    model_config["pretrained"] = False  # The checkpoint already contains the initialized backbone.
    model = ChestXrayClassifier(**{key: value for key, value in model_config.items()
                                   if key != "name"}).to(device).eval()
    model.load_state_dict(state["model"])

    warmup_studies = max(int(args.warmup_studies), 0)
    total = min(len(dataset), max(args.max_studies, 1) + warmup_studies)
    preprocessing_times, transfer_times, forward_times, image_counts = [], [], [], []
    with torch.inference_mode():
        for index in range(total):
            started = time.perf_counter()
            item = dataset[index]
            batch = collate_studies([item])
            preprocessing_elapsed = time.perf_counter() - started
            transfer_started = time.perf_counter()
            images = batch["images"].to(device, non_blocking=True)
            study_index = batch["study_index"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_elapsed = time.perf_counter() - transfer_started
            forward_started = time.perf_counter()
            _ = model(images, study_index, batch["num_studies"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_elapsed = time.perf_counter() - forward_started
            if index >= warmup_studies:
                preprocessing_times.append(preprocessing_elapsed)
                transfer_times.append(transfer_elapsed)
                forward_times.append(forward_elapsed)
                image_counts.append(int(batch["image_counts"][0]))

    if not forward_times:
        raise ValueError("No measured Studies; increase --max-studies or reduce --warmup-studies")
    report = {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "study_fusion": model_config.get("study_fusion", "logit_mean"),
        "input_size": data_cfg["image_size"],
        "measured_studies": len(forward_times),
        "study_image_count": {"mean": float(np.mean(image_counts)),
                              "p50": float(np.percentile(image_counts, 50)),
                              "max": int(np.max(image_counts))},
        "preprocessing_plus_decode": percentile_summary(preprocessing_times),
        "host_to_device_transfer": percentile_summary(transfer_times),
        "model_forward_and_fusion_per_study": percentile_summary(forward_times),
        "model_forward_ms_per_input_image_mean": float(
            np.mean(np.asarray(forward_times) / np.asarray(image_counts)) * 1000),
        "timing_scope": ("Study-level single-item call; the official rule is <=100 ms end-to-end "
                         "per image, while multi-image Study accounting is unspecified."),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = (PROJECT_DIR / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
