from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, collate_studies
from datasets.hybrid import build_hybrid_transform
from hybrid.backbone_factory import build_hybrid_backbone
from hybrid.common import PROJECT_DIR
from model.hybrid import HybridClassifier


COLUMNS = ["Subject_id", "Study_id", "Predict_class", "Probability"]
CLASS_IDS = list(range(1, 11))


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Study-level predictions from a trained Hybrid H1 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_DIR / "outputs/hybrid/h1/best.pth"),
        help="Trained Hybrid checkpoint.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_DIR.parent / "cxr_first_test/test.csv"),
        help="Test-copy CSV. Labels are not read.",
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_DIR.parent / "cxr_first_test"),
        help="Root containing test/<Subject_id>/<Study_id>/<Dicom_id>.jpg.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=str(PROJECT_DIR / "outputs/hybrid/shared/imagenet_initial.pth"),
        help="Local shared initialization used to instantiate the H1 architecture.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "outputs/hybrid/h1/predictions/submission.csv"),
        help="Destination submission CSV.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-budget", type=int, default=8)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the destination only when explicitly requested.",
    )
    args = parser.parse_args()

    checkpoint_path = project_path(args.checkpoint)
    test_csv = Path(args.csv).resolve()
    image_root = Path(args.image_root).resolve()
    initial_checkpoint = project_path(args.initial_checkpoint)
    output_path = project_path(args.output)

    for path in (checkpoint_path, test_csv, initial_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite to replace it."
        )

    with (PROJECT_DIR / "configs/data.yaml").open("r", encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream) or {}
    if data_config.get("class_ids", CLASS_IDS) != CLASS_IDS:
        raise ValueError("Official class order must be IDs 1 through 10.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config", {})
    model_config = saved_config.get("model", {})
    if model_config.get("convnext_model_name") != "convnextv2_tiny.fcmae_ft_in1k":
        raise ValueError("Checkpoint is not the expected ConvNeXtV2-Tiny H1 model.")
    if model_config.get("vit_model_name") != "vit_small_patch16_224.augreg_in1k":
        raise ValueError("Checkpoint does not contain the expected ViT-Small/16 branch.")
    if int(model_config.get("num_labels", 10)) != 10:
        raise ValueError("Checkpoint must predict exactly ten labels.")

    training_config = saved_config.get("training", {})
    image_size = int(model_config.get("image_size", 320))
    image_budget = int(args.image_budget or training_config.get("image_budget", 8))
    if image_budget < 1:
        raise ValueError("--image-budget must be positive.")

    backbone = build_hybrid_backbone(initial_checkpoint, map_location="cpu")
    model = HybridClassifier(backbone, num_labels=10)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    dataset = StudyDataset(
        test_csv,
        image_root,
        transform=build_hybrid_transform(image_size, train=False),
        label_column=None,
        label_semantics="unresolved",
        image_id_column=data_config.get("image_id_column", "Dicom_id"),
        subject_column=data_config.get("subject_column", "Subject_id"),
        study_column=data_config.get("study_column", "Study_id"),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=StudyBatchSampler(dataset, image_budget, shuffle=False),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_studies,
    )

    rows = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            study_index = batch["study_index"].to(device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 else nullcontext()
            )
            with autocast:
                logits = model(images, study_index, batch["num_studies"])
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            if not torch.isfinite(torch.from_numpy(probabilities)).all():
                raise ValueError("Model produced non-finite probabilities.")
            for index, (subject_id, study_id) in enumerate(
                zip(batch["subject_id"], batch["study_id"])
            ):
                for class_index, class_id in enumerate(CLASS_IDS):
                    rows.append({
                        "Subject_id": str(subject_id),
                        "Study_id": str(study_id),
                        "Predict_class": class_id,
                        "Probability": float(probabilities[index, class_index]),
                    })

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if len(frame) != len(dataset) * 10:
        raise ValueError("Expected exactly ten prediction rows for every test Study.")
    if not frame["Probability"].between(0.0, 1.0).all():
        raise ValueError("Probability values must be in [0, 1].")
    if not (frame.groupby(["Subject_id", "Study_id"], sort=False).size() == 10).all():
        raise ValueError("Every Study must have exactly ten class rows.")
    expected_keys = set(dataset.study_keys)
    actual_keys = set(zip(frame["Subject_id"], frame["Study_id"]))
    if actual_keys != expected_keys:
        raise ValueError("Predicted Study keys do not match the test-copy CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.name + f".{os.getpid()}.tmp"
    )
    frame.to_csv(temporary_path, index=False, columns=COLUMNS)
    temporary_path.replace(output_path)
    print(
        f"Wrote {len(frame)} rows for {len(dataset)} Studies to {output_path} "
        f"(device={device}, fusion=arithmetic mean of image logits)."
    )


if __name__ == "__main__":
    main()
