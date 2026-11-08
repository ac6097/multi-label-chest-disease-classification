from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_diagnostics as diagnostics


HERE = Path(__file__).resolve().parent
WORK_DIR = diagnostics.WORK_DIR
REPO_DIR = diagnostics.REPO_DIR
TEST_ROOT = REPO_DIR / "cxr_first_test"
TEST_CSV = TEST_ROOT / "test.csv"
OUTPUT = diagnostics.RUNS["R3"] / "submission.csv"
IMAGE_BUDGET = 8
SEED = 42
LABEL_IDS = tuple(range(1, 11))

sys.path.insert(0, str(WORK_DIR.parent))

from datasets.dataset import StudyDataset
from dino_data import EpochTaggedStudyDataset, make_loader, shutdown_loader
from dino_feature_models import MaskedStudyDataset, collate_masked_studies
from dino_losses import study_logit_mean


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the R3 512x512 Test inference")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"Test-copy CSV is missing: {TEST_CSV}")

    # The Test copy is used only for image IDs and Study grouping; no labels are loaded.
    base = StudyDataset(
        TEST_CSV,
        TEST_ROOT,
        transform=None,
        label_column=None,
        label_semantics="unresolved",
        image_id_column="Dicom_id",
        subject_column="Subject_id",
        study_column="Study_id",
    )
    if base.split != "test":
        raise RuntimeError(f"Expected test split, got {base.split!r}")
    if not base.studies:
        raise RuntimeError("Test copy has no Studies")
    keys = list(base.study_keys)
    if len(keys) != len(set(keys)):
        raise RuntimeError("Test copy contains duplicate (Subject_id, Study_id) keys")
    missing = [str(path) for study in base.studies for path in study["image_paths"]
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Test copy is missing {len(missing)} images; examples: {missing[:10]}")
    image_count = int(sum(base.image_counts))

    device = torch.device("cuda")
    model, checkpoint_metadata, _ = diagnostics.build_model("R3", device)
    if checkpoint_metadata["checkpoint"] != str(diagnostics.CHECKPOINTS["R3"]):
        raise RuntimeError("Loaded checkpoint is not the R3 Best Checkpoint")
    if checkpoint_metadata["checkpoint_epoch"] != 6:
        raise RuntimeError(f"R3 Best Checkpoint epoch is unexpected: {checkpoint_metadata}")
    if not checkpoint_metadata["training_runtime_channels_last"]:
        raise RuntimeError("R3 training runtime records channels_last=true; refusing a mismatched inference layout")

    dataset = EpochTaggedStudyDataset(
        MaskedStudyDataset(base, size=512, train=False), SEED)
    loader, _, _ = make_loader(
        dataset,
        image_budget=IMAGE_BUDGET,
        shuffle=False,
        seed=SEED,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_masked_studies,
    )
    model.eval()
    predictions = {}
    processed_images = 0
    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                images = batch["images"].to(device, non_blocking=True)
                images = images.contiguous(memory_format=torch.channels_last)
                masks = batch["valid_masks"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    image_logits = model(images, masks)
                study_logits = study_logit_mean(
                    image_logits.float(),
                    batch["study_index"].to(device, non_blocking=True),
                    batch["num_studies"],
                    batch["image_counts"].to(device, non_blocking=True),
                )
                probabilities = torch.sigmoid(study_logits).float().cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise FloatingPointError("R3 generated non-finite Test probabilities")
                if ((probabilities < 0.0) | (probabilities > 1.0)).any():
                    raise ValueError("R3 generated probabilities outside [0,1]")
                for index, key in enumerate(zip(batch["subject_id"], batch["study_id"])):
                    key = (str(key[0]), str(key[1]))
                    if key in predictions:
                        raise RuntimeError(f"Test Study was inferred twice: {key}")
                    predictions[key] = probabilities[index]
                processed_images += int(images.shape[0])
                if batch_number % 50 == 0:
                    print(f"R3 Test inference: {batch_number} batches, "
                          f"{len(predictions)} Studies, {processed_images} images",
                          flush=True)
    finally:
        shutdown_loader(loader)

    if processed_images != image_count:
        raise RuntimeError(f"Processed {processed_images} images, expected {image_count}")
    if set(predictions) != set(keys):
        missing_keys = sorted(set(keys) - set(predictions))
        extra_keys = sorted(set(predictions) - set(keys))
        raise RuntimeError(
            f"Test Study coverage mismatch: missing={missing_keys[:10]}, extra={extra_keys[:10]}")

    rows = []
    for key in keys:
        for index, class_id in enumerate(LABEL_IDS):
            rows.append({
                "Subject_id": key[0],
                "Study_id": key[1],
                "Predict_class": class_id,
                "Probability": float(predictions[key][index]),
            })
    frame = pd.DataFrame(
        rows,
        columns=["Subject_id", "Study_id", "Predict_class", "Probability"],
    )
    if len(frame) != len(base) * len(LABEL_IDS):
        raise RuntimeError("Submission row count is not exactly ten rows per Test Study")
    if frame.duplicated(["Subject_id", "Study_id", "Predict_class"]).any():
        raise RuntimeError("Submission contains duplicate Study/class rows")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, OUTPUT)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"R3 Best Checkpoint: {checkpoint_metadata['checkpoint']}")
    print(f"Test-copy Studies: {len(base)}; images: {image_count}; rows: {len(frame)}")
    print(f"Submission: {OUTPUT}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
