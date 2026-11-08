"""Generate the Study-level submission CSV with the R11 Best checkpoint.

Uses the test copy at repo-root ``cxr_first_test`` (test.csv + test/ images).
Only test IDs are read; the test CSV has no labels and no labels are used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

from genopt_common import EXPERIMENT_ROOT, LABEL_NAMES, SEED, study_logit_mean, write_json
from dino_data import EpochTaggedStudyDataset, make_loader, shutdown_loader
from dino_feature_models import MaskedStudyDataset, collate_masked_studies
from datasets.dataset import StudyDataset
from r11_model import build_r11_router

R11_DIR = EXPERIMENT_ROOT / "R11_DiseaseSpecific_SparseFeatureRouting_512"
TEST_ROOT = REPO_ROOT / "cxr_first_test"
TEST_CSV = TEST_ROOT / "test.csv"
OUTPUT_CSV = R11_DIR / "submission.csv"


def _load_router(device: torch.device):
    router = build_r11_router(device)
    payload = torch.load(R11_DIR / "best_checkpoint.pth", map_location="cpu",
                         weights_only=False)
    router.base.load_state_dict(payload["base_state"], strict=True)
    router.branches.load_state_dict(payload["branch_state"], strict=True)
    router.eval()
    router.base.eval()
    return router, payload


def main() -> None:
    device = torch.device("cuda")
    router, payload = _load_router(device)
    base_dataset = StudyDataset(
        TEST_CSV,
        TEST_ROOT,
        transform=None,
        label_semantics="unresolved",
    )
    dataset = EpochTaggedStudyDataset(
        MaskedStudyDataset(base_dataset, 512, train=False), SEED)
    loader, sampler, _ = make_loader(
        dataset, image_budget=8, shuffle=False, seed=SEED, num_workers=4,
        pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    study_logits = []
    subjects = []
    study_ids = []
    started = time.perf_counter()
    image_count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
            logits, _, _ = router(images)
            study = study_logit_mean(
                logits, batch["study_index"].to(device),
                batch["num_studies"], batch["image_counts"].to(device))
            study_logits.append(study.float().cpu())
            subjects.extend([str(value) for value in batch["subject_id"]])
            study_ids.extend([str(value) for value in batch["study_id"]])
            image_count += int(images.shape[0])
    shutdown_loader(loader)
    logits = torch.cat(study_logits, 0).numpy().astype(np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    if probability.shape != (len(dataset), len(LABEL_NAMES)):
        raise RuntimeError(f"Unexpected probability shape {probability.shape}")
    rows = []
    for index, (subject, study_id) in enumerate(zip(subjects, study_ids)):
        for class_index, name in enumerate(LABEL_NAMES):
            rows.append({
                "Subject_id": subject,
                "Study_id": study_id,
                "Predict_class": class_index + 1,
                "Probability": float(probability[index, class_index]),
            })
    frame = pd.DataFrame(rows, columns=["Subject_id", "Study_id",
                                        "Predict_class", "Probability"])
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8",
                 float_format="%.10f")
    metadata = {
        "experiment": "R11",
        "checkpoint": str(R11_DIR / "best_checkpoint.pth"),
        "checkpoint_best_epoch": int(payload.get("best_epoch", 0)),
        "checkpoint_best_macro_auroc": float(payload.get("best_macro_auroc", 0.0)),
        "test_csv": str(TEST_CSV),
        "test_studies": int(len(dataset)),
        "test_images": int(image_count),
        "test_subjects": int(len(set(subjects))),
        "mode": "R11 Full (all ten disease gates enabled)",
        "class_mapping": {str(index + 1): name for index, name in
                          enumerate(LABEL_NAMES)},
        "probability_columns": "one row per (Subject_id, Study_id, Predict_class)",
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "elapsed_seconds": time.perf_counter() - started,
        "output_csv": str(OUTPUT_CSV),
        "test_access": "test images and IDs only; no test labels used",
    }
    write_json(R11_DIR / "submission_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
