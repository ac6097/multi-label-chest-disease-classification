"""Measure R11 inference throughput, peak memory and score-distribution stats."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genopt_common import EXPERIMENT_ROOT, LABEL_NAMES, build_h_datasets, write_json
from dino_data import make_loader, shutdown_loader
from dino_feature_models import collate_masked_studies
from r11_analysis import _load_router

R11_DIR = EXPERIMENT_ROOT / "R11_DiseaseSpecific_SparseFeatureRouting_512"


def main() -> None:
    device = torch.device("cuda")
    router = _load_router(device, "best")
    valid_data = build_h_datasets()[1]
    loader, _, _ = make_loader(valid_data, image_budget=8, shuffle=False, seed=42,
                            num_workers=4, pin_memory=True, prefetch_factor=2,
                            collate_fn=collate_masked_studies)
    score_stats = []
    images_seen = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device)
            images = images.contiguous(memory_format=torch.channels_last)
            base_logits, stage3 = router.base_logits_and_stage3(images)
            for index, branch in enumerate(router.branches):
                channel_scores, spatial_scores = branch.scores(stage3)
                channel_soft = torch.softmax(channel_scores.float(), dim=-1)
                spatial_soft = torch.softmax(
                    spatial_scores.flatten(1).float(), dim=-1)
                score_stats.append({
                    "disease": LABEL_NAMES[index],
                    "channel_score_mean": float(channel_scores.mean()),
                    "channel_score_std": float(channel_scores.std()),
                    "channel_score_p05": float(torch.quantile(
                        channel_scores.flatten(), 0.05)),
                    "channel_score_p95": float(torch.quantile(
                        channel_scores.flatten(), 0.95)),
                    "channel_entropy": float(-(channel_soft * channel_soft.clamp_min(1e-12).log()).sum(-1).mean()),
                    "spatial_score_mean": float(spatial_scores.mean()),
                    "spatial_score_std": float(spatial_scores.std()),
                    "spatial_entropy": float(-(spatial_soft * spatial_soft.clamp_min(1e-12).log()).sum(-1).mean()),
                })
            images_seen += int(images.shape[0])
            if batch_index >= 39:
                break
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak = 0.0
    shutdown_loader(loader)
    # Aggregate score stats by disease.
    aggregated = {}
    for name in LABEL_NAMES:
        rows = [row for row in score_stats if row["disease"] == name]
        aggregated[name] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0] if key != "disease"
        }
    write_json(R11_DIR / "routing_score_diagnostics.json", aggregated)
    write_json(R11_DIR / "deployment_metrics.json", {
        "measured_images": images_seen,
        "measured_batches": 40,
        "seconds": elapsed,
        "images_per_second": images_seen / elapsed,
        "peak_gpu_allocated_gib": peak,
        "best_checkpoint_mb": (R11_DIR / "best_checkpoint.pth").stat().st_size
        / (1024 ** 2),
        "branch_parameters": sum(p.numel() for p in router.parameters()
                                 if p.requires_grad),
        "note": "training-loop peak memory was not logged by the training "
                "driver; this is the inference measurement on the same 8GB GPU",
        "test_access": False,
    })
    print(json.dumps({"images_per_second": images_seen / elapsed,
                      "peak_gpu_allocated_gib": peak}, indent=1))


if __name__ == "__main__":
    main()
