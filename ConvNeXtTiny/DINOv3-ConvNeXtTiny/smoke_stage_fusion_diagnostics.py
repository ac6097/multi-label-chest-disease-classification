"""Smoke-test the per-epoch diagnostics callback for R8/R9/R10 on real data."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dino_experiment import load_pretrained_backbone, save_json_atomic
from dino_feature_models import build_masked_datasets
from hybrid.common import seed_everything
from stage_fusion_experiment import (
    IMAGE_SIZE,
    SEED,
    StageFusionEpochDiagnostics,
    capture_fixed_validation_batch,
    freeze_layout,
    load_r5_contract,
    make_optimizer,
    write_json,
)
from stage_fusion_models import build_model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the diagnostics smoke test")
    contract = load_r5_contract()
    runtime = contract["runtime"]
    source = (HERE / contract["experiment_config"]["data"]["source_config"]).resolve()
    _, valid_data, _ = build_masked_datasets(source, SEED, IMAGE_SIZE)
    sample = capture_fixed_validation_batch(valid_data, runtime)
    device = torch.device("cuda")
    for experiment in ("R8", "R9", "R10"):
        seed_everything(SEED)
        backbone, verification = load_pretrained_backbone("cpu")
        model = build_model(experiment, backbone).to(device)
        freeze_layout(experiment, model)
        config = dict(contract["experiment_config"])
        config["freeze"] = contract["freeze"]
        optimizer = make_optimizer(experiment)(model, config)
        optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = StageFusionEpochDiagnostics(
                experiment, Path(directory), sample, runtime)
            diagnostics(model, 1, "start", None)
            row = diagnostics(model, 1, "end", {"lr_head": 2e-4})
            print(experiment, "diagnostics row keys:", sorted(row.keys()))
            for filename in ("branch_diagnostics.csv", "alpha_by_epoch.csv",
                             "stage3_logit_contribution.csv",
                             "alpha_by_disease.png",
                             "branch_contribution_curves.png"):
                path = Path(directory) / filename
                print(" ", filename, "exists" if path.is_file() else "MISSING",
                      path.stat().st_size if path.is_file() else "")
            if experiment in {"R9", "R10"}:
                for filename in ("attention_statistics.csv", "query_similarity.csv"):
                    path = Path(directory) / filename
                    print(" ", filename, "exists" if path.is_file() else "MISSING")
            if experiment == "R10":
                for filename in ("stage4_query_conditioning.csv",
                                 "conditioned_vs_unconditioned_attention.csv"):
                    path = Path(directory) / filename
                    print(" ", filename, "exists" if path.is_file() else "MISSING")
        del model, optimizer
        torch.cuda.empty_cache()
    print("DIAGNOSTICS_SMOKE_OK")


if __name__ == "__main__":
    main()
