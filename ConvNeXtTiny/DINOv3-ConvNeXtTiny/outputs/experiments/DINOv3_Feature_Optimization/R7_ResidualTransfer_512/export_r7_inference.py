"""Build the size-limited, standalone R7 inference package from its Best checkpoint."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file


HERE = Path(__file__).resolve().parent
WORK_DIR = HERE.parents[3]
PROJECT_DIR = WORK_DIR.parent
R5_PACKAGE = (HERE.parent / "R5_Global_Only_512" / "submission_export" / "R5_Inference")
EXPORT_DIR = HERE / "submission_export"
PACKAGE_DIR = EXPORT_DIR / "R7_Inference"
ARCHIVE_PATH = EXPORT_DIR / "DINOv3_ConvNeXtTiny_R7_Inference.zip"
SIZE_LIMIT_BYTES = 500_000_000
WEIGHT_SHA256 = "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main():
    status = read_json(HERE / "training_status.json")
    config = read_json(HERE / "config.json")
    runtime = read_json(HERE / "selected_runtime_config.json")["selected_runtime"]
    groups = read_json(HERE / "optimizer_parameter_groups.json")
    if not status.get("complete"):
        raise RuntimeError("Cannot export an incomplete R7 run")
    if config.get("initialization_sha256") != WEIGHT_SHA256:
        raise RuntimeError("R7 did not use the pinned original DINOv3 weights")
    if not R5_PACKAGE.is_dir():
        raise FileNotFoundError(f"R5 inference package is missing: {R5_PACKAGE}")

    checkpoint = torch.load(HERE / "best.pth", map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != int(status["best_epoch"]):
        raise RuntimeError("R7 Best checkpoint epoch differs from training_status.json")
    state = checkpoint.get("model_state")
    if not state:
        raise RuntimeError("R7 Best checkpoint has no model_state")
    if not any(name.startswith("transfer.projection.") for name in state):
        raise RuntimeError("R7 Best checkpoint is missing the residual projection")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    (PACKAGE_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    for name in ("dino_model.py", "dinov3_meta_convnext.py", "LICENSE.md",
                 "requirements-inference.txt"):
        shutil.copy2(R5_PACKAGE / name, PACKAGE_DIR / name)
    shutil.copy2(PROJECT_DIR / "datasets" / "transforms.py",
                 PACKAGE_DIR / "datasets" / "transforms.py")
    shutil.copy2(PROJECT_DIR / "datasets" / "dataset.py",
                 PACKAGE_DIR / "datasets" / "dataset.py")
    (PACKAGE_DIR / "datasets" / "__init__.py").write_text(
        '"""Minimal dataset package used by standalone R7 inference."""\n',
        encoding="utf-8")
    shutil.copy2(HERE / "r7_model.py", PACKAGE_DIR / "r7_model.py")
    shutil.copy2(HERE / "r7_inference.py", PACKAGE_DIR / "r7_inference.py")

    state_cpu = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items()
    }
    save_file(state_cpu, str(PACKAGE_DIR / "model.safetensors"))

    model_config = {
        "group": "R7_ResidualTransfer_512",
        "head_type": "r7_stage3_stage4_residual_transfer",
        "backbone": "DINOv3 ConvNeXt-Tiny",
        "architecture": "convnext_tiny",
        "image_size": 512,
        "num_labels": 10,
        "label_names": [
            "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
            "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
            "Lung Opacity", "Pleural Effusion", "No Finding",
        ],
        "layer_norm_eps": 1e-6,
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.0,
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "study_fusion": "mean raw image logits before sigmoid",
        "channels_last": bool(runtime["channels_last"]),
        "best_epoch": int(status["best_epoch"]),
        "best_validation_macro_auroc": float(status["best_macro_auroc"]),
        "pretrained_initialization_sha256": WEIGHT_SHA256,
        "residual_transfer": {
            "operation": "AvgPool2d(2,2) -> Conv2d(384,768,1,bias=False)",
            "fusion": "F4_fused = F4 + gamma*T(F3)",
            "gamma_best": float(state["transfer.gamma"].detach().cpu().item()),
        },
        "parameter_count": int(groups["total_parameters"]),
        "trainable_parameter_count": int(groups["trainable_parameters"]),
        "frozen_parameter_count": int(groups["frozen_parameters"]),
        "optimizer_or_scheduler_included": False,
        "test_data_accessed": False,
    }
    write_json(PACKAGE_DIR / "model_config.json", {"model": model_config})
    (PACKAGE_DIR / "README.md").write_text(
        "# DINOv3 ConvNeXt-Tiny R7 inference package\n\n"
        "`R7Predictor` loads the packaged Best checkpoint and returns ten Study-level "
        "probabilities. Pass the PIL images or file paths belonging to one Study to "
        "`predict_study_images` or `predict_study_paths`. Study aggregation is the "
        "arithmetic mean of raw image logits followed by sigmoid.\n\n"
        "The package contains the Best model weights only; optimizer and scheduler "
        "states are excluded. The model was initialized from the pinned general-vision "
        "DINOv3 ConvNeXt-Tiny checkpoint identified in `model_config.json`.\n",
        encoding="utf-8")

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(EXPORT_DIR))
    archive_bytes = ARCHIVE_PATH.stat().st_size
    if archive_bytes >= SIZE_LIMIT_BYTES:
        raise RuntimeError(
            f"R7 inference archive is {archive_bytes:,} bytes; must be below "
            f"{SIZE_LIMIT_BYTES:,} bytes")

    metrics = {
        "best_checkpoint_bytes": (HERE / "best.pth").stat().st_size,
        "inference_weights_bytes": (PACKAGE_DIR / "model.safetensors").stat().st_size,
        "archive_path": str(ARCHIVE_PATH),
        "archive_bytes": archive_bytes,
        "size_limit_bytes": SIZE_LIMIT_BYTES,
        "passes_strict_limit": archive_bytes < SIZE_LIMIT_BYTES,
        "total_parameters": int(groups["total_parameters"]),
        "trainable_parameters": int(groups["trainable_parameters"]),
        "frozen_parameters": int(groups["frozen_parameters"]),
        "added_projection_parameters": int(groups["r7_projection_parameters"]),
        "added_gamma_parameters": int(groups["r7_gamma_parameters"]),
        "test_data_accessed": False,
    }
    write_json(HERE / "r7_model_size_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
