"""Collect environment / checkpoint facts for the G0/G1 audit (read-only)."""

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from dino_experiment import load_pretrained_backbone  # noqa: E402

OUT = HERE / "outputs" / "experiments" / "DINOv3_Feature_Optimization"
R3 = OUT / "R3_Global_Local"
R5 = OUT / "R5_Global_Only_512"


def checkpoint_summary(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    summary = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "keys": sorted(payload.keys()),
        "epoch": payload.get("epoch"),
        "best_epoch": payload.get("best_epoch"),
        "best_macro_auroc": payload.get("best_macro_auroc"),
        "profile": payload.get("profile"),
        "experiment": payload.get("experiment"),
        "runtime": payload.get("runtime"),
        "run_context": payload.get("run_context"),
        "initial_weights_sha256": payload.get("initial_weights_sha256"),
        "state_tensor_count": len(payload.get("model_state", {})),
    }
    state = payload["model_state"]
    if "alpha.logit" in state:
        alpha_raw = state["alpha.logit"].detach().float().item()
        import math

        summary["alpha_logit_raw"] = alpha_raw
        summary["alpha_value"] = 0.2 * (1.0 / (1.0 + math.exp(-alpha_raw)))
    return summary


def main():
    r3_config = json.loads((R3 / "config.json").read_text(encoding="utf-8"))
    r5_config = json.loads((R5 / "config.json").read_text(encoding="utf-8"))
    r3_status = json.loads((R3 / "training_status.json").read_text(encoding="utf-8"))
    r5_status = json.loads((R5 / "training_status.json").read_text(encoding="utf-8"))
    r3_verify = json.loads((R3 / "model_verification.json").read_text(encoding="utf-8"))
    r5_verify = json.loads((R5 / "model_verification.json").read_text(encoding="utf-8"))
    r3_model_config = json.loads(
        (R3 / "submission_export" / "R3_Inference" / "model_config.json").read_text(
            encoding="utf-8"))
    r5_model_config = json.loads(
        (R5 / "submission_export" / "R5_Inference" / "model_config.json").read_text(
            encoding="utf-8"))
    backbone, verification = load_pretrained_backbone("cpu")
    payload = {
        "workspace": str(HERE),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "pretrained_weight_verification": verification,
        "data_config": {
            "source": "../configs/hybrid_data.yaml",
            "train_csv": "../cxr_first/train.csv",
            "val_csv": "../cxr_first/valid.csv",
            "image_root": "../cxr_first",
            "label_semantics": "closed_world",
            "image_size_override": 512,
        },
        "r3": {
            "config": r3_config,
            "status": r3_status,
            "model_verification": r3_verify,
            "package_model_config": r3_model_config,
            "checkpoint": checkpoint_summary(R3 / "best.pth"),
        },
        "r5": {
            "config": r5_config,
            "status": r5_status,
            "model_verification": r5_verify,
            "package_model_config": r5_model_config,
            "checkpoint": checkpoint_summary(R5 / "best.pth"),
        },
    }
    del backbone
    output = HERE / "outputs" / "experiments" / "DINOv3_Generalization_Diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment_and_checkpoint_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "r3_checkpoint": payload["r3"]["checkpoint"],
        "r5_checkpoint": payload["r5"]["checkpoint"],
        "pretrained": {key: verification[key] for key in (
            "repository", "revision", "weights_sha256", "weights_size_bytes")},
        "torch": payload["torch"],
        "gpu": payload["gpu"],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
