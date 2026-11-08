"""Smoke-test the three planned ImageNet-initialized DenseNet experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from engine.losses import MaskedBCEWithLogitsLoss
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path
from utils.seed import seed_everything


EXPERIMENTS = [
    ("gap", "densenet_low_lr_gap_model.yaml", "vector_classifier.fc.weight"),
    ("lca", "densenet_low_lr_lca_model.yaml", "classifier.queries"),
    ("cbam", "densenet_low_lr_cbam_model.yaml", "cbam.spatial.weight"),
]
EXPECTED_WEIGHT_SHA256 = (
    "A639EC97D7C33B07AE66F0B5FB7D0192F95A3B11B7576C66C0126C2A727C4395"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This smoke test requires the configured CUDA BF16 device")
    device = torch.device("cuda")
    weight_path = Path(torch.hub.get_dir()) / "checkpoints/densenet121-a639ec97.pth"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    weight_digest = sha256(weight_path)
    if weight_digest != EXPECTED_WEIGHT_SHA256:
        raise ValueError(f"Unexpected DenseNet ImageNet weight SHA-256: {weight_digest}")

    reference_backbone = None
    results = []
    criterion = MaskedBCEWithLogitsLoss()
    images = torch.randn(3, 3, 320, 320, device=device)
    study_index = torch.tensor([0, 0, 1], dtype=torch.long, device=device)
    targets = torch.tensor(
        [[1, 0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]],
        dtype=torch.float32,
        device=device,
    )
    known = torch.ones_like(targets, dtype=torch.bool)

    for name, model_config_name, required_gradient in EXPERIMENTS:
        seed_everything(42)
        model_config = read_yaml(
            resolve_config_path(PROJECT_DIR / "configs/experiments" / model_config_name)
        )
        if model_config.get("pretrained") is not True or model_config.get("study_fusion") != "logit_mean":
            raise ValueError(f"{name} smoke config does not use ImageNet weights and Logit Mean")

        model = ChestXrayClassifier(**{key: value for key, value in model_config.items() if key != "name"})
        model = model.to(device).train()
        current_backbone = model.backbone.state_dict()
        if reference_backbone is None:
            reference_backbone = {
                key: value.detach().cpu().clone() for key, value in current_backbone.items()
            }
        elif any(
            not torch.equal(reference_backbone[key], value.detach().cpu())
            for key, value in current_backbone.items()
        ):
            raise ValueError(f"{name} did not receive the same torchvision ImageNet backbone weights")

        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images, study_index, num_studies=2)
        loss = criterion(logits.float(), targets, known)
        if logits.shape != (2, 10) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced invalid Study logits or loss")
        loss.backward()

        named_parameters = dict(model.named_parameters())
        required_parameter = named_parameters[required_gradient]
        if (
            required_parameter.grad is None
            or not torch.isfinite(required_parameter.grad).all()
            or required_parameter.grad.detach().norm().item() == 0.0
        ):
            raise RuntimeError(f"{name} required parameter has missing or non-finite gradient")
        if name == "cbam":
            second = model.cbam.channel_mlp[0].weight
            if (
                second.grad is None
                or not torch.isfinite(second.grad).all()
                or second.grad.detach().norm().item() == 0.0
            ):
                raise RuntimeError("CBAM channel-attention gradient is missing or non-finite")
        if name == "lca":
            attention_gradient = model.classifier.cross_attention.in_proj_weight.grad
            if (
                attention_gradient is None
                or not torch.isfinite(attention_gradient).all()
                or attention_gradient.detach().norm().item() == 0.0
            ):
                raise RuntimeError("LCA cross-attention gradient is missing or non-finite")

        results.append(
            {
                "experiment": name,
                "logits_shape": list(logits.shape),
                "loss": float(loss.detach().cpu()),
                "required_gradient_parameter": required_gradient,
                "required_gradient_l2": float(required_parameter.grad.detach().norm().cpu()),
                "backbone_sha256": weight_digest,
            }
        )
        del model, logits, loss
        torch.cuda.empty_cache()

    report = {
        "device": torch.cuda.get_device_name(0),
        "input_size": 320,
        "study_index": [0, 0, 1],
        "study_count": 2,
        "amp_dtype": "bfloat16",
        "imagenet_weight_file": str(weight_path),
        "imagenet_weight_sha256": weight_digest,
        "results": results,
    }
    output = PROJECT_DIR / "outputs/experiments/attention_smoke_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
