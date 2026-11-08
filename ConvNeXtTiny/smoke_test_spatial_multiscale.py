import argparse
import io
import json
from pathlib import Path

import torch

from model import ChestXrayClassifier
from utils.config import PROJECT_DIR


VARIANTS = (
    ("a0", "stage4", "gap", "ConvNeXt_Spatial_A0_GAP_BCE20"),
    ("a1", "stage4", "spp", "ConvNeXt_Spatial_A1_SPP_BCE20"),
    ("b1", "stage3_stage4", "gap", "ConvNeXt_Spatial_B1_Stage3Stage4_GAP_BCE20"),
    ("c1", "stage3_stage4", "spp", "ConvNeXt_Spatial_C1_Stage3Stage4_SPP_BCE20"),
)


def smoke_variant(name, feature_mode, pooling_mode, output_name, device):
    model_config = {
        "num_labels": 10,
        "pretrained": False,
        "head": "spatial",
        "feature_mode": feature_mode,
        "pooling_mode": pooling_mode,
        "cbam": False,
        "study_fusion": "logit_mean",
    }
    model = ChestXrayClassifier(**model_config).to(device).train()
    images = torch.randn(2, 3, 320, 320, device=device)
    study_index = torch.tensor([0, 0], device=device)
    captured = {}
    call_count = 0

    def capture_backbone(_module, _inputs, output):
        nonlocal call_count
        call_count += 1
        for stage in ("stage3", "stage4"):
            captured[stage] = tuple(output[stage].shape)
            if output[stage].requires_grad:
                output[stage].retain_grad()

    per_image_logits = []
    def capture_head(_module, _inputs, output):
        per_image_logits.append(output.detach())

    backbone_hook = model.backbone.register_forward_hook(capture_backbone)
    head_hook = model.classifier.register_forward_hook(capture_head)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logits = model(images, study_index, num_studies=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    expected_mean = per_image_logits[0].mean(dim=0, keepdim=True)
    if tuple(logits.shape) != (1, 10):
        raise AssertionError(f"{name}: expected Study logits [1,10], got {tuple(logits.shape)}")
    if not torch.allclose(logits.detach(), expected_mean, atol=1e-6, rtol=1e-5):
        raise AssertionError(f"{name}: Study Logit Mean differs from the mean image logits")
    if call_count != 1:
        raise AssertionError(f"{name}: backbone executed {call_count} times in one model forward")
    if captured["stage3"] != (2, 384, 20, 20):
        raise AssertionError(f"{name}: unexpected Stage 3 shape {captured['stage3']}")
    if captured["stage4"] != (2, 768, 10, 10):
        raise AssertionError(f"{name}: unexpected Stage 4 shape {captured['stage4']}")

    logits.square().mean().backward()
    for stage in ("stage3", "stage4"):
        gradient = next(getattr(model.backbone, stage).parameters()).grad
        if gradient is None or not torch.isfinite(gradient).all() or float(gradient.abs().sum()) == 0.0:
            raise AssertionError(f"{name}: {stage} backbone gradients are missing, nonfinite, or zero")
    if feature_mode == "stage3_stage4":
        for parameter in model.classifier.fusion.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"{name}: spatial fusion gradients are missing or nonfinite")
    for parameter in model.classifier.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"{name}: classifier gradients are missing or nonfinite")
    backbone_hook.remove()
    head_hook.remove()

    # Verify the full model state can be serialized and reloaded without a filesystem copy.
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "model_config": model_config}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location=device, weights_only=False)
    reloaded = ChestXrayClassifier(**saved["model_config"]).to(device)
    reloaded.load_state_dict(saved["model"], strict=True)

    peak_gpu_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                   if device.type == "cuda" else 0.0)
    summary = {
        "passed": True,
        "variant": name,
        "model_config": model_config,
        "smoke_initialization": "random weights; architecture only",
        "pretrained_weights_loaded_in_smoke_test": False,
        "backbone_forward_calls_per_call": call_count,
        "backbone_stage3_shape": list(captured["stage3"]),
        "backbone_stage4_shape": list(captured["stage4"]),
        "study_logit_shape": list(logits.shape),
        "study_logit_mean_verified": True,
        "backward_gradients_verified": True,
        "checkpoint_serialization_reload_verified": True,
        "smoke_peak_gpu_gb": peak_gpu_gb,
        "device": str(device),
    }
    out_dir = PROJECT_DIR / "outputs" / "experiments" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    del model, reloaded, images, logits, expected_mean
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Forward/backward smoke tests for spatial ablations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    summaries = [smoke_variant(*variant, device) for variant in VARIANTS]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
