"""Optional synthetic ConvNeXt Study-bag and AMP smoke check; it reads no competition data."""
import argparse

import torch

from engine.losses import MaskedBCEWithLogitsLoss
from model import ChestXrayClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--head", choices=["gap", "multiscale_gap", "label_cross_attention"], default="gap")
    parser.add_argument("--fusion", choices=["logit_mean", "probability_mean", "probability_max",
        "feature_mean", "feature_max", "attention_pooling"], default="logit_mean")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXrayClassifier(num_labels=10, pretrained=False, head=args.head,
                                study_fusion=args.fusion).to(device).train()
    images = torch.randn(3, 3, args.size, args.size, device=device)
    study_index = torch.tensor([0, 0, 1], device=device)
    targets = torch.randint(0, 2, (2, 10), device=device).float()
    known = torch.ones_like(targets, dtype=torch.bool)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(images, study_index, num_studies=2)
        loss = MaskedBCEWithLogitsLoss()(logits.float(), targets, known)
    loss.backward()
    if logits.shape != (2, 10) or not torch.isfinite(loss):
        raise RuntimeError(f"Invalid synthetic output: shape={tuple(logits.shape)}, loss={loss.item()}")
    print(f"OK device={device} amp={amp_enabled} dtype={amp_dtype} shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
