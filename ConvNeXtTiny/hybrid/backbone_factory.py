from pathlib import Path

import timm
import torch

from hybrid.common import PROJECT_DIR, sha256_file
from model.hybrid import HybridBackbone


def build_hybrid_backbone(initial_checkpoint=None, map_location="cpu",
                          initial_checkpoint_payload=None, initial_checkpoint_sha256=None):
    initial_checkpoint = Path(initial_checkpoint or
                              PROJECT_DIR / "outputs/hybrid/shared/imagenet_initial.pth")
    if not initial_checkpoint.is_absolute():
        initial_checkpoint = PROJECT_DIR / initial_checkpoint
    payload = initial_checkpoint_payload
    if payload is None:
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(f"Shared ImageNet initialization is missing: {initial_checkpoint}")
        payload = torch.load(initial_checkpoint, map_location=map_location, weights_only=False)
    if payload.get("convnext_model_name") != "convnextv2_tiny.fcmae_ft_in1k":
        raise ValueError("Unexpected ConvNeXtV2 initialization identifier")
    if payload.get("vit_model_name") != "vit_small_patch16_224.augreg_in1k":
        raise ValueError("Unexpected ViT initialization identifier")
    if payload.get("image_size") != 320:
        raise ValueError("Shared initialization has the wrong input resolution")

    convnext_model = timm.create_model(payload["convnext_model_name"], pretrained=False)
    convnext_model.reset_classifier(0)
    convnext_model.head = torch.nn.Identity()
    vit_model = timm.create_model(payload["vit_model_name"], pretrained=False)
    vit_model.set_input_size(img_size=320)
    vit_model.reset_classifier(0)
    vit_model.head = torch.nn.Identity()
    convnext_model.load_state_dict(payload["convnext_state_dict"], strict=True)
    vit_model.load_state_dict(payload["vit_state_dict"], strict=True)

    metadata = payload["metadata"]
    cnn_meta, vit_meta = metadata["convnext"], metadata["vit"]
    backbone = HybridBackbone(
        convnext_model, vit_model,
        cnn_meta["input_mean"], cnn_meta["input_std"],
        vit_meta["input_mean"], vit_meta["input_std"],
        image_size=payload["image_size"],
    )
    backbone.initial_checkpoint = str(initial_checkpoint)
    backbone.initial_checkpoint_sha256 = (initial_checkpoint_sha256 or
                                          payload.get("initial_checkpoint_sha256") or
                                          (sha256_file(initial_checkpoint)
                                           if initial_checkpoint.is_file() else None))
    backbone.initialization_metadata = metadata
    return backbone
