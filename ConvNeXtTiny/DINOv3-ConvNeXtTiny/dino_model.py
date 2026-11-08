"""DINOv3 ConvNeXt-Tiny classifier using Meta's reference implementation."""

from __future__ import annotations

import re

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dinov3_meta_convnext import get_convnext_arch


FEATURE_DIM = 768
NUM_LABELS = 10


def map_hf_checkpoint_to_meta(state_dict: dict[str, torch.Tensor]):
    """Translate Hugging Face DINOv3ConvNext tensor names to Meta reference names."""
    mapped = {}
    for name, tensor in state_dict.items():
        if name.startswith("layer_norm."):
            suffix = name.removeprefix("layer_norm.")
            mapped[f"norm.{suffix}"] = tensor
            # The official feature-extractor API registers this same LayerNorm
            # under norms.3 as well; both names share one Parameter object.
            mapped[f"norms.3.{suffix}"] = tensor
            continue
        match = re.fullmatch(r"stages\.(\d+)\.downsample_layers\.(\d+)\.(.+)", name)
        if match:
            stage, layer, suffix = match.groups()
            mapped[f"downsample_layers.{stage}.{layer}.{suffix}"] = tensor
            continue
        match = re.fullmatch(r"stages\.(\d+)\.layers\.(\d+)\.(.+)", name)
        if not match:
            raise RuntimeError(f"Unrecognized DINOv3 checkpoint tensor: {name}")
        stage, block, suffix = match.groups()
        replacements = (
            ("depthwise_conv.", "dwconv."),
            ("layer_norm.", "norm."),
            ("pointwise_conv1.", "pwconv1."),
            ("pointwise_conv2.", "pwconv2."),
        )
        for source, destination in replacements:
            if suffix.startswith(source):
                suffix = destination + suffix[len(source):]
                break
        mapped[f"stages.{stage}.{block}.{suffix}"] = tensor
    if len(mapped) != len(state_dict) + 2:
        # The two extra names are aliases for the final norm in the official API.
        raise RuntimeError(f"Unexpected checkpoint mapping cardinality: {len(state_dict)} -> {len(mapped)}")
    return mapped


class DINOv3ConvNeXtBackbone(nn.Module):
    """Meta official DINOv3 ConvNeXt trunk returning pooled normalized features."""

    def __init__(self, layer_norm_eps: float = 1e-6,
                 layer_scale_init_value: float = 1e-6,
                 drop_path_rate: float = 0.0,
                 architecture: str = "convnext_tiny"):
        super().__init__()
        self.architecture = str(architecture)
        self.model = get_convnext_arch(self.architecture)(
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.model.norm.eps = layer_norm_eps
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def load_hf_state_dict(self, state_dict: dict[str, torch.Tensor]):
        mapped = map_hf_checkpoint_to_meta(state_dict)
        result = self.model.load_state_dict(mapped, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict official DINOv3 load was incomplete: {result}")
        return result

    def _forward_feature_maps(self, images: torch.Tensor,
                              capture_stage3: bool = False):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")
        input_height, input_width = map(int, images.shape[-2:])
        if input_height < 32 or input_width < 32 or input_height % 32 or input_width % 32:
            raise ValueError(
                f"Expected H and W to be positive multiples of 32, got {input_height}x{input_width}")
        features = images
        stage3 = None
        for stage_index in range(4):
            features = self.model.downsample_layers[stage_index](features)
            for block in self.model.stages[stage_index]:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    features = checkpoint(block, features, use_reentrant=False)
                else:
                    features = block(features)
            if stage_index == 2 and capture_stage3:
                stage3 = features
        expected_channels = int(self.model.norm.normalized_shape[-1])
        expected_spatial = (input_height // 32, input_width // 32)
        if tuple(features.shape[1:]) != (expected_channels, *expected_spatial):
            raise RuntimeError(
                f"Expected Stage4 [B,{expected_channels},{expected_spatial[0]},{expected_spatial[1]}], "
                f"got {tuple(features.shape)}")
        if capture_stage3 and stage3 is None:
            raise RuntimeError("Stage 3 features were not captured")
        return stage3, features

    def _normalize_global_features(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=(-2, -1))
        patch_tokens = torch.flatten(features, 2).transpose(1, 2)
        normalized_tokens = self.model.norm(
            torch.cat([pooled.unsqueeze(1), patch_tokens], dim=1))
        return normalized_tokens[:, 0]

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, stage4 = self._forward_feature_maps(images, capture_stage3=False)
        return self._normalize_global_features(stage4), stage4

    def forward_features_with_stage3(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage3, stage4 = self._forward_feature_maps(images, capture_stage3=True)
        return self._normalize_global_features(stage4), stage3, stage4


class DINOv3StudyClassifier(nn.Module):
    """Image classifier; caller applies the official Study Logit Mean."""

    def __init__(self, backbone: DINOv3ConvNeXtBackbone | None = None,
                 num_labels: int = NUM_LABELS, dropout: float = 0.0):
        super().__init__()
        if int(num_labels) != NUM_LABELS:
            raise ValueError("The official task has exactly ten labels")
        self.backbone = backbone or DINOv3ConvNeXtBackbone()
        # H4-only optional regularization: dropout on the final pooled feature,
        # immediately before the linear classifier. No parameters are added, so
        # R5/H2 checkpoints remain strictly loadable.
        self.head_dropout = float(dropout)
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.classifier = nn.Linear(FEATURE_DIM, NUM_LABELS)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        pooled, stage4 = self.backbone.forward_features(images)
        logits = self.classifier(self.dropout(pooled))
        if return_features:
            return logits, pooled, stage4
        return logits


class StudyProjectionHead(nn.Module):
    """D0-C1 training-only projection head; excluded from inference export."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(features), p=2, dim=-1)
