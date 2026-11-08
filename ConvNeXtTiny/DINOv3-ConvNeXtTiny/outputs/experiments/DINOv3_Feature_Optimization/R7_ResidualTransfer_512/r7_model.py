from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier, NUM_LABELS


class Stage3ToStage4Residual(nn.Module):
    """AvgPool2d(2) + 1×1 projection + one learnable global residual scale."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(384, 768, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, stage3: torch.Tensor, gamma_override: float | None = None):
        if stage3.ndim != 4 or tuple(stage3.shape[1:]) != (384, 32, 32):
            raise ValueError(f"Expected Stage 3 [B,384,32,32], got {tuple(stage3.shape)}")
        pooled = F.avg_pool2d(stage3, kernel_size=2, stride=2)
        projected = self.projection(pooled)
        if tuple(projected.shape[1:]) != (768, 16, 16):
            raise RuntimeError(f"R7 projection shape is {tuple(projected.shape)}")
        gamma = (self.gamma.to(device=projected.device, dtype=projected.dtype)
                 if gamma_override is None else
                 torch.as_tensor(gamma_override, device=projected.device,
                                 dtype=projected.dtype))
        contribution = gamma * projected
        return projected, contribution, gamma


class R7Stage3Stage4ResidualClassifier(DINOv3StudyClassifier):
    """R5 Global-only classifier plus a zero-initialized Stage3→Stage4 residual."""

    def __init__(self, backbone: DINOv3ConvNeXtBackbone | None = None,
                 num_labels: int = NUM_LABELS):
        # Keep the original `backbone.*` and `classifier.*` state-dict names.
        super().__init__(backbone=backbone, num_labels=num_labels)
        self.transfer = Stage3ToStage4Residual()

    def forward_components(self, images: torch.Tensor, gamma_override: float | None = None,
                           include_gamma_zero_logits: bool = False):
        stage3, stage4 = self.backbone._forward_feature_maps(
            images, capture_stage3=True)
        projected, residual, gamma = self.transfer(
            stage3, gamma_override=gamma_override)
        fused_stage4 = stage4 + residual
        pooled = self.backbone._normalize_global_features(fused_stage4)
        fused_logits = self.classifier(pooled)
        result = {
            "logits": fused_logits,
            "fused_logits": fused_logits,
            "pooled_features": pooled,
            "stage3": stage3,
            "stage4": stage4,
            "projected_stage3": projected,
            "residual_contribution": residual,
            "fused_stage4": fused_stage4,
            "gamma": gamma,
        }
        if include_gamma_zero_logits:
            pooled_gamma_zero = self.backbone._normalize_global_features(stage4)
            result["gamma_zero_pooled_features"] = pooled_gamma_zero
            result["gamma_zero_logits"] = self.classifier(pooled_gamma_zero)
        return result

    def forward(self, images: torch.Tensor, return_features: bool = False,
                return_components: bool = False,
                gamma_override: float | None = None,
                include_gamma_zero_logits: bool = False):
        result = self.forward_components(
            images, gamma_override=gamma_override,
            include_gamma_zero_logits=include_gamma_zero_logits)
        if return_components:
            return result
        if return_features:
            return result["fused_logits"], result["pooled_features"], result["fused_stage4"]
        return result["fused_logits"]
