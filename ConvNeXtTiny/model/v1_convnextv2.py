import torch
from torch import nn
from torch.nn import functional as F

from model.study_pooling import pool_by_study


class V1ConvNeXtV2Classifier(nn.Module):
    """Single-branch ConvNeXtV2-Tiny, Stage 4 GAP, and a ten-logit linear head."""

    def __init__(self, convnext, input_mean, input_std, num_labels=10):
        super().__init__()
        if num_labels != 10:
            raise ValueError("V1 is fixed to the official ten-label task")
        self.convnext = convnext
        self.classifier = nn.Linear(768, num_labels)
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.as_tensor(input_std, dtype=torch.float32).view(1, 3, 1, 1))

    def image_features(self, raw_images):
        if raw_images.ndim != 4 or tuple(raw_images.shape[1:]) != (3, 320, 320):
            raise ValueError(f"Expected raw images [B,3,320,320], got {tuple(raw_images.shape)}")
        normalized = (raw_images - self.input_mean) / self.input_std
        stage4 = self.convnext.forward_features(normalized)
        if tuple(stage4.shape[1:]) != (768, 10, 10):
            raise RuntimeError(f"Expected ConvNeXtV2 Stage 4 [B,768,10,10], got {tuple(stage4.shape)}")
        return F.adaptive_avg_pool2d(stage4, 1).flatten(1)

    def image_logits(self, raw_images):
        return self.classifier(self.image_features(raw_images))

    def forward(self, raw_images, study_index=None, num_studies=None):
        logits = self.image_logits(raw_images)
        if study_index is None:
            return logits
        if num_studies is None:
            raise ValueError("num_studies is required for Study Logit Mean")
        if study_index.device != logits.device:
            study_index = study_index.to(logits.device)
        return pool_by_study(logits, study_index, num_studies, mode="mean")

    def set_grad_checkpointing(self, enabled):
        if not hasattr(self.convnext, "set_grad_checkpointing"):
            if enabled:
                raise RuntimeError("This timm ConvNeXtV2 model does not support gradient checkpointing")
            return
        self.convnext.set_grad_checkpointing(bool(enabled))
