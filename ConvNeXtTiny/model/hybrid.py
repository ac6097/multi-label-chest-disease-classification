import torch
from torch import nn
from torch.nn import functional as F

from .study_pooling import pool_by_study


class HybridBackbone(nn.Module):
    """ImageNet-initialized ConvNeXtV2-Tiny and ViT-Small/16 feature encoders."""

    def __init__(self, convnext_model, vit_model, convnext_mean, convnext_std,
                 vit_mean, vit_std, image_size=320):
        super().__init__()
        self.convnext = convnext_model
        self.vit = vit_model
        self.image_size = int(image_size)
        self.register_buffer("convnext_mean", torch.tensor(convnext_mean).view(1, 3, 1, 1))
        self.register_buffer("convnext_std", torch.tensor(convnext_std).view(1, 3, 1, 1))
        self.register_buffer("vit_mean", torch.tensor(vit_mean).view(1, 3, 1, 1))
        self.register_buffer("vit_std", torch.tensor(vit_std).view(1, 3, 1, 1))

    def normalize_convnext(self, raw_images):
        return (raw_images - self.convnext_mean) / self.convnext_std

    def normalize_vit(self, raw_images):
        return (raw_images - self.vit_mean) / self.vit_std

    def convnext_stage3(self, raw_images):
        x = self.convnext.stem(self.normalize_convnext(raw_images))
        for stage_index in range(3):
            x = self.convnext.stages[stage_index](x)
        return x

    def convnext_stage4_from_stage3(self, stage3):
        return self.convnext.stages[3](stage3)

    def convnext_features(self, raw_images):
        stage3 = self.convnext_stage3(raw_images)
        stage4 = self.convnext_stage4_from_stage3(stage3)
        return stage3, stage4

    def vit_input_tokens(self, raw_images):
        """Patchify, add the resized absolute position embedding, and apply input norms."""
        images = self.normalize_vit(raw_images)
        tokens = self.vit.patch_embed(images)
        tokens = self.vit._pos_embed(tokens)
        tokens = self.vit.patch_drop(tokens)
        return self.vit.norm_pre(tokens)

    def vit_features(self, raw_images):
        return self.vit.forward_features(self.normalize_vit(raw_images))

    def set_grad_checkpointing(self, enable=True):
        self.convnext.set_grad_checkpointing(enable)
        self.vit.set_grad_checkpointing(enable)


class HybridClassifier(nn.Module):
    """ConvNeXtV2 Stage 4 GAP + ViT CLS concatenation followed by Linear(1152, 10)."""

    def __init__(self, backbone, num_labels=10):
        super().__init__()
        if num_labels != 10:
            raise ValueError("The official chest X-ray route requires exactly 10 labels")
        self.backbone = backbone
        self.classifier = nn.Linear(768 + 384, num_labels)

    def image_logits(self, raw_images):
        _, stage4 = self.backbone.convnext_features(raw_images)
        convnext_vector = F.adaptive_avg_pool2d(stage4, 1).flatten(1)
        vit_tokens = self.backbone.vit_features(raw_images)
        if vit_tokens.ndim != 3 or vit_tokens.shape[1] != 401:
            raise RuntimeError(f"Expected ViT tokens [B, 401, 384], got {tuple(vit_tokens.shape)}")
        vit_cls = vit_tokens[:, 0]
        if convnext_vector.shape[1] != 768 or vit_cls.shape[1] != 384:
            raise RuntimeError("Hybrid branch feature widths do not match 768 + 384")
        return self.classifier(torch.cat((convnext_vector, vit_cls), dim=1))

    def forward(self, raw_images, study_index=None, num_studies=None):
        logits = self.image_logits(raw_images)
        if study_index is None:
            return logits
        if num_studies is None:
            raise ValueError("num_studies is required for Study-level Logit Mean")
        if study_index.device != logits.device:
            study_index = study_index.to(logits.device)
        return pool_by_study(logits, study_index, num_studies, mode="mean")

    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)

