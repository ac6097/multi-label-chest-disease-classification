import torch
from torch import nn

from .backbone import build_backbone
from .classifier import CBAM, LabelCrossAttentionHead, MultiLabelHead, StudyAttentionPool
from .study_pooling import FEATURE_POOLINGS, PREDICTION_POOLINGS, pool_by_study


class ChestXrayClassifier(nn.Module):
    """DenseNet-121 route with Study-level multi-image prediction and ablation heads."""
    def __init__(self, num_labels=10, pretrained=False, head="gap", cbam=False,
                 study_fusion="logit_mean", label_d_model=256, label_heads=4,
                 label_dropout=0.1):
        super().__init__()
        if num_labels != 10:
            raise ValueError("This competition route is defined for exactly 10 labels")
        if head not in {"gap", "multiscale_gap", "label_cross_attention"}:
            raise ValueError("head must be gap, multiscale_gap, or label_cross_attention")
        supported_fusions = set(FEATURE_POOLINGS) | set(PREDICTION_POOLINGS) | {"attention_pooling"}
        if study_fusion not in supported_fusions:
            raise ValueError(f"study_fusion must be one of {sorted(supported_fusions)}")
        if cbam and head != "gap":
            raise ValueError("CBAM is kept as an independent experiment with the standard GAP head")
        if head == "label_cross_attention" and study_fusion in set(FEATURE_POOLINGS) | {"attention_pooling"}:
            raise ValueError("Label Cross-Attention and feature/attention pooling are separate ablations")

        self.backbone, widths = build_backbone(pretrained)
        self.head_name, self.study_fusion = head, study_fusion
        self.cbam = CBAM(widths["stage4"]) if cbam else nn.Identity()
        if head == "multiscale_gap":
            self.representation_width = widths["stage2"] + widths["stage3"] + widths["stage4"]
        else:
            self.representation_width = widths["stage4"]
        if head == "label_cross_attention":
            self.classifier = LabelCrossAttentionHead(
                widths["stage4"], num_labels, label_d_model, label_heads, label_dropout)
            self.vector_classifier = None
        else:
            self.classifier = None
            self.vector_classifier = MultiLabelHead(self.representation_width, num_labels)
        self.study_attention = (StudyAttentionPool(self.representation_width)
                                if study_fusion == "attention_pooling" else None)

    def _represent(self, maps):
        stage4 = self.cbam(maps["stage4"])
        if self.head_name == "multiscale_gap":
            return torch.cat((
                self.vector_classifier.pool_map(maps["stage2"]),
                self.vector_classifier.pool_map(maps["stage3"]),
                self.vector_classifier.pool_map(stage4),
            ), dim=1), stage4
        return self.vector_classifier.pool_map(stage4) if self.vector_classifier else None, stage4

    def forward(self, images, study_index=None, num_studies=None):
        maps = self.backbone(images)
        representation, stage4 = self._represent(maps)
        if self.head_name == "label_cross_attention":
            image_logits = self.classifier(stage4)
        else:
            image_logits = self.vector_classifier.forward_vector(representation)
        if study_index is None:
            return image_logits
        if num_studies is None:
            raise ValueError("num_studies is required when study_index is provided")
        if study_index.device != images.device:
            study_index = study_index.to(images.device)
        if self.study_fusion in FEATURE_POOLINGS:
            pooled_features = pool_by_study(representation, study_index, num_studies,
                                            FEATURE_POOLINGS[self.study_fusion])
            return self.vector_classifier.forward_vector(pooled_features)
        if self.study_fusion == "attention_pooling":
            pooled_features = self.study_attention(representation, study_index, num_studies)
            return self.vector_classifier.forward_vector(pooled_features)
        if self.study_fusion == "logit_mean":
            return pool_by_study(image_logits, study_index, num_studies, "mean")
        probabilities = torch.sigmoid(image_logits)
        mode = PREDICTION_POOLINGS[self.study_fusion]
        pooled_probabilities = pool_by_study(probabilities, study_index, num_studies, mode)
        return torch.logit(pooled_probabilities.clamp(1e-6, 1.0 - 1e-6))
