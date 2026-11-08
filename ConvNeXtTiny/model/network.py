import torch
from torch import nn

from .backbone import build_backbone
from .classifier import CBAM, LabelCrossAttentionHead, MultiLabelHead, StudyAttentionPool
from .dsap_rpb import DiseaseSpecificAttentionPooling
from .spatial_multiscale import SpatialMultiscaleHead
from .study_pooling import FEATURE_POOLINGS, PREDICTION_POOLINGS, pool_by_study


class ChestXrayClassifier(nn.Module):
    """ConvNeXt-Tiny route with Study-level multi-image prediction and ablation heads."""
    def __init__(self, num_labels=10, pretrained=False, head="gap", cbam=False,
                 study_fusion="logit_mean", label_d_model=256, label_heads=4,
                 label_dropout=0.1, dsap_rpb_d_model=256, dsap_rpb_heads=4,
                 dsap_rpb_dropout=0.1, dsap_rpb_max_relative_height=16,
                 dsap_rpb_max_relative_width=16, dsap_rpb_position_scale_init=1.0,
                 dsap_rpb_alpha_init=0.1, feature_mode="stage4", pooling_mode="gap"):
        super().__init__()
        if num_labels != 10:
            raise ValueError("This competition route is defined for exactly 10 labels")
        if head not in {"gap", "multiscale_gap", "label_cross_attention", "dsap_rpb", "spatial"}:
            raise ValueError("head must be gap, multiscale_gap, label_cross_attention, dsap_rpb, or spatial")
        supported_fusions = set(FEATURE_POOLINGS) | set(PREDICTION_POOLINGS) | {"attention_pooling"}
        if study_fusion not in supported_fusions:
            raise ValueError(f"study_fusion must be one of {sorted(supported_fusions)}")
        if cbam and head != "gap":
            raise ValueError("CBAM is kept as an independent experiment with the standard GAP head")
        if head == "label_cross_attention" and study_fusion in set(FEATURE_POOLINGS) | {"attention_pooling"}:
            raise ValueError("Label Cross-Attention and feature/attention pooling are separate ablations")
        if head == "dsap_rpb" and (cbam or study_fusion != "logit_mean"):
            raise ValueError("DSAP_RPB requires CBAM disabled and Study logit_mean fusion")
        if head == "spatial" and (cbam or study_fusion != "logit_mean"):
            raise ValueError("Spatial ablations require CBAM disabled and Study logit_mean fusion")
        if head != "spatial" and (feature_mode != "stage4" or pooling_mode != "gap"):
            raise ValueError("feature_mode and pooling_mode are only configurable for the spatial head")

        self.backbone, widths = build_backbone(pretrained)
        self.head_name, self.study_fusion = head, study_fusion
        self.feature_mode, self.pooling_mode = feature_mode, pooling_mode
        self.cbam = CBAM(widths["stage4"]) if cbam else nn.Identity()
        if head == "spatial":
            self.representation_width = None
        elif head == "multiscale_gap":
            self.representation_width = widths["stage2"] + widths["stage3"] + widths["stage4"]
        else:
            self.representation_width = widths["stage4"]
        if head == "spatial":
            self.classifier = SpatialMultiscaleHead(
                widths["stage3"], widths["stage4"], num_labels,
                feature_mode, pooling_mode,
            )
            self.representation_width = self.classifier.embedding_width
            self.vector_classifier = None
        elif head == "label_cross_attention":
            self.classifier = LabelCrossAttentionHead(
                widths["stage4"], num_labels, label_d_model, label_heads, label_dropout)
            self.vector_classifier = None
        elif head == "dsap_rpb":
            self.classifier = DiseaseSpecificAttentionPooling(
                widths["stage4"], num_labels, dsap_rpb_d_model, dsap_rpb_heads,
                dsap_rpb_dropout, dsap_rpb_max_relative_height,
                dsap_rpb_max_relative_width, dsap_rpb_position_scale_init,
                dsap_rpb_alpha_init,
            )
            self.vector_classifier = None
        else:
            self.classifier = None
            self.vector_classifier = MultiLabelHead(self.representation_width, num_labels, normalize=True)
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

    def forward(self, images, study_index=None, num_studies=None, return_attention=False):
        maps = self.backbone(images)
        diagnostics = None
        if self.head_name == "spatial":
            if return_attention:
                raise ValueError("return_attention is only available for the dsap_rpb head")
            image_logits = self.classifier(maps)
            representation, stage4 = None, maps["stage4"]
        elif self.head_name == "dsap_rpb":
            stage4 = self.cbam(maps["stage4"])
            result = self.classifier(stage4, return_attention=return_attention)
            if return_attention:
                image_logits, diagnostics = result
            else:
                image_logits = result
            representation = None
        else:
            if return_attention:
                raise ValueError("return_attention is only available for the dsap_rpb head")
            representation, stage4 = self._represent(maps)
        if self.head_name == "label_cross_attention":
            image_logits = self.classifier(stage4)
        elif self.head_name not in {"dsap_rpb", "spatial"}:
            image_logits = self.vector_classifier.forward_vector(representation)
        if study_index is None:
            return (image_logits, diagnostics) if return_attention else image_logits
        if num_studies is None:
            raise ValueError("num_studies is required when study_index is provided")
        if study_index.device != images.device:
            study_index = study_index.to(images.device)
        if self.study_fusion in FEATURE_POOLINGS:
            if representation is None:
                raise ValueError("Feature-level Study fusion is not available for the spatial head")
            pooled_features = pool_by_study(representation, study_index, num_studies,
                                            FEATURE_POOLINGS[self.study_fusion])
            output = self.vector_classifier.forward_vector(pooled_features)
            return (output, diagnostics) if return_attention else output
        if self.study_fusion == "attention_pooling":
            pooled_features = self.study_attention(representation, study_index, num_studies)
            output = self.vector_classifier.forward_vector(pooled_features)
            return (output, diagnostics) if return_attention else output
        if self.study_fusion == "logit_mean":
            output = pool_by_study(image_logits, study_index, num_studies, "mean")
            return (output, diagnostics) if return_attention else output
        probabilities = torch.sigmoid(image_logits)
        mode = PREDICTION_POOLINGS[self.study_fusion]
        pooled_probabilities = pool_by_study(probabilities, study_index, num_studies, mode)
        output = torch.logit(pooled_probabilities.clamp(1e-6, 1.0 - 1e-6))
        return (output, diagnostics) if return_attention else output
