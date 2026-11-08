"""R8/R9/R10 disease-specific Stage3/Stage4 fusion models.

All three models keep the unchanged R5 main path:

    Stage3 -> original Downsample -> Stage4
    Stage4 -> official pooled-token normalization -> Linear(768, 10)  -> z4

and add a controlled auxiliary Stage3 path on top:

    R8  : z3,k = Linear10(LayerNorm(GAP(F3)))            (zero-initialized head)
          z,k  = z4,k + alpha_k * z3,k
    R9  : H,k  = CrossAttention(e_k, K(F3), V(F3))       (10 disease queries)
          z,k  = z4,k + alpha_k * w_k^T H,k + b_k        (zero-initialized head)
    R10 : q_k(x) = e_k + W_q LayerNorm(g4(x))            (W_q zero-initialized)
          z,k  = z4,k + alpha_k * w_k^T H_k + b_k

alpha_k = sigmoid(theta_k) with theta initialized at logit(0.1).  The
auxiliary output head starts at exactly zero, so at initialization every model
is bit-identical to the independently seeded R5 Global-only classifier; the
zero-initialized head still receives a non-zero gradient because alpha > 0,
which is what makes the delayed-gradient audit meaningful.

No R7 shared residual transfer, no auxiliary loss, no extra Stage3 module, no
Study-level pooling change.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from dino_model import (
    DINOv3ConvNeXtBackbone,
    DINOv3StudyClassifier,
    FEATURE_DIM,
    NUM_LABELS,
)

STAGE3_CHANNELS = 384
STAGE3_TOKENS = 32 * 32
D_MODEL = 128
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
ALPHA_INIT = 0.1
THETA_INIT = math.log(ALPHA_INIT / (1.0 - ALPHA_INIT))

R8_ARCHITECTURE = "DINOv3ConvNeXtTinyR8DiseaseSpecificStageLogitFusion"
R9_ARCHITECTURE = "DINOv3ConvNeXtTinyR9DiseaseSpecificQueryAttention"
R10_ARCHITECTURE = "DINOv3ConvNeXtTinyR10Stage4GuidedCrossStageAttention"

ARCHITECTURES = {
    "R8": R8_ARCHITECTURE,
    "R9": R9_ARCHITECTURE,
    "R10": R10_ARCHITECTURE,
}


def _zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


class DiseaseAlpha(nn.Module):
    """Ten independent per-disease fusion scalars alpha_k = sigmoid(theta_k)."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(
            torch.full((NUM_LABELS,), float(THETA_INIT), dtype=torch.float32))

    def alpha(self, alpha_override: torch.Tensor | None = None) -> torch.Tensor:
        if alpha_override is None:
            return torch.sigmoid(self.theta)
        override = torch.as_tensor(
            alpha_override, dtype=self.theta.dtype, device=self.theta.device)
        if override.shape != (NUM_LABELS,):
            raise ValueError(
                f"alpha_override must have shape ({NUM_LABELS},), got {tuple(override.shape)}")
        return override


class Stage3GAPBranch(nn.Module):
    """R8 auxiliary Stage3 branch: LayerNorm(384) + Linear(384, 10), zero-init."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(STAGE3_CHANNELS)
        self.head = nn.Linear(STAGE3_CHANNELS, NUM_LABELS)
        _zero_init_linear(self.head)

    def forward(self, stage3: torch.Tensor):
        if stage3.ndim != 4 or int(stage3.shape[1]) != STAGE3_CHANNELS:
            raise ValueError(f"Expected Stage3 [N,{STAGE3_CHANNELS},H,W], got {tuple(stage3.shape)}")
        pooled = F.adaptive_avg_pool2d(stage3, 1).flatten(1)
        logits = self.head(self.norm(pooled))
        return logits, pooled


class DiseaseQueryCrossAttention(nn.Module):
    """Stage3 token cross-attention with ten learnable disease queries."""

    def __init__(self, with_stage4_conditioning: bool):
        super().__init__()
        self.with_stage4_conditioning = bool(with_stage4_conditioning)
        self.token_norm = nn.LayerNorm(STAGE3_CHANNELS)
        self.key_proj = nn.Linear(STAGE3_CHANNELS, D_MODEL)
        self.value_proj = nn.Linear(STAGE3_CHANNELS, D_MODEL)
        self.disease_queries = nn.Parameter(torch.empty(NUM_LABELS, D_MODEL))
        nn.init.normal_(self.disease_queries, std=0.02)
        self.condition_norm = nn.LayerNorm(FEATURE_DIM) if self.with_stage4_conditioning else None
        self.condition_proj = nn.Linear(FEATURE_DIM, D_MODEL) if self.with_stage4_conditioning else None
        if self.condition_proj is not None:
            _zero_init_linear(self.condition_proj)
        # Per-disease auxiliary logit head applied to the attended representation.
        self.aux_head_weight = nn.Parameter(torch.zeros(NUM_LABELS, D_MODEL))
        self.aux_head_bias = nn.Parameter(torch.zeros(NUM_LABELS))

    @property
    def head_dim(self) -> int:
        return HEAD_DIM

    def build_queries(self, batch_size: int,
                      global_stage4: torch.Tensor | None = None,
                      use_conditioning: bool = True):
        """Return queries [B,10,128] and the (possibly zero) conditioning term."""
        queries = self.disease_queries.unsqueeze(0).expand(batch_size, -1, -1)
        conditioning = torch.zeros_like(queries)
        if self.condition_proj is not None and use_conditioning and global_stage4 is not None:
            projected = self.condition_proj(self.condition_norm(global_stage4))
            conditioning = projected.unsqueeze(1).expand(-1, NUM_LABELS, -1)
            queries = queries + conditioning
        return queries, conditioning

    def _tokens(self, stage3: torch.Tensor):
        if stage3.ndim != 4 or int(stage3.shape[1]) != STAGE3_CHANNELS:
            raise ValueError(
                f"Expected Stage3 [N,{STAGE3_CHANNELS},H,W], got {tuple(stage3.shape)}")
        tokens = stage3.flatten(2).transpose(1, 2).contiguous()
        normalized = self.token_norm(tokens)
        return self.key_proj(normalized), self.value_proj(normalized)

    @staticmethod
    def _split_heads(tensor: torch.Tensor) -> torch.Tensor:
        # [N, T, 128] -> [N, heads, T, 32]
        return tensor.reshape(
            tensor.shape[0], tensor.shape[1], NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    def explicit_attention(self, queries: torch.Tensor,
                           keys: torch.Tensor,
                           values: torch.Tensor) -> torch.Tensor:
        """Attention weights [B,heads,10,1024] computed in FP32 for diagnostics only."""
        query_heads = self._split_heads(queries.float())
        key_heads = self._split_heads(keys.float())
        scale = 1.0 / math.sqrt(float(HEAD_DIM))
        scores = torch.matmul(query_heads, key_heads.transpose(-1, -2)) * scale
        return torch.softmax(scores, dim=-1)

    def forward(self, stage3: torch.Tensor, queries: torch.Tensor,
                attention_mode: str = "learned",
                return_attention_weights: bool = False):
        keys, values = self._tokens(stage3)
        batch_size = keys.shape[0]
        if queries.shape[0] != batch_size:
            raise ValueError("Query batch does not match Stage3 token batch")
        if attention_mode not in {"learned", "uniform"}:
            raise ValueError(f"Unknown attention mode: {attention_mode}")

        query_heads = self._split_heads(queries)
        key_heads = self._split_heads(keys)
        value_heads = self._split_heads(values)
        weights = None
        if attention_mode == "uniform":
            attended = value_heads.mean(dim=2, keepdim=True).expand(
                -1, -1, NUM_LABELS, -1)
        else:
            attended = F.scaled_dot_product_attention(
                query_heads, key_heads, value_heads, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch_size, NUM_LABELS, D_MODEL)
        logits = (attended * self.aux_head_weight.unsqueeze(0)).sum(-1) + self.aux_head_bias
        if return_attention_weights:
            if attention_mode == "uniform":
                weights = torch.full(
                    (batch_size, NUM_HEADS, NUM_LABELS, keys.shape[1]),
                    1.0 / float(keys.shape[1]), dtype=queries.dtype, device=queries.device)
            else:
                weights = self.explicit_attention(queries, keys, values)
        return logits, attended, weights


class _StageFusionBase(DINOv3StudyClassifier):
    """Shared plumbing: keep the original DINOv3Classifier parameter names intact."""

    experiment: str = ""

    def _forward_maps(self, images: torch.Tensor, need_stage3: bool):
        stage3, stage4 = self.backbone._forward_feature_maps(
            images, capture_stage3=need_stage3)
        pooled = self.backbone._normalize_global_features(stage4)
        stage4_logits = self.classifier(pooled)
        return stage3, stage4, pooled, stage4_logits


class R8DiseaseSpecificStageLogitFusion(_StageFusionBase):
    """R8: main Stage4 head plus a per-disease weighted Stage3 GAP logit."""

    experiment = "R8"

    def __init__(self, backbone: DINOv3ConvNeXtBackbone | None = None,
                 num_labels: int = NUM_LABELS):
        # The original Global Head must be created first so its seeded
        # initialization matches the R5 Global-only classifier exactly.
        super().__init__(backbone=backbone, num_labels=num_labels)
        self.stage3_branch = Stage3GAPBranch()
        self.fusion_alpha = DiseaseAlpha()

    def forward_components(self, images: torch.Tensor, *, branch_off: bool = False,
                           alpha_override: torch.Tensor | None = None):
        if branch_off:
            pooled, stage4 = self.backbone.forward_features(images)
            stage4_logits = self.classifier(pooled)
            alpha = torch.zeros(NUM_LABELS, dtype=stage4_logits.dtype,
                                device=stage4_logits.device)
            return {
                "logits": stage4_logits,
                "stage4_logits": stage4_logits,
                "stage3_logits": None,
                "stage3_pooled": None,
                "alpha": alpha,
                "stage4_pooled": pooled,
                "stage4": stage4,
            }
        stage3, stage4, pooled, stage4_logits = self._forward_maps(images, True)
        alpha = self.fusion_alpha.alpha(alpha_override)
        stage3_logits, stage3_pooled = self.stage3_branch(stage3)
        logits = stage4_logits + alpha.unsqueeze(0) * stage3_logits
        return {
            "logits": logits,
            "stage4_logits": stage4_logits,
            "stage3_logits": stage3_logits,
            "stage3_pooled": stage3_pooled,
            "alpha": alpha,
            "stage4_pooled": pooled,
            "stage4": stage4,
            "stage3": stage3,
        }

    def forward(self, images: torch.Tensor, return_features: bool = False):
        components = self.forward_components(images)
        if return_features:
            return components["logits"], components["stage4_pooled"], components["stage4"]
        return components["logits"]


class _QueryAttentionBase(_StageFusionBase):
    """Shared R9/R10 attention body; only query construction differs."""

    requires_stage4_conditioning = False

    def __init__(self, backbone: DINOv3ConvNeXtBackbone | None = None,
                 num_labels: int = NUM_LABELS):
        super().__init__(backbone=backbone, num_labels=num_labels)
        self.query_attention = DiseaseQueryCrossAttention(self.requires_stage4_conditioning)
        self.fusion_alpha = DiseaseAlpha()

    def forward_components(self, images: torch.Tensor, *, branch_off: bool = False,
                           alpha_override: torch.Tensor | None = None,
                           attention_mode: str = "learned",
                           use_conditioning: bool = True,
                           return_attention_weights: bool = False):
        if branch_off:
            pooled, stage4 = self.backbone.forward_features(images)
            stage4_logits = self.classifier(pooled)
            alpha = torch.zeros(NUM_LABELS, dtype=stage4_logits.dtype,
                                device=stage4_logits.device)
            return {
                "logits": stage4_logits,
                "stage4_logits": stage4_logits,
                "stage3_logits": None,
                "attended": None,
                "alpha": alpha,
                "queries": None,
                "conditioning": None,
                "attention_weights": None,
                "stage4_pooled": pooled,
                "stage4": stage4,
            }
        stage3, stage4, pooled, stage4_logits = self._forward_maps(images, True)
        alpha = self.fusion_alpha.alpha(alpha_override)
        queries, conditioning = self.query_attention.build_queries(
            int(images.shape[0]), pooled, use_conditioning=use_conditioning)
        stage3_logits, attended, weights = self.query_attention(
            stage3, queries, attention_mode=attention_mode,
            return_attention_weights=return_attention_weights)
        logits = stage4_logits + alpha.unsqueeze(0) * stage3_logits
        return {
            "logits": logits,
            "stage4_logits": stage4_logits,
            "stage3_logits": stage3_logits,
            "attended": attended,
            "alpha": alpha,
            "queries": queries,
            "conditioning": conditioning,
            "attention_weights": weights,
            "stage4_pooled": pooled,
            "stage4": stage4,
            "stage3": stage3,
        }

    def forward(self, images: torch.Tensor, return_features: bool = False):
        components = self.forward_components(images)
        if return_features:
            return components["logits"], components["stage4_pooled"], components["stage4"]
        return components["logits"]


class R9DiseaseSpecificQueryAttention(_QueryAttentionBase):
    """R9: disease-query cross-attention reads Stage3 tokens instead of GAP."""

    experiment = "R9"
    requires_stage4_conditioning = False


class R10Stage4GuidedCrossStageAttention(_QueryAttentionBase):
    """R10: R9 plus a zero-initialized Stage4-global query conditioning term."""

    experiment = "R10"
    requires_stage4_conditioning = True


MODEL_CLASSES = {
    "R8": R8DiseaseSpecificStageLogitFusion,
    "R9": R9DiseaseSpecificQueryAttention,
    "R10": R10Stage4GuidedCrossStageAttention,
}


def build_model(experiment: str, backbone: DINOv3ConvNeXtBackbone | None = None):
    try:
        model_class = MODEL_CLASSES[experiment]
    except KeyError as error:
        raise ValueError(f"Unknown stage-fusion experiment: {experiment}") from error
    return model_class(backbone=backbone)


def new_branch_parameter_names(experiment: str) -> tuple[str, ...]:
    """Top-level state-dict namespaces added on top of the R5 Global-only model."""
    if experiment == "R8":
        return ("stage3_branch", "fusion_alpha")
    if experiment in {"R9", "R10"}:
        return ("query_attention", "fusion_alpha")
    raise ValueError(f"Unknown stage-fusion experiment: {experiment}")
