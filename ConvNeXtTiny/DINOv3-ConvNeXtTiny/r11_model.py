"""R11: disease-specific sparse feature routing on top of a frozen H2 model.

Only the ten disease branches are trainable.  The frozen H2 backbone is run
exactly once per batch; Stage3 features are detached and routed through ten
independent Top-K (channel + spatial) selectors with a straight-through
estimator, then projected/pooled into one scalar residual per disease.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from genopt_common import (  # noqa: E402
    EXPERIMENT_ROOT,
    LABEL_NAMES,
    load_h_checkpoint_model,
    load_r5_reference,
)


class DiseaseSparseBranch(nn.Module):
    """One disease's channel selector + spatial selector + residual head."""

    def __init__(self, channels: int = 384, spatial: int = 32,
                 hidden: int = 64, projection: int = 64,
                 head_hidden: int = 32, temperature: float = 1.0):
        super().__init__()
        self.channels = int(channels)
        self.spatial = int(spatial)
        self.channel_k = int(math.ceil(0.5 * channels))
        self.spatial_k = int(math.ceil(0.5 * spatial * spatial))
        self.temperature = float(temperature)
        self.channel_selector = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )
        self.spatial_selector = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.projection = nn.Conv2d(self.channels, projection, kernel_size=1)
        self.norm = nn.LayerNorm(projection)
        self.head = nn.Sequential(
            nn.Linear(projection, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        # Exact zero residual at initialization; the last layer still gets a
        # gradient on the first update because dL/dr is non-zero.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ------------------------------------------------------------------
    def scores(self, stage3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gap = stage3.mean(dim=(-2, -1))
        channel_scores = self.channel_selector(gap)
        spatial_scores = self.spatial_selector(stage3).squeeze(1)
        return channel_scores, spatial_scores

    def hard_masks(self, channel_scores: torch.Tensor,
                   spatial_scores: torch.Tensor):
        batch = channel_scores.shape[0]
        channel_top = torch.topk(channel_scores, self.channel_k, dim=-1).indices
        channel_mask = torch.zeros_like(channel_scores, dtype=torch.bool)
        channel_mask.scatter_(1, channel_top, True)
        flat = spatial_scores.flatten(1)
        spatial_top = torch.topk(flat, self.spatial_k, dim=-1).indices
        spatial_mask = torch.zeros_like(flat, dtype=torch.bool)
        spatial_mask.scatter_(1, spatial_top, True)
        return (channel_mask, spatial_mask.view(batch, self.spatial, self.spatial))

    def ste_masks(self, channel_scores: torch.Tensor,
                  spatial_scores: torch.Tensor,
                  channel_mask: torch.Tensor | None = None,
                  spatial_mask: torch.Tensor | None = None):
        if channel_mask is None or spatial_mask is None:
            channel_mask, spatial_mask = self.hard_masks(
                channel_scores, spatial_scores)
        # Independent sigmoid surrogates; the channel surrogate is scaled by C
        # so that its per-element gradient is not diluted by the 384-way sum
        # (the forward value is the hard mask either way).
        channel_soft = torch.sigmoid(
            channel_scores / self.temperature) * float(self.channels)
        spatial_soft = torch.sigmoid(spatial_scores / self.temperature)
        channel_ste = channel_mask.to(channel_soft.dtype) - channel_soft.detach() + channel_soft
        spatial_ste = spatial_mask.to(spatial_soft.dtype) - spatial_soft.detach() + spatial_soft
        return channel_ste, spatial_ste, channel_mask, spatial_mask

    # ------------------------------------------------------------------
    def forward(self, stage3: torch.Tensor,
                channel_mask: torch.Tensor | None = None,
                spatial_mask: torch.Tensor | None = None,
                return_diagnostics: bool = False):
        if stage3.ndim != 4:
            raise ValueError("Stage3 must be [B,C,H,W]")
        if stage3.shape[1] != self.channels or stage3.shape[-1] != self.spatial:
            raise RuntimeError(
                f"Expected Stage3 [B,{self.channels},{self.spatial},{self.spatial}], "
                f"got {tuple(stage3.shape)}")
        channel_scores, spatial_scores = self.scores(stage3)
        channel_ste, spatial_ste, channel_hard, spatial_hard = self.ste_masks(
            channel_scores, spatial_scores, channel_mask, spatial_mask)
        routed = stage3 * channel_ste.unsqueeze(-1).unsqueeze(-1)
        routed = routed * spatial_ste.unsqueeze(1)
        # Masked mean over selected channel/spatial elements (denominator fixed
        # by the Top-K budget; zero-filled entries cannot inflate the scale).
        pooled_map = self.projection(routed)
        pooled = (pooled_map * spatial_hard.unsqueeze(1)).sum(dim=(2, 3))
        pooled = pooled / float(self.spatial_k)
        pooled = self.norm(pooled)
        residual = self.head(pooled).squeeze(-1)
        if return_diagnostics:
            return residual, {
                "channel_mask": channel_hard,
                "spatial_mask": spatial_hard,
                "channel_scores": channel_scores,
                "spatial_scores": spatial_scores,
                "residual": residual,
            }
        return residual


class R11Router(nn.Module):
    """Frozen H2 base + ten independent disease residual branches."""

    def __init__(self, base_model: nn.Module, channels: int = 384,
                 spatial: int = 32, branch_hidden: int = 64,
                 projection: int = 64, head_hidden: int = 32):
        super().__init__()
        self.base = base_model
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.channels = int(channels)
        self.spatial = int(spatial)
        self.num_diseases = len(LABEL_NAMES)
        self.branches = nn.ModuleList([
            DiseaseSparseBranch(channels=channels, spatial=spatial,
                                hidden=branch_hidden,
                                projection=projection,
                                head_hidden=head_hidden)
            for _ in range(self.num_diseases)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        # The H2 base must stay in eval mode in every phase.
        self.base.eval()
        return self

    def base_logits_and_stage3(self, images: torch.Tensor):
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             dtype=torch.bfloat16):
            pooled, stage3, _ = self.base.backbone.forward_features_with_stage3(images)
            base_logits = self.base.classifier(pooled)
        return base_logits.float(), stage3.detach().float()

    def forward(self, images: torch.Tensor, gates: torch.Tensor | None = None,
                mask_override: dict | None = None,
                return_diagnostics: bool = False):
        base_logits, stage3 = self.base_logits_and_stage3(images)
        residuals = []
        diagnostics = []
        for index, branch in enumerate(self.branches):
            override = None if mask_override is None else mask_override.get(index)
            if override is None:
                if return_diagnostics:
                    output, info = branch(stage3, return_diagnostics=True)
                    diagnostics.append(info)
                else:
                    output = branch(stage3)
            else:
                channel_mask, spatial_mask = override
                if return_diagnostics:
                    output, info = branch(stage3, channel_mask, spatial_mask,
                                          return_diagnostics=True)
                    diagnostics.append(info)
                else:
                    output = branch(stage3, channel_mask, spatial_mask)
            residuals.append(output)
        residual = torch.stack(residuals, dim=-1)
        if gates is None:
            gates = torch.ones(self.num_diseases, device=residual.device,
                               dtype=residual.dtype)
        logits = base_logits + gates.unsqueeze(0) * residual
        if return_diagnostics:
            return logits, base_logits, residual, diagnostics
        return logits, base_logits, residual

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters()
                if parameter.requires_grad]


def load_h2_base() -> tuple[nn.Module, dict]:
    runtime = load_r5_reference()["runtime"]
    model, payload = load_h_checkpoint_model(
        EXPERIMENT_ROOT / "H2_Single_Study_Loss_Weight", "best", runtime)
    return model.eval(), payload


def build_r11_router(device: torch.device | None = None) -> R11Router:
    base, _ = load_h2_base()
    router = R11Router(base)
    if device is not None:
        router = router.to(device)
    router.eval()
    return router


def count_branch_parameters(router: R11Router) -> dict:
    per_branch = []
    for branch in router.branches:
        per_branch.append(int(sum(p.numel() for p in branch.parameters())))
    total = int(sum(per_branch))
    return {
        "per_branch": per_branch,
        "total_branch_parameters": total,
        "base_parameters": int(sum(p.numel() for p in router.base.parameters())),
        "total_with_base": total + int(
            sum(p.numel() for p in router.base.parameters())),
    }


if __name__ == "__main__":
    router = build_r11_router(torch.device("cuda"))
    images = torch.randn(2, 3, 512, 512, device="cuda")
    logits, base_logits, residual = router(images)
    print("logits", tuple(logits.shape), "base", tuple(base_logits.shape),
          "residual", tuple(residual.shape))
    print(count_branch_parameters(router))
