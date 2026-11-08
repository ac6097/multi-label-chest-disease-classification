"""R12: fixed H2 + five frozen R11 branches + three new disease modules."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from genopt_common import EXPERIMENT_ROOT, LABEL_NAMES, load_r5_reference  # noqa: E402
from r11_model import DiseaseSparseBranch, load_h2_base  # noqa: E402
from dino_model import DINOv3StudyClassifier  # noqa: E402

R11_DIR = EXPERIMENT_ROOT / "R11_DiseaseSpecific_SparseFeatureRouting_512"
KEEP = ["Consolidation", "Pneumonia", "Cardiomegaly", "Atelectasis",
        "Lung Opacity"]
KEEP_INDICES = [LABEL_NAMES.index(name) for name in KEEP]
TARGETS = {"A": "Enlarged Cardiomediastinum", "B": "Pneumothorax",
           "C": "No Finding"}
TARGET_INDICES = {key: LABEL_NAMES.index(name) for key, name in TARGETS.items()}
VIEW_CATEGORIES = {"": 3, "AP": 0, "PA": 1, "LATERAL": 2, "LL": 2,
                   "PA LLD": 2, "UNKNOWN": 3}
VIEW_NAMES = ["AP", "PA", "Lateral", "Unknown"]


class GeometryRegionModule(nn.Module):
    """R12-A: region/geometry/view-conditioned residual for EC."""

    def __init__(self, channels: int = 384, stage4_dim: int = 768,
                 region_dim: int = 32):
        super().__init__()
        self.region_dim = region_dim
        self.project3 = nn.Conv2d(channels, region_dim, 1)
        self.stage4_projection = nn.Linear(stage4_dim, 64)
        self.view_embedding = nn.Embedding(4, 8)
        # Fixed normalized coordinates for global / 2x2 / center / left-right /
        # top-bottom region groups (2 coords per region group, 9 groups).
        self.register_buffer("region_coordinates", torch.tensor([
            [0.5, 0.5],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
            [0.5, 0.5],
            [0.25, 0.5], [0.75, 0.5],
            [0.5, 0.25], [0.5, 0.75],
        ], dtype=torch.float32))
        input_dim = region_dim * (1 + 4 + 1 + 2 + 2) + 20 + 8 + 64
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, stage3: torch.Tensor, stage4: torch.Tensor,
                view_ids: torch.Tensor, use_view: bool = True) -> torch.Tensor:
        projected = self.project3(stage3)
        regions = [projected.mean(dim=(-2, -1))]
        regions.extend(F.adaptive_avg_pool2d(projected, (2, 2)).flatten(2).unbind(-1))
        h, w = projected.shape[-2:]
        regions.append(projected[:, :, h // 4:h * 3 // 4, w // 4:w * 3 // 4]
                       .mean(dim=(-2, -1)))
        regions.append(projected[:, :, :, :w // 2].mean(dim=(-2, -1)))
        regions.append(projected[:, :, :, w // 2:].mean(dim=(-2, -1)))
        regions.append(projected[:, :, :h // 2, :].mean(dim=(-2, -1)))
        regions.append(projected[:, :, h // 2:, :].mean(dim=(-2, -1)))
        region_features = torch.cat(regions, dim=-1)
        stage4_pooled = self.stage4_projection(stage4.mean(dim=(-2, -1)))
        if use_view:
            view_features = self.view_embedding(view_ids)
        else:
            view_features = torch.zeros(stage3.shape[0], 8, device=stage3.device)
        view_features = view_features.to(stage3.dtype)
        geometry = self.region_coordinates.to(stage3.dtype).unsqueeze(0).expand(
            stage3.shape[0], -1, -1).flatten(1)
        features = torch.cat([region_features, geometry, view_features,
                              stage4_pooled], dim=-1)
        return self.head(features).squeeze(-1)


class LocalEdgeModule(nn.Module):
    """R12-B: high-resolution Stage2 local difference residual for PTX."""

    def __init__(self, stage2_channels: int = 192, stage4_dim: int = 768):
        super().__init__()
        self.average = nn.AvgPool2d(3, stride=1, padding=1)
        self.dw3 = nn.Conv2d(stage2_channels, stage2_channels, 3, padding=1,
                             groups=stage2_channels)
        self.dw5 = nn.Conv2d(stage2_channels, stage2_channels, 5, padding=2,
                             groups=stage2_channels)
        self.stage4_projection = nn.Linear(stage4_dim, 64)
        self.head = nn.Sequential(
            nn.Linear(stage2_channels * 3 + 64, 64), nn.GELU(),
            nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, stage2: torch.Tensor, stage4: torch.Tensor,
                use_local: bool = True, use_difference: bool = True) -> torch.Tensor:
        if use_local:
            difference = stage2 - self.average(stage2) if use_difference else stage2
            edges = 0.5 * (self.dw3(difference) + self.dw5(difference))
            average = edges.mean(dim=(-2, -1))
            maximum = edges.amax(dim=(-2, -1))
            flat = edges.flatten(2)
            topk = flat.topk(min(64, flat.shape[-1]), dim=-1).values.mean(-1)
            local_features = torch.cat([average, maximum, topk], dim=-1)
        else:
            local_features = torch.zeros(stage2.shape[0], stage2.shape[1] * 3,
                                         device=stage2.device,
                                         dtype=stage2.dtype)
        stage4_pooled = self.stage4_projection(stage4.mean(dim=(-2, -1)))
        features = torch.cat([local_features.to(stage4_pooled.dtype),
                              stage4_pooled], dim=-1)
        return self.head(features).squeeze(-1)


class LabelRelationModule(nn.Module):
    """R12-C: residual from the ten frozen H2 base logits for No Finding."""

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        return self.network(base_logits.float()).squeeze(-1)


def load_r11_keep_branches() -> nn.ModuleList:
    payload = torch.load(R11_DIR / "best_checkpoint.pth", map_location="cpu",
                         weights_only=False)
    branch_state = payload["branch_state"]
    branches = nn.ModuleList()
    for index in KEEP_INDICES:
        branch = DiseaseSparseBranch()
        local = {key.split(".", 1)[1]: value for key, value in branch_state.items()
                 if key.startswith(f"{index}.")}
        branch.load_state_dict(local, strict=True)
        for parameter in branch.parameters():
            parameter.requires_grad_(False)
        branches.append(branch)
    return branches


class R12Model(nn.Module):
    """Frozen H2 + frozen five R11 branches + optional A/B/C modules."""

    def __init__(self, enable_a: bool = False, enable_b: bool = False,
                 enable_c: bool = False):
        super().__init__()
        base, _ = load_h2_base()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.keep_branches = load_r11_keep_branches().eval()
        for parameter in self.keep_branches.parameters():
            parameter.requires_grad_(False)
        self.module_a = GeometryRegionModule() if enable_a else None
        self.module_b = LocalEdgeModule() if enable_b else None
        self.module_c = LabelRelationModule() if enable_c else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.keep_branches.eval()
        return self

    def backbone_features(self, images: torch.Tensor):
        captured = {}
        handle = self.base.backbone.model.stages[1][-1].register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("stage2", output))
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda",
                                                 dtype=torch.bfloat16):
                pooled, stage3, stage4 = (
                    self.base.backbone.forward_features_with_stage3(images))
                base_logits = self.base.classifier(pooled)
        finally:
            handle.remove()
        return (base_logits.float(), captured["stage2"].detach().float(),
                stage3.detach().float(), stage4.detach().float())

    def forward(self, images: torch.Tensor, view_ids: torch.Tensor | None = None,
                gates: dict | None = None,
                module_off: dict | None = None,
                use_view: bool = True, use_local: bool = True,
                use_difference: bool = True):
        gates = gates or {"A": True, "B": True, "C": True}
        module_off = module_off or {}
        base_logits, stage2, stage3, stage4 = self.backbone_features(images)
        logits = base_logits.clone()
        keep_residuals = torch.stack(
            [branch(stage3) for branch in self.keep_branches], dim=-1)
        logits[:, KEEP_INDICES] = logits[:, KEEP_INDICES] + keep_residuals
        residual_parts = {}
        if self.module_a is not None and gates.get("A", True) and not module_off.get("A", False):
            if view_ids is None:
                view_ids = torch.full((images.shape[0],), 3, dtype=torch.long,
                                      device=images.device)
            residual_a = self.module_a(stage3, stage4, view_ids,
                                       use_view=use_view)
            logits[:, TARGET_INDICES["A"]] += residual_a
            residual_parts["A"] = residual_a
        if self.module_b is not None and gates.get("B", True) and not module_off.get("B", False):
            residual_b = self.module_b(stage2, stage4, use_local=use_local,
                                       use_difference=use_difference)
            logits[:, TARGET_INDICES["B"]] += residual_b
            residual_parts["B"] = residual_b
        if self.module_c is not None and gates.get("C", True) and not module_off.get("C", False):
            residual_c = self.module_c(base_logits)
            logits[:, TARGET_INDICES["C"]] += residual_c
            residual_parts["C"] = residual_c
        return logits, base_logits, residual_parts


def fixed_logits_from_r11() -> dict:
    """Return reference paths/hashes for the frozen R11 keep branches."""
    payload = torch.load(R11_DIR / "best_checkpoint.pth", map_location="cpu",
                         weights_only=False)
    return {
        "r11_checkpoint": str(R11_DIR / "best_checkpoint.pth"),
        "r11_best_epoch": int(payload.get("best_epoch", 0)),
        "keep_diseases": KEEP,
        "keep_indices": KEEP_INDICES,
        "base_state_key": "base_state",
        "branch_state_key": "branch_state",
    }


def count_module_parameters(model: R12Model) -> dict:
    counts = {}
    for name in ("module_a", "module_b", "module_c"):
        module = getattr(model, name)
        counts[name] = int(sum(p.numel() for p in module.parameters())) if module else 0
    counts["keep_r11_branches"] = int(sum(p.numel() for p in model.keep_branches.parameters()))
    counts["base"] = int(sum(p.numel() for p in model.base.parameters()))
    return counts


if __name__ == "__main__":
    model = R12Model(enable_a=True, enable_b=True, enable_c=True).to(torch.device("cuda"))
    images = torch.randn(2, 3, 512, 512, device="cuda")
    view = torch.tensor([0, 1], device="cuda")
    logits, base, parts = model(images, view)
    print(logits.shape, base.shape, {key: value.shape for key, value in parts.items()})
    print(count_module_parameters(model))
