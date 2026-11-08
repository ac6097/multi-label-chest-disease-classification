import torch
import torch.nn.functional as F
from torch import nn


class SpatialPyramidPooling(nn.Module):
    """Concatenate fixed 1x1 global and 2x2 regional average pools."""
    def __init__(self, channels):
        super().__init__()
        self.output_width = int(channels) * 5

    def forward(self, feature_map):
        global_features = F.adaptive_avg_pool2d(feature_map, output_size=(1, 1)).flatten(1)
        regional_features = F.adaptive_avg_pool2d(feature_map, output_size=(2, 2)).flatten(1)
        return torch.cat((global_features, regional_features), dim=1)


class Stage3Stage4SpatialFusion(nn.Module):
    """Project Stage 3/4 maps, align Stage 4 spatially, and fuse them."""
    def __init__(self, stage3_channels, stage4_channels, projected_channels=256):
        super().__init__()
        self.projected_channels = int(projected_channels)
        self.stage3_projection = nn.Conv2d(
            stage3_channels, self.projected_channels, kernel_size=1, bias=False)
        self.stage4_projection = nn.Conv2d(
            stage4_channels, self.projected_channels, kernel_size=1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.projected_channels * 2, self.projected_channels,
                      kernel_size=1, bias=False),
            nn.GroupNorm(1, self.projected_channels),
            nn.GELU(),
        )

    def forward(self, stage3, stage4):
        stage3_projected = self.stage3_projection(stage3)
        stage4_projected = self.stage4_projection(stage4)
        stage4_aligned = F.interpolate(
            stage4_projected, size=stage3_projected.shape[-2:],
            mode="bilinear", align_corners=False)
        return self.fusion(torch.cat((stage3_projected, stage4_aligned), dim=1))


class SpatialMultiscaleHead(nn.Module):
    """GAP/SPP classifier over Stage 4 or fused Stage 3+4 spatial maps."""
    def __init__(self, stage3_channels, stage4_channels, num_labels=10,
                 feature_mode="stage4", pooling_mode="gap", projected_channels=256):
        super().__init__()
        if feature_mode not in {"stage4", "stage3_stage4"}:
            raise ValueError("feature_mode must be stage4 or stage3_stage4")
        if pooling_mode not in {"gap", "spp"}:
            raise ValueError("pooling_mode must be gap or spp")

        self.feature_mode = feature_mode
        self.pooling_mode = pooling_mode
        if feature_mode == "stage3_stage4":
            self.fusion = Stage3Stage4SpatialFusion(
                stage3_channels, stage4_channels, projected_channels)
            channels = int(projected_channels)
        else:
            self.fusion = None
            channels = int(stage4_channels)

        if pooling_mode == "spp":
            self.pooling = SpatialPyramidPooling(channels)
            self.embedding_width = self.pooling.output_width
        else:
            self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.embedding_width = channels
        self.normalization = nn.LayerNorm(self.embedding_width)
        self.classifier = nn.Linear(self.embedding_width, num_labels)

    def feature_map(self, maps):
        if self.fusion is None:
            return maps["stage4"]
        return self.fusion(maps["stage3"], maps["stage4"])

    def pooled_features(self, feature_map):
        if self.pooling_mode == "gap":
            return self.pooling(feature_map).flatten(1)
        return self.pooling(feature_map)

    def forward(self, maps):
        features = self.pooled_features(self.feature_map(maps))
        return self.classifier(self.normalization(features))
