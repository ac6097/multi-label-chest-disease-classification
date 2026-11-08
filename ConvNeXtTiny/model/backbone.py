from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTinyBackbone(nn.Module):
    """Torchvision ConvNeXt-Tiny exposing Stage2/3/4 feature maps."""

    def __init__(self, pretrained=False):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        features = model.features
        self.patchify_stem = features[0]
        self.stage1 = features[1]
        self.downsample2, self.stage2 = features[2], features[3]
        self.downsample3, self.stage3 = features[4], features[5]
        self.downsample4, self.stage4 = features[6], features[7]
        self.channels = {
            "stage2": self._infer_stage_channels(self.stage2),
            "stage3": self._infer_stage_channels(self.stage3),
            "stage4": self._infer_stage_channels(self.stage4),
        }

    @staticmethod
    def _infer_stage_channels(stage):
        for module in reversed(list(stage.modules())):
            if (isinstance(module, nn.Conv2d) and module.groups == module.in_channels
                    and module.out_channels == module.in_channels):
                return int(module.out_channels)
        raise RuntimeError("Unable to infer ConvNeXt stage channels from its depthwise blocks")

    def forward(self, images):
        x = self.stage1(self.patchify_stem(images))
        stage2 = self.stage2(self.downsample2(x))
        stage3 = self.stage3(self.downsample3(stage2))
        stage4 = self.stage4(self.downsample4(stage3))
        return {"stage2": stage2, "stage3": stage3, "stage4": stage4}


def build_backbone(pretrained=False):
    backbone = ConvNeXtTinyBackbone(pretrained=pretrained)
    return backbone, dict(backbone.channels)
