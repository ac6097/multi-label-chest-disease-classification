from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTinyBackbone(nn.Module):
    """Torchvision ConvNeXt-Tiny exposing Stage2/3/4 feature maps."""
    channels = {"stage2": 192, "stage3": 384, "stage4": 768}

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

    def forward(self, images):
        x = self.stage1(self.patchify_stem(images))
        stage2 = self.stage2(self.downsample2(x))
        stage3 = self.stage3(self.downsample3(stage2))
        stage4 = self.stage4(self.downsample4(stage3))
        return {"stage2": stage2, "stage3": stage3, "stage4": stage4}


def build_backbone(pretrained=False):
    return ConvNeXtTinyBackbone(pretrained=pretrained), ConvNeXtTinyBackbone.channels
