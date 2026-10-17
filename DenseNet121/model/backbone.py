from torch import nn
from torch.nn import functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Backbone(nn.Module):
    """Torchvision DenseNet-121 exposing the route's Block2/3/4 feature maps."""
    channels = {"stage2": 512, "stage3": 1024, "stage4": 1024}

    def __init__(self, pretrained=False):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        features = model.features
        self.conv0, self.norm0 = features.conv0, features.norm0
        self.relu0, self.pool0 = features.relu0, features.pool0
        self.denseblock1, self.transition1 = features.denseblock1, features.transition1
        self.denseblock2, self.transition2 = features.denseblock2, features.transition2
        self.denseblock3, self.transition3 = features.denseblock3, features.transition3
        self.denseblock4, self.norm5 = features.denseblock4, features.norm5

    def forward(self, images):
        x = self.pool0(self.relu0(self.norm0(self.conv0(images))))
        x = self.transition1(self.denseblock1(x))
        stage2 = self.denseblock2(x)
        x = self.transition2(stage2)
        stage3 = self.denseblock3(x)
        x = self.transition3(stage3)
        stage4 = F.relu(self.norm5(self.denseblock4(x)), inplace=False)
        return {"stage2": stage2, "stage3": stage3, "stage4": stage4}


def build_backbone(pretrained=False):
    return DenseNet121Backbone(pretrained=pretrained), DenseNet121Backbone.channels
