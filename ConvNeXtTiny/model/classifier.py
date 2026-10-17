import torch
from torch import nn


class CBAM(nn.Module):
    """Convolutional Block Attention Module for an independent ablation."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        channel_gate = self.sigmoid(
            self.channel_mlp(torch.nn.functional.adaptive_avg_pool2d(features, 1)) +
            self.channel_mlp(torch.nn.functional.adaptive_max_pool2d(features, 1)))
        features = features * channel_gate
        spatial = torch.cat((features.mean(dim=1, keepdim=True),
                             features.amax(dim=1, keepdim=True)), dim=1)
        return features * self.sigmoid(self.spatial(spatial))


class MultiLabelHead(nn.Module):
    def __init__(self, in_features, num_labels=10, normalize=False):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(in_features) if normalize else nn.Identity()
        self.fc = nn.Linear(in_features, num_labels)

    def pool_map(self, features):
        return self.pool(features).flatten(1)

    def forward_vector(self, vector):
        return self.fc(self.norm(vector))

    def forward(self, features):
        return self.forward_vector(self.pool_map(features))


class LabelCrossAttentionHead(nn.Module):
    """Ten learned label queries attend to spatial backbone tokens."""
    def __init__(self, in_features, num_labels=10, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.token_projection = nn.Linear(in_features, d_model)
        self.queries = nn.Parameter(torch.empty(1, num_labels, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                     batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, 1)

    def forward(self, feature_map):
        batch, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens)
        queries = self.queries.expand(batch, -1, -1)
        attended, _ = self.cross_attention(queries, tokens, tokens, need_weights=False)
        label_features = self.norm(queries + self.dropout(attended))
        return self.output(label_features).squeeze(-1)


class StudyAttentionPool(nn.Module):
    """Learned attention over image vectors belonging to each Study."""
    def __init__(self, in_features):
        super().__init__()
        hidden = max(in_features // 4, 64)
        self.score = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, features, study_index, num_studies):
        scores = self.score(features).squeeze(-1)
        pooled = []
        for study in range(num_studies):
            selected = study_index == study
            if not torch.any(selected):
                raise ValueError(f"Study {study} has no images in the batch")
            weights = torch.softmax(scores[selected], dim=0)
            pooled.append(torch.sum(features[selected] * weights[:, None], dim=0))
        return torch.stack(pooled, dim=0)
