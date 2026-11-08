import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_position_encoding_2d(height, width, dimension, device):
    """Return a fixed [1, H*W, D] encoding with independent height/width axes."""
    if dimension % 4:
        raise ValueError("2D sinusoidal encoding requires dimension divisible by 4")
    axis_dimension = dimension // 2
    frequencies = torch.exp(
        torch.arange(0, axis_dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / axis_dimension)
    )
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequencies[None, :]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequencies[None, :]
    y_encoding = torch.stack((y.sin(), y.cos()), dim=-1).flatten(1)
    x_encoding = torch.stack((x.sin(), x.cos()), dim=-1).flatten(1)
    grid = torch.cat((
        y_encoding[:, None, :].expand(height, width, axis_dimension),
        x_encoding[None, :, :].expand(height, width, axis_dimension),
    ), dim=-1)
    return grid.reshape(1, height * width, dimension)


class RelativePositionBias2D(nn.Module):
    """Learnable per-head bias indexed by two-dimensional relative coordinates."""

    def __init__(self, num_heads, max_height=16, max_width=16):
        super().__init__()
        if num_heads < 1 or max_height < 1 or max_width < 1:
            raise ValueError("num_heads and relative-position bounds must be positive")
        self.num_heads = int(num_heads)
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        height_offsets = 2 * self.max_height - 1
        width_offsets = 2 * self.max_width - 1
        self.bias_table = nn.Parameter(torch.zeros(height_offsets * width_offsets, self.num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def relative_position_index(self, height, width, device):
        if height < 1 or width < 1:
            raise ValueError("Feature map height and width must be positive")
        bias_height = max(height, self.max_height)
        bias_width = max(width, self.max_width)
        coords_h = torch.arange(height, device=device)
        coords_w = torch.arange(width, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = coords.flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative[0] += bias_height - 1
        relative[1] += bias_width - 1
        return relative[0] * (2 * bias_width - 1) + relative[1]

    def forward(self, height, width, dtype=None):
        index = self.relative_position_index(height, width, self.bias_table.device)
        bias_height = max(height, self.max_height)
        bias_width = max(width, self.max_width)
        table = self.bias_table.view(2 * self.max_height - 1,
                                     2 * self.max_width - 1,
                                     self.num_heads).permute(2, 0, 1).unsqueeze(0)
        target_size = (2 * bias_height - 1, 2 * bias_width - 1)
        if table.shape[-2:] != target_size:
            table = F.interpolate(table, size=target_size, mode="bicubic", align_corners=False)
        table = table.squeeze(0).permute(1, 2, 0).reshape(-1, self.num_heads)
        token_count = height * width
        bias = table[index.reshape(-1)].view(
            token_count, token_count, self.num_heads
        ).permute(2, 0, 1).unsqueeze(0)
        return bias if dtype is None else bias.to(dtype=dtype)


class SpatialSelfAttention2D(nn.Module):
    def __init__(self, dimension, num_heads=4, dropout=0.1,
                 max_relative_height=16, max_relative_width=16, position_scale_init=1.0):
        super().__init__()
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.head_dimension = self.dimension // self.num_heads
        self.position_scale = nn.Parameter(torch.tensor(float(position_scale_init)))
        self.q_projection = nn.Linear(self.dimension, self.dimension)
        self.k_projection = nn.Linear(self.dimension, self.dimension)
        self.v_projection = nn.Linear(self.dimension, self.dimension)
        self.output_projection = nn.Linear(self.dimension, self.dimension)
        self.relative_position_bias = RelativePositionBias2D(
            self.num_heads, max_relative_height, max_relative_width
        )
        self.attention_dropout = float(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.dimension)

    def _split_heads(self, tensor):
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dimension).transpose(1, 2)

    def forward(self, tokens, height, width, return_attention=False):
        positions = sinusoidal_position_encoding_2d(
            height, width, self.dimension, tokens.device
        ).to(dtype=tokens.dtype)
        positioned = tokens + self.position_scale.to(dtype=tokens.dtype) * positions
        query = self._split_heads(self.q_projection(positioned))
        key = self._split_heads(self.k_projection(positioned))
        value = self._split_heads(self.v_projection(tokens))

        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * (self.head_dimension ** -0.5)
        scores = scores + self.relative_position_bias(height, width).float()
        weights = torch.softmax(scores, dim=-1)
        if return_attention and not torch.isfinite(weights).all():
            raise FloatingPointError("Spatial self-attention contains NaN or Inf")
        used_weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        context = torch.matmul(used_weights.to(dtype=value.dtype), value)
        context = context.transpose(1, 2).contiguous().view_as(tokens)
        context = self.norm(tokens + self.output_dropout(self.output_projection(context)))
        if return_attention:
            return context, weights
        return context, None


class DiseaseSpecificAttentionPooling(nn.Module):
    """Disease queries pool contextual spatial tokens and fuse them with raw GAP."""

    def __init__(self, in_channels, num_labels=10, dimension=256, num_heads=4,
                 dropout=0.1, max_relative_height=16, max_relative_width=16,
                 position_scale_init=1.0, alpha_init=0.1):
        super().__init__()
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be strictly between 0 and 1")
        if dimension % 4:
            raise ValueError("dimension must be divisible by 4 for 2D encoding")
        self.in_channels = int(in_channels)
        self.num_labels = int(num_labels)
        self.dimension = int(dimension)
        self.token_projection = nn.Linear(self.in_channels, self.dimension)
        self.spatial_attention = SpatialSelfAttention2D(
            self.dimension, num_heads, dropout, max_relative_height,
            max_relative_width, position_scale_init
        )
        self.disease_queries = nn.Parameter(torch.empty(self.num_labels, self.dimension))
        nn.init.trunc_normal_(self.disease_queries, std=0.02)
        self.pool_key = nn.Linear(self.dimension, self.dimension)
        self.pool_value = nn.Linear(self.dimension, self.in_channels)
        self.alpha_logits = nn.Parameter(torch.full(
            (self.num_labels,), math.log(alpha_init / (1.0 - alpha_init))
        ))
        self.classifier_weight = nn.Parameter(torch.empty(self.num_labels, self.in_channels))
        self.classifier_bias = nn.Parameter(torch.zeros(self.num_labels))
        nn.init.xavier_uniform_(self.classifier_weight)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logits)

    def forward(self, feature_map, return_attention=False):
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [batch, channels, height, width]")
        batch, channels, height, width = feature_map.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}")
        raw_tokens = feature_map.flatten(2).transpose(1, 2)
        projected = self.token_projection(raw_tokens)
        context, spatial_weights = self.spatial_attention(
            projected, height, width, return_attention=return_attention
        )

        keys = self.pool_key(context)
        scores = torch.einsum("kd,bnd->bkn", self.disease_queries, keys)
        scores = scores.float() * (self.dimension ** -0.5)
        disease_weights = torch.softmax(scores, dim=-1)
        values = self.pool_value(context)
        attended = torch.bmm(disease_weights.to(dtype=values.dtype), values)
        gap = feature_map.mean(dim=(2, 3))
        fused = gap[:, None, :] + self.alpha.to(dtype=gap.dtype)[None, :, None] * attended
        logits = (fused * self.classifier_weight[None, :, :]).sum(dim=-1)
        logits = logits + self.classifier_bias[None, :]
        if return_attention:
            if not torch.isfinite(disease_weights).all():
                raise FloatingPointError("Disease attention contains NaN or Inf")
            diagnostics = {
                "disease_attention": disease_weights.view(batch, self.num_labels, height, width),
                "spatial_attention": spatial_weights,
                "alpha": self.alpha,
                "feature_shape": (batch, channels, height, width),
            }
            return logits, diagnostics
        return logits
