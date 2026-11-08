import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


MASK_GRID_SIZE = 10
MASK_CELLS = 60


def sample_spatial_mask(batch_size, device, generator=None):
    """Return an exact 60/100 random mask for 32x32 regions in a 320x320 image."""
    order = torch.rand((batch_size, MASK_GRID_SIZE * MASK_GRID_SIZE),
                       device=device, generator=generator).argsort(dim=1)
    mask = torch.zeros_like(order, dtype=torch.bool)
    mask.scatter_(1, order[:, :MASK_CELLS], True)
    return mask.view(batch_size, MASK_GRID_SIZE, MASK_GRID_SIZE)


def sample_spatial_mask_for_ids(subject_ids, study_ids, image_ids, epoch, seed, device):
    """Generate masks keyed by image identity so H1/H2 masks match across microbatch sizes."""
    if not (len(subject_ids) == len(study_ids) == len(image_ids)):
        raise ValueError("Subject, Study and image ID lists must have the same length")
    masks = []
    for subject_id, study_id, image_id in zip(subject_ids, study_ids, image_ids):
        key = f"{int(seed)}:{int(epoch)}:{subject_id}:{study_id}:{image_id}".encode("utf-8")
        per_image_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        generator = torch.Generator(device="cpu").manual_seed(per_image_seed & ((1 << 63) - 1))
        order = torch.randperm(MASK_GRID_SIZE * MASK_GRID_SIZE, generator=generator)
        mask = torch.zeros(MASK_GRID_SIZE * MASK_GRID_SIZE, dtype=torch.bool)
        mask[order[:MASK_CELLS]] = True
        masks.append(mask.view(MASK_GRID_SIZE, MASK_GRID_SIZE))
    return torch.stack(masks).to(device=device)


def mask_to_vit_grid(mask_10):
    return mask_10.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def mask_to_pixels(mask, patch_size):
    return (mask.repeat_interleave(patch_size, dim=1)
            .repeat_interleave(patch_size, dim=2))


def masked_pixel_mse(prediction, target, mask, patch_size):
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    pixel_mask = mask_to_pixels(mask, patch_size).unsqueeze(1)
    squared_error = (prediction.float() - target.float()).square()
    per_image_sum = (squared_error * pixel_mask).sum(dim=(1, 2, 3))
    per_image_count = pixel_mask.sum(dim=(1, 2, 3)) * prediction.shape[1]
    if torch.any(per_image_count == 0):
        raise ValueError("Masked pixel loss received a sample with no masked pixels")
    return (per_image_sum / per_image_count).mean()


def normalized_latents(features):
    """Normalize each channel vector to give both latent objectives a common scale."""
    if features.ndim == 4:
        channels_last = features.permute(0, 2, 3, 1)
        normalized = F.layer_norm(channels_last, (features.shape[1],))
        return normalized.permute(0, 3, 1, 2)
    if features.ndim == 3:
        return F.layer_norm(features, (features.shape[-1],))
    raise ValueError("Latent features must be a spatial map or token sequence")


def masked_latent_mse(prediction, target, spatial_mask):
    if prediction.shape != target.shape:
        raise ValueError(f"Latent prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    squared_error = (prediction.float() - target.float()).square()
    if prediction.ndim == 4:
        per_position = squared_error.mean(dim=1)
        mask = spatial_mask.to(per_position.dtype)
    elif prediction.ndim == 3:
        per_position = squared_error.mean(dim=2)
        mask = spatial_mask.to(per_position.dtype)
    else:
        raise ValueError("Latent features must be [B,C,H,W] or [B,N,C]")
    counts = mask.flatten(1).sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Masked latent loss received a sample with no masked positions")
    return ((per_position * mask).flatten(1).sum(dim=1) / counts).mean()


def unpatchify_vit_pixels(patches, grid_size=20, patch_size=16, channels=3):
    batch, token_count, pixel_values = patches.shape
    expected = grid_size * grid_size
    if token_count != expected or pixel_values != channels * patch_size * patch_size:
        raise ValueError("ViT patch decoder output has an unexpected size")
    patches = patches.view(batch, grid_size, grid_size, channels, patch_size, patch_size)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(
        batch, channels, grid_size * patch_size, grid_size * patch_size)


class ConvPixelDecoder(nn.Module):
    """Predict each 32x32 pixel tile from its corresponding final CNN feature."""

    def __init__(self, channels=768, hidden=128, patch_size=32):
        super().__init__()
        self.patch_size = int(patch_size)
        self.project = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3 * self.patch_size * self.patch_size, kernel_size=1),
        )

    def forward(self, features):
        batch, _, grid_h, grid_w = features.shape
        p = self.patch_size
        tiles = self.project(features).view(batch, 3, p, p, grid_h, grid_w)
        return tiles.permute(0, 1, 4, 2, 5, 3).reshape(batch, 3, grid_h * p, grid_w * p)


class VitPixelDecoder(nn.Module):
    def __init__(self, width=384, patch_size=16):
        super().__init__()
        self.patch_size = int(patch_size)
        self.project = nn.Linear(width, 3 * self.patch_size * self.patch_size)

    def forward(self, patch_tokens):
        return unpatchify_vit_pixels(self.project(patch_tokens))


class ConvLatentDecoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=384):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, stage4):
        features = F.interpolate(stage4, scale_factor=2, mode="bilinear", align_corners=False)
        return self.project(features)


class VitLatentDecoder(nn.Module):
    def __init__(self, width=384):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width),
        )

    def forward(self, tokens):
        return self.project(tokens)


class PixelReconstructionPretrainer(nn.Module):
    """H1: dense pixel masking and two branch-specific pixel reconstruction decoders."""

    def __init__(self, backbone, checkpoint_blocks=False):
        super().__init__()
        self.backbone = backbone
        self.cnn_decoder = ConvPixelDecoder()
        self.vit_decoder = VitPixelDecoder()
        self.vit_mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.checkpoint_blocks = bool(checkpoint_blocks)
        nn.init.trunc_normal_(self.vit_mask_token, std=0.02)

    def _run_block(self, block, tokens):
        if self.checkpoint_blocks and self.training and tokens.requires_grad:
            return checkpoint(block, tokens, use_reentrant=False)
        return block(tokens)

    def forward(self, raw_images, mask_10):
        if raw_images.shape[-2:] != (320, 320):
            raise ValueError("H1 pixel reconstruction requires 320x320 images")
        if mask_10.shape != (raw_images.shape[0], 10, 10):
            raise ValueError("H1 mask must have shape [B,10,10]")
        pixel_mask = mask_to_pixels(mask_10, 32).unsqueeze(1)
        masked_images = raw_images * (~pixel_mask).to(raw_images.dtype)

        _, cnn_stage4 = self.backbone.convnext_features(masked_images)
        cnn_pixels = self.cnn_decoder(cnn_stage4)

        tokens = self.backbone.vit_input_tokens(masked_images)
        vit_mask = mask_to_vit_grid(mask_10).flatten(1)
        position_tokens = self.backbone.vit.pos_embed[:, self.backbone.vit.num_prefix_tokens:]
        replacement = self.vit_mask_token.to(tokens.dtype) + position_tokens.to(tokens.dtype)
        patch_tokens = torch.where(vit_mask.unsqueeze(-1), replacement, tokens[:, 1:])
        tokens = torch.cat((tokens[:, :1], patch_tokens), dim=1)
        for block in self.backbone.vit.blocks:
            tokens = self._run_block(block, tokens)
        tokens = self.backbone.vit.norm(tokens)
        vit_pixels = self.vit_decoder(tokens[:, self.backbone.vit.num_prefix_tokens:])

        cnn_loss = masked_pixel_mse(cnn_pixels, raw_images, mask_10, patch_size=32)
        vit_loss = masked_pixel_mse(vit_pixels, raw_images, mask_10, patch_size=32)
        total_loss = 0.5 * cnn_loss + 0.5 * vit_loss
        return {
            "loss": total_loss,
            "cnn_loss": cnn_loss,
            "vit_loss": vit_loss,
            "mask_ratio": mask_10.float().mean().detach(),
            "cnn_prediction_std": cnn_pixels.float().std(unbiased=False).detach(),
            "vit_prediction_std": vit_pixels.float().std(unbiased=False).detach(),
            "cnn_pixels": cnn_pixels,
            "vit_pixels": vit_pixels,
            "masked_images": masked_images,
            "mask_10": mask_10,
        }


class LatentReconstructionPretrainer(nn.Module):
    """H2: masked Stage 3 and ViT block-6 feature reconstruction without a teacher."""

    def __init__(self, backbone, checkpoint_blocks=False):
        super().__init__()
        self.backbone = backbone
        self.cnn_decoder = ConvLatentDecoder()
        self.vit_decoder = VitLatentDecoder()
        self.cnn_mask_token = nn.Parameter(torch.zeros(1, 384, 1, 1))
        self.vit_mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.checkpoint_blocks = bool(checkpoint_blocks)
        nn.init.trunc_normal_(self.cnn_mask_token, std=0.02)
        nn.init.trunc_normal_(self.vit_mask_token, std=0.02)

    def _run_block(self, block, tokens):
        if self.checkpoint_blocks and self.training and tokens.requires_grad:
            return checkpoint(block, tokens, use_reentrant=False)
        return block(tokens)

    def forward(self, raw_images, mask_10):
        if raw_images.shape[-2:] != (320, 320):
            raise ValueError("H2 latent reconstruction requires 320x320 images")
        if mask_10.shape != (raw_images.shape[0], 10, 10):
            raise ValueError("H2 mask must have shape [B,10,10]")
        mask_20 = mask_to_vit_grid(mask_10)

        cnn_stage3 = self.backbone.convnext_stage3(raw_images)
        if cnn_stage3.shape[1:] != (384, 20, 20):
            raise RuntimeError(f"Expected Stage 3 [B,384,20,20], got {tuple(cnn_stage3.shape)}")
        cnn_target = normalized_latents(cnn_stage3).detach()
        cnn_masked = torch.where(
            mask_20.unsqueeze(1), self.cnn_mask_token.to(cnn_stage3.dtype), cnn_stage3)
        cnn_stage4 = self.backbone.convnext_stage4_from_stage3(cnn_masked)
        cnn_prediction = normalized_latents(self.cnn_decoder(cnn_stage4))
        cnn_loss = masked_latent_mse(cnn_prediction, cnn_target, mask_20)

        tokens = self.backbone.vit_input_tokens(raw_images)
        if tokens.shape[1:] != (401, 384):
            raise RuntimeError(f"Expected ViT input tokens [B,401,384], got {tuple(tokens.shape)}")
        for block in self.backbone.vit.blocks[:6]:
            tokens = self._run_block(block, tokens)
        t6 = tokens[:, self.backbone.vit.num_prefix_tokens:]
        vit_target = normalized_latents(t6).detach()
        mask_flat = mask_20.flatten(1)
        position_tokens = self.backbone.vit.pos_embed[:, self.backbone.vit.num_prefix_tokens:]
        replacement = self.vit_mask_token.to(tokens.dtype) + position_tokens.to(tokens.dtype)
        masked_t6 = torch.where(mask_flat.unsqueeze(-1), replacement, t6)
        tokens = torch.cat((tokens[:, :self.backbone.vit.num_prefix_tokens], masked_t6), dim=1)
        for block in self.backbone.vit.blocks[6:]:
            tokens = self._run_block(block, tokens)
        tokens = self.backbone.vit.norm(tokens)
        vit_prediction = normalized_latents(
            self.vit_decoder(tokens[:, self.backbone.vit.num_prefix_tokens:]))
        vit_loss = masked_latent_mse(vit_prediction, vit_target, mask_flat)

        total_loss = 0.5 * cnn_loss + 0.5 * vit_loss
        return {
            "loss": total_loss,
            "cnn_loss": cnn_loss,
            "vit_loss": vit_loss,
            "mask_ratio": mask_10.float().mean().detach(),
            "cnn_feature_variance": cnn_target.var(unbiased=False).detach(),
            "vit_feature_variance": vit_target.var(unbiased=False).detach(),
            "cnn_spatial_variance": cnn_stage3.float().var(dim=(2, 3), unbiased=False).mean().detach(),
            "vit_spatial_variance": t6.float().var(dim=1, unbiased=False).mean().detach(),
            "cnn_prediction_spatial_std": cnn_prediction.float().std(dim=(2, 3), unbiased=False).mean().detach(),
            "vit_prediction_spatial_std": vit_prediction.float().std(dim=1, unbiased=False).mean().detach(),
            "cnn_prediction": cnn_prediction,
            "cnn_target": cnn_target,
            "cnn_mask": mask_20,
            "vit_prediction": vit_prediction,
            "vit_target": vit_target,
            "vit_mask": mask_flat,
        }
