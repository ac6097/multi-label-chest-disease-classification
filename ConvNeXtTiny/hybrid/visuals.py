from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def _image_from_tensor(tensor):
    array = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = np.rint(array * 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array, mode="RGB")


def save_pixel_reconstruction(path, target, masked, cnn_prediction, vit_prediction):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Original target [0,1]", _image_from_tensor(target)),
        ("Same 60% mask to both branches", _image_from_tensor(masked)),
        ("ConvNeXtV2 pixel reconstruction", _image_from_tensor(cnn_prediction)),
        ("ViT pixel reconstruction", _image_from_tensor(vit_prediction)),
    ]
    width, height = panels[0][1].size
    caption_height = 32
    canvas = Image.new("RGB", (width * len(panels), height + caption_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for index, (label, image) in enumerate(panels):
        left = index * width
        canvas.paste(image, (left, caption_height))
        draw.text((left + 6, 7), label, fill="black", font=font)
    canvas.save(path)


def _heatmap(values, size=320):
    values = torch.as_tensor(values).float().cpu()
    values = torch.nan_to_num(values)
    low, high = values.min(), values.max()
    normalized = ((values - low) / (high - low).clamp_min(1e-12)).numpy()
    red = (255 * normalized).astype(np.uint8)
    blue = (255 * (1.0 - normalized)).astype(np.uint8)
    green = (180 * (1.0 - np.abs(2 * normalized - 1))).astype(np.uint8)
    image = Image.fromarray(np.stack((red, green, blue), axis=-1), mode="RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def save_latent_reconstruction(path, target, prediction, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target.detach().float().cpu()
    prediction = prediction.detach().float().cpu()
    mask = mask.detach().cpu().bool()
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    target_map = target[0].square().mean(dim=0).sqrt()
    error_map = (target[0] - prediction[0]).square().mean(dim=0).sqrt()
    mask_map = mask[0].float()
    panels = [
        ("Stage3 latent target norm", _heatmap(target_map)),
        ("Stage3 masked reconstruction error", _heatmap(error_map)),
        ("Stage3 spatial mask (white=masked)", Image.fromarray(
            np.repeat((mask_map.numpy() * 255).astype(np.uint8)[:, :, None], 3, axis=2), "RGB"
        ).resize((320, 320), Image.Resampling.NEAREST)),
    ]
    width, height = panels[0][1].size
    caption_height = 32
    canvas = Image.new("RGB", (width * len(panels), height + caption_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for index, (label, image) in enumerate(panels):
        left = index * width
        canvas.paste(image, (left, caption_height))
        draw.text((left + 6, 7), label, fill="black", font=font)
    canvas.save(path)

