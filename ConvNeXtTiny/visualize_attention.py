import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from datasets import build_transform
from datasets.transforms import ResizePad
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


def _heat_overlay(image, attention):
    attention = np.asarray(attention, dtype=np.float32)
    low, high = float(attention.min()), float(attention.max())
    normalized = ((attention - low) / (high - low)) if high > low else np.zeros_like(attention)
    heat = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.asarray(255 * normalized, dtype=np.uint8)
    heat[..., 1] = np.asarray(110 * (1.0 - np.abs(2.0 * normalized - 1.0)), dtype=np.uint8)
    heat[..., 2] = np.asarray(255 * (1.0 - normalized), dtype=np.uint8)
    heat_image = Image.fromarray(heat, mode="RGB").resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), heat_image, alpha=0.42)


def main():
    parser = argparse.ArgumentParser(description="Visualize DSAP-RPB disease attention on one validation image")
    parser.add_argument("--image", required=True, help="Path to one image; use a validation image for diagnostics")
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "outputs/experiments/ConvNeXt_DSAP_RPB/best.pth"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint_path = resolve_config_path(args.checkpoint)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    model_config = dict(config.get("model_config", {}))
    if model_config.get("head") != "dsap_rpb":
        raise ValueError("The supplied checkpoint is not a DSAP-RPB model")
    model_config["pretrained"] = False
    model = ChestXrayClassifier(**{key: value for key, value in model_config.items()
                                   if key != "name"})
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    data_path = resolve_config_path(PROJECT_DIR / "configs/data.yaml")
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    original = Image.open(args.image).convert("RGB")
    input_tensor = build_transform(int(data_cfg["image_size"]))(original).unsqueeze(0).to(device)
    amp_enabled = device.type == "cuda" and bool(config.get("amp", True))
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else torch.float16
    if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                                enabled=amp_enabled):
        logits, diagnostics = model(input_tensor, return_attention=True)
        probabilities = torch.sigmoid(logits.float())[0].cpu().numpy()
        attention_maps = diagnostics["disease_attention"][0].float().cpu().numpy()
        alpha = diagnostics["alpha"].float().cpu().numpy()
    if not np.isfinite(probabilities).all() or not np.isfinite(attention_maps).all():
        raise FloatingPointError("The checkpoint generated non-finite probabilities or attention maps")

    tile_size = 320
    header = 40
    canvas = Image.new("RGB", (tile_size * 4, (tile_size + header) * 3), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    input_image = ResizePad(int(data_cfg["image_size"]))(original).convert("L")
    input_image = input_image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    class_names = config.get("class_names", data_cfg["class_names"])
    tiles = [("Original", input_image.convert("RGB"))]
    for index, class_name in enumerate(class_names):
        tiles.append((
            f"{class_name}: {probabilities[index]:.3f}",
            _heat_overlay(input_image, attention_maps[index]),
        ))
    for index, (title, tile) in enumerate(tiles):
        row, column = divmod(index, 4)
        x, y = column * tile_size, row * (tile_size + header)
        draw.text((x + 6, y + 7), title, fill="black", font=font)
        canvas.paste(tile.resize((tile_size, tile_size)), (x, y + header))
    final_index = 11
    row, column = divmod(final_index, 4)
    x, y = column * tile_size, row * (tile_size + header)
    draw.text((x + 8, y + 8), "Learned alpha by disease", fill="black", font=font)
    for index, (class_name, value) in enumerate(zip(class_names, alpha)):
        draw.text((x + 8, y + header + 8 + 26 * index), f"{class_name}: {value:.4f}",
                  fill="black", font=font)
    output = Path(args.output) if args.output else (
        PROJECT_DIR / "outputs/experiments/ConvNeXt_DSAP_RPB/attention_visualizations"
        / f"{Path(args.image).stem}_attention.png"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(json.dumps({"image": str(Path(args.image).resolve()),
                      "output": str(output),
                      "probabilities": probabilities.tolist(),
                      "alpha": alpha.tolist(),
                      "attention_shape": list(attention_maps.shape)},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
