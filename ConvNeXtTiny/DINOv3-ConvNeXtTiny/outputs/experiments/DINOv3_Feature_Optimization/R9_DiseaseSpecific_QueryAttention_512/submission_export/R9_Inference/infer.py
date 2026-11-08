"""Standalone inference for one Study; Study Logit Mean before sigmoid."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from dino_model import DINOv3ConvNeXtBackbone
from stage_fusion_models import build_model


def preprocess(path, size):
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = T.functional.resize(image, (resized_height, resized_width),
                                InterpolationMode.BILINEAR)
    pad_w = size - image.size[0]
    pad_h = size - image.size[1]
    left, top = pad_w // 2, pad_h // 2
    image = T.functional.pad(image, [left, top, pad_w - left, pad_h - top], fill=0)
    image = T.Grayscale(num_output_channels=3)(image)
    tensor = T.functional.to_tensor(image)
    return T.functional.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("model_config.json"))
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["image_size"] != 512 or config["num_labels"] != 10:
        raise RuntimeError("Unexpected package configuration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DINOv3ConvNeXtBackbone(
        layer_norm_eps=float(config["layer_norm_eps"]),
        layer_scale_init_value=float(config["layer_scale_init_value"]),
        drop_path_rate=float(config["drop_path_rate"]),
        architecture=config["architecture_name"])
    model = build_model(config["experiment"], backbone)
    state = load_file(str(args.config.parent / "model.safetensors"), device="cpu")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    model.to(device).eval()
    images = torch.stack([preprocess(path, int(config["image_size"]))
                          for path in args.images]).to(device)
    with torch.inference_mode():
        logits = model(images).float().mean(dim=0)
        probabilities = torch.sigmoid(logits).cpu().tolist()
    payload = {
        "experiment": config["experiment"],
        "label_names": config["label_names"],
        "probabilities": probabilities,
        "study_fusion": "mean_raw_image_logits_before_sigmoid",
        "test_data_accessed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
