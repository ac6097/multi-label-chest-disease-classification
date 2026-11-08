"""Standalone DINOv3 ConvNeXt-Tiny study inference entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from dino_model import DINOv3StudyClassifier


class Predictor:
    def __init__(self, package_dir: str | Path, device: str | None = None):
        self.package_dir = Path(package_dir)
        self.config = json.loads((self.package_dir / "model_config.json").read_text(encoding="utf-8"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DINOv3StudyClassifier(num_labels=10)
        state = load_file(str(self.package_dir / "model.safetensors"), device="cpu")
        result = self.model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Inference checkpoint load was incomplete: {result}")
        self.model.to(self.device).eval()
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(tuple(self.config["input_mean"]), tuple(self.config["input_std"])),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        size = int(self.config["image_size"])
        scale = size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = T.functional.resize(image, (new_height, new_width),
                                    InterpolationMode.BILINEAR)
        pad_width, pad_height = size - new_width, size - new_height
        left, top = pad_width // 2, pad_height // 2
        image = T.functional.pad(image, [left, top, pad_width - left,
                                         pad_height - top], fill=0)
        tensor = self.transform(image)
        if tuple(tensor.shape) != (3, size, size):
            raise RuntimeError(f"Preprocessor returned unexpected shape {tuple(tensor.shape)}")
        return tensor

    @torch.inference_mode()
    def predict_study_tensor(self, normalized_images: torch.Tensor) -> np.ndarray:
        if normalized_images.ndim != 4 or normalized_images.shape[1:] != (3, 320, 320):
            raise ValueError("Expected normalized Study images with shape [N,3,320,320]")
        images = normalized_images.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda" and torch.cuda.is_bf16_supported()):
            image_logits = self.model(images)
        study_logits = image_logits.float().mean(dim=0)
        return torch.sigmoid(study_logits).cpu().numpy()

    @torch.inference_mode()
    def predict_study_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("A Study must contain at least one image")
        tensor = torch.stack([self.preprocess(image) for image in images])
        return self.predict_study_tensor(tensor)

    @torch.inference_mode()
    def predict_study_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        if not paths:
            raise ValueError("A Study must contain at least one image path")
        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        return self.predict_study_images(images)
