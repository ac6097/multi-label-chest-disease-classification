"""Standalone R7 inference for one Study at a time."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from datasets.transforms import build_transform
from dino_model import DINOv3ConvNeXtBackbone, NUM_LABELS
from r7_model import R7Stage3Stage4ResidualClassifier


class R7Predictor:
    def __init__(self, package_dir: str | Path, device: str | None = None):
        self.package_dir = Path(package_dir)
        self.config = json.loads(
            (self.package_dir / "model_config.json").read_text(encoding="utf-8"))
        model_config = self.config["model"]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.channels_last = bool(model_config["channels_last"])
        backbone = DINOv3ConvNeXtBackbone(
            layer_norm_eps=float(model_config["layer_norm_eps"]),
            layer_scale_init_value=float(model_config["layer_scale_init_value"]),
            drop_path_rate=float(model_config["drop_path_rate"]),
            architecture=str(model_config["architecture"]),
        )
        self.model = R7Stage3Stage4ResidualClassifier(
            backbone, num_labels=NUM_LABELS)
        state = load_file(str(self.package_dir / "model.safetensors"), device="cpu")
        result = self.model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict R7 inference load failed: {result}")
        self.model.to(self.device).eval()
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = build_transform(int(model_config["image_size"]), train=False)
        self.last_inference_ms = 0.0
        self.last_end_to_end_ms = 0.0
        self.last_peak_gpu_gib = 0.0

    @torch.inference_mode()
    def predict_study_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("A Study must contain at least one image")
        started = time.perf_counter()
        batch = torch.stack([
            self.transform(image.convert("RGB")) for image in images
        ]).to(self.device)
        if self.channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        inference_started = time.perf_counter()
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_logits = self.model(batch)
        else:
            image_logits = self.model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_ms = (time.perf_counter() - inference_started) * 1000.0
        probabilities = torch.sigmoid(image_logits.float().mean(dim=0)).cpu().numpy()
        self.last_end_to_end_ms = (time.perf_counter() - started) * 1000.0
        self.last_peak_gpu_gib = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if self.device.type == "cuda" else 0.0)
        return probabilities

    def predict_study_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        if not paths:
            raise ValueError("A Study must contain at least one image path")
        started = time.perf_counter()
        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        probabilities = self.predict_study_images(images)
        self.last_end_to_end_ms = (time.perf_counter() - started) * 1000.0
        return probabilities
