"""Standalone study inference for DINOv3 R1/R2/R3/R4 feature groups."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from dino_feature_models import DINOv3GlobalLocalClassifier, ResizePadWithMask
from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier, NUM_LABELS
from datasets.transforms import build_transform


class FeaturePredictor:
    def __init__(self, package_dir: str | Path, device: str | None = None):
        self.package_dir = Path(package_dir)
        self.config = json.loads(
            (self.package_dir / "model_config.json").read_text(encoding="utf-8"))
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_config = self.config["model"]
        self.channels_last = bool(model_config.get("channels_last", False))
        backbone = DINOv3ConvNeXtBackbone(
            layer_norm_eps=float(model_config["layer_norm_eps"]),
            layer_scale_init_value=float(model_config["layer_scale_init_value"]),
            drop_path_rate=float(model_config["drop_path_rate"]),
            architecture=str(model_config["architecture"]),
        )
        if model_config["head_type"] == "global_local":
            r3_config = model_config["r3"]
            self.model = DINOv3GlobalLocalClassifier(
                backbone,
                topk_ratio=float(r3_config["topk_ratio"]),
                initial_alpha=float(r3_config["initial_alpha"]),
                maximum_alpha=float(r3_config["maximum_alpha"]),
                mask_valid_fraction_threshold=float(
                    r3_config["mask_valid_fraction_threshold"]),
            )
            self.mask_transform = ResizePadWithMask(
                int(model_config["image_size"]), train=False)
            self.transform = None
        elif model_config["head_type"] == "gap":
            self.model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS)
            self.transform = build_transform(int(model_config["image_size"]), train=False)
            self.mask_transform = None
        else:
            raise ValueError(f"Unsupported feature inference head: {model_config['head_type']}")
        state = load_file(str(self.package_dir / "model.safetensors"), device="cpu")
        result = self.model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict standalone load failed: {result}")
        self.model.to(self.device).eval()
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def predict_study_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("A Study must contain at least one image")
        image_tensors, mask_tensors = [], []
        for image in images:
            image = image.convert("RGB")
            if self.mask_transform is None:
                image_tensors.append(self.transform(image))
            else:
                tensor, mask = self.mask_transform(image)
                image_tensors.append(tensor)
                mask_tensors.append(mask)
        batch = torch.stack(image_tensors).to(self.device)
        if self.channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        masks = torch.stack(mask_tensors).to(self.device) if mask_tensors else None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            logits = self.model(batch) if masks is None else self.model(batch, masks)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        mean_logits = logits.float().mean(dim=0)
        probabilities = torch.sigmoid(mean_logits).cpu().numpy()
        self.last_peak_gpu_gib = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if self.device.type == "cuda" else 0.0)
        return probabilities

    def predict_study_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        if not paths:
            raise ValueError("A Study must contain at least one image path")
        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        start = time.perf_counter()
        result = self.predict_study_images(images)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return result
