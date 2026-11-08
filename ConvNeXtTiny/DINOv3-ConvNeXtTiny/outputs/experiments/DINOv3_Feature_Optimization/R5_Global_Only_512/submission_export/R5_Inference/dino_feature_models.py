"""Feature-optimization data adapters and model variants for R3/R4."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from datasets.dataset import StudyDataset, collate_studies
from datasets.transforms import ResizePad
from dino_data import EpochTaggedStudyDataset, read_data_config
from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier, FEATURE_DIM, NUM_LABELS
from hybrid.common import seed_everything, sha256_file


class ResizePadWithMask:
    """Apply the project ResizePad and carry its valid image region through augmentation."""

    def __init__(self, size: int, train: bool):
        self.size = int(size)
        self.train = bool(train)
        self.color_jitter = T.ColorJitter(brightness=0.05, contrast=0.05)
        self.normalize = T.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))

    def __call__(self, image: Image.Image):
        width, height = image.size
        scale = self.size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        image = T.functional.resize(
            image, (resized_height, resized_width), InterpolationMode.BILINEAR)
        pad_width = self.size - resized_width
        pad_height = self.size - resized_height
        left, top = pad_width // 2, pad_height // 2
        right, bottom = pad_width - left, pad_height - top
        image = T.functional.pad(image, [left, top, right, bottom], fill=0)

        mask = Image.new("L", (self.size, self.size), 0)
        mask.paste(255, (left, top, left + resized_width, top + resized_height))
        image = T.Grayscale(num_output_channels=3)(image)
        if self.train:
            if random.random() < 0.5:
                image = self.color_jitter(image)
            angle, translate, scale_factor, shear = T.RandomAffine.get_params(
                degrees=(0.0, 0.0), translate=(0.02, 0.02),
                scale_ranges=(0.97, 1.03), shears=None,
                img_size=[self.size, self.size])
            image = T.functional.affine(
                image, angle, translate, scale_factor, shear,
                interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = T.functional.affine(
                mask, angle, translate, scale_factor, shear,
                interpolation=InterpolationMode.NEAREST, fill=0)
        image_tensor = self.normalize(T.ToTensor()(image))
        mask_tensor = T.functional.to_tensor(mask) > 0.5
        return image_tensor, mask_tensor


class MaskedStudyDataset(Dataset):
    """StudyDataset adapter that returns an explicit post-ResizePad valid region mask."""

    def __init__(self, base_dataset: StudyDataset, size: int, train: bool):
        self.base = base_dataset
        self.transform = ResizePadWithMask(size, train=train)

    def __getattr__(self, name):
        if name in {"base", "transform"}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        study = self.base.studies[int(index)]
        images, masks = [], []
        for path in study["image_paths"]:
            with Image.open(path) as source:
                image = source.convert("RGB")
            image_tensor, valid_mask = self.transform(image)
            images.append(image_tensor)
            masks.append(valid_mask)
        return {
            "images": torch.stack(images),
            "valid_masks": torch.stack(masks),
            "labels": study["target"],
            "known": study["valid"],
            "subject_id": study["subject_id"],
            "study_id": study["study_id"],
            "image_ids": study["image_ids"],
        }


def collate_masked_studies(items):
    batch = collate_studies(items)
    batch["valid_masks"] = torch.cat(
        [item["valid_masks"] for item in items], dim=0)
    return batch


def build_masked_datasets(config_path: Path, seed: int, image_size: int):
    config = read_data_config(config_path, image_size=int(image_size))
    shared = {
        "label_column": config["label_column"],
        "label_semantics": config["label_semantics"],
        "image_id_column": config["image_id_column"],
        "subject_column": config["subject_column"],
        "study_column": config["study_column"],
    }
    train_base = StudyDataset(config["train_csv"], config["image_root"],
                              transform=None, **shared)
    valid_base = StudyDataset(config["val_csv"], config["image_root"],
                              transform=None, **shared)
    if train_base.split != "train" or valid_base.split != "valid":
        raise RuntimeError("R3 is restricted to the official Train and Validation splits")
    train_data = EpochTaggedStudyDataset(
        MaskedStudyDataset(train_base, image_size, train=True), int(seed))
    valid_data = EpochTaggedStudyDataset(
        MaskedStudyDataset(valid_base, image_size, train=False), int(seed))
    return train_data, valid_data, config


class DINOv3GlobalLocalClassifier(nn.Module):
    """GAP logits plus masked Stage 3 Top-K disease-response logits."""

    def __init__(self, backbone: DINOv3ConvNeXtBackbone, topk_ratio: float = 0.1,
                 initial_alpha: float = 0.05, maximum_alpha: float = 0.2,
                 mask_valid_fraction_threshold: float = 0.5):
        super().__init__()
        if not (0.0 < float(topk_ratio) <= 1.0):
            raise ValueError("topk_ratio must be in (0,1]")
        if not (0.0 < float(initial_alpha) < float(maximum_alpha) <= 1.0):
            raise ValueError("R3 alpha must satisfy 0 < initial < maximum <= 1")
        self.backbone = backbone
        self.classifier = nn.Linear(FEATURE_DIM, NUM_LABELS)
        self.local_classifier = nn.Conv2d(384, NUM_LABELS, kernel_size=1)
        self.alpha = nn.ParameterDict({
            "logit": nn.Parameter(torch.tensor(
                math.log(initial_alpha / (maximum_alpha - initial_alpha)),
                dtype=torch.float32))
        })
        self.topk_ratio = float(topk_ratio)
        self.maximum_alpha = float(maximum_alpha)
        self.mask_valid_fraction_threshold = float(mask_valid_fraction_threshold)

    def _masked_topk_pool(self, response_maps: torch.Tensor,
                          valid_masks: torch.Tensor):
        batch, labels, height, width = response_maps.shape
        valid = F.adaptive_avg_pool2d(
            valid_masks.float(), output_size=(height, width))[:, 0]
        valid = valid >= self.mask_valid_fraction_threshold
        flattened = response_maps.flatten(2)
        valid_flat = valid.flatten(1)
        local_logits, selected_masks, valid_counts, selected_counts = [], [], [], []
        for row in range(batch):
            valid_indices = torch.nonzero(valid_flat[row], as_tuple=False).flatten()
            count = int(valid_indices.numel())
            if count == 0:
                raise RuntimeError("ResizePad mask removed every Stage 3 position")
            k = min(count, max(1, math.ceil(count * self.topk_ratio)))
            valid_responses = flattened[row].index_select(1, valid_indices)
            top_values, top_indices = torch.topk(valid_responses, k=k, dim=1)
            local_logits.append(top_values.mean(dim=1))
            chosen_indices = valid_indices.index_select(0, top_indices.reshape(-1))
            chosen = torch.zeros(
                (labels, height * width), dtype=torch.bool, device=response_maps.device)
            chosen.scatter_(1, chosen_indices.reshape(labels, k), True)
            selected_masks.append(chosen.reshape(labels, height, width))
            valid_counts.append(count)
            selected_counts.append(k)
        return (torch.stack(local_logits), valid, torch.stack(selected_masks),
                valid_counts, selected_counts)

    def forward(self, images: torch.Tensor, valid_masks: torch.Tensor | None = None,
                return_components: bool = False):
        if valid_masks is None:
            raise ValueError("R3 requires a ResizePad valid-region mask per image")
        global_features, stage3, stage4 = self.backbone.forward_features_with_stage3(images)
        if stage3.shape[1] != self.local_classifier.in_channels:
            raise RuntimeError(
                f"Stage 3 channels {stage3.shape[1]} != local head channels "
                f"{self.local_classifier.in_channels}")
        global_logits = self.classifier(global_features)
        response_maps = self.local_classifier(stage3)
        local_logits, valid_grid, topk_grid, valid_counts, selected_counts = (
            self._masked_topk_pool(response_maps, valid_masks))
        alpha = self.maximum_alpha * torch.sigmoid(self.alpha["logit"])
        fused_logits = (1.0 - alpha) * global_logits + alpha * local_logits
        if return_components:
            return {
                "logits": fused_logits,
                "global_logits": global_logits,
                "local_logits": local_logits,
                "response_maps": response_maps,
                "valid_grid": valid_grid,
                "topk_grid": topk_grid,
                "alpha": alpha,
                "valid_counts": valid_counts,
                "selected_counts": selected_counts,
                "stage4": stage4,
            }
        return fused_logits


def load_small_pretrained_backbone(repo_id: str, revision: str = "main", device="cpu",
                                   expected_sha256: str | None = None,
                                   local_files_only: bool = False):
    """Download and strictly verify the official general-vision DINOv3 Small weights."""
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file

    if revision == "main" and not local_files_only:
        api = HfApi(token=True)
        model_info = api.model_info(repo_id=repo_id, revision=revision, token=True)
        resolved_revision = str(model_info.sha)
    else:
        resolved_revision = str(revision)
    paths = {}
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
        paths[filename] = Path(hf_hub_download(
            repo_id=repo_id, filename=filename, revision=resolved_revision,
            token=True, local_files_only=bool(local_files_only)))
    import json
    with paths["config.json"].open("r", encoding="utf-8") as stream:
        hf_config = json.load(stream)
    expected = {
        "architectures": ["DINOv3ConvNextModel"],
        "depths": [3, 3, 27, 3],
        "hidden_sizes": [96, 192, 384, 768],
        "layer_norm_eps": 1e-6,
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.0,
        "num_channels": 3,
        "model_type": "dinov3_convnext",
    }
    for key, expected_value in expected.items():
        if hf_config.get(key) != expected_value:
            raise RuntimeError(f"Unexpected DINOv3 Small {key}: {hf_config.get(key)!r}")
    with paths["preprocessor_config.json"].open("r", encoding="utf-8") as stream:
        processor = json.load(stream)
    if processor.get("image_mean") != [0.485, 0.456, 0.406]:
        raise RuntimeError("DINOv3 Small processor mean does not match the configured normalize")
    if processor.get("image_std") != [0.229, 0.224, 0.225]:
        raise RuntimeError("DINOv3 Small processor std does not match the configured normalize")
    sha256 = sha256_file(paths["model.safetensors"])
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise RuntimeError(
            f"DINOv3 Small weights SHA-256 mismatch: expected {expected_sha256}, got {sha256}")
    state = load_file(str(paths["model.safetensors"]), device="cpu")
    backbone = DINOv3ConvNeXtBackbone(
        layer_norm_eps=float(hf_config["layer_norm_eps"]),
        layer_scale_init_value=float(hf_config["layer_scale_init_value"]),
        drop_path_rate=float(hf_config["drop_path_rate"]),
        architecture="convnext_small")
    result = backbone.load_hf_state_dict(state)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict DINOv3 Small load was incomplete: {result}")
    backbone.to(device)
    verification = {
        "repository": repo_id,
        "revision": resolved_revision,
        "weights_file": str(paths["model.safetensors"]),
        "weights_sha256": sha256,
        "depths": hf_config["depths"],
        "hidden_sizes": hf_config["hidden_sizes"],
        "layer_norm_eps": float(hf_config["layer_norm_eps"]),
        "layer_scale_init_value": float(hf_config["layer_scale_init_value"]),
        "drop_path_rate": float(hf_config["drop_path_rate"]),
        "input_mean": processor["image_mean"],
        "input_std": processor["image_std"],
        "initialization_type": "official general-vision self-supervised DINOv3",
    }
    return backbone, verification


def feature_profile_layout(model: nn.Module):
    from dino_freeze_experiment import profile_layout

    trainable_modules, frozen_modules, _ = profile_layout(model, "p2")
    for parameter in model.local_classifier.parameters():
        parameter.requires_grad_(True)
    for parameter in model.alpha.parameters():
        parameter.requires_grad_(True)
    trainable_modules.extend([
        ("local_classifier", model.local_classifier),
        ("alpha", model.alpha),
    ])
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    expected_prefixes = (
        "classifier.", "local_classifier.", "alpha.",
        "backbone.model.downsample_layers.2.", "backbone.model.stages.2.",
        "backbone.model.downsample_layers.3.", "backbone.model.stages.3.",
        "backbone.model.norm.", "backbone.model.norms.3.",
    )
    unexpected = [name for name in trainable_names
                  if not name.startswith(expected_prefixes)]
    if unexpected:
        raise RuntimeError(f"R3 unexpectedly unfroze parameters: {unexpected[:10]}")
    return trainable_modules, frozen_modules, trainable_names


def build_r3_model(profile, device, runtime, seed, feature):
    from dino_experiment import load_pretrained_backbone

    seed_everything(seed)
    backbone, verification = load_pretrained_backbone("cpu")
    model = DINOv3GlobalLocalClassifier(
        backbone,
        topk_ratio=float(feature["r3"]["topk_ratio"]),
        initial_alpha=float(feature["r3"]["initial_alpha"]),
        maximum_alpha=float(feature["r3"]["maximum_alpha"]),
        mask_valid_fraction_threshold=float(
            feature["r3"]["mask_valid_fraction_threshold"]),
    ).to(device)
    trainable_modules, frozen_modules, trainable_names = feature_profile_layout(model)
    if runtime is not None:
        model.backbone.set_gradient_checkpointing(
            bool(runtime["gradient_checkpointing"]))
        if runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)
    return model, verification, trainable_modules, frozen_modules, trainable_names


def build_small_model(profile, device, runtime, seed, feature,
                      pinned_verification: dict | None = None):
    from dino_freeze_experiment import profile_layout

    seed_everything(seed)
    revision = (feature["source"]["small_revision"] if pinned_verification is None
                else pinned_verification["revision"])
    backbone, verification = load_small_pretrained_backbone(
        feature["source"]["small_repo"], revision, "cpu",
        expected_sha256=(None if pinned_verification is None
                         else pinned_verification["weights_sha256"]),
        local_files_only=pinned_verification is not None)
    model = DINOv3StudyClassifier(backbone, num_labels=NUM_LABELS).to(device)
    trainable_modules, frozen_modules, trainable_names = profile_layout(model, "p2")
    if runtime is not None:
        model.backbone.set_gradient_checkpointing(
            bool(runtime["gradient_checkpointing"]))
        if runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)
    return model, verification, trainable_modules, frozen_modules, trainable_names


def feature_profile_optimizer(model, config):
    training = config["training"]
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [p for name, p in model.named_parameters()
                       if p.requires_grad and not name.startswith("backbone.")]
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters,
                       "lr": float(training["peak_lr_backbone"]),
                       "name": "backbone"})
    if head_parameters:
        groups.append({"params": head_parameters,
                       "lr": float(training["peak_lr_head"]),
                       "name": "head"})
    if not groups:
        raise RuntimeError("Feature experiment has no trainable optimizer parameters")
    return torch.optim.AdamW(
        groups, weight_decay=float(training["weight_decay"]),
        fused=bool(config["freeze"]["training"]["fused_adamw"]))


def r3_batch_forward(model, images, batch):
    masks = batch["valid_masks"].to(images.device, non_blocking=True)
    return model(images, masks)
