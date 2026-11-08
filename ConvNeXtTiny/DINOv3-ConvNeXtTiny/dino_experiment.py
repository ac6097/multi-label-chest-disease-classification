"""Reproducible DINOv3 ConvNeXt-Tiny D0 / D0-C1 experiment runner."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)
from torch.utils.data import DataLoader


WORK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORK_DIR.parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dino_data import (build_datasets, build_transform, make_loader,
                       shutdown_loader)  # noqa: E402
from dino_analysis import compare_runs  # noqa: E402
from dino_losses import (contrastive_relation_masks, multi_positive_info_nce,
                         study_bce_per_study, study_logit_mean)  # noqa: E402
from dino_model import (FEATURE_DIM, NUM_LABELS, DINOv3StudyClassifier,
                        DINOv3ConvNeXtBackbone, StudyProjectionHead)  # noqa: E402
from hybrid.common import (json_sanitize, seed_everything, sha256_file,
                           write_csv, write_json)  # noqa: E402


CONFIG_PATH = WORK_DIR / "configs" / "dino_experiment.json"
LABEL_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
]
MODEL_REPO = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
MODEL_REVISION = "10d30274b4d445111e2d5bf75ac93bbd94db274b"
MODEL_SHA256 = "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config["source"]["huggingface_repo"] != MODEL_REPO:
        raise ValueError("Unexpected DINOv3 model repository in experiment config")
    if config["source"]["weights_sha256"] != MODEL_SHA256:
        raise ValueError("Unexpected DINOv3 initialization SHA-256 in experiment config")
    if int(config["data"]["image_size"]) != 320:
        raise ValueError("DINOv3 experiment is fixed to 320x320")
    return config


def output_root(config):
    root = Path(config["outputs"]["root"])
    return root if root.is_absolute() else WORK_DIR / root


def setup_logging(output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dino.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def save_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(json_sanitize(payload), indent=2,
                                ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def save_torch_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def parameter_counts(module):
    return {
        "total": sum(p.numel() for p in module.parameters()),
        "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
    }


def write_csv_union(path: Path, rows: list[dict]):
    fieldnames = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    write_csv(path, rows, fieldnames=fieldnames)


def hf_cached_file(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        MODEL_REPO, filename, revision=MODEL_REVISION,
        token=True, local_files_only=True,
    )
    return Path(path)


def load_pretrained_backbone(device="cpu"):
    config_path = hf_cached_file("config.json")
    with config_path.open("r", encoding="utf-8") as stream:
        hf_config = json.load(stream)
    expected = {
        "architectures": ["DINOv3ConvNextModel"],
        "depths": [3, 3, 9, 3],
        "hidden_sizes": [96, 192, 384, 768],
        "layer_norm_eps": 1e-6,
        "layer_scale_init_value": 1e-6,
        "drop_path_rate": 0.0,
        "num_channels": 3,
        "model_type": "dinov3_convnext",
    }
    for key, expected_value in expected.items():
        if hf_config.get(key) != expected_value:
            raise RuntimeError(f"Unexpected DINOv3 {key}: {hf_config.get(key)!r}")

    processor_path = hf_cached_file("preprocessor_config.json")
    with processor_path.open("r", encoding="utf-8") as stream:
        processor_config = json.load(stream)
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    if processor_config.get("image_mean") != expected_mean:
        raise RuntimeError("DINOv3 processor mean differs from the configured input normalization")
    if processor_config.get("image_std") != expected_std:
        raise RuntimeError("DINOv3 processor std differs from the configured input normalization")

    weights_path = hf_cached_file("model.safetensors")
    sha256 = sha256_file(weights_path)
    if sha256 != MODEL_SHA256:
        raise RuntimeError(f"DINOv3 weight SHA-256 mismatch: {sha256}")
    state_dict = load_file(str(weights_path), device="cpu")
    backbone = DINOv3ConvNeXtBackbone(
        layer_norm_eps=float(hf_config["layer_norm_eps"]),
        layer_scale_init_value=float(hf_config["layer_scale_init_value"]),
        drop_path_rate=float(hf_config["drop_path_rate"]),
    )
    result = backbone.load_hf_state_dict(state_dict)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict DINOv3 checkpoint load was incomplete: {result}")
    backbone.to(device)
    verification = {
        "repository": MODEL_REPO,
        "revision": MODEL_REVISION,
        "weights_file": str(weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256,
        "config_file": str(config_path),
        "processor_file": str(processor_path),
        "model_type": hf_config["model_type"],
        "architecture": hf_config["architectures"][0],
        "depths": hf_config["depths"],
        "dims": hf_config["hidden_sizes"],
        "weight_tensor_count": len(state_dict),
        "missing_tensors": result.missing_keys,
        "unexpected_tensors": result.unexpected_keys,
        "backbone_parameters": parameter_counts(backbone)["total"],
        "feature_dim": FEATURE_DIM,
        "hf_default_image_size": hf_config.get("image_size"),
        "experiment_input_size": 320,
        "input_mean": processor_config["image_mean"],
        "input_std": processor_config["image_std"],
        "preprocessor_resize_disabled_in_experiment": True,
    }
    del state_dict
    return backbone, verification


def environment_report():
    import safetensors
    import timm

    cuda = torch.cuda.is_available()
    if not cuda:
        raise RuntimeError("DINOv3 training requires the configured CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured experiment requires CUDA BF16 support")
    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": sys.platform,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "torch_cpu_threads": torch.get_num_threads(),
        "cpu_logical_count": os.cpu_count(),
    }


def write_verification_report(output_root_path: Path, verification: dict,
                              environment: dict, experiment_config: dict,
                              data_config: dict,
                              train_data, valid_data):
    path = output_root_path / "model_and_weight_verification.md"
    class_counts = {name: int(train_data.label_tensors()[0][:, i].sum())
                    for i, name in enumerate(LABEL_NAMES)}
    train_images = int(sum(train_data.image_counts))
    valid_images = int(sum(valid_data.image_counts))
    text = [
        "# DINOv3 ConvNeXt-Tiny 模型与权重核验", "",
        "## 初始化来源", "",
        f"- Hugging Face：[{verification['repository']}](https://huggingface.co/{verification['repository']})",
        f"- 官方 Meta 实现：[{MODEL_REPO.split('/')[0]} DINOv3](https://github.com/facebookresearch/dinov3)",
        f"- Revision: `{verification['revision']}`",
        f"- Meta ConvNeXt source blob: `{experiment_config['source']['meta_convnext_file_blob_sha']}`",
        f"- 文件大小：{verification['weights_size_bytes']:,} bytes",
        f"- SHA-256：`{verification['weights_sha256']}`",
        f"- 权重张量：{verification['weight_tensor_count']}；missing={len(verification['missing_tensors'])}；unexpected={len(verification['unexpected_tensors'])}",
        f"- 参数量：{verification['backbone_parameters']:,} backbone + {NUM_LABELS * (FEATURE_DIM + 1):,} classifier = {verification['backbone_parameters'] + NUM_LABELS * (FEATURE_DIM + 1):,}",
        f"- 官方特征：{verification['feature_dim']} 维，GAP 后使用官方 LayerNorm；分类头前未额外重复池化或归一化。",
        f"- HF processor 默认尺寸为 {verification['hf_default_image_size']}；本实验显式保留 320×320，并禁用默认 Resize/Crop。",
        f"- 预处理均值/标准差：`{verification['input_mean']}` / `{verification['input_std']}`",
        "- 权重来源为 Meta 通用网络图像自监督/蒸馏模型；未用医学预训练权重。",
        "- 官方许可：[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)；模型使用者已在 Hugging Face 获得 gated 仓库访问权。",
        "", "## 运行环境", "",
    ]
    text.extend(f"- {key}: `{value}`" for key, value in environment.items())
    text += [
        "", "## 数据核验", "",
        f"- Train: {len(train_data):,} Studies / {train_images:,} images / {len({s for s, _ in train_data.study_keys}):,} Subjects",
        f"- Valid: {len(valid_data):,} Studies / {valid_images:,} images / {len({s for s, _ in valid_data.study_keys}):,} Subjects",
        f"- Train 多图 Study: {sum(count > 1 for count in train_data.image_counts):,}；单 Study 最大图像数：{max(train_data.image_counts)}",
        f"- Train 阳性 Study 数（1..10）: `{class_counts}`",
        f"- 标签语义：`{data_config['label_semantics']}`；Test 数据未读取。",
        "- Study sampler 保留完整 Study、每 epoch 每 Study 恰好一次；D0 与 D0-C1 共用相同采样器和随机种子。",
        "", "## 权重加载结果", "",
        "- 采用 Meta 官方 DINOv3 ConvNeXt-Tiny 运算结构，将 HF state tensors 严格加载；未安装或引入新运行时依赖。",
        "- 模型实际张量形状、十分类 logits、320×320 输入、Study Logit Mean 和真实 Train BCE/对比梯度检查由后续 `smoke` 阶段记录。",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def test_contrastive_semantics():
    targets = torch.zeros((7, 10), dtype=torch.float32)
    targets[0, 3] = 1.0
    targets[1, 9] = 1.0
    targets[2, 3] = 1.0
    targets[3, 4] = 1.0
    targets[4, 9] = 1.0
    targets[5, 9] = 1.0
    targets[6, 5] = 1.0
    batch = {
        "study_index": torch.tensor([0, 0, 1, 2, 3, 4, 5, 6]),
        "labels": targets,
        "known": torch.ones_like(targets, dtype=torch.bool),
        "subject_id": ["S1", "S2", "S3", "S1", "S4", "S5", "S6"],
    }
    relations = contrastive_relation_masks(batch)
    checks = {
        "same_study_is_positive": bool(relations["positive"][0, 1]),
        "disjoint_disease_different_subject_is_negative": bool(relations["negative"][0, 2]),
        "overlapping_disease_not_negative": not bool(relations["negative"][0, 3]),
        "same_subject_not_negative": not bool(relations["negative"][0, 4]),
        "two_no_finding_studies_not_negative": not bool(relations["negative"][5, 6]),
        "no_finding_vs_disease_kept_distinct": bool(relations["negative"][5, 7]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Synthetic Study contrastive semantics check failed: {checks}")
    projected = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1).requires_grad_()
    loss_sum, count = multi_positive_info_nce(projected, relations, 0.1)
    if count <= 0 or not torch.isfinite(loss_sum):
        raise RuntimeError("Synthetic Study contrastive loss produced no finite anchor")
    loss_sum.backward()
    if projected.grad is None or not torch.isfinite(projected.grad).all():
        raise RuntimeError("Synthetic Study contrastive loss failed to backpropagate")
    no_negative_batch = {
        "study_index": torch.tensor([0, 0]),
        "labels": targets[:1],
        "known": torch.ones((1, 10), dtype=torch.bool),
        "subject_id": ["S1"],
    }
    no_negative = contrastive_relation_masks(no_negative_batch)
    no_negative_loss, no_negative_count = multi_positive_info_nce(
        torch.nn.functional.normalize(torch.randn(2, 128), dim=-1),
        no_negative, 0.1)
    checks["anchor_without_negative_skipped"] = (
        no_negative_count == 0 and bool(torch.isfinite(no_negative_loss).item()))
    if not checks["anchor_without_negative_skipped"]:
        raise RuntimeError("An anchor without a reliable negative was not safely skipped")
    return {"passed": True, "checks": checks,
            "synthetic_valid_anchors": count,
            "synthetic_positive_pairs": relations["positive_pairs"],
            "synthetic_negative_pairs": relations["negative_pairs"]}


def run_verification(config):
    root = output_root(config)
    logger = setup_logging(root, "verification")
    environment = environment_report()
    logger.info("GPU=%s memory=%d CUDA=%s BF16=%s",
                environment["gpu_name"], environment["gpu_total_memory_bytes"],
                environment["cuda_runtime"], environment["bf16_supported"])
    data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
    train_data, valid_data, data_config = build_datasets(data_path, config["seed"])
    backbone, verification = load_pretrained_backbone("cpu")
    verification["classifier_parameter_count"] = NUM_LABELS * (FEATURE_DIM + 1)
    verification["total_inference_parameter_count"] = (
        verification["backbone_parameters"] + verification["classifier_parameter_count"])
    write_verification_report(root, verification, environment, config, data_config,
                              train_data, valid_data)
    save_json_atomic(root / "environment_and_weight_verification.json", {
        "environment": environment,
        "weights": verification,
        "train_studies": len(train_data),
        "train_images": int(sum(train_data.image_counts)),
        "valid_studies": len(valid_data),
        "valid_images": int(sum(valid_data.image_counts)),
    })
    logger.info("Strict weight load: tensors=%d missing=%d unexpected=%d SHA256=%s",
                verification["weight_tensor_count"],
                len(verification["missing_tensors"]),
                len(verification["unexpected_tensors"]),
                verification["weights_sha256"])
    del backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_data, valid_data, data_config, verification, environment


def _find_contrastive_smoke_batch(train_data, seed: int):
    loader, sampler, _ = make_loader(
        train_data, image_budget=8, shuffle=True, seed=seed,
        num_workers=0, pin_memory=True)
    sampler.set_epoch(0)
    try:
        for batch in loader:
            relations = contrastive_relation_masks(batch)
            if relations["valid_anchor_count"] > 0:
                return batch, relations
    finally:
        shutdown_loader(loader)
    raise RuntimeError("No real Train batch has a valid Study positive and reliable negative")


def _gradient_norm(gradients):
    values = [grad.detach().float().square().sum() for grad in gradients
              if grad is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu())


def run_smoke(config, train_data=None, valid_data=None):
    root = output_root(config)
    logger = setup_logging(root, "smoke")
    if train_data is None or valid_data is None:
        data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
        train_data, valid_data, _ = build_datasets(data_path, config["seed"])
    semantic_test = test_contrastive_semantics()
    device = torch.device("cuda")
    batch, relations = _find_contrastive_smoke_batch(train_data, config["seed"])
    images = batch["images"].to(device, non_blocking=True,
                                 memory_format=torch.channels_last)
    study_index = batch["study_index"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    known = batch["known"].to(device, non_blocking=True)
    image_counts = batch["image_counts"].to(device, non_blocking=True)

    seed_everything(config["seed"])
    backbone, verification = load_pretrained_backbone("cpu")
    model = DINOv3StudyClassifier(backbone).to(device)
    projection = StudyProjectionHead().to(device)
    model.train()
    projection.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_logits, pooled, stage4 = model(images, return_features=True)
        study_logits = study_logit_mean(image_logits, study_index,
                                        batch["num_studies"], image_counts)
        expected_study_logits = torch.stack([
            image_logits[study_index == index].float().mean(dim=0)
            for index in range(batch["num_studies"])
        ])
        official_features = model.backbone.model.forward_features(images)["x_norm_clstoken"]
        per_study, eligible = study_bce_per_study(study_logits, labels, known)
        bce = per_study[eligible].mean()
        projected = projection(pooled)
        contrast_sum, anchor_count = multi_positive_info_nce(
            projected, relations, config["contrastive"]["temperature"])
        contrast = contrast_sum / max(anchor_count, 1)
    if tuple(images.shape[1:]) != (3, 320, 320):
        raise RuntimeError(f"Train transform shape mismatch: {tuple(images.shape)}")
    if tuple(stage4.shape[1:]) != (768, 10, 10):
        raise RuntimeError(f"DINOv3 Stage4 feature mismatch: {tuple(stage4.shape)}")
    if tuple(pooled.shape) != (images.shape[0], 768):
        raise RuntimeError(f"GAP pooled feature mismatch: {tuple(pooled.shape)}")
    if tuple(image_logits.shape) != (images.shape[0], 10):
        raise RuntimeError(f"Ten-label image logits mismatch: {tuple(image_logits.shape)}")
    if tuple(study_logits.shape) != (batch["num_studies"], 10):
        raise RuntimeError("Study Logit Mean returned an invalid shape")
    if not torch.allclose(study_logits.float(), expected_study_logits.float(),
                          atol=2e-5, rtol=2e-5):
        raise RuntimeError("Study Logit Mean differs from the mean of raw image logits")
    if not torch.allclose(pooled.float(), official_features.float(),
                          atol=2e-5, rtol=2e-5):
        raise RuntimeError("Pooled features differ from Meta's official ConvNeXt output")
    if not torch.isfinite(bce) or not torch.isfinite(contrast):
        raise RuntimeError("BCE or Contrastive Loss is non-finite")
    if anchor_count != relations["valid_anchor_count"] or anchor_count == 0:
        raise RuntimeError("The real smoke batch did not exercise reliable contrastive anchors")

    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    bce_grads = torch.autograd.grad(bce, backbone_parameters, retain_graph=True,
                                    allow_unused=True)
    contrast_grads = torch.autograd.grad(contrast, backbone_parameters,
                                         retain_graph=True, allow_unused=True)
    bce_norm = _gradient_norm(bce_grads)
    contrast_norm = _gradient_norm(contrast_grads)
    weighted_contrast_to_bce_gradient_ratio = (
        config["contrastive"]["lambda"] * contrast_norm / max(bce_norm, 1e-12))
    total_loss = bce + config["contrastive"]["lambda"] * contrast
    total_loss.backward()
    backbone_grad_norm = _gradient_norm([p.grad for p in model.backbone.parameters()])
    classifier_grad_norm = _gradient_norm([p.grad for p in model.classifier.parameters()])
    projection_grad_norm = _gradient_norm([p.grad for p in projection.parameters()])
    if min(backbone_grad_norm, classifier_grad_norm, projection_grad_norm) <= 0:
        raise RuntimeError("A required backbone/classifier/projection gradient is missing")
    if weighted_contrast_to_bce_gradient_ratio > 1.0:
        raise RuntimeError(
            "The configured contrastive term dominates the BCE backbone gradient; "
            "do not start formal training until the implementation is reviewed")

    validation_sample = valid_data[0]
    if tuple(validation_sample["images"].shape[1:]) != (3, 320, 320):
        raise RuntimeError("Validation transform did not preserve 320x320 input")
    square_sample = Image.new("RGB", (320, 320), (127, 127, 127))
    if tuple(build_transform(320, train=False)(square_sample).shape) != (3, 320, 320):
        raise RuntimeError("320x320 evaluation preprocessing contract failed")

    result = {
        "passed": True,
        "device": torch.cuda.get_device_name(device),
        "real_train_batch_images": int(images.shape[0]),
        "real_train_batch_studies": int(batch["num_studies"]),
        "real_valid_contrastive_anchors": anchor_count,
        "positive_pairs": relations["positive_pairs"],
        "reliable_negative_pairs": relations["negative_pairs"],
        "skipped_anchors_without_negative": relations["skipped_no_negative"],
        "study4_stage_shape": list(stage4.shape),
        "pooled_feature_shape": list(pooled.shape),
        "image_logit_shape": list(image_logits.shape),
        "study_logit_shape": list(study_logits.shape),
        "bce": float(bce.detach()),
        "contrastive_loss": float(contrast.detach()),
        "total_loss": float(total_loss.detach()),
        "bce_backbone_gradient_norm": bce_norm,
        "contrastive_backbone_gradient_norm": contrast_norm,
        "lambda_weighted_contrast_to_bce_gradient_ratio": weighted_contrast_to_bce_gradient_ratio,
        "total_backbone_gradient_norm": backbone_grad_norm,
        "classifier_gradient_norm": classifier_grad_norm,
        "projection_gradient_norm": projection_grad_norm,
        "synthetic_semantics": semantic_test,
        "weight_sha256": verification["weights_sha256"],
        "test_split_read": False,
    }
    save_json_atomic(root / "dino_smoke_test.json", result)
    logger.info("Smoke passed: BCE=%.6f contrast=%.6f anchors=%d gradient ratio=%.4f",
                result["bce"], result["contrastive_loss"], anchor_count,
                weighted_contrast_to_bce_gradient_ratio)
    del model, projection, backbone, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_optimizer(model, projection, training, fused=True):
    groups = [
        {"params": list(model.backbone.parameters()),
         "lr": float(training["peak_lr_backbone"]), "name": "backbone"},
        {"params": list(model.classifier.parameters()),
         "lr": float(training["peak_lr_head"]), "name": "head"},
    ]
    if projection is not None:
        groups.append({"params": list(projection.parameters()),
                       "lr": float(training["peak_lr_head"]), "name": "projection"})
    return torch.optim.AdamW(groups, weight_decay=float(training["weight_decay"]),
                             fused=bool(fused))


def update_learning_rates(optimizer, step: int, total_steps: int, training: dict):
    warmup_steps = max(1, int(training["warmup_updates"]))
    if step < warmup_steps:
        factor = float(step + 1) / warmup_steps
        for group in optimizer.param_groups:
            peak = (float(training["peak_lr_backbone"])
                    if group["name"] == "backbone"
                    else float(training["peak_lr_head"]))
            group["lr"] = peak * factor
    else:
        cosine_steps = max(1, int(total_steps) - warmup_steps - 1)
        progress = min(1.0, max(0.0, (step - warmup_steps) / cosine_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            if group["name"] == "backbone":
                peak, minimum = (float(training["peak_lr_backbone"]),
                                 float(training["minimum_lr_backbone"]))
            else:
                peak, minimum = (float(training["peak_lr_head"]),
                                 float(training["minimum_lr_head"]))
            group["lr"] = minimum + (peak - minimum) * cosine
    return {group["name"]: float(group["lr"]) for group in optimizer.param_groups}


def _sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _classification_metrics(targets, probabilities, known):
    class_rows = []
    aucs, auprcs = [], []
    for index, class_name in enumerate(LABEL_NAMES):
        mask = known[:, index].astype(bool)
        truth = targets[mask, index].astype(np.int64)
        predicted = probabilities[mask, index]
        n_positive = int(truth.sum())
        n_negative = int(len(truth) - n_positive)
        auc = float("nan")
        auprc = float("nan")
        if n_positive > 0 and n_negative > 0:
            auc = float(roc_auc_score(truth, predicted))
            auprc = float(average_precision_score(truth, predicted))
            aucs.append(auc)
            auprcs.append(auprc)
        f1 = (float(f1_score(truth, predicted >= 0.5, zero_division=0))
              if truth.size else float("nan"))
        class_rows.append({
            "class_id": index + 1,
            "class_name": class_name,
            "auroc": auc,
            "auprc": auprc,
            "f1": f1,
            "n_positive": n_positive,
            "n_negative": n_negative,
        })
    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "macro_f1": float(np.nanmean([row["f1"] for row in class_rows])),
        "per_class": class_rows,
    }


@torch.no_grad()
def validate(model, valid_data, device, image_budget, num_workers,
             pin_memory, prefetch_factor, seed, channels_last,
             batch_forward=None, collate_fn=None):
    loader_kwargs = {
        "image_budget": image_budget,
        "shuffle": False,
        "seed": seed,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader, sampler, _ = make_loader(
        valid_data, **loader_kwargs)
    sampler.set_epoch(0)
    logits_rows, target_rows, known_rows = [], [], []
    subject_ids, study_ids = [], []
    loss_sum, eligible_total = 0.0, 0
    model.eval()
    started = time.perf_counter()
    try:
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            study_index = batch["study_index"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            known = batch["known"].to(device, non_blocking=True)
            image_counts = batch["image_counts"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                image_logits = (model(images) if batch_forward is None
                                else batch_forward(model, images, batch))
            study_logits = study_logit_mean(image_logits.float(), study_index,
                                            batch["num_studies"], image_counts)
            per_study, eligible = study_bce_per_study(study_logits, labels, known)
            eligible_count = int(eligible.sum())
            if eligible_count:
                loss_sum += float(per_study[eligible].sum().cpu())
                eligible_total += eligible_count
            logits_rows.append(study_logits.float().cpu().numpy())
            target_rows.append(batch["labels"].numpy())
            known_rows.append(batch["known"].numpy())
            subject_ids.extend(str(value) for value in batch["subject_id"])
            study_ids.extend(str(value) for value in batch["study_id"])
    finally:
        shutdown_loader(loader)
    if not logits_rows:
        raise RuntimeError("Validation produced no Study logits")
    duration = time.perf_counter() - started
    logits = np.concatenate(logits_rows, axis=0)
    targets = np.concatenate(target_rows, axis=0).astype(np.int64)
    known = np.concatenate(known_rows, axis=0).astype(bool)
    probabilities = _sigmoid(logits)
    metrics = _classification_metrics(targets, probabilities, known)
    metrics.update({
        "valid_loss": loss_sum / eligible_total if eligible_total else float("nan"),
        "valid_seconds": duration,
        "valid_studies": int(len(subject_ids)),
        "valid_studies_per_s": len(subject_ids) / duration if duration else 0.0,
        "subject_ids": subject_ids,
        "study_ids": study_ids,
        "targets": targets,
        "known": known,
        "probabilities": probabilities,
        "logits": logits,
    })
    return metrics


def prediction_frame(metrics):
    payload = {
        "Subject_id": metrics["subject_ids"],
        "Study_id": metrics["study_ids"],
    }
    for index, name in enumerate(LABEL_NAMES):
        safe_name = name.replace(" ", "_")
        payload[f"target_{safe_name}"] = metrics["targets"][:, index].astype(np.int64)
        payload[f"prob_{safe_name}"] = metrics["probabilities"][:, index].astype(np.float64)
    return pd.DataFrame(payload)


def _epoch_report(run_dir: Path, history: list[dict], variant: str,
                  config: dict, complete: bool):
    best = max(history, key=lambda row: float(row["macro_auroc"]))
    status = "completed 20 epochs" if complete else f"in progress through epoch {history[-1]['epoch']}"
    lines = [
        f"# DINOv3 ConvNeXt-Tiny {variant.upper()} Training Report", "",
        f"- Status: {status}",
        f"- Best Epoch: {int(best['epoch'])}",
        f"- Best Validation Macro-AUROC: {float(best['macro_auroc']):.6f}",
        f"- Best Train BCE / Valid BCE: {float(best['train_bce']):.6f} / {float(best['valid_loss']):.6f}",
        f"- Best Macro-AUPRC / Macro-F1: {float(best['macro_auprc']):.6f} / {float(best['macro_f1']):.6f}",
        f"- Initialization SHA-256: `{config['source']['weights_sha256']}`",
        f"- Study sampling: complete Studies, shared sampler; image budget {best['image_budget']}, accumulation {best['grad_accumulation']}.",
        "- Test split was not read.",
        "", "## Epoch metrics", "",
        "| Epoch | Train BCE | Contrastive | Total Loss | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | Images/s | Studies/s | Pos pairs | Neg pairs | Valid anchors | LR backbone/head | Peak alloc/reserved GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in history:
        lines.append(
            f"| {int(row['epoch'])} | {float(row['train_bce']):.6f} | "
            f"{float(row['train_contrastive']):.6f} | {float(row['train_total']):.6f} | "
            f"{float(row['valid_loss']):.6f} | {float(row['macro_auroc']):.6f} | "
            f"{float(row['macro_auprc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['images_per_s']):.2f} | {float(row['studies_per_s']):.2f} | "
            f"{int(row['positive_pairs'])} | {int(row['negative_pairs'])} | "
            f"{int(row['valid_contrastive_anchors'])} | "
            f"{float(row['lr_backbone']):.3g}/{float(row['lr_head']):.3g} | "
            f"{float(row['peak_gpu_allocated_gib']):.3f}/{float(row['peak_gpu_reserved_gib']):.3f} |"
        )
    best_class_path = run_dir / "best_per_class_metrics.csv"
    if best_class_path.is_file():
        best_classes = pd.read_csv(best_class_path).to_dict("records")
    else:
        best_classes = [row for row in pd.read_csv(
            run_dir / "per_class_metrics.csv").to_dict("records")
            if int(row["epoch"]) == int(best["epoch"])]
    lines.extend(["", "## Best-epoch per-class metrics", "",
                  "| Class | AUROC | AUPRC | F1 | Positive | Negative |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in best_classes:
        lines.append(f"| {row['class_name']} | {float(row['auroc']):.6f} | {float(row['auprc']):.6f} | {float(row['f1']):.6f} | {int(row['n_positive'])} | {int(row['n_negative'])} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_checkpoint(model, projection, optimizer, epoch, best_auc,
                     best_epoch, global_step, history, initial_sha,
                     experiment_config, sampler, loader_generator):
    return {
        "format_version": 1,
        "variant": "d0c1" if projection is not None else "d0",
        "epoch": int(epoch),
        "best_macro_auroc": float(best_auc),
        "best_epoch": int(best_epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "projection_state": projection.state_dict() if projection is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "initial_weights_sha256": initial_sha,
        "experiment_config": experiment_config,
        "sampler_epoch": int(sampler.epoch),
        "loader_generator_state": loader_generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state):
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def train_variant(config, variant: str, train_data, valid_data,
                  verification: dict, runtime: dict, resume: bool = True):
    if variant not in {"d0", "d0c1"}:
        raise ValueError("variant must be d0 or d0c1")
    root = output_root(config)
    run_dir = root / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir, "train")
    save_json_atomic(run_dir / "config.json", {
        "variant": variant,
        "experiment_config": config,
        "selected_runtime": runtime,
        "initialization": verification,
        "train_studies": len(train_data),
        "train_images": int(sum(train_data.image_counts)),
        "valid_studies": len(valid_data),
        "valid_images": int(sum(valid_data.image_counts)),
        "test_split_read": False,
    })
    training = dict(config["training"])
    training["early_stopping"] = False
    device = torch.device("cuda")
    seed_everything(config["seed"])
    backbone, weight_report = load_pretrained_backbone("cpu")
    model = DINOv3StudyClassifier(backbone).to(device)
    projection = StudyProjectionHead().to(device) if variant == "d0c1" else None
    model.backbone.set_gradient_checkpointing(bool(runtime["gradient_checkpointing"]))
    channels_last = bool(runtime["channels_last"])
    if channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, projection, training, fused=True)
    train_loader, sampler, loader_generator = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=int(config["seed"]), num_workers=int(runtime["num_workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        prefetch_factor=int(runtime["prefetch_factor"]))
    valid_loader_config = {
        "image_budget": int(runtime["image_budget"]),
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
        "prefetch_factor": int(runtime["prefetch_factor"]),
        "seed": int(config["seed"]),
        "channels_last": channels_last,
    }
    updates_by_epoch = []
    for epoch_index in range(int(training["epochs"])):
        sampler.set_epoch(epoch_index)
        updates_by_epoch.append(math.ceil(len(sampler) / int(runtime["grad_accumulation"])))
    total_updates = sum(updates_by_epoch)
    if total_updates <= 0:
        raise RuntimeError("Study sampler emitted no optimizer updates")
    warmup_epochs = int(training["warmup_epochs"])
    training["warmup_updates"] = sum(updates_by_epoch[:warmup_epochs])

    last_path, best_path = run_dir / "last.pth", run_dir / "best.pth"
    history = []
    epoch_class_rows = []
    checkpoint = None
    best_auc, best_epoch, global_step, start_epoch = -math.inf, 0, 0, 0
    if resume and last_path.is_file():
        checkpoint_data = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint_data["variant"] != variant:
            raise RuntimeError("Checkpoint variant does not match requested D0/D0-C1 run")
        if checkpoint_data["initial_weights_sha256"] != verification["weights_sha256"]:
            raise RuntimeError("Checkpoint was initialized from a different DINOv3 weight file")
        if checkpoint_data["experiment_config"]["runtime"] != runtime:
            raise RuntimeError("Checkpoint runtime differs from selected shared benchmark configuration")
        model.load_state_dict(checkpoint_data["model_state"], strict=True)
        if projection is not None:
            if checkpoint_data["projection_state"] is None:
                raise RuntimeError("D0-C1 checkpoint is missing its training-only projection head")
            projection.load_state_dict(checkpoint_data["projection_state"], strict=True)
        optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        history = checkpoint_data["history"]
        best_auc = float(checkpoint_data["best_macro_auroc"])
        best_epoch = int(checkpoint_data["best_epoch"])
        global_step = int(checkpoint_data["global_step"])
        start_epoch = int(checkpoint_data["epoch"])
        sampler.set_epoch(start_epoch)
        loader_generator.set_state(checkpoint_data["loader_generator_state"])
        _restore_rng(checkpoint_data)
        per_class_path = run_dir / "per_class_metrics.csv"
        if per_class_path.is_file():
            epoch_class_rows = pd.read_csv(per_class_path).to_dict("records")
        logger.info("Resuming %s after epoch %d", variant.upper(), start_epoch)

    try:
        for epoch_index in range(start_epoch, int(training["epochs"])):
            epoch_number = epoch_index + 1
            sampler.set_epoch(epoch_index)
            sampler_iter = iter(train_loader)
            model.train()
            if projection is not None:
                projection.train()
            train_start = time.perf_counter()
            data_wait_seconds = 0.0
            train_image_count = train_study_count = 0
            bce_sum_epoch = contrast_sum_epoch = 0.0
            eligible_studies_epoch = contrast_anchors_epoch = 0
            positive_pairs_epoch = negative_pairs_epoch = 0
            skipped_no_negative_epoch = no_positive_anchor_epoch = 0
            accumulation = int(runtime["grad_accumulation"])
            torch.cuda.reset_peak_memory_stats(device)
            while True:
                window = []
                for _ in range(accumulation):
                    fetch_started = time.perf_counter()
                    try:
                        batch = next(sampler_iter)
                    except StopIteration:
                        break
                    data_wait_seconds += time.perf_counter() - fetch_started
                    window.append(batch)
                if not window:
                    break
                eligible_window = sum(int(batch["known"].any(dim=1).sum()) for batch in window)
                relations_window = [contrastive_relation_masks(batch) for batch in window]
                anchors_window = sum(item["valid_anchor_count"] for item in relations_window)
                optimizer.zero_grad(set_to_none=True)
                update_learning_rates(optimizer, global_step, total_updates, training)
                update_started = time.perf_counter()
                cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                cuda_start.record()
                for batch, relations in zip(window, relations_window):
                    images = batch["images"].to(device, non_blocking=True)
                    if channels_last:
                        images = images.contiguous(memory_format=torch.channels_last)
                    study_index = batch["study_index"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    known = batch["known"].to(device, non_blocking=True)
                    image_counts = batch["image_counts"].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        image_logits, pooled, _ = model(images, return_features=True)
                        study_logits = study_logit_mean(image_logits, study_index,
                                                        batch["num_studies"], image_counts)
                        per_study, eligible = study_bce_per_study(study_logits, labels, known)
                        this_bce_sum = per_study[eligible].sum()
                        denominator = max(eligible_window, 1)
                        total_loss = this_bce_sum / denominator
                        if projection is not None and anchors_window:
                            projected = projection(pooled)
                            this_contrast_sum, _ = multi_positive_info_nce(
                                projected, relations,
                                config["contrastive"]["temperature"])
                            total_loss = total_loss + (
                                float(config["contrastive"]["lambda"])
                                * this_contrast_sum / anchors_window)
                        elif projection is not None:
                            total_loss = total_loss + pooled.sum() * 0.0
                    if not torch.isfinite(total_loss):
                        raise FloatingPointError(
                            f"Non-finite DINOv3 training loss at epoch {epoch_number}")
                    total_loss.backward()
                    bce_sum_epoch += float(this_bce_sum.detach().float().cpu())
                    eligible_studies_epoch += int(eligible.sum())
                    train_image_count += int(batch["images"].shape[0])
                    train_study_count += int(batch["num_studies"])
                    positive_pairs_epoch += relations["positive_pairs"]
                    negative_pairs_epoch += relations["negative_pairs"]
                    skipped_no_negative_epoch += relations["skipped_no_negative"]
                    no_positive_anchor_epoch += relations["anchors_without_positive"]
                    if projection is not None and anchors_window:
                        contrast_sum_epoch += float(this_contrast_sum.detach().float().cpu())
                        contrast_anchors_epoch += int(relations["valid_anchor_count"])
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + (list(projection.parameters()) if projection else []),
                    float(training["gradient_clip_norm"]), error_if_nonfinite=True)
                optimizer.step()
                cuda_end.record()
                torch.cuda.synchronize(device)
                del window, relations_window
                global_step += 1
                _ = time.perf_counter() - update_started
            train_seconds = time.perf_counter() - train_start
            validation = validate(model, valid_data, device, **valid_loader_config)
            score = float(validation["macro_auroc"])
            improved = math.isfinite(score) and score > best_auc
            if improved:
                best_auc, best_epoch = score, epoch_number
            train_bce = bce_sum_epoch / max(eligible_studies_epoch, 1)
            train_contrast = (contrast_sum_epoch / max(contrast_anchors_epoch, 1)
                              if projection is not None else 0.0)
            train_total = train_bce + float(config["contrastive"]["lambda"]) * train_contrast
            lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
            row = {
                "epoch": epoch_number,
                "train_bce": train_bce,
                "train_contrastive": train_contrast,
                "train_total": train_total,
                "valid_loss": validation["valid_loss"],
                "macro_auroc": score,
                "macro_auprc": validation["macro_auprc"],
                "macro_f1": validation["macro_f1"],
                "optimizer_updates": updates_by_epoch[epoch_index],
                "train_images": train_image_count,
                "train_studies": train_study_count,
                "train_seconds": train_seconds,
                "validation_seconds": validation["valid_seconds"],
                "validation_studies_per_s": validation["valid_studies_per_s"],
                "images_per_s": train_image_count / train_seconds if train_seconds else 0.0,
                "studies_per_s": train_study_count / train_seconds if train_seconds else 0.0,
                "data_loader_wait_seconds": data_wait_seconds,
                "positive_pairs": positive_pairs_epoch,
                "negative_pairs": negative_pairs_epoch,
                "valid_contrastive_anchors": contrast_anchors_epoch,
                "skipped_anchors_without_negative": skipped_no_negative_epoch,
                "anchors_without_positive": no_positive_anchor_epoch,
                "participating_image_fraction": (
                    contrast_anchors_epoch / max(train_image_count, 1)
                    if projection is not None else 0.0),
                "lr_backbone": lrs["backbone"],
                "lr_head": lrs["head"],
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
                "parameters": parameter_counts(model)["total"]
                              + (parameter_counts(projection)["total"] if projection else 0),
                "best_epoch": best_epoch,
                "best_macro_auroc": best_auc,
                "image_budget": int(runtime["image_budget"]),
                "grad_accumulation": accumulation,
            }
            history.append(row)
            for class_row in validation["per_class"]:
                epoch_class_rows.append({"epoch": epoch_number, **class_row})
            write_csv(run_dir / "train_log.csv", history)
            write_csv(run_dir / "per_class_metrics.csv", epoch_class_rows)
            val_dir = run_dir / "predictions"
            val_dir.mkdir(parents=True, exist_ok=True)
            prediction_frame(validation).to_csv(
                val_dir / f"val_epoch_{epoch_number:02d}.csv",
                index=False, encoding="utf-8")
            checkpoint = _make_checkpoint(
                model, projection, optimizer, epoch_number, best_auc, best_epoch,
                global_step, history, verification["weights_sha256"],
                {"runtime": runtime, "config": config}, sampler, loader_generator)
            save_torch_atomic(last_path, checkpoint)
            if improved:
                save_torch_atomic(best_path, checkpoint)
                prediction_frame(validation).to_csv(
                    val_dir / "val_best.csv", index=False, encoding="utf-8")
                per_class_best = validation["per_class"]
                write_csv(run_dir / "best_per_class_metrics.csv", per_class_best)
            logger.info(
                "epoch=%02d/20 train_bce=%.6f contrast=%.6f valid_bce=%.6f "
                "macro_auroc=%.6f macro_auprc=%.6f macro_f1=%.6f "
                "images_s=%.2f studies_s=%.2f valid_anchors=%d best_epoch=%d best_auc=%.6f",
                epoch_number, train_bce, train_contrast, validation["valid_loss"],
                score, validation["macro_auprc"], validation["macro_f1"],
                row["images_per_s"], row["studies_per_s"],
                contrast_anchors_epoch, best_epoch, best_auc)
            _epoch_report(run_dir, history, variant, config,
                          complete=(epoch_number == int(training["epochs"])))
            save_json_atomic(run_dir / "training_status.json", {
                "variant": variant, "completed_epochs": epoch_number,
                "best_epoch": best_epoch, "best_macro_auroc": best_auc,
                "last_checkpoint": str(last_path),
                "complete": epoch_number == int(training["epochs"]),
            })
            if epoch_number == int(training["epochs"]):
                save_torch_atomic(run_dir / "epoch20.pth", checkpoint)
    finally:
        shutdown_loader(train_loader)
        del model, projection, optimizer, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    final = {
        "variant": variant,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_macro_auroc": best_auc,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "output_dir": str(run_dir),
    }
    save_json_atomic(run_dir / "training_result.json", final)
    return final


def _benchmark_candidate(config, variant: str, train_data, candidate: dict,
                         warmup_updates: int, measured_updates: int,
                         repeat_index: int, logger):
    device = torch.device("cuda")
    seed_everything(config["seed"])
    model = None
    projection = None
    backbone = None
    optimizer = None
    loader = None
    row = {
        "phase": "long" if warmup_updates > config["benchmark"]["short_warmup_updates"] else "short",
        "variant": variant,
        "repeat": int(repeat_index),
        **candidate,
        "warmup_updates": int(warmup_updates),
        "measured_updates": int(measured_updates),
    }
    try:
        backbone, verification = load_pretrained_backbone("cpu")
        model = DINOv3StudyClassifier(backbone).to(device)
        model.backbone.set_gradient_checkpointing(candidate["gradient_checkpointing"])
        if candidate["channels_last"]:
            model.to(memory_format=torch.channels_last)
        projection = (StudyProjectionHead().to(device) if variant == "d0c1" else None)
        optimizer = build_optimizer(model, projection, config["training"], fused=True)
        loader, sampler, _ = make_loader(
            train_data, image_budget=candidate["image_budget"], shuffle=True,
            seed=config["seed"], num_workers=candidate["num_workers"],
            pin_memory=candidate["pin_memory"],
            prefetch_factor=candidate["prefetch_factor"])
        sampler.set_epoch(0)
        iterator = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        wall_steps, compute_steps, images_steps, studies_steps = [], [], [], []
        anchor_counts, anchor_denominators = [], []
        positive_pair_counts, negative_pair_counts = [], []
        data_wait_total = 0.0
        for update_index in range(int(warmup_updates) + int(measured_updates)):
            wall_started = time.perf_counter()
            window = []
            update_data_wait = 0.0
            for _ in range(candidate["grad_accumulation"]):
                fetch_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    raise RuntimeError("Short benchmark exhausted the train sampler unexpectedly")
                update_data_wait += time.perf_counter() - fetch_start
                window.append(batch)
            eligible_window = sum(int(batch["known"].any(dim=1).sum()) for batch in window)
            relations_window = [contrastive_relation_masks(batch) for batch in window]
            anchors_window = sum(item["valid_anchor_count"] for item in relations_window)
            update_images = sum(int(batch["images"].shape[0]) for batch in window)
            update_studies = sum(int(batch["num_studies"]) for batch in window)
            optimizer.zero_grad(set_to_none=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record()
            for batch, relations in zip(window, relations_window):
                images = batch["images"].to(device, non_blocking=True)
                if candidate["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                study_index = batch["study_index"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                known = batch["known"].to(device, non_blocking=True)
                image_counts = batch["image_counts"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_logits, pooled, _ = model(images, return_features=True)
                    study_logits = study_logit_mean(image_logits, study_index,
                                                    batch["num_studies"], image_counts)
                    per_study, eligible = study_bce_per_study(study_logits, labels, known)
                    loss = per_study[eligible].sum() / max(eligible_window, 1)
                    if projection is not None and anchors_window:
                        projected = projection(pooled)
                        contrast_sum, _ = multi_positive_info_nce(
                            projected, relations, config["contrastive"]["temperature"])
                        loss = loss + float(config["contrastive"]["lambda"]) * contrast_sum / anchors_window
                    elif projection is not None:
                        loss = loss + pooled.sum() * 0.0
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss during GPU benchmark")
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + (list(projection.parameters()) if projection else []),
                float(config["training"]["gradient_clip_norm"]), error_if_nonfinite=True)
            optimizer.step()
            compute_end.record()
            torch.cuda.synchronize(device)
            wall_elapsed = time.perf_counter() - wall_started
            compute_elapsed = compute_start.elapsed_time(compute_end) / 1000.0
            if update_index >= int(warmup_updates):
                wall_steps.append(wall_elapsed)
                compute_steps.append(compute_elapsed)
                data_wait_total += update_data_wait
                images_steps.append(update_images)
                studies_steps.append(update_studies)
                anchor_counts.append(anchors_window)
                anchor_denominators.append(update_images)
                positive_pair_counts.append(sum(r["positive_pairs"] for r in relations_window))
                negative_pair_counts.append(sum(r["negative_pairs"] for r in relations_window))
        measured_wall = float(sum(wall_steps))
        measured_images = int(sum(images_steps))
        measured_studies = int(sum(studies_steps))
        total_anchors = int(sum(anchor_counts))
        total_anchor_images = int(sum(anchor_denominators))
        total_seconds = max(measured_wall, 1e-9)
        row.update({
            "status": "ok",
            "end_to_end_images_per_s": measured_images / total_seconds,
            "compute_images_per_s": measured_images / max(sum(compute_steps), 1e-9),
            "studies_per_s": measured_studies / total_seconds,
            "median_step_seconds": float(np.median(wall_steps)),
            "p95_step_seconds": float(np.percentile(wall_steps, 95)),
            "mean_compute_step_seconds": float(np.mean(compute_steps)),
            "data_loader_wait_seconds": data_wait_total,
            "measured_images": measured_images,
            "measured_studies": measured_studies,
            "effective_images_per_update": float(np.median(images_steps)),
            "valid_contrastive_anchors": total_anchors,
            "valid_anchor_fraction": (total_anchors / max(total_anchor_images, 1)
                                      if variant == "d0c1" else 0.0),
            "positive_pairs": int(sum(positive_pair_counts)),
            "negative_pairs": int(sum(negative_pair_counts)),
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_reserved_memory_fraction": (
                torch.cuda.max_memory_reserved(device)
                / torch.cuda.get_device_properties(device).total_memory),
            "weight_sha256": verification["weights_sha256"],
        })
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("OOM variant=%s candidate=%s repeat=%d: %s",
                       variant, candidate, repeat_index, exc)
        row.update({"status": "oom", "error": str(exc)[:500]})
    finally:
        shutdown_loader(loader)
        del model, projection, backbone, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return row


def _benchmark_candidates(config):
    candidates = []
    target = int(config["benchmark"]["target_effective_images_per_update"])
    for budget in config["benchmark"]["image_budgets"]:
        accumulation = max(1, math.ceil(target / int(budget)))
        for workers in config["benchmark"]["num_workers"]:
            for checkpointing in config["training"]["gradient_checkpointing_candidates"]:
                candidates.append({
                    "image_budget": int(budget),
                    "grad_accumulation": int(accumulation),
                    "effective_batch_target": target,
                    "num_workers": int(workers),
                    "pin_memory": bool(config["benchmark"]["pin_memory"]),
                    "persistent_workers": bool(workers > 0),
                    "prefetch_factor": int(config["benchmark"]["prefetch_factor"]),
                    "gradient_checkpointing": bool(checkpointing),
                    "channels_last": False,
                })
    return candidates


def _candidate_key(row):
    return tuple(row[name] for name in (
        "image_budget", "grad_accumulation", "num_workers",
        "gradient_checkpointing", "channels_last"))


def run_benchmark(config, train_data=None, force: bool = False):
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root, "benchmark")
    progress_path = root / "dino_gpu_benchmark_progress.json"
    selected_path = root / "dino_selected_runtime_config.json"
    short_path = root / "dino_gpu_benchmark_short.csv"
    long_path = root / "dino_gpu_benchmark_long.csv"
    if selected_path.is_file() and not force:
        with selected_path.open("r", encoding="utf-8") as stream:
            selected = json.load(stream)
        logger.info("Using existing selected runtime config: %s", selected)
        return selected
    if train_data is None:
        data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
        train_data, _, _ = build_datasets(data_path, config["seed"])
    short_rows, long_rows = [], []
    candidates = _benchmark_candidates(config)
    total = len(candidates) * 2
    completed = 0
    for candidate_index, candidate in enumerate(candidates):
        for variant in ("d0", "d0c1"):
            row = _benchmark_candidate(
                config, variant, train_data, candidate,
                config["benchmark"]["short_warmup_updates"],
                config["benchmark"]["short_measured_updates"],
                repeat_index=0, logger=logger)
            short_rows.append(row)
            write_csv_union(short_path, short_rows)
            completed += 1
            save_json_atomic(progress_path, {
                "phase": "short", "completed_candidates": completed,
                "total_candidates": total, "last_variant": variant,
                "last_candidate": candidate, "last_status": row["status"],
            })
            logger.info("short %d/%d variant=%s budget=%d accum=%d workers=%d checkpoint=%s status=%s images_s=%s anchors=%s",
                        completed, total, variant, candidate["image_budget"],
                        candidate["grad_accumulation"], candidate["num_workers"],
                        candidate["gradient_checkpointing"], row["status"],
                        row.get("end_to_end_images_per_s"), row.get("valid_anchor_fraction"))

    valid_short = [row for row in short_rows if row.get("status") == "ok"]
    if not valid_short:
        raise RuntimeError("Every short DINOv3 GPU benchmark candidate failed")
    finalists = set()
    count = int(config["benchmark"]["finalists"])
    for variant in ("d0", "d0c1"):
        rows = [row for row in valid_short if row["variant"] == variant]
        if variant == "d0c1":
            rows = [row for row in rows if row.get("valid_contrastive_anchors", 0) > 0]
        rows.sort(key=lambda row: float(row["end_to_end_images_per_s"]), reverse=True)
        finalists.update(_candidate_key(row) for row in rows[:count])
    if not finalists:
        raise RuntimeError("No short benchmark configuration yielded a valid contrastive anchor")

    finalist_rows = []
    for key in sorted(finalists):
        candidate = next(row for row in valid_short if _candidate_key(row) == key)
        finalist_rows.append({
            key_name: candidate[key_name]
            for key_name in ("image_budget", "grad_accumulation", "effective_batch_target",
                             "num_workers", "pin_memory", "persistent_workers",
                             "prefetch_factor", "gradient_checkpointing", "channels_last")
        })
    long_repeats = int(config["benchmark"]["long_repeats_per_finalist"])
    long_total = len(finalist_rows) * long_repeats * 2
    long_completed = 0
    for candidate in finalist_rows:
        for variant in ("d0", "d0c1"):
            for repeat in range(long_repeats):
                row = _benchmark_candidate(
                    config, variant, train_data, candidate,
                    config["benchmark"]["long_warmup_updates"],
                    config["benchmark"]["long_measured_updates"],
                    repeat_index=repeat + 1, logger=logger)
                long_rows.append(row)
                write_csv_union(long_path, long_rows)
                long_completed += 1
                save_json_atomic(progress_path, {
                    "phase": "long", "completed_candidates": long_completed,
                    "total_candidates": long_total, "last_variant": variant,
                    "last_candidate": candidate, "last_repeat": repeat + 1,
                    "last_status": row["status"],
                })
                logger.info("long %d/%d variant=%s budget=%d accum=%d workers=%d checkpoint=%s repeat=%d status=%s images_s=%s anchors=%s",
                            long_completed, long_total, variant,
                            candidate["image_budget"], candidate["grad_accumulation"],
                            candidate["num_workers"], candidate["gradient_checkpointing"],
                            repeat + 1, row["status"],
                            row.get("end_to_end_images_per_s"), row.get("valid_anchor_fraction"))

    benchmark_rows = short_rows + long_rows
    write_csv_union(root / "dino_gpu_benchmark.csv", benchmark_rows)
    selected = _select_shared_runtime(config, long_rows)
    save_json_atomic(selected_path, selected)
    write_json(root / "dino_gpu_benchmark_summary.json", {
        "short_candidates": short_rows,
        "long_repeats": long_rows,
        "selected_runtime": selected,
    })
    _write_benchmark_report(root, config, short_rows, long_rows, selected)
    save_json_atomic(progress_path, {"phase": "complete", "selected_runtime": selected})
    return selected


def _select_shared_runtime(config, long_rows):
    okay = [row for row in long_rows if row.get("status") == "ok"]
    grouped = defaultdict(list)
    for row in okay:
        grouped[(_candidate_key(row), row["variant"])].append(row)
    candidates = []
    max_reserved = float(config["benchmark"]["maximum_reserved_memory_fraction"])
    for key in sorted({_candidate_key(row) for row in okay}):
        d0 = grouped.get((key, "d0"), [])
        c1 = grouped.get((key, "d0c1"), [])
        expected = int(config["benchmark"]["long_repeats_per_finalist"])
        if len(d0) != expected or len(c1) != expected:
            continue
        d0_speeds = [float(row["end_to_end_images_per_s"]) for row in d0]
        c1_speeds = [float(row["end_to_end_images_per_s"]) for row in c1]
        d0_reserved = max(float(row["peak_reserved_memory_fraction"]) for row in d0)
        c1_reserved = max(float(row["peak_reserved_memory_fraction"]) for row in c1)
        contrast_anchors = sum(int(row["valid_contrastive_anchors"]) for row in c1)
        contrast_fraction = float(np.mean([float(row["valid_anchor_fraction"]) for row in c1]))
        effective = float(np.mean([float(row["effective_images_per_update"])
                                   for row in d0 + c1]))
        target = int(config["benchmark"]["target_effective_images_per_update"])
        if d0_reserved > max_reserved or c1_reserved > max_reserved or contrast_anchors <= 0:
            continue
        if abs(effective - target) / target > 0.15:
            continue
        d0_mean, c1_mean = float(np.mean(d0_speeds)), float(np.mean(c1_speeds))
        d0_cv = float(np.std(d0_speeds, ddof=1) / d0_mean) if len(d0_speeds) > 1 and d0_mean else 0.0
        c1_cv = float(np.std(c1_speeds, ddof=1) / c1_mean) if len(c1_speeds) > 1 and c1_mean else 0.0
        candidates.append({
            "key": key,
            "d0_mean_images_s": d0_mean,
            "d0_std_images_s": float(np.std(d0_speeds, ddof=1)),
            "d0_cv": d0_cv,
            "d0c1_mean_images_s": c1_mean,
            "d0c1_std_images_s": float(np.std(c1_speeds, ddof=1)),
            "d0c1_cv": c1_cv,
            "conservative_shared_images_s": min(d0_mean, c1_mean),
            "contrastive_valid_anchor_fraction": contrast_fraction,
            "valid_contrastive_anchors_across_repeats": contrast_anchors,
            "effective_images_per_update": effective,
            "peak_reserved_memory_fraction": max(d0_reserved, c1_reserved),
        })
    if not candidates:
        raise RuntimeError(
            "No shared configuration met memory margin, effective batch tolerance "
            "and nonzero valid contrastive anchors")
    best_speed = max(row["conservative_shared_images_s"] for row in candidates)
    near_limit = best_speed * (1.0 - float(config["benchmark"]["near_max_throughput_fraction"]))
    near = [row for row in candidates if row["conservative_shared_images_s"] >= near_limit]
    near.sort(key=lambda row: (row["contrastive_valid_anchor_fraction"],
                               row["conservative_shared_images_s"]), reverse=True)
    best = near[0]
    budget, accumulation, workers, checkpointing, channels_last = best["key"]
    return {
        "image_budget": int(budget),
        "grad_accumulation": int(accumulation),
        "effective_batch_target": int(config["benchmark"]["target_effective_images_per_update"]),
        "num_workers": int(workers),
        "pin_memory": bool(config["benchmark"]["pin_memory"]),
        "persistent_workers": bool(workers > 0),
        "prefetch_factor": int(config["benchmark"]["prefetch_factor"]),
        "gradient_checkpointing": bool(checkpointing),
        "channels_last": bool(channels_last),
        "measured_d0_images_per_s": best["d0_mean_images_s"],
        "measured_d0c1_images_per_s": best["d0c1_mean_images_s"],
        "d0_repeat_cv": best["d0_cv"],
        "d0c1_repeat_cv": best["d0c1_cv"],
        "contrastive_valid_anchor_fraction": best["contrastive_valid_anchor_fraction"],
        "effective_images_per_update": best["effective_images_per_update"],
        "peak_reserved_memory_fraction": best["peak_reserved_memory_fraction"],
        "selection_rule": "within 3% of highest conservative shared end-to-end images/s, then maximize valid C1 anchor fraction; require <=90% reserved memory, effective batch within 15%, and nonzero valid anchors",
        "near_max_candidates": near,
    }


def _write_benchmark_report(root, config, short_rows, long_rows, selected):
    lines = [
        "# DINOv3 D0 / D0-C1 GPU Benchmark", "",
        "- End-to-end timing covers Data Loading, DINOv3 Forward, Study Logit Mean, loss, backward, gradient accumulation, clipping and optimizer update.",
        "- Short screening ran both variants for each candidate; top candidates were repeated with longer warmup/measurement.",
        "- All candidates used whole Studies, the same seeded sampler and 320×320 images. D0-C1 contrast was restricted to each microbatch; gradient accumulation did not create cross-microbatch contrastive pairs.",
        "- D0 and D0-C1 are each benchmarked from the same official initial weight SHA.",
        "", "## Short screen", "",
        "| Variant | Budget | Accum | Workers | Checkpoint | Status | E2E images/s | Compute images/s | Studies/s | Anchor fraction | Reserved GiB |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in short_rows:
        lines.append(
            f"| {row['variant']} | {row['image_budget']} | {row['grad_accumulation']} | "
            f"{row['num_workers']} | {row['gradient_checkpointing']} | {row['status']} | "
            f"{float(row.get('end_to_end_images_per_s', 0.0)):.2f} | "
            f"{float(row.get('compute_images_per_s', 0.0)):.2f} | "
            f"{float(row.get('studies_per_s', 0.0)):.2f} | "
            f"{float(row.get('valid_anchor_fraction', 0.0)):.3f} | "
            f"{float(row.get('peak_gpu_reserved_bytes', 0.0)) / (1024 ** 3):.3f} |"
        )
    lines.extend(["", "## Long repeated finalists", "",
                  "| Variant | Budget | Accum | Workers | Checkpoint | Repeat | E2E images/s | Median step s | P95 step s | Data wait s | Anchors | Reserved GiB |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in long_rows:
        lines.append(
            f"| {row['variant']} | {row['image_budget']} | {row['grad_accumulation']} | "
            f"{row['num_workers']} | {row['gradient_checkpointing']} | {row['repeat']} | "
            f"{float(row.get('end_to_end_images_per_s', 0.0)):.2f} | "
            f"{float(row.get('median_step_seconds', 0.0)):.3f} | "
            f"{float(row.get('p95_step_seconds', 0.0)):.3f} | "
            f"{float(row.get('data_loader_wait_seconds', 0.0)):.2f} | "
            f"{int(row.get('valid_contrastive_anchors', 0))} | "
            f"{float(row.get('peak_gpu_reserved_bytes', 0.0)) / (1024 ** 3):.3f} |"
        )
    lines.extend(["", "## Shared formal training configuration", "",
                  f"- Image budget / accumulation: {selected['image_budget']} / {selected['grad_accumulation']}",
                  f"- DataLoader workers / persistent / prefetch: {selected['num_workers']} / {selected['persistent_workers']} / {selected['prefetch_factor']}",
                  f"- Gradient checkpointing / channels-last: {selected['gradient_checkpointing']} / {selected['channels_last']}",
                  f"- D0 mean end-to-end throughput: {selected['measured_d0_images_per_s']:.2f} images/s (CV {selected['d0_repeat_cv']:.3f})",
                  f"- D0-C1 mean end-to-end throughput: {selected['measured_d0c1_images_per_s']:.2f} images/s (CV {selected['d0c1_repeat_cv']:.3f})",
                  f"- Valid-anchor fraction: {selected['contrastive_valid_anchor_fraction']:.3f}",
                  f"- Effective images/update: {selected['effective_images_per_update']:.2f}",
                  f"- Peak reserved memory fraction: {selected['peak_reserved_memory_fraction']:.3f}",
                  f"- Selection rule: {selected['selection_rule']}", "",
                  "Selection balances the conservative shared end-to-end rate against valid D0-C1 anchors, memory margin and effective-batch tolerance.", ""])
    (root / "dino_gpu_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def export_submission_package(config, summary: dict | None = None,
                              valid_data=None):
    root = output_root(config)
    if summary is None:
        with (root / "d0_d0c1_comparison.json").open("r", encoding="utf-8") as stream:
            comparison_payload = json.load(stream)
        summary = comparison_payload["summary"]
    variants = {
        "d0": (summary["d0_best_macro_auroc"], summary["d0_best_epoch"]),
        "d0c1": (summary["d0c1_best_macro_auroc"], summary["d0c1_best_epoch"]),
    }
    selected_variant = max(variants, key=lambda name: variants[name][0])
    selected_epoch = int(variants[selected_variant][1])
    checkpoint_path = root / selected_variant / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != selected_variant:
        raise RuntimeError("Selected best checkpoint variant metadata is inconsistent")
    if int(checkpoint["best_epoch"]) != selected_epoch:
        raise RuntimeError("Selected best checkpoint epoch differs from comparison analysis")

    export_root = root / "submission_export"
    package_dir = export_root / "DINOv3_ConvNeXtTiny_Inference"
    package_dir.mkdir(parents=True, exist_ok=True)
    inference_model = DINOv3StudyClassifier(num_labels=NUM_LABELS)
    inference_model.load_state_dict(checkpoint["model_state"], strict=True)
    state_dict = {key: tensor.detach().cpu().contiguous().clone()
                  for key, tensor in inference_model.state_dict().items()}
    model_path = package_dir / "model.safetensors"
    save_file(state_dict, str(model_path))
    shutil.copyfile(WORK_DIR / "dino_model.py", package_dir / "dino_model.py")
    shutil.copyfile(WORK_DIR / "dinov3_meta_convnext.py",
                    package_dir / "dinov3_meta_convnext.py")
    shutil.copyfile(WORK_DIR / "submission_inference.py",
                    package_dir / "submission_inference.py")
    license_text = (WORK_DIR / "META_DINOV3_LICENSE.md").read_text(encoding="utf-8")
    if "DINOv3 License" not in license_text:
        raise RuntimeError("Bundled DINOv3 license text is missing or invalid")
    (package_dir / "LICENSE.md").write_text(license_text, encoding="utf-8")
    model_config = {
        "model": "DINOv3 ConvNeXt-Tiny single-branch",
        "variant_selected_by_best_validation_macro_auroc": selected_variant,
        "best_epoch": selected_epoch,
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "image_size": 320,
        "study_fusion": "mean of raw per-image logits before sigmoid",
        "feature_pooling": "global average pooling then the official DINOv3 ConvNeXt layer_norm",
        "input_mean": config["source"]["input_mean"],
        "input_std": config["source"]["input_std"],
        "source_repository": MODEL_REPO,
        "source_revision": MODEL_REVISION,
        "initialization_sha256": config["source"]["weights_sha256"],
        "trained_checkpoint_sha256": sha256_file(checkpoint_path),
        "inference_parameters": parameter_counts(inference_model)["total"],
        "training_projection_head_included": False,
        "test_split_read": False,
    }
    (package_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (package_dir / "requirements-inference.txt").write_text(
        "torch>=2.0\ntorchvision>=0.15\nsafetensors>=0.4\nPillow>=10\nnumpy>=1.24\n",
        encoding="utf-8")

    archive_path = export_root / "DINOv3_ConvNeXtTiny_Inference.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        package_names = (
            "model.safetensors", "model_config.json", "dino_model.py",
            "dinov3_meta_convnext.py", "submission_inference.py",
            "LICENSE.md", "requirements-inference.txt",
        )
        for name in package_names:
            path = package_dir / name
            archive.write(path, arcname=f"DINOv3_ConvNeXtTiny_Inference/{path.name}")
    archive_bytes = archive_path.stat().st_size
    limit = int(config["outputs"]["submission_size_limit_bytes"])
    if archive_bytes >= limit:
        raise RuntimeError(f"Inference archive is {archive_bytes} bytes, not below {limit}")

    if valid_data is None:
        data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
        _, valid_data, _ = build_datasets(data_path, config["seed"])
    multi_indices = [index for index, count in enumerate(valid_data.image_counts)
                     if count > 1]
    if not multi_indices:
        raise RuntimeError("Validation split has no multi-image Study for export verification")
    selected_index = min(multi_indices, key=lambda index: valid_data.image_counts[index])
    sample = valid_data[(selected_index, 0)]
    normalized_images = sample["images"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_model = DINOv3StudyClassifier(num_labels=NUM_LABELS)
    source_model.load_state_dict(checkpoint["model_state"], strict=True)
    source_model.to(device).eval()
    source_input = normalized_images.to(device)
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
        expected = torch.sigmoid(source_model(source_input).float().mean(dim=0)).cpu().numpy()
    del source_model, source_input, inference_model, state_dict, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    from submission_inference import Predictor

    predictor = Predictor(package_dir, device=str(device))
    actual_tensor = predictor.predict_study_tensor(normalized_images)
    np.testing.assert_allclose(actual_tensor, expected, rtol=2e-5, atol=2e-6)
    image_paths = valid_data.dataset.studies[selected_index]["image_paths"]
    actual_paths = predictor.predict_study_paths(image_paths)
    np.testing.assert_allclose(actual_paths, expected, rtol=2e-5, atol=2e-6)
    if len(image_paths) < 2 or actual_paths.shape != (NUM_LABELS,):
        raise RuntimeError("Standalone multi-image Study inference verification failed")

    uncompressed_bytes = sum(path.stat().st_size for path in package_dir.iterdir())
    report = [
        "# DINOv3 Inference Package Size and Load Verification", "",
        f"- Selected variant: `{selected_variant}`; Best Epoch: {selected_epoch}.",
        f"- Archive bytes: {archive_bytes:,}.",
        f"- Limit: strictly less than {limit:,} bytes.",
        f"- Compressed archive passes: {archive_bytes < limit}.",
        f"- Uncompressed package file bytes: {uncompressed_bytes:,}.",
        f"- Files: `{[path.name for path in sorted(package_dir.iterdir())]}`.",
        f"- DINOv3 backbone + classifier parameters: {model_config['inference_parameters']:,}.",
        "- D0-C1 training-only projection head, optimizer, scheduler, training sampler and validation predictions are excluded.",
        f"- Standalone strict load and validation multi-image Study ({len(image_paths)} images) succeeded.",
        f"- Packaged probabilities match the selected best checkpoint: max absolute error {float(np.max(np.abs(actual_tensor - expected))):.8g}.",
        f"- PIL preprocessing and multi-image prediction match: max absolute error {float(np.max(np.abs(actual_paths - expected))):.8g}.",
        "- Test split was not read.", "",
    ]
    (root / "submission_size_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "selected_variant": selected_variant,
        "best_epoch": selected_epoch,
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "limit_bytes": limit,
        "passes_limit": archive_bytes < limit,
        "uncompressed_bytes": uncompressed_bytes,
        "multi_image_study_images": len(image_paths),
        "max_probability_absolute_error": float(np.max(np.abs(actual_paths - expected))),
        "parameters": model_config["inference_parameters"],
    }
    save_json_atomic(root / "submission_export_verification.json", result)
    return result


def run_all(config, force_benchmark=False):
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root, "run")
    train_data, valid_data, data_config, verification, environment = run_verification(config)
    smoke = run_smoke(config, train_data, valid_data)
    runtime = run_benchmark(config, train_data, force=force_benchmark)
    d0 = train_variant(config, "d0", train_data, valid_data,
                       verification, runtime, resume=True)
    d0c1 = train_variant(config, "d0c1", train_data, valid_data,
                         verification, runtime, resume=True)
    comparison = compare_runs(config)
    export = export_submission_package(config, comparison, valid_data)
    outcome = {
        "experiment": config["experiment"],
        "verification": verification,
        "environment": environment,
        "smoke": smoke,
        "runtime": runtime,
        "d0": d0,
        "d0c1": d0c1,
        "comparison": comparison,
        "submission_export": export,
        "test_split_read": False,
    }
    save_json_atomic(root / "experiment_complete.json", outcome)
    logger.info("DINOv3 D0/D0-C1 experiment completed; D0=%.6f D0-C1=%.6f export_bytes=%d",
                comparison["d0_best_macro_auroc"],
                comparison["d0c1_best_macro_auroc"], export["archive_bytes"])
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "verify", "smoke", "benchmark",
                                              "train", "compare", "export"),
                        default="all")
    parser.add_argument("--variant", choices=("d0", "d0c1"), default="d0")
    parser.add_argument("--force-benchmark", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.stage == "all":
        run_all(config, force_benchmark=args.force_benchmark)
    elif args.stage == "verify":
        run_verification(config)
    elif args.stage == "smoke":
        train_data, valid_data, _, _, _ = run_verification(config)
        run_smoke(config, train_data, valid_data)
    elif args.stage == "benchmark":
        data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
        train_data, _, _ = build_datasets(data_path, config["seed"])
        run_benchmark(config, train_data, force=args.force_benchmark)
    elif args.stage == "train":
        root = output_root(config)
        runtime_path = root / "dino_selected_runtime_config.json"
        if not runtime_path.is_file():
            raise FileNotFoundError("Run the required GPU benchmark before formal training")
        with runtime_path.open("r", encoding="utf-8") as stream:
            runtime = json.load(stream)
        data_path = (WORK_DIR / config["data"]["source_config"]).resolve()
        train_data, valid_data, _ = build_datasets(data_path, config["seed"])
        _, verification = load_pretrained_backbone("cpu")
        train_variant(config, args.variant, train_data, valid_data,
                      verification, runtime, resume=not args.no_resume)
    elif args.stage == "compare":
        compare_runs(config)
    elif args.stage == "export":
        export_submission_package(config)


if __name__ == "__main__":
    main()
