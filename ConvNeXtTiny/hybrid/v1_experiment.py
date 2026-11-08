"""V1: ConvNeXtV2-Tiny fine-tuning from the final H1 pixel-pretrained CNN encoder."""

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, collate_studies
from datasets.hybrid import build_hybrid_transform
from datasets.transforms import ResizePad
from engine.trainer import _draw_curve
from engine.validator import validate
from hybrid.benchmark import NvidiaSampler
from hybrid.common import (PROJECT_DIR, StandardStudyBCE, autocast_context,
                           build_adamw, build_update_scheduler, count_parameters,
                           current_lrs, seed_everything, seed_worker, sha256_file,
                           sha256_state_dict, write_csv, write_json)
from model.v1_convnextv2 import V1ConvNeXtV2Classifier
from utils.metrics import multilabel_metrics


LABEL_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion",
    "No Finding",
]
TIMM_MODEL = "convnextv2_tiny.fcmae_ft_in1k"


def _read_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                         encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _resolve_paths(data_config):
    resolved = dict(data_config)
    for key in ("image_root", "train_csv", "val_csv"):
        value = Path(data_config[key])
        resolved[key] = value if value.is_absolute() else (PROJECT_DIR / value).resolve()
    return resolved


def _build_datasets(data_config):
    if int(data_config.get("image_size", 0)) != 320:
        raise ValueError("V1 reuses the H1 320x320 ResizePad input contract")
    if data_config.get("label_semantics") != "closed_world":
        raise ValueError("V1 requires the audited closed-world ten-label semantics")
    if list(data_config.get("class_names", [])) != LABEL_NAMES:
        raise ValueError("V1 class ordering/names must match the Hybrid H1 ten-label ordering")
    if list(data_config.get("class_ids", [])) != list(range(1, 11)):
        raise ValueError("V1 class IDs must remain 1..10 in the established order")
    options = {
        "label_column": data_config["label_column"],
        "label_semantics": data_config["label_semantics"],
        "image_id_column": data_config["image_id_column"],
        "subject_column": data_config["subject_column"],
        "study_column": data_config["study_column"],
    }
    train = StudyDataset(
        data_config["train_csv"], data_config["image_root"],
        transform=build_hybrid_transform(320, train=True), **options)
    valid = StudyDataset(
        data_config["val_csv"], data_config["image_root"],
        transform=build_hybrid_transform(320, train=False), **options)
    for name, dataset in (("Train", train), ("Validation", valid)):
        if not dataset.studies:
            raise ValueError(f"{name} split contains no Study samples")
        if dataset.label_tensors()[1].logical_not().any():
            raise ValueError(f"{name} must have complete closed-world ten-label targets")
    train_subjects = {subject for subject, _ in train.study_keys}
    valid_subjects = {subject for subject, _ in valid.study_keys}
    overlap = train_subjects & valid_subjects
    if overlap:
        raise ValueError(f"Train/Validation Subject overlap detected: {len(overlap)}")
    return train, valid


def _verify_split_audit(data_config, train_data, valid_data, h1_verification=None):
    audit_path = PROJECT_DIR / "outputs" / "hybrid" / "shared" / "manifests" / "data_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Official Train/Validation data audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("test_split_read") is not False:
        raise ValueError("Existing split audit does not confirm that Test remained unread")
    if int(audit.get("train_valid_subject_overlap", -1)) != 0:
        raise ValueError("Existing split audit reports Train/Validation Subject overlap")
    if audit.get("label_semantics") != "closed_world":
        raise ValueError("V1 requires the same confirmed closed-world label semantics as H1")
    if Path(audit["train_csv"]).resolve() != Path(data_config["train_csv"]).resolve():
        raise ValueError("V1 Train CSV differs from the audited official Train split")
    if Path(audit["valid_csv"]).resolve() != Path(data_config["val_csv"]).resolve():
        raise ValueError("V1 Validation CSV differs from the audited official Validation split")
    expected = {
        "train": (len(train_data), sum(train_data.image_counts), len({s for s, _ in train_data.study_keys})),
        "valid": (len(valid_data), sum(valid_data.image_counts), len({s for s, _ in valid_data.study_keys})),
    }
    for split, (studies, images, subjects) in expected.items():
        audited = audit[split]
        actual = (int(audited["studies"]), int(audited["image_rows"]), int(audited["subjects"]))
        if actual != (studies, images, subjects):
            raise ValueError(f"V1 {split} Dataset counts differ from the audited split: {actual} != {(studies, images, subjects)}")
    if h1_verification is not None:
        train_studies, train_images, train_subjects = expected["train"]
        if h1_verification["source_train_manifest_sha256"] != audit["train_manifest_sha256"]:
            raise ValueError("V1 Train manifest differs from the images used by H1 self-supervised pretraining")
        if Path(h1_verification["source_train_csv"]).resolve() != Path(data_config["train_csv"]).resolve():
            raise ValueError("V1 Train CSV differs from the H1 self-supervised Train CSV")
        if Path(h1_verification["source_image_root"]).resolve() != Path(data_config["image_root"]).resolve():
            raise ValueError("V1 image root differs from the H1 self-supervised image root")
        if (h1_verification["source_train_image_rows"] != train_images or
                h1_verification["source_train_studies"] != train_studies or
                h1_verification["source_train_subjects"] != train_subjects):
            raise ValueError("V1 Train image/Study/Subject counts differ from H1 self-supervised pretraining")
    return audit


def _validate_contract(config, require_benchmark=False):
    model = config["model"]
    training = config["training"]
    if (model["timm_model_name"] != TIMM_MODEL or int(model["image_size"]) != 320 or
            int(model["num_labels"]) != 10 or model["pooling"] != "gap" or
            model["classifier"] != "linear" or model["study_fusion"] != "logit_mean" or
            model["uses_vit"] or model["uses_decoder"]):
        raise ValueError("V1 architecture must remain ConvNeXtV2-Tiny Stage4 GAP + Linear(768,10), without ViT/decoder")
    if (int(training["epochs"]) != 20 or int(training["warmup_epochs"]) != 2 or
            int(training["cosine_end_epoch"]) != 20 or training["early_stopping_enabled"]):
        raise ValueError("V1 supervised fine-tuning is fixed to 20 epochs, 2+18 schedule, and no early stopping")
    fixed = {
        "seed": 42,
        "learning_rate_backbone": 1e-5,
        "learning_rate_head": 2e-4,
        "min_lr_backbone": 5e-7,
        "min_lr_head": 1e-5,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "loss": "standard_bce",
        "weighting": "none",
        "scheduler_step_unit": "optimizer_update",
        "amp": True,
        "amp_dtype": "bfloat16",
    }
    for key, expected in fixed.items():
        actual = training[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"V1 training setting {key} must be {expected}")
        elif actual != expected:
            raise ValueError(f"V1 training setting {key} must be {expected}")
    if require_benchmark and (training.get("image_budget") is None or
                              training.get("grad_accumulation") is None):
        raise RuntimeError("V1 formal training cannot start before the real GPU benchmark selects batch settings")


def _parameter_state_hash(state):
    return sha256_state_dict(state)


def _load_h1_encoder(config, output_dir):
    init = config["initialization"]
    encoder_path = PROJECT_DIR / init["h1_encoder_last"]
    full_path = PROJECT_DIR / init["h1_full_last"]
    h1_config_path = PROJECT_DIR / init["h1_config"]
    h1_report_path = PROJECT_DIR / init["h1_report"]
    required = (encoder_path, full_path, h1_config_path, h1_report_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "V1 requires the final H1 self-supervised encoder and its provenance; missing: "
            + ", ".join(missing))

    h1_config = json.loads(h1_config_path.read_text(encoding="utf-8"))
    report_text = h1_report_path.read_text(encoding="utf-8")
    encoder_payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
    full_payload = torch.load(full_path, map_location="cpu", weights_only=False)
    if encoder_payload.get("method") != "h1_pixel" or int(encoder_payload.get("epoch", -1)) != 20:
        raise ValueError("H1 encoder_last is not the final epoch-20 pixel-pretraining export")
    if encoder_payload.get("decoder_excluded") is not True:
        raise ValueError("H1 encoder export does not confirm decoder exclusion")
    if encoder_payload.get("teacher_student_used") is not False:
        raise ValueError("H1 pixel-pretraining export has unexpected Teacher–Student provenance")
    if int(full_payload.get("epoch", -1)) != 20:
        raise ValueError("H1 full last checkpoint is not from epoch 20")
    full_config = full_payload.get("config", {})
    if full_config.get("method") != "h1_pixel":
        raise ValueError("H1 full last checkpoint is not a pixel reconstruction run")
    if h1_config.get("method") != "h1_pixel" or int(h1_config.get("config", {}).get("epochs", -1)) != 20:
        raise ValueError("H1 run config does not attest to a 20-epoch pixel reconstruction run")
    if "Data source: official train images only" not in report_text or "Status: completed 20 epochs" not in report_text:
        raise ValueError("H1 report does not confirm a completed official-train-only run")
    if h1_config.get("test_split_read") is not False or h1_config.get("valid_split_read") is not False:
        raise ValueError("H1 pretraining provenance indicates non-train data access")

    exported_backbone = encoder_payload.get("backbone")
    full_model = full_payload.get("model", {})
    if not isinstance(exported_backbone, dict) or not exported_backbone:
        raise ValueError("H1 encoder export has no backbone state")
    convnext_state = {key[len("convnext."):]: value
                      for key, value in exported_backbone.items()
                      if key.startswith("convnext.")}
    if not convnext_state:
        raise ValueError("H1 export contains no ConvNeXtV2 state")
    for key in ("convnext_mean", "convnext_std"):
        if key not in exported_backbone:
            raise ValueError(f"H1 encoder is missing the ConvNeXt normalization buffer {key}")
    if any("vit" in key.lower() or "decoder" in key.lower() or "mask_token" in key.lower()
           for key in convnext_state):
        raise ValueError("Filtered V1 source unexpectedly includes ViT or decoder state")

    for key, tensor in exported_backbone.items():
        if key.startswith("convnext.") or key in {"convnext_mean", "convnext_std"}:
            full_key = "backbone." + key
            if full_key not in full_model or not torch.equal(tensor, full_model[full_key]):
                raise ValueError(f"H1 final encoder export differs from H1 epoch-20 checkpoint at {key}")

    shared_init_hash = h1_config.get("initial_checkpoint_sha256")
    if not shared_init_hash:
        raise ValueError("H1 ImageNet initialization SHA-256 is missing")
    if encoder_payload.get("config", {}).get("initial_checkpoint_sha256") != shared_init_hash:
        raise ValueError("H1 encoder export provenance differs from the H1 run config")
    if full_config.get("initial_checkpoint_sha256") != shared_init_hash:
        raise ValueError("H1 full checkpoint provenance differs from the H1 run config")
    shared_initial_path = PROJECT_DIR / "outputs" / "hybrid" / "shared" / "imagenet_initial.pth"
    if not shared_initial_path.is_file() or sha256_file(shared_initial_path) != shared_init_hash:
        raise ValueError("The shared H1 ImageNet initialization checkpoint is missing or has a different SHA-256")
    initial_metadata_path = shared_initial_path.with_suffix(".json")
    if not initial_metadata_path.is_file():
        raise FileNotFoundError(f"Shared ImageNet initialization metadata is missing: {initial_metadata_path}")
    shared_metadata = json.loads(initial_metadata_path.read_text(encoding="utf-8"))
    h1_cnn_metadata = shared_metadata.get("convnext", {})
    source_mean = h1_cnn_metadata.get("input_mean")
    source_std = h1_cnn_metadata.get("input_std")
    export_mean = exported_backbone["convnext_mean"].reshape(-1).tolist()
    export_std = exported_backbone["convnext_std"].reshape(-1).tolist()
    if source_mean is None or source_std is None:
        raise ValueError("H1 provenance does not record ConvNeXtV2 normalization values")
    if not np.allclose(export_mean, source_mean, rtol=0.0, atol=1e-7):
        raise ValueError("H1 export normalization mean differs from its ImageNet provenance")
    if not np.allclose(export_std, source_std, rtol=0.0, atol=1e-7):
        raise ValueError("H1 export normalization standard deviation differs from its ImageNet provenance")

    model = timm.create_model(TIMM_MODEL, pretrained=False)
    model.reset_classifier(0)
    model.head = nn.Identity()
    incompatible = model.load_state_dict(convnext_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Strict ConvNeXtV2 load was incomplete: {incompatible}")
    grn_keys = [key for key in convnext_state if ".grn." in key]
    if not grn_keys:
        raise ValueError("Loaded ConvNeXtV2 state has no GRN parameters")
    if count_parameters(model) != 27_864_960:
        raise ValueError(f"Unexpected ConvNeXtV2 parameter count: {count_parameters(model)}")

    # Persist a CNN-only initialization payload so subsequent V1 stages never carry the ViT tensors.
    v1_encoder = {
        "model_name": TIMM_MODEL,
        "state_dict": {key: value.detach().cpu().contiguous()
                       for key, value in convnext_state.items()},
        "input_mean": export_mean,
        "input_std": export_std,
        "source_checkpoint": str(encoder_path),
        "source_checkpoint_sha256": sha256_file(encoder_path),
        "source_method": "h1_pixel",
        "source_epoch": 20,
        "imagenet_initial_checkpoint_sha256": shared_init_hash,
    }
    extracted_path = Path(output_dir) / "convnextv2_h1_encoder.pth"
    _atomic_torch_save(extracted_path, v1_encoder)

    verification = {
        "status": "verified",
        "selected_source": str(encoder_path),
        "source_role": "H1 pixel-pretraining final encoder export, epoch 20",
        "source_file_sha256": sha256_file(encoder_path),
        "source_full_checkpoint": str(full_path),
        "source_full_checkpoint_sha256": sha256_file(full_path),
        "source_full_checkpoint_epoch": int(full_payload["epoch"]),
        "source_method": encoder_payload["method"],
        "source_encoder_epoch": int(encoder_payload["epoch"]),
        "source_initial_imagenet_sha256": shared_init_hash,
        "source_train_manifest_sha256": h1_config.get("train_manifest_sha256"),
        "source_train_csv": h1_config.get("train_csv"),
        "source_image_root": h1_config.get("image_root"),
        "source_train_image_rows": int(h1_config.get("train_image_rows", -1)),
        "source_train_studies": int(h1_config.get("train_studies", -1)),
        "source_train_subjects": int(h1_config.get("train_subjects", -1)),
        "source_valid_split_read": h1_config.get("valid_split_read"),
        "source_test_split_read": h1_config.get("test_split_read"),
        "decoder_excluded": encoder_payload["decoder_excluded"],
        "teacher_student_used": encoder_payload.get("teacher_student_used"),
        "convnext_model_name": TIMM_MODEL,
        "timm_version": timm.__version__,
        "convnext_state_key_count": len(convnext_state),
        "convnext_parameter_count": count_parameters(model),
        "grn_state_key_count": len(grn_keys),
        "loaded_missing_keys": incompatible.missing_keys,
        "loaded_unexpected_keys": incompatible.unexpected_keys,
        "normalization_mean": export_mean,
        "normalization_std": export_std,
        "weight_selection_policy": "final predetermined H1 self-supervised epoch, not classifier-selected",
        "v1_extracted_encoder_checkpoint": str(extracted_path),
        "v1_extracted_encoder_sha256": sha256_file(extracted_path),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / "h1_pretrain_weight_verification.md"
    lines = [
        "# H1 Encoder Weight Verification", "",
        f"- Status: {verification['status']}",
        f"- V1 source: `{encoder_path}`",
        f"- Source role: {verification['source_role']}",
        f"- Source SHA-256: `{verification['source_file_sha256']}`",
        f"- Paired H1 full checkpoint: `{full_path}` (epoch {verification['source_full_checkpoint_epoch']})",
        f"- H1 method / encoder epoch: `{verification['source_method']}` / {verification['source_encoder_epoch']}",
        f"- Shared ImageNet initialization SHA-256: `{shared_init_hash}`",
        f"- Official Train manifest SHA-256: `{verification['source_train_manifest_sha256']}`",
        f"- Validation/Test read during H1 pretraining: {verification['source_valid_split_read']} / {verification['source_test_split_read']}",
        f"- Encoder export excludes pixel decoder: {verification['decoder_excluded']}",
        f"- ConvNeXtV2 state tensors loaded strictly: {len(convnext_state)}; missing: {len(incompatible.missing_keys)}; unexpected: {len(incompatible.unexpected_keys)}",
        f"- GRN state tensors verified: {len(grn_keys)}",
        f"- ConvNeXtV2 parameters: {count_parameters(model):,}",
        f"- Normalization mean/std: `{export_mean}` / `{export_std}`",
        f"- CNN-only V1 initialization export: `{extracted_path}`",
        f"- CNN-only V1 initialization SHA-256: `{verification['v1_extracted_encoder_sha256']}`",
        "- The H1 export was inspected for provenance, then only ConvNeXtV2 tensors were extracted. ViT weights, the H1 classifier, pixel decoder, and mask tokens are excluded from the V1 model and its initialization export.",
    ]
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    del model, exported_backbone, full_payload, encoder_payload
    gc.collect()
    return v1_encoder, verification


def _make_model(v1_encoder, seed):
    torch.manual_seed(int(seed))
    convnext = timm.create_model(TIMM_MODEL, pretrained=False)
    convnext.reset_classifier(0)
    convnext.head = nn.Identity()
    if (v1_encoder.get("model_name") != TIMM_MODEL or
            v1_encoder.get("source_method") != "h1_pixel" or
            int(v1_encoder.get("source_epoch", -1)) != 20):
        raise ValueError("V1 initialization must be the extracted final H1 pixel-pretrained ConvNeXtV2 encoder")
    result = convnext.load_state_dict(v1_encoder["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"V1 ConvNeXtV2 initialization was incomplete: {result}")
    return V1ConvNeXtV2Classifier(
        convnext,
        v1_encoder["input_mean"],
        v1_encoder["input_std"],
    )


def _loader(dataset, image_budget, shuffle, seed, workers, pin_memory, generator=None):
    sampler = StudyBatchSampler(dataset, image_budget, shuffle=shuffle, seed=seed)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers, pin_memory=pin_memory,
        collate_fn=collate_studies, generator=generator,
        worker_init_fn=seed_worker if workers else None,
        persistent_workers=bool(workers),
    )
    return loader, sampler


def _shutdown_loader(loader, iterator=None):
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    if loader is not None:
        loader_iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(loader_iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def _next_window(loader, iterator, sampler, epoch, count):
    batches = []
    for _ in range(count):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        batches.append(batch)
    return batches, iterator, epoch


def _prepare_model(config, v1_encoder, device, checkpointing=False):
    model = _make_model(v1_encoder, config["training"]["seed"]).to(device)
    model.set_grad_checkpointing(checkpointing)
    return model


def run_smoke_test(config, v1_encoder, train_data, valid_data, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("V1 requires the verified CUDA BF16-capable GPU")
    train_multi = [index for index, count in enumerate(train_data.image_counts) if count > 1]
    valid_multi = [index for index, count in enumerate(valid_data.image_counts) if count > 1]
    train_index = (min(train_multi, key=lambda index: train_data.image_counts[index])
                   if train_multi else min(range(len(train_data)), key=lambda index: train_data.image_counts[index]))
    valid_index = (min(valid_multi, key=lambda index: valid_data.image_counts[index])
                   if valid_multi else min(range(len(valid_data)), key=lambda index: valid_data.image_counts[index]))
    train_item = train_data[train_index]
    valid_item = valid_data[valid_index]
    if tuple(train_item["images"].shape[1:]) != (3, 320, 320):
        raise RuntimeError(f"Train transform shape mismatch: {tuple(train_item['images'].shape)}")
    if tuple(valid_item["images"].shape[1:]) != (3, 320, 320):
        raise RuntimeError(f"Validation transform shape mismatch: {tuple(valid_item['images'].shape)}")

    model = _prepare_model(config, v1_encoder, device)
    if any(name.startswith("vit") or "decoder" in name.lower() or "mask_token" in name.lower()
           for name, _ in model.named_parameters()):
        raise RuntimeError("V1 model unexpectedly contains a ViT or pretraining-only parameters")
    train_batch = collate_studies([train_item])
    images = train_batch["images"].to(device)
    study_index = train_batch["study_index"].to(device)
    model.eval()
    with torch.no_grad(), autocast_context(device, True, torch.bfloat16):
        stage4 = model.convnext.forward_features((images - model.input_mean) / model.input_std)
        image_logits = model.image_logits(images)
        pooled_logits = model(images, study_index, train_batch["num_studies"])
        feature_shape = tuple(model.image_features(images).shape)
    expected_pool = image_logits.mean(dim=0, keepdim=True)
    if tuple(stage4.shape[1:]) != (768, 10, 10):
        raise RuntimeError(f"V1 Stage4 shape mismatch: {tuple(stage4.shape)}")
    if feature_shape != (images.shape[0], 768):
        raise RuntimeError("V1 GAP feature width is not 768")
    if tuple(image_logits.shape) != (images.shape[0], 10):
        raise RuntimeError("V1 classifier did not emit ten image logits")
    if tuple(pooled_logits.shape) != (1, 10) or not torch.allclose(
            pooled_logits.float(), expected_pool.float(), atol=2e-3, rtol=2e-3):
        raise RuntimeError("V1 Study Logit Mean does not match the arithmetic mean of raw image logits")

    model.train()
    optimizer = build_adamw([
        {"params": model.convnext.parameters(), "lr": config["training"]["learning_rate_backbone"], "name": "backbone"},
        {"params": model.classifier.parameters(), "lr": config["training"]["learning_rate_head"], "name": "head"},
    ], config["training"]["weight_decay"], fused=config["training"]["optimizer_fused"])
    criterion = StandardStudyBCE()
    optimizer.zero_grad(set_to_none=True)
    labels = train_batch["labels"].to(device)
    known = train_batch["known"].to(device)
    backbone_parameters = list(model.convnext.parameters())
    backbone_before = [parameter.detach().float().cpu().clone()
                       for parameter in backbone_parameters]
    head_before = model.classifier.weight.detach().clone()
    with autocast_context(device, True, torch.bfloat16):
        logits = model(images, study_index, train_batch["num_studies"])
        loss = criterion(logits, labels, known)
    if not torch.isfinite(loss):
        raise RuntimeError("V1 real-sample smoke loss is non-finite")
    loss.backward()
    backbone_grad = sum(float(p.grad.detach().float().square().sum())
                        for p in model.convnext.parameters() if p.grad is not None) ** 0.5
    head_grad = sum(float(p.grad.detach().float().square().sum())
                    for p in model.classifier.parameters() if p.grad is not None) ** 0.5
    if not math.isfinite(backbone_grad) or backbone_grad <= 0 or not math.isfinite(head_grad) or head_grad <= 0:
        raise RuntimeError(f"V1 encoder/head gradients missing: {backbone_grad}/{head_grad}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"],
                                  error_if_nonfinite=True)
    optimizer.step()
    backbone_change_sq = sum(float((parameter.detach().float().cpu() - before).square().sum())
                             for parameter, before in zip(backbone_parameters, backbone_before))
    backbone_change = math.sqrt(backbone_change_sq)
    head_change = float((model.classifier.weight.detach() - head_before).float().norm())
    if backbone_change <= 0 or head_change <= 0:
        raise RuntimeError(f"V1 real-sample optimizer step did not update parameters: {backbone_change}/{head_change}")

    output = {
        "passed": True,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "train_study_image_count": int(images.shape[0]),
        "train_study_id": train_item["study_id"],
        "train_shape": list(train_item["images"].shape),
        "validation_shape": list(valid_item["images"].shape),
        "stage4_shape": list(stage4.shape),
        "gap_shape": [int(images.shape[0]), 768],
        "image_logits_shape": list(image_logits.shape),
        "study_logits_shape": list(pooled_logits.shape),
        "study_fusion": "mean of raw image logits before sigmoid",
        "loss": float(loss.detach()),
        "backbone_gradient_norm": backbone_grad,
        "head_gradient_norm": head_grad,
        "backbone_parameter_change": backbone_change,
        "head_parameter_change": head_change,
        "contains_vit": False,
        "contains_decoder": False,
        "contains_mask_token": False,
    }
    write_json(Path(output_dir) / "v1_smoke_test.json", output)

    # Validate the fixed aspect-ratio preprocessing contract without reading more NAS data.
    from PIL import Image
    for width, height in ((2544, 3056), (3056, 2544), (2021, 2021)):
        sample = ResizePad(320)(Image.new("RGB", (width, height), (255, 255, 255)))
        if sample.size != (320, 320):
            raise RuntimeError(f"ResizePad returned {sample.size}, expected (320,320)")
        box = sample.getbbox()
        if box is None:
            raise RuntimeError("ResizePad test image unexpectedly became empty")
        resized_w, resized_h = box[2] - box[0], box[3] - box[1]
        expected = width / height
        if abs(resized_w / resized_h - expected) > 0.02:
            raise RuntimeError(f"ResizePad changed aspect ratio for {width}x{height}")
        if width < height and not (box[0] > 0 and box[1] == 0 and box[3] == 320):
            raise RuntimeError(f"ResizePad portrait padding direction is wrong for {width}x{height}: {box}")
        if width > height and not (box[0] == 0 and box[1] > 0 and box[2] == 320):
            raise RuntimeError(f"ResizePad landscape padding direction is wrong for {width}x{height}: {box}")
        if width == height and box != (0, 0, 320, 320):
            raise RuntimeError(f"ResizePad square image should not be padded: {box}")
    return output


def _candidate_configs(config):
    target = int(config["benchmark"]["target_effective_images"])
    result = []
    for budget in config["benchmark"]["image_budgets"]:
        budget = int(budget)
        accumulation = max(1, int(round(target / budget)))
        result.append({
            "image_budget": budget,
            "grad_accumulation": accumulation,
            "effective_batch_target": budget * accumulation,
            "num_workers": int(config["training"]["num_workers"]),
            "pin_memory": bool(config["training"]["pin_memory"]),
            "optimizer_fused": bool(config["training"]["optimizer_fused"]),
            "gradient_checkpointing": False,
        })
    return result


def _benchmark_candidate(candidate, config, v1_encoder, train_data, device):
    benchmark = config["benchmark"]
    training = config["training"]
    seed_everything(int(training["seed"]))
    model = _prepare_model(config, v1_encoder, device,
                           checkpointing=candidate["gradient_checkpointing"])
    optimizer = build_adamw([
        {"params": model.convnext.parameters(), "lr": float(training["learning_rate_backbone"]), "name": "backbone"},
        {"params": model.classifier.parameters(), "lr": float(training["learning_rate_head"]), "name": "head"},
    ], training["weight_decay"], fused=candidate["optimizer_fused"])
    criterion = StandardStudyBCE().to(device)
    generator = torch.Generator().manual_seed(int(training["data_loader_worker_seed"]))
    loader, sampler = _loader(train_data, candidate["image_budget"], True,
                              int(training["seed"]), candidate["num_workers"],
                              candidate["pin_memory"], generator)
    iterator = iter(loader)
    epoch = 0
    warmup = int(benchmark["warmup_updates"])
    measured = int(benchmark["measured_updates"])
    update_seconds, compute_seconds = [], []
    images_seen = studies_seen = 0
    subjects_seen = set()
    data_loader_wait_seconds = 0.0
    sampler_stats = NvidiaSampler(interval=0.5)
    sampler_stats.start()
    try:
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        for update_index in range(warmup + measured):
            wall_start = time.perf_counter()
            window, iterator, epoch = _next_window(
                loader, iterator, sampler, epoch, candidate["grad_accumulation"])
            fetch_elapsed = time.perf_counter() - wall_start
            study_total = sum(int(batch["num_studies"]) for batch in window)
            images_in_window = sum(int(batch["images"].shape[0]) for batch in window)
            event_pairs = []
            optimizer.zero_grad(set_to_none=True)
            for batch in window:
                images = batch["images"].to(device, non_blocking=True)
                study_index = batch["study_index"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                known = batch["known"].to(device, non_blocking=True)
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
                with autocast_context(device, True, torch.bfloat16):
                    logits = model(images, study_index, batch["num_studies"])
                    loss = criterion(logits, labels, known)
                eligible = int(known.any(dim=1).sum().item())
                (loss * (eligible / study_total)).backward()
                end_event.record()
                event_pairs.append((start_event, end_event))
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]),
                                          error_if_nonfinite=True)
            step_start, step_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            step_start.record()
            optimizer.step()
            step_end.record()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            wall_elapsed = time.perf_counter() - wall_start
            compute_elapsed = sum(start.elapsed_time(end) / 1000.0 for start, end in event_pairs)
            compute_elapsed += step_start.elapsed_time(step_end) / 1000.0
            if update_index >= warmup:
                update_seconds.append(wall_elapsed)
                compute_seconds.append(compute_elapsed)
                images_seen += images_in_window
                studies_seen += study_total
                subjects_seen.update(subject for batch in window for subject in batch["subject_id"])
                data_loader_wait_seconds += fetch_elapsed
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        sampled = sampler_stats.summary()
        wall_total, compute_total = sum(update_seconds), sum(compute_seconds)
        return {
            **candidate,
            "status": "ok",
            "images": images_seen,
            "studies": studies_seen,
            "subjects": len(subjects_seen),
            "optimizer_updates": measured,
            "effective_images_measured": images_seen / measured,
            "end_to_end_images_per_s": images_seen / wall_total if wall_total else 0.0,
            "gpu_compute_images_per_s": images_seen / compute_total if compute_total else 0.0,
            "update_mean_seconds": statistics.fmean(update_seconds),
            "update_p50_seconds": float(np.percentile(update_seconds, 50)),
            "update_p95_seconds": float(np.percentile(update_seconds, 95)),
            "update_cv": (float(np.std(update_seconds) / np.mean(update_seconds))
                          if np.mean(update_seconds) else 0.0),
            "peak_allocated_mib": allocated,
            "peak_reserved_mib": reserved,
            "peak_reserved_fraction": reserved / torch.cuda.get_device_properties(device).total_memory * 1024**2,
            "gpu_utilization_mean_pct": sampled["gpu_utilization_mean_pct"],
            "gpu_memory_used_mean_mib": sampled["gpu_memory_used_mean_mib"],
            "data_loader_wait_seconds": data_loader_wait_seconds,
            "data_and_transfer_seconds": max(wall_total - compute_total, 0.0),
            "error": "",
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {**candidate, "status": "oom", "error": str(exc),
                "images": images_seen, "studies": studies_seen, "subjects": len(subjects_seen),
                "optimizer_updates": len(update_seconds), "end_to_end_images_per_s": 0.0,
                "gpu_compute_images_per_s": 0.0, "peak_allocated_mib": float("nan"),
                "peak_reserved_mib": float("nan"), "peak_reserved_fraction": float("nan"),
                "effective_images_measured": float("nan"), "update_mean_seconds": float("nan"),
                "update_p50_seconds": float("nan"), "update_p95_seconds": float("nan"),
                "update_cv": float("nan"), "gpu_utilization_mean_pct": float("nan"),
                "gpu_memory_used_mean_mib": float("nan"),
                "data_loader_wait_seconds": float("nan"), "data_and_transfer_seconds": float("nan")}
    finally:
        sampler_stats.stop()
        _shutdown_loader(loader, iterator)
        batches = window = event_pairs = None
        batch = images = study_index = labels = known = loss = None
        del model, optimizer, loader, iterator, criterion
        gc.collect()
        torch.cuda.empty_cache()


def run_benchmark(config, v1_encoder, train_data, output_dir, config_path):
    device = torch.device("cuda")
    rows = []
    initial_candidates = _candidate_configs(config)
    for candidate in initial_candidates:
        result = _benchmark_candidate(candidate, config, v1_encoder, train_data, device)
        result["candidate_group"] = "base"
        rows.append(result)
        print(f"V1 benchmark budget={candidate['image_budget']} accum={candidate['grad_accumulation']} "
              f"status={result['status']} e2e={result.get('end_to_end_images_per_s', 0):.2f} images/s")

    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("All V1 real-training throughput benchmark candidates failed")
    target_effective = int(config["benchmark"]["target_effective_images"])
    effective_tolerance = float(config["benchmark"]["effective_batch_tolerance_fraction"])

    def effective_batch_comparable(row):
        measured = float(row.get("effective_images_measured", float("nan")))
        nominal = int(row["effective_batch_target"])
        return (math.isfinite(measured) and
                abs(nominal - target_effective) <= target_effective * effective_tolerance and
                abs(measured - target_effective) <= target_effective * effective_tolerance)

    comparable = [row for row in successful if effective_batch_comparable(row)]
    if not comparable:
        raise RuntimeError("No successful V1 benchmark candidate retained an effective batch comparable to the 32-image target")
    top_candidates = sorted(comparable, key=lambda row: row["end_to_end_images_per_s"], reverse=True)[:2]
    for prior in top_candidates:
        candidate = {key: prior[key] for key in (
            "image_budget", "grad_accumulation", "effective_batch_target", "num_workers",
            "pin_memory", "optimizer_fused")}
        candidate["gradient_checkpointing"] = True
        result = _benchmark_candidate(candidate, config, v1_encoder, train_data, device)
        result["candidate_group"] = "checkpointing_recheck"
        rows.append(result)
        print(f"V1 checkpointing benchmark budget={candidate['image_budget']} "
              f"status={result['status']} e2e={result.get('end_to_end_images_per_s', 0):.2f} images/s")

    total_memory_mib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    margin = float(config["benchmark"]["oom_memory_margin_fraction"])
    feasible = [row for row in rows if row["status"] == "ok" and effective_batch_comparable(row) and
                math.isfinite(float(row["peak_reserved_mib"])) and
                row["peak_reserved_mib"] <= total_memory_mib * (1.0 - margin)]
    if not feasible:
        raise RuntimeError("No V1 benchmark candidate met the required GPU-memory headroom")
    fastest = max(row["end_to_end_images_per_s"] for row in feasible)
    tolerance = float(config["benchmark"]["near_max_throughput_tolerance"])
    near_fastest = [row for row in feasible
                    if row["end_to_end_images_per_s"] >= fastest * (1.0 - tolerance)]
    selected = min(near_fastest, key=lambda row: (
        row["peak_reserved_mib"], -row["end_to_end_images_per_s"]))
    train = config["training"]
    for key in ("image_budget", "grad_accumulation", "num_workers", "pin_memory",
                "optimizer_fused", "gradient_checkpointing"):
        train[key] = selected[key]
    config["benchmark_result"] = {
        "selected": selected,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_mib": total_memory_mib,
        "torch_version": str(torch.__version__),
        "timm_version": str(timm.__version__),
    }
    write_json(Path(output_dir) / "config.json", config)
    benchmark_fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(Path(output_dir) / "v1_gpu_benchmark.csv", rows, fieldnames=benchmark_fields)
    lines = [
        "# V1 ConvNeXtV2 GPU Training Benchmark", "",
        f"- GPU: {torch.cuda.get_device_name(device)}; total VRAM: {total_memory_mib / 1024:.2f} GiB",
        f"- PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; timm: {timm.__version__}",
        "- Each candidate includes real NAS DataLoader batches, H2D transfer, ConvNeXtV2 forward, Study Logit Mean, Standard BCE, backward, clipping, gradient accumulation, and AdamW step.",
        f"- Each candidate used {config['benchmark']['warmup_updates']} warmup updates and {config['benchmark']['measured_updates']} measured updates.",
        "- Gradient accumulation targets roughly 32 images/update where the discrete budget permits (for example 12×3=36 and 24×1=24); actual images, Studies, and effective images/update are recorded per candidate.",
        f"- Only candidates within {effective_tolerance:.0%} of the 32-image target, using both nominal and measured effective images/update, are eligible for selection. A 24×1 candidate is measured but excluded from selection because it changes the effective batch materially.",
        "- Compute images/s sums CUDA-event timing around model/loss/backward and optimizer step. End-to-end images/s includes batch loading and transfer and is the selection metric.",
        f"- A candidate needs at least {margin:.0%} reserved-memory headroom. Within {tolerance:.0%} of the fastest feasible candidate, selection prefers more headroom.",
        f"- Selected: image_budget={selected['image_budget']}, grad_accumulation={selected['grad_accumulation']}, checkpointing={selected['gradient_checkpointing']}; {selected['end_to_end_images_per_s']:.2f} end-to-end images/s, {selected['gpu_compute_images_per_s']:.2f} compute images/s, reserved peak {selected['peak_reserved_mib']:.0f} MiB.",
        "", "## Candidate results", "",
        "| Budget | Accumulation | Effective images/update | Checkpointing | Status | Images/Studies/Subjects | Updates | E2E images/s | Compute images/s | Update P50/P95 sec | Update CV | Alloc/Reserved MiB | Mean GPU memory MiB | GPU util % | DataLoader wait sec | Error |",
        "|---:|---:|---:|:---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['image_budget']} | {row['grad_accumulation']} | {row.get('effective_images_measured', float('nan')):.1f} | {row['gradient_checkpointing']} | {row['status']} | {row.get('images', 0)}/{row.get('studies', 0)}/{row.get('subjects', 0)} | {row.get('optimizer_updates', 0)} | {row.get('end_to_end_images_per_s', 0):.2f} | {row.get('gpu_compute_images_per_s', 0):.2f} | {row.get('update_p50_seconds', float('nan')):.3f}/{row.get('update_p95_seconds', float('nan')):.3f} | {row.get('update_cv', float('nan')):.3f} | {row.get('peak_allocated_mib', float('nan')):.0f}/{row.get('peak_reserved_mib', float('nan')):.0f} | {row.get('gpu_memory_used_mean_mib', float('nan')):.0f} | {row.get('gpu_utilization_mean_pct', float('nan')):.1f} | {row.get('data_loader_wait_seconds', float('nan')):.2f} | {str(row.get('error', '')).replace('|', '/')[:120]} |"
        )
    Path(output_dir, "v1_gpu_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Keep the training YAML typed and concise; the full candidate table and selected
    # benchmark metadata are persisted separately in the CSV, Markdown, and JSON artifacts.
    _write_yaml(config_path, {key: value for key, value in config.items()
                              if key not in {"resolved_data", "benchmark_result"}})
    return selected, rows


def _state_rng():
    import numpy.random
    return {
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def _restore_rng(state):
    torch.set_rng_state(state["rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])


def _train_epoch(model, loader, sampler, optimizer, scheduler, criterion, config,
                 train_generator, device):
    model.train()
    model.set_grad_checkpointing(bool(config["training"]["gradient_checkpointing"]))
    accumulation = int(config["training"]["grad_accumulation"])
    sampler.set_epoch(int(config.get("current_epoch_index", 0)))
    iterator = iter(loader)
    batch_count = len(loader)
    windows = math.ceil(batch_count / accumulation)
    total_loss = 0.0
    total_studies = 0
    images_seen = 0
    update_norms = []
    update_durations = []
    epoch_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for window_index in range(windows):
        update_started = time.perf_counter()
        batches = []
        for _ in range(min(accumulation, batch_count - window_index * accumulation)):
            batches.append(next(iterator))
        study_count = sum(int(batch["num_studies"]) for batch in batches)
        for batch in batches:
            images = batch["images"].to(device, non_blocking=True)
            study_index = batch["study_index"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            known = batch["known"].to(device, non_blocking=True)
            with autocast_context(device, True, torch.bfloat16):
                logits = model(images, study_index, batch["num_studies"])
                loss = criterion(logits, labels, known)
            n_studies = int(batch["num_studies"])
            (loss * (n_studies / study_count)).backward()
            total_loss += float(loss.detach()) * n_studies
            total_studies += n_studies
            images_seen += int(images.shape[0])
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"]),
            error_if_nonfinite=True)
        update_norms.append(float(norm))
        start = time.perf_counter()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_durations.append(time.perf_counter() - update_started)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    else:
        allocated = reserved = 0.0
    elapsed = time.perf_counter() - epoch_started
    return {
        "loss": total_loss / total_studies if total_studies else float("nan"),
        "studies": total_studies,
        "images": images_seen,
        "optimizer_updates": len(update_norms),
        "gradient_norm": float(np.mean(update_norms)) if update_norms else 0.0,
        "elapsed_s": elapsed,
        "images_per_s": images_seen / elapsed if elapsed else 0.0,
        "mean_update_seconds": float(np.mean(update_durations)) if update_durations else 0.0,
        "peak_gpu_allocated_gib": allocated,
        "peak_gpu_reserved_gib": reserved,
    }


def _prediction_rows(validation, class_names):
    rows = []
    for index, (subject_id, study_id) in enumerate(zip(validation["subject_ids"], validation["study_ids"])):
        row = {"Subject_id": str(subject_id), "Study_id": str(study_id)}
        for label_index, name in enumerate(class_names):
            row[f"target_{name}"] = float(validation["targets"][index, label_index])
            row[f"prob_{name}"] = float(validation["probabilities"][index, label_index])
        rows.append(row)
    return rows


def _write_epoch_curves(output_dir, history, per_class_rows):
    epochs = [int(row["epoch"]) for row in history]
    _draw_curve(Path(output_dir) / "train_valid_loss.png", [
        ("Train Loss", epochs, [float(row["train_loss"]) for row in history], (39, 110, 220)),
        ("Validation Loss", epochs, [float(row["val_loss"]) for row in history], (220, 75, 55)),
    ], "Study-mean Standard BCE", markers=(5, 7, 10, 15, 20))
    _draw_curve(Path(output_dir) / "macro_auroc_curve.png", [
        ("Validation Macro-AUROC", epochs,
         [float(row["macro_auroc"]) for row in history], (39, 145, 88)),
    ], "Macro-AUROC", markers=(5, 7, 10, 15, 20))
    if per_class_rows:
        colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                  (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
                  (188, 189, 34), (23, 190, 207)]
        series = []
        for index, name in enumerate(LABEL_NAMES):
            rows = sorted((row for row in per_class_rows if row["class_name"] == name),
                          key=lambda row: int(row["epoch"]))
            series.append((name, [int(row["epoch"]) for row in rows],
                           [float(row["auroc"]) for row in rows], colors[index]))
        _draw_curve(Path(output_dir) / "v1_per_class_auroc_curve.png", series,
                    "Validation per-class AUROC", markers=(5, 7, 10, 15, 20))


def _checkpoint_state(model, optimizer, scheduler, epoch, best_auc, best_epoch,
                      config, train_generator, val_generator):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_macro_auroc": float(best_auc),
        "best_epoch": int(best_epoch),
        "config": config,
        "rng": _state_rng(),
        "train_loader_generator_state": train_generator.get_state(),
        "val_loader_generator_state": val_generator.get_state(),
    }


def _train(config, config_path, v1_encoder, verification, train_data,
           valid_data, output_dir, resume=False):
    output_dir = Path(output_dir)
    best_path, last_path = output_dir / "best.pth", output_dir / "last.pth"
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing V1 training checkpoints in {output_dir}")
    training = config["training"]
    _validate_contract(config, require_benchmark=True)
    if training.get("image_budget") is None or training.get("grad_accumulation") is None:
        raise RuntimeError("V1 training cannot start before the real-data GPU benchmark selects its batch settings")
    seed_everything(int(training["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("V1 requires the verified BF16-capable CUDA GPU")
    model = _prepare_model(config, v1_encoder, device,
                           checkpointing=bool(training["gradient_checkpointing"]))
    train_sampler = StudyBatchSampler(train_data, int(training["image_budget"]),
                                      shuffle=True, seed=int(training["seed"]))
    val_sampler = StudyBatchSampler(valid_data, int(training["image_budget"]),
                                    shuffle=False, seed=int(training["seed"]))
    loader_options = {
        "num_workers": int(training["num_workers"]),
        "pin_memory": bool(training["pin_memory"]),
        "collate_fn": collate_studies,
        "persistent_workers": bool(training["num_workers"]),
        "worker_init_fn": seed_worker if training["num_workers"] else None,
    }
    train_generator = torch.Generator().manual_seed(int(training["data_loader_worker_seed"]))
    val_generator = torch.Generator().manual_seed(int(training["data_loader_worker_seed"]) + 1)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler,
                              generator=train_generator, **loader_options)
    valid_loader = DataLoader(valid_data, batch_sampler=val_sampler,
                              generator=val_generator, **loader_options)
    groups = [
        {"params": model.convnext.parameters(), "lr": float(training["learning_rate_backbone"]), "name": "backbone"},
        {"params": model.classifier.parameters(), "lr": float(training["learning_rate_head"]), "name": "head"},
    ]
    optimizer = build_adamw(groups, training["weight_decay"], fused=training["optimizer_fused"])
    updates_by_epoch = [math.ceil(len(train_sampler) / int(training["grad_accumulation"]))
                        for _ in range(int(training["epochs"]))]
    scheduler, scheduler_info = build_update_scheduler(
        optimizer, updates_by_epoch, int(training["warmup_epochs"]),
        int(training["cosine_end_epoch"]), min_factor=0.05)
    criterion = StandardStudyBCE().to(device)
    history_path = output_dir / "train_log.csv"
    classes_path = output_dir / "per_class_metrics.csv"
    history = _read_rows(history_path) if resume else []
    per_class_rows = _read_rows(classes_path) if resume else []
    start_epoch, best_auc, best_epoch = 0, -math.inf, 0
    audit = _verify_split_audit(config["resolved_data"], train_data, valid_data, verification)
    run_config = {
        "experiment": "V1_ConvNeXtV2_PixelPretrained",
        "data": {key: str(config["resolved_data"][key]) for key in ("train_csv", "val_csv", "image_root")},
        "data_label_semantics": "closed_world",
        "train_subjects": len({subject for subject, _ in train_data.study_keys}),
        "train_studies": len(train_data),
        "train_images": int(sum(train_data.image_counts)),
        "valid_subjects": len({subject for subject, _ in valid_data.study_keys}),
        "valid_studies": len(valid_data),
        "valid_images": int(sum(valid_data.image_counts)),
        "train_validation_subject_overlap": 0,
        "train_manifest_sha256": audit["train_manifest_sha256"],
        "valid_manifest_sha256": audit["valid_manifest_sha256"],
        "data_audit_path": str(PROJECT_DIR / "outputs" / "hybrid" / "shared" / "manifests" / "data_audit.json"),
        "label_mapping": [{"id": index + 1, "name": name}
                          for index, name in enumerate(LABEL_NAMES)],
        "test_split_read": False,
        "model": config["model"],
        "training": training,
        "h1_encoder_verification": verification,
        "scheduler": scheduler_info,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "parameter_counts": {"backbone": count_parameters(model.convnext),
                             "head": count_parameters(model.classifier),
                             "total": count_parameters(model)},
        "input_shape": [1, 3, 320, 320],
        "gap_width": 768,
        "output_shape": [1, 10],
        "study_fusion": "mean raw image logits before sigmoid",
        "loss": "unweighted StandardStudyBCE",
        "uses_vit": False,
        "uses_pretraining_decoder": False,
        "uses_mask_token": False,
        "benchmark_result": config.get("benchmark_result"),
    }
    write_json(output_dir / "config.json", run_config)
    from hybrid.common import create_logger
    logger = create_logger(output_dir / "logs" / "train.log", "v1_convnextv2")
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"V1 resume checkpoint is missing: {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config") != run_config:
            raise ValueError("V1 resume config differs from the saved run")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        train_generator.set_state(state["train_loader_generator_state"].cpu())
        val_generator.set_state(state["val_loader_generator_state"].cpu())
        start_epoch = int(state["epoch"])
        best_auc = float(state["best_macro_auroc"])
        best_epoch = int(state["best_epoch"])
        logger.info("resuming V1 at epoch %d", start_epoch)

    logger.info("V1 device=%s gpu=%s batch=%d accumulation=%d init_sha256=%s",
                device, torch.cuda.get_device_name(device), training["image_budget"],
                training["grad_accumulation"], verification["source_file_sha256"])
    for epoch_index in range(start_epoch, int(training["epochs"])):
        epoch = epoch_index + 1
        train_sampler.set_epoch(epoch_index)
        train_stats = _train_epoch(model, train_loader, train_sampler, optimizer, scheduler,
                                   criterion, {**config, "current_epoch_index": epoch_index},
                                   train_generator, device)
        validation_start = time.perf_counter()
        validation = validate(model, valid_loader, criterion, device, True, torch.bfloat16)
        validation_seconds = time.perf_counter() - validation_start
        score = float(validation["macro_auroc"])
        auprcs = [item["auprc"] for item in validation["per_label"]
                  if math.isfinite(float(item["auprc"]))]
        macro_auprc = float(np.mean(auprcs)) if auprcs else float("nan")
        improved = math.isfinite(score) and score > best_auc
        if improved:
            best_auc, best_epoch = score, epoch
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": float(validation["loss"]),
            "macro_auroc": score,
            "macro_auprc": macro_auprc,
            "macro_f1": float(validation["macro_f1"]),
            "lr_backbone": current_lrs(optimizer)["backbone"],
            "lr_head": current_lrs(optimizer)["head"],
            "optimizer_updates": train_stats["optimizer_updates"],
            "train_images": train_stats["images"],
            "train_studies": train_stats["studies"],
            "train_seconds": train_stats["elapsed_s"],
            "validation_seconds": validation_seconds,
            "validation_studies_per_s": len(valid_data) / validation_seconds if validation_seconds else 0.0,
            "validation_images_per_s": sum(valid_data.image_counts) / validation_seconds if validation_seconds else 0.0,
            "images_per_s": train_stats["images_per_s"],
            "mean_update_seconds": train_stats["mean_update_seconds"],
            "peak_gpu_allocated_gib": train_stats["peak_gpu_allocated_gib"],
            "peak_gpu_reserved_gib": train_stats["peak_gpu_reserved_gib"],
            "parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "best_macro_auroc": best_auc,
        }
        history.append(row)
        for index, item in enumerate(validation["per_label"]):
            per_class_rows.append({"epoch": epoch, "class_name": LABEL_NAMES[index], **item})
        write_csv(history_path, history)
        write_csv(classes_path, per_class_rows)
        predictions = _prediction_rows(validation, LABEL_NAMES)
        write_csv(output_dir / "predictions" / f"val_epoch_{epoch:02d}.csv", predictions)
        _write_epoch_curves(output_dir, history, per_class_rows)
        state = _checkpoint_state(model, optimizer, scheduler, epoch, best_auc,
                                  best_epoch, run_config, train_generator, val_generator)
        _atomic_torch_save(last_path, state)
        if improved:
            _atomic_torch_save(best_path, state)
            write_csv(output_dir / "predictions" / "val_best.csv", predictions)
        if epoch == int(training["epochs"]):
            _atomic_torch_save(output_dir / "epoch20.pth", state)
        logger.info("epoch=%d/20 train_loss=%.6f valid_loss=%.6f macro_auroc=%.6f "
                    "macro_auprc=%.6f macro_f1=%.6f images_s=%.2f best_epoch=%d best_auc=%.6f",
                    epoch, row["train_loss"], row["val_loss"], score, macro_auprc,
                    row["macro_f1"], row["images_per_s"], best_epoch, best_auc)
        _write_training_report(output_dir, run_config, history, per_class_rows,
                               complete=(epoch == int(training["epochs"])))
    _shutdown_loader(train_loader)
    _shutdown_loader(valid_loader)
    return {"completed_epochs": len(history), "best_epoch": best_epoch,
            "best_macro_auroc": best_auc, "output_dir": str(output_dir)}


def _write_training_report(output_dir, config, history, per_class_rows, complete):
    best = max(history, key=lambda row: float(row["macro_auroc"]))
    status = "completed 20 epochs" if complete else f"in progress through epoch {history[-1]['epoch']}"
    lines = [
        "# V1 ConvNeXtV2 Pixel-Pretrained Fine-Tuning Report", "",
        f"- Status: {status}", f"- Best Epoch: {best['epoch']}",
        f"- Best Validation Macro-AUROC: {float(best['macro_auroc']):.6f}",
        f"- Best Train Loss / Valid Loss: {float(best['train_loss']):.6f} / {float(best['val_loss']):.6f}",
        f"- Best Macro-AUPRC / Macro-F1: {float(best['macro_auprc']):.6f} / {float(best['macro_f1']):.6f}",
        f"- H1 pretraining source: `{config['h1_encoder_verification']['selected_source']}`",
        f"- H1 encoder SHA-256: `{config['h1_encoder_verification']['source_file_sha256']}`",
        f"- Parameters: {config['parameter_counts']['total']:,} ({config['parameter_counts']['backbone']:,} backbone + {config['parameter_counts']['head']:,} head)",
        f"- Train/Validation images: {config['train_images']:,}/{config['valid_images']:,}; Studies: {config['train_studies']:,}/{config['valid_studies']:,}; Subjects: {config['train_subjects']:,}/{config['valid_subjects']:,}",
        "- Test split was not read.",
        "", "## Epoch metrics", "",
        "| Epoch | Train Loss | Valid Loss | Macro-AUROC | Macro-AUPRC | Macro-F1 | Images/s | Backbone/Head LR | Peak Alloc/Reserved GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in history:
        lines.append(
            f"| {int(row['epoch'])} | {float(row['train_loss']):.6f} | "
            f"{float(row['val_loss']):.6f} | {float(row['macro_auroc']):.6f} | "
            f"{float(row['macro_auprc']):.6f} | {float(row['macro_f1']):.6f} | "
            f"{float(row['images_per_s']):.2f} | "
            f"{float(row['lr_backbone']):.3g}/{float(row['lr_head']):.3g} | "
            f"{float(row['peak_gpu_allocated_gib']):.3f}/"
            f"{float(row['peak_gpu_reserved_gib']):.3f} |"
        )
    best_classes = [row for row in per_class_rows if int(row["epoch"]) == int(best["epoch"])]
    lines.extend(["", "## Best-epoch class metrics", "",
                  "| Class | AUROC | AUPRC | F1 | Positive | Negative |", "|---|---:|---:|---:|---:|---:|"])
    for row in best_classes:
        lines.append(f"| {row['class_name']} | {float(row['auroc']):.6f} | {float(row['auprc']):.6f} | {float(row['f1']):.6f} | {int(row['n_positive'])} | {int(row['n_negative'])} |")
    Path(output_dir, "v1_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _class_auc_from_predictions(frame):
    targets = frame[[f"target_{name}" for name in LABEL_NAMES]].to_numpy(dtype=np.int64)
    probs = frame[[f"prob_{name}" for name in LABEL_NAMES]].to_numpy(dtype=np.float64)
    aucs = []
    for index in range(len(LABEL_NAMES)):
        if np.unique(targets[:, index]).size != 2:
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(targets[:, index], probs[:, index])))
    finite = [value for value in aucs if math.isfinite(value)]
    return targets, probs, aucs, float(np.mean(finite)) if finite else float("nan")


def _paired_h1_bootstrap(h1_frame, v1_frame, replicates=2000, seed=42):
    h1 = h1_frame.copy()
    v1 = v1_frame.copy()
    for frame in (h1, v1):
        frame["Subject_id"] = frame["Subject_id"].astype(str)
        frame["Study_id"] = frame["Study_id"].astype(str)
    h1_keys = list(zip(h1["Subject_id"], h1["Study_id"]))
    v1_keys = list(zip(v1["Subject_id"], v1["Study_id"]))
    if set(h1_keys) != set(v1_keys) or len(h1_keys) != len(v1_keys):
        raise ValueError("H1 and V1 validation predictions do not cover the same Study keys")
    v1 = v1.set_index(["Subject_id", "Study_id"]).loc[h1_keys].reset_index()
    targets, h1_probs, h1_aucs, h1_macro = _class_auc_from_predictions(h1)
    v1_targets, v1_probs, v1_aucs, v1_macro = _class_auc_from_predictions(v1)
    if not np.array_equal(targets, v1_targets):
        raise ValueError("H1 and V1 validation Study labels differ")
    subjects = np.asarray(sorted(h1["Subject_id"].unique()))
    subject_array = h1["Subject_id"].to_numpy()
    subject_rows = {subject: np.flatnonzero(subject_array == subject) for subject in subjects}
    rng = np.random.default_rng(seed)
    deltas = []
    rejected = 0
    for _ in range(int(replicates)):
        sample_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([subject_rows[subject] for subject in sample_subjects])
        sampled_targets = targets[indices]
        if any(np.unique(sampled_targets[:, index]).size != 2 for index in range(10)):
            rejected += 1
            continue
        class_deltas = []
        for index in range(10):
            class_deltas.append(
                roc_auc_score(sampled_targets[:, index], v1_probs[indices, index]) -
                roc_auc_score(sampled_targets[:, index], h1_probs[indices, index]))
        deltas.append(float(np.mean(class_deltas)))
    if not deltas:
        raise RuntimeError("V1−H1 paired Subject bootstrap produced no valid replicates")
    return {
        "method": "paired bootstrap resampling of Subjects with replacement",
        "seed": int(seed), "requested_replicates": int(replicates),
        "valid_replicates": len(deltas), "rejected_replicates": rejected,
        "subject_count": len(subjects), "study_count": len(h1),
        "h1_macro_auroc": h1_macro, "v1_macro_auroc": v1_macro,
        "delta_v1_minus_h1": v1_macro - h1_macro,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "h1_class_auroc": dict(zip(LABEL_NAMES, h1_aucs)),
        "v1_class_auroc": dict(zip(LABEL_NAMES, v1_aucs)),
        "class_delta_auroc": {name: v1_aucs[i] - h1_aucs[i]
                              for i, name in enumerate(LABEL_NAMES)},
    }


def _draw_class_comparison(path, h1_aucs, v1_aucs):
    width, height = 1500, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
        title = ImageFont.truetype("arial.ttf", 25)
    except OSError:
        font = title = ImageFont.load_default()
    draw.text((100, 25), "Best-checkpoint per-class validation AUROC: H1 vs V1", fill="black", font=title)
    left, top, right, bottom = 100, 95, 1450, 660
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    for tick in range(6):
        value = tick / 5
        y = bottom - int(value * (bottom - top))
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((55, y - 9), f"{value:.1f}", fill="black", font=font)
    band = (right - left) / len(LABEL_NAMES)
    for index, name in enumerate(LABEL_NAMES):
        center = left + band * (index + 0.5)
        for offset, value, color in ((-13, h1_aucs[index], (44, 160, 44)),
                                     (13, v1_aucs[index], (31, 119, 180))):
            if math.isfinite(float(value)):
                y = bottom - int(float(value) * (bottom - top))
                draw.rectangle((center + offset - 8, y, center + offset + 8, bottom), fill=color)
        short = name.replace(" ", "\n")
        draw.multiline_text((center - 50, bottom + 10), short, fill="black", font=font,
                            align="center", spacing=0)
    draw.rectangle((left + 5, top + 8, left + 23, top + 26), fill=(44, 160, 44))
    draw.text((left + 30, top + 7), "H1", fill="black", font=font)
    draw.rectangle((left + 90, top + 8, left + 108, top + 26), fill=(31, 119, 180))
    draw.text((left + 115, top + 7), "V1", fill="black", font=font)
    image.save(path)


def _draw_throughput_comparison(path, h1_images_s, v1_images_s, h1_memory, v1_memory):
    width, height = 1200, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        title = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = title = ImageFont.load_default()
    draw.text((90, 28), "H1 vs V1: training throughput and peak allocated GPU memory", fill="black", font=title)
    for panel, top, values, label, unit in (
        (0, 115, [h1_images_s, v1_images_s], "End-to-end train throughput", "images/s"),
        (1, 405, [h1_memory, v1_memory], "Peak allocated memory", "GiB"),
    ):
        draw.text((100, top - 28), f"{label} ({unit})", fill="black", font=font)
        maximum = max(values) * 1.15 if max(values) > 0 else 1.0
        base = top + 175
        bar_width = 150
        for index, (name, value, color) in enumerate((
                ("Hybrid H1", values[0], (44, 160, 44)), ("V1", values[1], (31, 119, 180)))):
            height_value = int(145 * float(value) / maximum)
            x = 300 + index * 400
            draw.rectangle((x, base - height_value, x + bar_width, base), fill=color)
            draw.text((x + 30, base + 10), name, fill="black", font=font)
            draw.text((x + 40, base - height_value - 30), f"{value:.2f}", fill="black", font=font)
        draw.line((240, base, 1000, base), fill="black", width=2)
    image.save(path)


def compare_v1_to_h1(config, output_dir, v1_history, v1_classes):
    h1_dir = PROJECT_DIR / config["comparison"]["h1_run_dir"]
    h1_log = pd.read_csv(h1_dir / "train_log.csv")
    h1_predictions_path = h1_dir / "predictions" / "val_best.csv"
    v1_predictions_path = Path(output_dir) / "predictions" / "val_best.csv"
    if not h1_predictions_path.is_file() or not v1_predictions_path.is_file():
        raise FileNotFoundError("Paired H1/V1 comparison requires both best-checkpoint validation prediction caches")
    h1_pred = pd.read_csv(h1_predictions_path, dtype={"Subject_id": "string", "Study_id": "string"})
    v1_pred = pd.read_csv(v1_predictions_path, dtype={"Subject_id": "string", "Study_id": "string"})
    bootstrap = _paired_h1_bootstrap(
        h1_pred, v1_pred,
        int(config["comparison"]["paired_subject_bootstrap_replicates"]),
        int(config["comparison"]["paired_subject_bootstrap_seed"]))
    write_json(Path(output_dir) / "v1_vs_h1_paired_subject_bootstrap.json", bootstrap)

    h1_best_epoch = int(h1_log.loc[h1_log["macro_auroc"].idxmax(), "epoch"])
    h1_best = h1_log.loc[h1_log["macro_auroc"].idxmax()]
    h1_classes_frame = pd.read_csv(h1_dir / "per_class_metrics.csv")
    h1_classes = h1_classes_frame[h1_classes_frame["epoch"] == h1_best_epoch]
    h1_auc_by_name = {row["class_name"]: float(row["auroc"])
                      for _, row in h1_classes.iterrows()}
    v1_best = max(v1_history, key=lambda row: float(row["macro_auroc"]))
    v1_best_class_rows = [row for row in v1_classes if int(row["epoch"]) == int(v1_best["epoch"])]
    v1_auc_by_name = {row["class_name"]: float(row["auroc"]) for row in v1_best_class_rows}
    h1_auc = [h1_auc_by_name[name] for name in LABEL_NAMES]
    v1_auc = [v1_auc_by_name[name] for name in LABEL_NAMES]
    _draw_class_comparison(Path(output_dir) / "per_class_auroc_comparison.png", h1_auc, v1_auc)

    h1_best_images_s = float(h1_best["images_per_s"])
    v1_best_images_s = float(v1_best["images_per_s"])
    h1_best_alloc = float(h1_best["peak_gpu_allocated_gib"])
    v1_best_alloc = float(v1_best["peak_gpu_allocated_gib"])
    h1_inference_ms = (1000.0 / float(h1_best["validation_studies_per_s"])
                       if float(h1_best["validation_studies_per_s"]) > 0 else float("nan"))
    v1_inference_ms = (1000.0 / float(v1_best["validation_studies_per_s"])
                       if float(v1_best["validation_studies_per_s"]) > 0 else float("nan"))
    v1_parameters = int(v1_best["parameters"])
    parameter_saving_pct = 100.0 * (49_620_490 - v1_parameters) / 49_620_490
    throughput_change_pct = 100.0 * (v1_best_images_s / h1_best_images_s - 1.0)
    h1_epochs = h1_log.set_index("epoch")
    v1_epochs = {int(row["epoch"]): row for row in v1_history}
    _draw_throughput_comparison(Path(output_dir) / "gpu_throughput_comparison.png",
                                h1_best_images_s, v1_best_images_s,
                                h1_best_alloc, v1_best_alloc)

    lines = [
        "# V1 vs Hybrid H1 Comparison", "",
        "V1 is a single-branch ConvNeXtV2-Tiny initialized from the final H1 pixel-pretrained CNN encoder. H1 is the completed ConvNeXtV2 + ViT pixel-pretrained hybrid.",
        "The difference includes ViT removal, classifier input width/head, and whether the CNN was jointly optimized with a ViT during downstream fine-tuning; this is a whole-solution comparison, not an isolated causal estimate of the ViT branch.",
        "", "## Best validation results", "",
        "| Metric | Hybrid H1 | V1 | V1−H1 |",
        "|---|---:|---:|---:|",
        f"| Best epoch | {h1_best_epoch} | {v1_best['epoch']} | — |",
        f"| Macro-AUROC | {float(h1_best['macro_auroc']):.6f} | {float(v1_best['macro_auroc']):.6f} | {float(v1_best['macro_auroc'])-float(h1_best['macro_auroc']):+.6f} |",
        f"| Pneumonia AUROC | {h1_auc_by_name['Pneumonia']:.6f} | {v1_auc_by_name['Pneumonia']:.6f} | {v1_auc_by_name['Pneumonia']-h1_auc_by_name['Pneumonia']:+.6f} |",
        f"| Lung Opacity AUROC | {h1_auc_by_name['Lung Opacity']:.6f} | {v1_auc_by_name['Lung Opacity']:.6f} | {v1_auc_by_name['Lung Opacity']-h1_auc_by_name['Lung Opacity']:+.6f} |",
        f"| Parameters | 49,620,490 | {v1_parameters:,} | {v1_parameters-49_620_490:+,} |",
        f"| Parameter reduction | — | — | {parameter_saving_pct:.2f}% |",
        f"| Best-epoch end-to-end images/s | {h1_best_images_s:.2f} | {v1_best_images_s:.2f} | {v1_best_images_s-h1_best_images_s:+.2f} |",
        f"| Throughput change vs H1 | — | — | {throughput_change_pct:+.2f}% |",
        f"| Peak allocated GPU memory (GiB) | {h1_best_alloc:.3f} | {v1_best_alloc:.3f} | {v1_best_alloc-h1_best_alloc:+.3f} |",
        f"| Validation inference latency (ms/Study) | {h1_inference_ms:.3f} | {v1_inference_ms:.3f} | {v1_inference_ms-h1_inference_ms:+.3f} |",
        "", "## Per-class AUROC difference at each model's best checkpoint", "",
        "| Disease | H1 AUROC | V1 AUROC | V1−H1 |",
        "|---|---:|---:|---:|",
    ]
    for name in LABEL_NAMES:
        lines.append(f"| {name} | {bootstrap['h1_class_auroc'][name]:.6f} | {bootstrap['v1_class_auroc'][name]:.6f} | {bootstrap['class_delta_auroc'][name]:+.6f} |")
    lines.extend([
        "", "## Validation trend after epoch 7", "",
        "| Model | AUROC epoch 7 | AUROC epoch 20 | AUROC change | Validation−Train loss gap epoch 7 | gap epoch 20 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, log in (("Hybrid H1", h1_epochs), ("V1", v1_epochs)):
        e7, e20 = log.loc[7], log.loc[20]
        e7_auc, e20_auc = float(e7["macro_auroc"]), float(e20["macro_auroc"])
        e7_gap = float(e7["val_loss"]) - float(e7["train_loss"])
        e20_gap = float(e20["val_loss"]) - float(e20["train_loss"])
        lines.append(f"| {name} | {e7_auc:.6f} | {e20_auc:.6f} | {e20_auc-e7_auc:+.6f} | {e7_gap:+.6f} | {e20_gap:+.6f} |")
    lines.extend([
        "", "## Paired Subject Bootstrap", "",
        f"Resampled {bootstrap['subject_count']} Subjects over the same {bootstrap['study_count']} Validation Studies; seed={bootstrap['seed']}, requested replicates={bootstrap['requested_replicates']}, valid={bootstrap['valid_replicates']}, rejected for missing polarity={bootstrap['rejected_replicates']}.",
        f"- V1−H1 Macro-AUROC: {bootstrap['delta_v1_minus_h1']:+.6f}",
        f"- Paired 95% CI: [{bootstrap['ci95_low']:+.6f}, {bootstrap['ci95_high']:+.6f}]",
        "", "## Interpretation constraints", "",
        "- Both systems use the same Validation split and H1 pixel-pretrained ConvNeXtV2 initialization for the CNN branch.",
        "- V1 does not establish the benefit of pixel pretraining versus ImageNet-only ConvNeXtV2; that would require the V0 control, which this task did not run.",
        "- Validation predictions were used for model selection and paired analysis. The official Test split and labels were not read.",
        "- H1 reference results: Macro-AUROC 0.793342, best epoch 6, Pneumonia AUROC 0.722252, Lung Opacity AUROC 0.712168.",
        "- H1 report confirms 20 supervised epochs and H1 self-supervised use of official Train images only.",
    ])
    Path(output_dir, "v1_vs_h1_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bootstrap


def run_all(config_path="configs/v1_convnextv2.yaml", resume=False):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = _read_yaml(config_path)
    _validate_contract(config)
    output_dir = PROJECT_DIR / config["output_dir"]
    if not resume and any((output_dir / name).exists() for name in ("best.pth", "last.pth")):
        raise FileExistsError(f"Refusing to overwrite existing V1 training checkpoints in {output_dir}")
    config["resolved_data"] = _resolve_paths(_read_yaml(config["data"]["data_config"]))

    v1_encoder, verification = _load_h1_encoder(config, output_dir)
    train_data, valid_data = _build_datasets(config["resolved_data"])
    _verify_split_audit(config["resolved_data"], train_data, valid_data, verification)
    smoke = run_smoke_test(config, v1_encoder, train_data, valid_data, output_dir)
    write_json(output_dir / "v1_smoke_test.json", smoke)
    print("V1 Smoke Test passed")

    selected, benchmark_rows = run_benchmark(config, v1_encoder, train_data,
                                             output_dir, config_path)
    write_json(output_dir / "config.json", config)
    if config["training"].get("image_budget") is None or config["training"].get("grad_accumulation") is None:
        raise RuntimeError("V1 batch configuration was not selected by the benchmark")

    training_result = _train(config, config_path, v1_encoder, verification,
                             train_data, valid_data, output_dir, resume=resume)
    final_config = _read_yaml(config_path)
    final_config["resolved_data"] = config["resolved_data"]
    # Read the persisted training history to keep final comparisons tied to the saved run.
    history = _read_rows(output_dir / "train_log.csv")
    per_class = _read_rows(output_dir / "per_class_metrics.csv")
    bootstrap = compare_v1_to_h1(final_config, output_dir, history, per_class)
    return {"training": training_result, "selected_benchmark": selected,
            "benchmark_candidates": len(benchmark_rows), "paired_bootstrap": bootstrap,
            "comparison_report": str(output_dir / "v1_vs_h1_comparison.md")}


def main():
    parser = argparse.ArgumentParser(description="Run V1 ConvNeXtV2 fine-tuning from H1 pixel-pretrained CNN weights")
    parser.add_argument("--config", default="configs/v1_convnextv2.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "verify", "smoke", "benchmark", "train", "compare"), default="all")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = _read_yaml(config_path)
    _validate_contract(config, require_benchmark=args.stage == "train")
    output_dir = PROJECT_DIR / config["output_dir"]
    if args.stage == "all" and not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing V1 artifacts in {output_dir}; choose a stage or resume explicitly")
    config["resolved_data"] = _resolve_paths(_read_yaml(config["data"]["data_config"]))
    if args.stage == "all":
        result = run_all(config_path, resume=args.resume)
    elif args.stage == "verify":
        _, _, verification = _load_h1_encoder(config, output_dir)
        result = verification
    elif args.stage in {"smoke", "benchmark", "train"}:
        v1_encoder, verification = _load_h1_encoder(config, output_dir)
        train_data, valid_data = _build_datasets(config["resolved_data"])
        _verify_split_audit(config["resolved_data"], train_data, valid_data, verification)
        if args.stage == "smoke":
            result = run_smoke_test(config, v1_encoder, train_data, valid_data, output_dir)
        elif args.stage == "benchmark":
            result = run_benchmark(config, v1_encoder, train_data, output_dir, config_path)
        else:
            result = _train(config, config_path, v1_encoder, verification,
                            train_data, valid_data, output_dir, resume=args.resume)
    else:
        history = _read_rows(output_dir / "train_log.csv")
        per_class = _read_rows(output_dir / "per_class_metrics.csv")
        result = compare_v1_to_h1(config, output_dir, history, per_class)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
