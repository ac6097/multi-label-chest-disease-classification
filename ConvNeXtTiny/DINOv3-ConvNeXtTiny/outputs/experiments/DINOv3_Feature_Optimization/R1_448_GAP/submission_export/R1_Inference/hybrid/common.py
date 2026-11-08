import csv
import hashlib
import json
import logging
import math
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent


class StandardStudyBCE(torch.nn.Module):
    """Unweighted BCE averaged across labels per Study, then equally across Studies."""

    def forward(self, logits, targets, known=None):
        if logits.shape != targets.shape:
            raise ValueError("Study logits and targets must have the same shape")
        if known is not None and known.shape != targets.shape:
            raise ValueError("Known-label mask and targets must have the same shape")
        element = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), reduction="none")
        return element.mean(dim=1).mean()


def read_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def resolve_data_config(path="configs/hybrid_data.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    config = read_yaml(path)
    if int(config.get("image_size", 0)) != 320:
        raise ValueError("The Hybrid ConvNeXtV2–ViT experiment is fixed to 320x320 input")
    resolved = dict(config)
    for key in ("image_root", "train_csv", "val_csv"):
        value = Path(config[key])
        resolved[key] = value if value.is_absolute() else (path.parent.parent / value).resolve()
    return resolved


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def json_sanitize(value):
    if isinstance(value, dict):
        return {key: json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return json_sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_sanitize(payload), indent=2, ensure_ascii=False,
                                    allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_state_dict(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def create_logger(log_path, name):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def device_and_amp(amp_enabled=True, amp_dtype="bfloat16"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Hybrid experiments require a CUDA GPU")
    if amp_dtype != "bfloat16":
        raise ValueError("Hybrid experiments are configured for BF16 without an implicit precision fallback")
    dtype = torch.bfloat16
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16; no precision fallback is configured")
    return device, bool(amp_enabled), dtype


def build_adamw(groups, weight_decay, fused=True):
    try:
        return torch.optim.AdamW(groups, weight_decay=float(weight_decay), fused=bool(fused))
    except (TypeError, RuntimeError) as exc:
        if fused:
            raise RuntimeError("Fused AdamW was selected but failed to initialize") from exc
        raise


def update_counts_for_epoch(batch_count, accumulation):
    if batch_count < 1 or accumulation < 1:
        raise ValueError("batch_count and accumulation must both be positive")
    return math.ceil(batch_count / accumulation)


def build_update_scheduler(optimizer, updates_by_epoch, warmup_epochs=2,
                           cosine_end_epoch=20, min_factor=0.05):
    epochs = len(updates_by_epoch)
    if not 0 < warmup_epochs < cosine_end_epoch <= epochs:
        raise ValueError("Expected 0 < warmup_epochs < cosine_end_epoch <= epochs")
    if any(int(count) < 1 for count in updates_by_epoch):
        raise ValueError("Each epoch must contain at least one optimizer update")
    total_steps = sum(int(count) for count in updates_by_epoch)
    warmup_steps = sum(int(count) for count in updates_by_epoch[:warmup_epochs])
    cosine_end_step = sum(int(count) for count in updates_by_epoch[:cosine_end_epoch])
    cosine_steps = cosine_end_step - warmup_steps

    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        cosine_index = step - warmup_steps
        if cosine_index < cosine_steps:
            progress = cosine_index / max(cosine_steps - 1, 1)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    metadata = {
        "unit": "optimizer_update",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "cosine_end_step": cosine_end_step,
        "cosine_steps": cosine_steps,
        "warmup_epochs": int(warmup_epochs),
        "cosine_end_epoch": int(cosine_end_epoch),
        "min_factor": float(min_factor),
        "updates_per_epoch": [int(count) for count in updates_by_epoch],
    }
    return scheduler, metadata


def current_lrs(optimizer):
    return {group.get("name", f"group_{index}"): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)}


def autocast_context(device, enabled=True, dtype=torch.bfloat16):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def count_parameters(module, trainable_only=False):
    return sum(parameter.numel() for parameter in module.parameters()
               if not trainable_only or parameter.requires_grad)

