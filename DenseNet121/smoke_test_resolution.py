"""Validate strict Epoch 6 initialization and one real 448 Study update."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from datasets import StudyDataset, build_transform, collate_studies
from engine.losses import MaskedBCEWithLogitsLoss
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths
from utils.seed import seed_everything


BASELINE_SHA256 = "8162986883188d168b013ad926f70eccddd559f61d9c922b05ab5db9eff7d477"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_DIR / "configs/experiments/densenet_448_data.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs/experiments/densenet_448_smoke"))
    args = parser.parse_args()

    seed_everything(42)
    data_path = resolve_config_path(args.data_config)
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    if int(data_cfg["image_size"]) != 448:
        raise ValueError("Smoke test requires image_size=448")
    checkpoint_path = resolve_config_path(args.checkpoint)
    digest = sha256(checkpoint_path)
    if digest.lower() != BASELINE_SHA256:
        raise ValueError(f"Baseline checkpoint SHA-256 mismatch: {digest}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = dict(state["config"]["model_config"])
    if int(state.get("epoch", 0)) != 6:
        raise ValueError("The source checkpoint is not the Epoch 6 best")
    if (model_cfg.get("head") != "gap" or model_cfg.get("cbam", False) or
            model_cfg.get("study_fusion") != "logit_mean" or model_cfg.get("num_labels") != 10):
        raise ValueError("The source checkpoint is not the standard DenseNet GAP model")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested 448 GPU smoke test")
    device = torch.device("cuda")
    model_cfg["pretrained"] = False
    model = ChestXrayClassifier(**{key: value for key, value in model_cfg.items()
                                   if key != "name"}).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.train()

    train_data = StudyDataset(
        data_cfg["train_csv"], data_cfg["image_root"],
        transform=build_transform(448, train=True),
        label_column=data_cfg["label_column"], label_semantics=data_cfg["label_semantics"],
        image_id_column=data_cfg["image_id_column"],
        subject_column=data_cfg["subject_column"], study_column=data_cfg["study_column"])
    batch = collate_studies([train_data[0]])
    images = batch["images"].to(device)
    study_index = batch["study_index"].to(device)
    labels = batch["labels"].to(device)
    known = batch["known"].to(device)
    amp_enabled = True
    amp_dtype = torch.bfloat16
    if amp_enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is not supported by the selected GPU")

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 5e-6},
        {"params": model.vector_classifier.parameters(), "lr": 5e-5},
    ], weight_decay=1e-4)
    criterion = MaskedBCEWithLogitsLoss().to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(images, study_index, batch["num_studies"])
    if tuple(logits.shape) != (batch["num_studies"], 10):
        raise RuntimeError(f"Unexpected Study logits shape: {tuple(logits.shape)}")
    loss = criterion(logits.float(), labels, known)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss.detach().cpu().item()}")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    output_dir = resolve_config_path(args.output_dir)
    result_path = output_dir / "smoke_result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing smoke result: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "input_size": int(data_cfg["image_size"]),
        "study_image_count": int(images.shape[0]),
        "study_logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "strict_state_load": True,
        "baseline_checkpoint_sha256": digest,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**result, "output": str(result_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
