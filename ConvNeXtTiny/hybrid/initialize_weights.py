import argparse
import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import timm
import torch
import timm.models._builder as timm_builder
import timm.models._hub as timm_hub

from hybrid.common import PROJECT_DIR, count_parameters, sha256_file, sha256_state_dict, write_json


CONVNEXT_NAME = "convnextv2_tiny.fcmae_ft_in1k"
VIT_NAME = "vit_small_patch16_224.augreg_in1k"
IMAGE_SIZE = 320


class _TimmLoadCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _load_report(messages):
    missing, unexpected = [], []
    for message in messages:
        match = re.search(r"Missing keys \((.*?)\)", message)
        if match and match.group(1):
            missing.extend(key.strip() for key in match.group(1).split(","))
        match = re.search(r"Unexpected keys \((.*?)\)", message)
        if match and match.group(1):
            unexpected.extend(key.strip() for key in match.group(1).split(","))
    return {"missing_keys": missing, "unexpected_keys": unexpected,
            "loader_messages": messages}


def _create_pretrained_model(name, logger):
    capture = _TimmLoadCapture()
    source_paths = []
    original_load_from_url = timm_builder.load_state_dict_from_url
    original_download_cached = timm_builder.download_cached_file
    original_hf_download = timm_hub.hf_hub_download

    def tracked_load_from_url(url, *args, **kwargs):
        state = original_load_from_url(url, *args, **kwargs)
        cache_dir = kwargs.get("model_dir") or (Path(torch.hub.get_dir()) / "checkpoints")
        source_paths.append(Path(cache_dir) / Path(urlparse(url).path).name)
        return state

    def tracked_download_cached(url, *args, **kwargs):
        source_path = original_download_cached(url, *args, **kwargs)
        source_paths.append(Path(source_path))
        return source_path

    def tracked_hf_download(*args, **kwargs):
        source_path = original_hf_download(*args, **kwargs)
        source_paths.append(Path(source_path))
        return source_path

    timm_builder.load_state_dict_from_url = tracked_load_from_url
    timm_builder.download_cached_file = tracked_download_cached
    timm_hub.hf_hub_download = tracked_hf_download
    logger.addHandler(capture)
    try:
        model = timm.create_model(name, pretrained=True)
    finally:
        logger.removeHandler(capture)
        timm_builder.load_state_dict_from_url = original_load_from_url
        timm_builder.download_cached_file = original_download_cached
        timm_hub.hf_hub_download = original_hf_download
    report = _load_report(capture.messages)
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"Pretrained weight mismatch for {name}: {report}")
    unique_paths = []
    for source_path in source_paths:
        if source_path.is_file() and all(not source_path.samefile(existing) for existing in unique_paths):
            unique_paths.append(source_path)
    if not unique_paths:
        raise RuntimeError(f"timm loaded {name}, but its source checkpoint file could not be identified for SHA-256")
    report["source_files"] = [{"path": str(path), "size_bytes": path.stat().st_size,
                               "sha256": sha256_file(path)} for path in unique_paths]
    model.eval()
    return model, report


def initialize_weights(output_path=None):
    output_path = Path(output_path or PROJECT_DIR / "outputs/hybrid/shared/imagenet_initial.pth")
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() or metadata_path.exists():
        if not output_path.is_file() or not metadata_path.is_file():
            raise FileExistsError("Partial ImageNet initialization artifacts exist; refusing to overwrite")
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(output_path)
        if existing_metadata.get("initial_checkpoint_sha256") != actual_hash:
            raise RuntimeError("Shared ImageNet initialization hash does not match its metadata")
        if (existing_metadata.get("convnext", {}).get("model_name") != CONVNEXT_NAME or
                existing_metadata.get("vit", {}).get("model_name") != VIT_NAME):
            raise RuntimeError("Existing ImageNet initialization metadata names different model weights")
        print(f"Using existing shared initialization: {output_path}")
        print(f"SHA-256: {actual_hash}")
        return str(output_path)

    logger = logging.getLogger("timm.models._builder")
    logger.setLevel(logging.INFO)
    cnn, cnn_report = _create_pretrained_model(CONVNEXT_NAME, logger)
    vit, vit_report = _create_pretrained_model(VIT_NAME, logger)

    cnn_cfg = cnn.pretrained_cfg
    vit_cfg = vit.pretrained_cfg
    original_cls_pos = vit.pos_embed[:, :vit.num_prefix_tokens].detach().clone()
    original_grid = tuple(vit.patch_embed.grid_size)
    if original_grid != (14, 14) or vit.num_prefix_tokens != 1:
        raise RuntimeError(f"Unexpected pretrained ViT token layout: grid={original_grid}, "
                           f"prefix_tokens={vit.num_prefix_tokens}")
    vit.set_input_size(img_size=IMAGE_SIZE)
    if tuple(vit.patch_embed.grid_size) != (20, 20):
        raise RuntimeError(f"Expected 20x20 ViT patch grid, got {vit.patch_embed.grid_size}")
    if not torch.equal(vit.pos_embed[:, :vit.num_prefix_tokens], original_cls_pos):
        raise RuntimeError("ViT CLS position embedding changed during patch-grid interpolation")
    vit_pos_report = {
        "source_grid": list(original_grid),
        "target_grid": list(vit.patch_embed.grid_size),
        "source_patch_tokens": 196,
        "target_patch_tokens": 400,
        "prefix_tokens": vit.num_prefix_tokens,
        "method": "timm VisionTransformer.set_input_size bicubic absolute-position resampling",
        "cls_position_embedding_preserved": True,
    }

    cnn.reset_classifier(0)
    vit.reset_classifier(0)
    cnn.head = torch.nn.Identity()
    vit.head = torch.nn.Identity()
    cnn_state = {key: value.detach().cpu() for key, value in cnn.state_dict().items()}
    vit_state = {key: value.detach().cpu() for key, value in vit.state_dict().items()}
    payload = {
        "format_version": 1,
        "convnext_model_name": CONVNEXT_NAME,
        "vit_model_name": VIT_NAME,
        "image_size": IMAGE_SIZE,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "convnext_state_dict": cnn_state,
        "vit_state_dict": vit_state,
        "metadata": {
            "convnext": {
                "model_name": CONVNEXT_NAME,
                "origin_url": cnn_cfg.get("origin_url"),
                "source_url": cnn_cfg.get("url"),
                "hf_hub_id": cnn_cfg.get("hf_hub_id"),
                "pretraining_history": "ConvNeXtV2 FCMAE pretraining followed by ImageNet-1K supervised fine-tuning, as named by timm tag fcmae_ft_in1k",
                "license": cnn_cfg.get("license"),
                "input_mean": list(cnn_cfg["mean"]),
                "input_std": list(cnn_cfg["std"]),
                "loaded_parameter_count": count_parameters(cnn),
                "loaded_state_sha256": sha256_state_dict(cnn_state),
                "load_report": cnn_report,
            },
            "vit": {
                "model_name": VIT_NAME,
                "source_url": vit_cfg.get("url"),
                "hf_hub_id": vit_cfg.get("hf_hub_id"),
                "pretraining_history": "ImageNet-1K supervised ViT-Small/16 with the AugReg recipe, as named by timm tag augreg_in1k",
                "license": vit_cfg.get("license"),
                "input_mean": list(vit_cfg["mean"]),
                "input_std": list(vit_cfg["std"]),
                "loaded_parameter_count": count_parameters(vit),
                "loaded_state_sha256": sha256_state_dict(vit_state),
                "load_report": vit_report,
                "position_embedding": vit_pos_report,
            },
            "heads_removed_after_loading": True,
            "all_experiments_load_this_same_initial_file": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    file_hash = sha256_file(output_path)
    report = {**payload["metadata"], "initial_checkpoint": str(output_path),
              "initial_checkpoint_sha256": file_hash,
              "timm_version": timm.__version__, "torch_version": torch.__version__}
    write_json(metadata_path, report)
    print(f"Saved shared ImageNet initialization: {output_path}")
    print(f"SHA-256: {file_hash}")
    print(f"ConvNeXt load missing/unexpected: {cnn_report['missing_keys']}/{cnn_report['unexpected_keys']}")
    print(f"ViT load missing/unexpected: {vit_report['missing_keys']}/{vit_report['unexpected_keys']}")
    print(f"ViT position embedding: {vit_pos_report}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    initialize_weights(args.output)


if __name__ == "__main__":
    main()
