import argparse
import json
import math
from pathlib import Path

import torch

from datasets import StudyDataset
from datasets.hybrid import OfficialTrainImageDataset, build_hybrid_transform
from hybrid.backbone_factory import build_hybrid_backbone
from hybrid.common import (PROJECT_DIR, autocast_context, build_adamw,
                           count_parameters, device_and_amp, read_yaml, StandardStudyBCE,
                           resolve_data_config, seed_everything, sha256_file,
                           write_json)
from hybrid.pretrain import _iter_group_parameters, _grad_norm
from model.hybrid import HybridClassifier
from model.hybrid_pretraining import (LatentReconstructionPretrainer,
                                      PixelReconstructionPretrainer,
                                      mask_to_vit_grid, sample_spatial_mask)


def _one_update(model, output, loss, device, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("Smoke test produced a zero or non-finite gradient norm")
    before = next(model.parameters()).detach().clone()
    optimizer.step()
    changed = float((next(model.parameters()).detach() - before).float().norm())
    if changed <= 0 or not math.isfinite(changed):
        raise RuntimeError("Smoke test optimizer did not update a model parameter")
    return gradient_norm, changed


def run_smoke_test(experiments_config_path="configs/hybrid_experiments.yaml"):
    experiments = read_yaml(experiments_config_path)
    data_config = resolve_data_config(experiments["data_config"])
    initial_path = Path(experiments["initial_weights"])
    if not initial_path.is_absolute():
        initial_path = PROJECT_DIR / initial_path
    if not initial_path.is_file():
        raise FileNotFoundError("Run hybrid.initialize_weights before the Smoke Test")
    initial_hash = sha256_file(initial_path)
    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    seed_everything(42)
    device, amp_enabled, amp_dtype = device_and_amp(True, "bfloat16")
    build_backbone = lambda: build_hybrid_backbone(
        initial_path, map_location="cpu", initial_checkpoint_payload=initial_payload,
        initial_checkpoint_sha256=initial_hash)
    initial_backbone = build_backbone()

    raw_transform = build_hybrid_transform(data_config["image_size"], train=True)
    study_data = StudyDataset(
        data_config["train_csv"], data_config["image_root"], transform=raw_transform,
        label_column=data_config["label_column"],
        label_semantics=data_config["label_semantics"],
        image_id_column=data_config["image_id_column"],
        subject_column=data_config["subject_column"],
        study_column=data_config["study_column"],
    )
    candidate_indices = [index for index, study in enumerate(study_data.studies)
                         if len(study["image_paths"]) >= 2]
    if not candidate_indices:
        raise RuntimeError("No multi-image Train Study is available to verify Study Logit Mean")
    study_index = min(candidate_indices,
                      key=lambda index: len(study_data.studies[index]["image_paths"]))
    sample = study_data[study_index]
    images = sample["images"].to(device)
    if images.ndim != 4 or images.shape[1:] != (3, 320, 320):
        raise RuntimeError(f"Raw train transform returned unexpected image shape: {tuple(images.shape)}")
    if images.min() < 0 or images.max() > 1:
        raise RuntimeError("Hybrid preprocessing must return raw image pixels in [0,1]")
    if len(sample["image_ids"]) != images.shape[0] or images.shape[0] < 2:
        raise RuntimeError("The selected multi-image Study did not preserve all of its images")
    valid_data = StudyDataset(
        data_config["val_csv"], data_config["image_root"],
        transform=build_hybrid_transform(data_config["image_size"], train=False),
        label_column=data_config["label_column"],
        label_semantics=data_config["label_semantics"],
        image_id_column=data_config["image_id_column"],
        subject_column=data_config["subject_column"],
        study_column=data_config["study_column"],
    )
    valid_sample = valid_data[0]
    valid_image = valid_sample["images"][0]
    if tuple(valid_image.shape) != (3, 320, 320) or valid_image.min() < 0 or valid_image.max() > 1:
        raise RuntimeError("Validation preprocessing must return uncropped RGB tensors in [0,1] at 320x320")

    model = HybridClassifier(initial_backbone).to(device)
    model.eval()
    with torch.no_grad(), autocast_context(device, amp_enabled, amp_dtype):
        cnn_stage3, cnn_stage4 = model.backbone.convnext_features(images[:1])
        vit_tokens = model.backbone.vit_features(images[:1])
        per_image_logits = model.image_logits(images)
        indices = torch.zeros(images.shape[0], dtype=torch.long, device=device)
        study_logits = model(images, indices, num_studies=1)
    if tuple(cnn_stage3.shape[1:]) != (384, 20, 20):
        raise RuntimeError(f"ConvNeXtV2 Stage 3 shape mismatch: {tuple(cnn_stage3.shape)}")
    if tuple(cnn_stage4.shape[1:]) != (768, 10, 10):
        raise RuntimeError(f"ConvNeXtV2 Stage 4 shape mismatch: {tuple(cnn_stage4.shape)}")
    if tuple(vit_tokens.shape[1:]) != (401, 384):
        raise RuntimeError(f"ViT positional grid/CLS shape mismatch: {tuple(vit_tokens.shape)}")
    if tuple(per_image_logits.shape) != (images.shape[0], 10) or tuple(study_logits.shape) != (1, 10):
        raise RuntimeError("Hybrid classifier must return 10 logits per image and Study")
    if not torch.allclose(study_logits[0].float(), per_image_logits.float().mean(dim=0),
                          atol=2e-3, rtol=2e-3):
        raise RuntimeError("Study fusion is not arithmetic mean of raw image logits")
    shape_info = {
        "convnext_stage3": list(cnn_stage3.shape),
        "convnext_stage4": list(cnn_stage4.shape),
        "vit_tokens": list(vit_tokens.shape),
        "image_logits": list(per_image_logits.shape),
        "study_logits": list(study_logits.shape),
    }

    class_output = Path(experiments["outputs"]["benchmark"])
    if not class_output.is_absolute():
        class_output = PROJECT_DIR / class_output
    smoke_dir = class_output.parent / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}

    model.train()
    cls_optimizer = build_adamw([
        {"params": model.backbone.convnext.parameters(), "lr": 1e-5},
        {"params": model.backbone.vit.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 2e-4},
    ], 0.01, fused=True)
    known = sample["known"].unsqueeze(0).to(device)
    targets = sample["labels"].unsqueeze(0).to(device)
    cls_optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, amp_enabled, amp_dtype):
        logits = model(images, indices, num_studies=1)
        cls_loss = StandardStudyBCE().to(device)(logits, targets, known)
    cls_grad, cls_change = _one_update(model, None, cls_loss, device, cls_optimizer)
    cls_state = {"model": model.state_dict(), "optimizer": cls_optimizer.state_dict(),
                 "loss": float(cls_loss.detach())}
    h0_checkpoint_path = smoke_dir / "h0_supervised_step.pth"
    torch.save(cls_state, h0_checkpoint_path)
    loaded_h0_state = torch.load(h0_checkpoint_path, map_location="cpu", weights_only=False)
    clone = HybridClassifier(build_backbone()).to(device)
    clone.load_state_dict(loaded_h0_state["model"], strict=True)
    metrics["h0"] = {"loss": float(cls_loss.detach()), "gradient_norm": cls_grad,
                      "parameter_change": cls_change, "checkpoint_reload": True}
    del clone, cls_optimizer, model, initial_backbone
    del cnn_stage3, cnn_stage4, vit_tokens
    del per_image_logits, study_logits, indices
    torch.cuda.empty_cache()

    image_dataset = OfficialTrainImageDataset(
        data_config["train_csv"], data_config["image_root"],
        transform=raw_transform,
        image_id_column=data_config["image_id_column"],
        subject_column=data_config["subject_column"],
        study_column=data_config["study_column"],
    )
    one_image = image_dataset[0]["image"].unsqueeze(0).to(device)
    mask_generator = torch.Generator(device=device).manual_seed(42)
    mask = sample_spatial_mask(1, device, generator=mask_generator)
    if int(mask.sum()) != 60 or int(mask_to_vit_grid(mask).sum()) != 240:
        raise RuntimeError("The exact 60% shared 32-pixel spatial mask is incorrect")
    metrics["mask"] = {"mask_cells_10x10": int(mask.sum()),
                       "mask_cells_vit_20x20": int(mask_to_vit_grid(mask).sum()),
                       "mask_ratio": float(mask.float().mean())}

    for method, model_class in (("h1_pixel", PixelReconstructionPretrainer),
                                ("h2_latent", LatentReconstructionPretrainer)):
        pretrainer = model_class(build_backbone()).to(device)
        pretrainer.train()
        pretrainer.backbone.set_grad_checkpointing(False)
        original_one_image = one_image.clone()
        output = pretrainer(one_image, mask)
        if not torch.isfinite(output["loss"]):
            raise RuntimeError(f"{method} reconstruction loss is non-finite")
        if not torch.equal(one_image, original_one_image):
            raise RuntimeError(f"{method} modified the unmasked reconstruction target")
        if abs(float(output["mask_ratio"]) - 0.60) > 1e-6:
            raise RuntimeError(f"{method} did not apply exactly 60% masking")
        if method == "h2_latent" and (output["cnn_target"].requires_grad or output["vit_target"].requires_grad):
            raise RuntimeError("H2 latent reconstruction targets must be detached")
        optimizer_groups = [
            {"params": pretrainer.backbone.convnext.parameters(), "lr": 1e-5},
            {"params": pretrainer.backbone.vit.parameters(), "lr": 1e-5},
            {"params": list(pretrainer.cnn_decoder.parameters()) +
                        list(pretrainer.vit_decoder.parameters()) +
                        ([pretrainer.cnn_mask_token, pretrainer.vit_mask_token]
                         if hasattr(pretrainer, "cnn_mask_token") else [pretrainer.vit_mask_token]),
             "lr": 2e-4},
        ]
        optimizer = build_adamw(optimizer_groups, 0.01, fused=True)
        grad_norm, parameter_change = _one_update(pretrainer, output, output["loss"], device, optimizer)
        groups = _iter_group_parameters(pretrainer, method)
        branch_norms = {name: _grad_norm(parameters) for name, parameters in groups.items()}
        if any(value <= 1e-12 or not math.isfinite(value) for value in branch_norms.values()):
            raise RuntimeError(f"{method} has a missing encoder/decoder gradient: {branch_norms}")
        if method == "h2_latent" and (
                branch_norms["cnn_stage3"] <= 0 or branch_norms["cnn_stage4"] <= 0 or
                branch_norms["vit_pre_block6"] <= 0 or branch_norms["vit_post_block6"] <= 0):
            raise RuntimeError(f"H2 did not train Stage3/Stage4 and both ViT block groups: {branch_norms}")
        checkpoint = {"model": pretrainer.state_dict(), "optimizer": optimizer.state_dict(),
                      "method": method}
        full_checkpoint_path = smoke_dir / f"{method}_full_step.pth"
        torch.save(checkpoint, full_checkpoint_path)
        loaded_checkpoint = torch.load(full_checkpoint_path, map_location="cpu", weights_only=False)
        reloaded = model_class(build_backbone()).to(device)
        reloaded.load_state_dict(loaded_checkpoint["model"], strict=True)
        encoder_export = {"backbone": pretrainer.backbone.state_dict(), "method": method}
        encoder_export_path = smoke_dir / f"{method}_encoder_export.pth"
        torch.save(encoder_export, encoder_export_path)
        loaded_encoder_export = torch.load(encoder_export_path, map_location="cpu", weights_only=False)
        supervised_backbone = build_backbone()
        supervised_backbone.load_state_dict(loaded_encoder_export["backbone"], strict=True)
        metrics[method] = {
            "loss": float(output["loss"].detach()),
            "cnn_loss": float(output["cnn_loss"].detach()),
            "vit_loss": float(output["vit_loss"].detach()),
            "gradient_norm": grad_norm,
            "parameter_change": parameter_change,
            "branch_gradient_norms": branch_norms,
            "checkpoint_reload": True,
            "encoder_only_reload": True,
            "decoder_excluded_from_encoder_export": "cnn_decoder" not in loaded_encoder_export["backbone"],
        }
        if method == "h1_pixel":
            metrics[method]["decoder_output_shape"] = list(output["cnn_pixels"].shape)
            metrics[method]["target_range"] = [float(one_image.min()), float(one_image.max())]
        else:
            metrics[method]["cnn_stage3_shape"] = list(output["cnn_target"].shape)
            metrics[method]["vit_block6_target_shape"] = list(output["vit_target"].shape)
            metrics[method]["feature_variance"] = {
                "cnn": float(output["cnn_spatial_variance"].detach()),
                "vit": float(output["vit_spatial_variance"].detach()),
            }
        del pretrainer, optimizer, reloaded, supervised_backbone, output
        torch.cuda.empty_cache()

    report = {
        "passed": True,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "timm_version": __import__("timm").__version__,
        "input_shape": list(images.shape[1:]),
        "multi_image_study_count": int(images.shape[0]),
        "train_preprocessing_shape": list(images.shape[1:]),
        "train_preprocessing_range": [float(images.min()), float(images.max())],
        "validation_preprocessing_shape": list(valid_image.shape),
        "validation_preprocessing_range": [float(valid_image.min()), float(valid_image.max())],
        "convnext_stage3_shape": shape_info["convnext_stage3"],
        "convnext_stage4_shape": shape_info["convnext_stage4"],
        "vit_token_shape": shape_info["vit_tokens"],
        "hybrid_fusion_width": 1152,
        "classifier_output_shape": [1, 10],
        "study_fusion": "raw image logits averaged before sigmoid",
        "initial_weight_metadata": str(initial_path.with_suffix(".json")),
        "metrics": metrics,
    }
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_flops=True,
        ) as profiler:
            model = HybridClassifier(build_backbone()).to(device).eval()
            with torch.no_grad(), autocast_context(device, amp_enabled, amp_dtype):
                model.image_logits(one_image)
            torch.cuda.synchronize(device)
        report["classifier_profiler_flops_per_image"] = sum(
            int(event.flops or 0) for event in profiler.key_averages())
        if report["classifier_profiler_flops_per_image"] <= 0:
            raise RuntimeError("torch.profiler did not report positive classifier FLOPs")
        report["classifier_parameter_count"] = count_parameters(model)
        del model
    except Exception as exc:
        report["passed"] = False
        report["classifier_profiler_flops_error"] = f"{type(exc).__name__}: {exc}"
    output_path = smoke_dir / "hybrid_smoke_test.json"
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Smoke Test report saved: {output_path}")
    if report.get("passed") is False:
        raise RuntimeError(report["classifier_profiler_flops_error"])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-config", default="configs/hybrid_experiments.yaml")
    args = parser.parse_args()
    run_smoke_test(args.experiments_config)


if __name__ == "__main__":
    main()
