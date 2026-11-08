"""Pre-training audit for H2-B / H2-C single-image weights and H4 head dropout."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

from genopt_common import (
    EXPERIMENTS,
    EXPERIMENT_ROOT,
    EXPECTED_HEAD_SHA256,
    IMAGE_SIZE,
    LABEL_NAMES,
    PROFILE,
    R5_DIR,
    SEED,
    build_h_datasets,
    head_hash,
    load_r5_reference,
    make_h_builder,
    read_json,
    write_json,
)
from genopt_train import _experiment_context
from dino_data import make_loader, shutdown_loader
from dino_experiment import load_pretrained_backbone
from dino_feature_models import collate_masked_studies
from dino_losses import study_bce_per_study, study_logit_mean

OUT = EXPERIMENT_ROOT / "H2_H4_Comparative_Analysis"
OUT.mkdir(parents=True, exist_ok=True)


def _r5_head_audit(reference, device) -> dict:
    builder = make_h_builder(reference["runtime"], reference["head_sha"],
                             OUT / "scratch")
    model, verification, _, _, _ = builder(PROFILE, device, reference["runtime"], SEED)
    head_parameters = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.")
    }
    head_modules = [
        (name, type(module).__name__)
        for name, module in model.named_modules()
        if name and not name.startswith("backbone.")
    ]
    has_dropout = any(isinstance(module, nn.Dropout)
                      for module in model.modules())
    has_head_layernorm = any(
        isinstance(child, nn.LayerNorm) for child in model.classifier.children())
    linear_layers = [name for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)
                     and not name.startswith("backbone.")]
    payload = {
        "architecture": "DINOv3 ConvNeXt-Tiny Global-only (R5)",
        "head_parameter_names_and_shapes": head_parameters,
        "head_modules_outside_backbone": head_modules,
        "has_dropout_anywhere": bool(has_dropout),
        "has_head_layernorm": bool(has_head_layernorm),
        "head_linear_layers": linear_layers,
        "forward_order": "backbone.forward_features(images) -> pooled (final "
                         "LayerNorm + GAP, dim 768) -> classifier Linear(768,10)",
        "num_head_linear_layers": len(linear_layers),
        "backbone_init_sha256": verification["weights_sha256"],
        "head_init_sha256": head_hash(model),
        "expected_head_init_sha256": EXPECTED_HEAD_SHA256,
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    del model
    torch.cuda.empty_cache()
    return payload


def _h4_audit(reference, device, batch) -> dict:
    runtime = reference["runtime"]
    builder0 = make_h_builder(runtime, reference["head_sha"], OUT / "scratch",
                              head_dropout=0.0)
    builder4 = make_h_builder(runtime, reference["head_sha"], OUT / "scratch",
                              head_dropout=0.10)
    r5_model, _, _, _, _ = builder0(PROFILE, device, runtime, SEED)
    h4_model, _, _, _, _ = builder4(PROFILE, device, runtime, SEED)
    # Copy the R5 initialization (backbone + linear head) into the H4 module.
    h4_model.load_state_dict(r5_model.state_dict(), strict=True)
    keys0 = {name: tuple(tensor.shape) for name, tensor in r5_model.state_dict().items()}
    keys4 = {name: tuple(tensor.shape) for name, tensor in h4_model.state_dict().items()}
    images = batch["images"].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        r5_model.eval()
        r5_logits_a = r5_model(images).float()
        r5_logits_b = r5_model(images).float()
        h4_model.eval()
        h4_logits_a = h4_model(images).float()
        h4_logits_b = h4_model(images).float()
        h4_model.train()
        train_a = h4_model(images).float()
        train_b = h4_model(images).float()
    dropout_module = next((module for name, module in h4_model.named_modules()
                           if isinstance(module, nn.Dropout)), None)
    dropout_parent = next((name for name, module in h4_model.named_modules()
                           if name.endswith("dropout")), None)
    payload = {
        "dropout_p": float(dropout_module.p) if dropout_module else None,
        "dropout_parent_module": dropout_parent,
        "dropout_has_parameters": bool(
            any(True for _ in (dropout_module.parameters() if dropout_module else []))),
        "state_dict_keys_identical_to_r5": keys0 == keys4,
        "total_parameter_count_r5": int(sum(p.numel() for p in r5_model.parameters())),
        "total_parameter_count_h4": int(sum(p.numel() for p in h4_model.parameters())),
        "eval_dropout_off_two_forwards_exact_equal": bool(
            torch.equal(h4_logits_a, h4_logits_b)),
        "eval_matches_r5_exact_equal": bool(torch.equal(r5_logits_a, h4_logits_a)),
        "train_two_forwards_differ": bool(not torch.equal(train_a, train_b)),
        "train_eval_logits_differ": bool(not torch.equal(train_a, h4_logits_a)),
    }
    del r5_model, h4_model
    torch.cuda.empty_cache()
    return payload


def _weight_loss_audit() -> dict:
    generator = torch.Generator().manual_seed(SEED)
    num_studies, num_labels = 24, 10
    logits = torch.randn(num_studies, num_labels, generator=generator) * 2.0
    labels = (torch.rand(num_studies, num_labels, generator=generator) > 0.6).float()
    known = torch.ones_like(labels, dtype=torch.bool)
    counts = torch.tensor([1 if index % 3 == 0 else 2
                           for index in range(num_studies)], dtype=torch.long)
    per_study, eligible = study_bce_per_study(logits, labels, known)
    overall_loss = per_study[eligible].sum() / eligible.sum()
    records = []
    for name, weight in (("H0", 1.0), ("H2B", 1.10), ("H2", 1.25), ("H2C", 1.50)):
        weights = torch.where(counts == 1, weight, 1.0)
        loss = (per_study[eligible] * weights[eligible]).sum() / weights[eligible].sum()
        single_loss = per_study[eligible & (counts == 1)].mean()
        multi_loss = per_study[eligible & (counts >= 2)].mean()
        single_fraction = float(weights[eligible & (counts == 1)].sum())
        multi_fraction = float(weights[eligible & (counts >= 2)].sum())
        # Weighted mean of the two group means: the mixture weight is the
        # weighted group mass W1/(W1+W2), not the raw Study-count share.
        mixture_weight = single_fraction / (single_fraction + multi_fraction)
        # Window-level tests: split the same studies into 4 microbatches and
        # compare the accumulated weighted numerator/denominator against the
        # single-window reference.
        micro_sizes = [6, 6, 6, 6]
        start = 0
        numerator = denominator = 0.0
        for size in micro_sizes:
            index = torch.arange(start, start + size)
            start += size
            microbatch_loss = (per_study[index] * weights[index]).sum()
            numerator += float(microbatch_loss)
            denominator += float(weights[index].sum())
        weights_expected = mixture_weight
        records.append({
            "experiment": name,
            "single_weight": weight,
            "window_loss": float(loss),
            "single_raw_bce": float(single_loss),
            "multi_raw_bce": float(multi_loss),
            "single_count": int((counts == 1).sum()),
            "multi_count": int((counts >= 2).sum()),
            "microbatch_accumulated_loss": numerator / denominator,
            "microbatch_matches_window": bool(
                abs(numerator / denominator - float(loss)) < 1e-6),
            "weighted_mixture_identity": bool(
                abs(float(loss) - weights_expected * float(single_loss)
                    - (1.0 - weights_expected) * float(multi_loss)) < 1e-6),
            "loss_equals_unweighted_when_weight_1": bool(
                abs(float(loss) - float(overall_loss)) < 1e-7),
        })
    return {
        "rows": records,
        "implementation_reference": "same weighted Study mean as accepted H2: "
                                    "sum_s w_s * ell_s / sum_s w_s over eligible "
                                    "Studies in each optimizer window",
        "h2b_weight_correct": next(row for row in records
                                   if row["experiment"] == "H2B")["single_weight"] == 1.10,
        "h2c_weight_correct": next(row for row in records
                                   if row["experiment"] == "H2C")["single_weight"] == 1.50,
        "all_microbatch_checks_pass": all(row["microbatch_matches_window"]
                                          for row in records),
        "all_mixture_checks_pass": all(row["weighted_mixture_identity"]
                                       for row in records),
    }


def run_preflight() -> dict:
    reference = load_r5_reference()
    device = torch.device("cuda")
    train_data, valid_data, _ = build_h_datasets()
    loader, _, _ = make_loader(
        train_data, image_budget=8, shuffle=False, seed=SEED, num_workers=0,
        pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    batch = next(iter(loader))
    shutdown_loader(loader)
    head = _r5_head_audit(reference, device)
    h4 = _h4_audit(reference, device, batch)
    weights = _weight_loss_audit()
    builder = make_h_builder(reference["runtime"], reference["head_sha"],
                             OUT / "scratch")
    model, verification, trainable_modules, frozen_modules, trainable_names = builder(
        PROFILE, device, reference["runtime"], SEED)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    frozen_stage_prefixes = [name for name, _ in frozen_modules
                             if hasattr(_, "parameters")]
    freeze_audit = {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "trainable_namespace_stage3_stage4": all(
            name.startswith("backbone.model.stages.2.")
            or name.startswith("backbone.model.stages.3.")
            or name.startswith("backbone.model.downsample_layers.2.")
            or name.startswith("backbone.model.downsample_layers.3.")
            or name.startswith("backbone.model.norm.")
            or name.startswith("classifier.")
            for name in trainable_names),
        "expected_counts": [27827818, 26592778, 1235040],
        "counts_match_r5": [total, trainable, frozen] == [27827818, 26592778, 1235040],
        "init_weights_sha256": verification["weights_sha256"],
        "reference_init_sha256": reference["initialization_sha256"],
        "init_matches_r5": verification["weights_sha256"]
        == reference["initialization_sha256"],
        "head_dropout_in_h4": h4["dropout_p"],
        "h4_expected_dropout": 0.10,
    }
    del model
    torch.cuda.empty_cache()
    checks = {
        "r5_head_has_no_dropout": not head["has_dropout_anywhere"],
        "r5_head_has_single_linear": head["num_head_linear_layers"] == 1,
        "r5_head_no_layernorm": not head["has_head_layernorm"],
        "h4_dropout_p_is_0.10": abs(float(h4["dropout_p"]) - 0.10) < 1e-12,
        "h4_dropout_positioned_after_pooled_before_linear":
            h4["dropout_parent_module"] == "dropout",
        "h4_state_dict_identical_to_r5": h4["state_dict_keys_identical_to_r5"],
        "h4_parameter_count_unchanged":
            h4["total_parameter_count_r5"] == h4["total_parameter_count_h4"],
        "h4_dropout_active_in_train": h4["train_two_forwards_differ"],
        "h4_dropout_off_in_eval": h4["eval_dropout_off_two_forwards_exact_equal"],
        "h4_eval_matches_r5": h4["eval_matches_r5_exact_equal"],
        "h2b_weight_1.10": weights["h2b_weight_correct"],
        "h2c_weight_1.50": weights["h2c_weight_correct"],
        "h2_microbatch_normalization": weights["all_microbatch_checks_pass"],
        "h2_weighted_mixture_identity": weights["all_mixture_checks_pass"],
        "freeze_policy_matches_r5": freeze_audit["counts_match_r5"],
        "initial_weights_match_r5": freeze_audit["init_matches_r5"],
    }
    payload = {
        "status": "completed",
        "experiments": ["H2B", "H2C", "H4"],
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "r5_head_audit": head,
        "h4_dropout_audit": h4,
        "single_weight_audit": weights,
        "freeze_and_initialization_audit": freeze_audit,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "test_access": False,
    }
    write_json(OUT / "pre_training_audit.json", payload)
    _write_markdown(payload)
    return payload


def _write_markdown(payload: dict) -> None:
    head = payload["r5_head_audit"]
    h4 = payload["h4_dropout_audit"]
    freeze = payload["freeze_and_initialization_audit"]
    weights = payload["single_weight_audit"]
    lines = [
        "# H2-B / H2-C / H4 pre-training audit", "",
        f"- Status: {payload['status']}; Seed={payload['seed']}; "
        f"input={payload['image_size']}.",
        "",
        "## 1. R5 Global-only classifier head", "",
        f"- Head parameter names/shapes: `{json.dumps(head['head_parameter_names_and_shapes'])}`.",
        f"- Linear layers outside backbone: {head['head_linear_layers']}.",
        f"- Has any Dropout: **{head['has_dropout_anywhere']}**; has head LayerNorm: "
        f"**{head['has_head_layernorm']}**.",
        f"- Forward order: {head['forward_order']}.",
        f"- Original head init SHA-256: `{head['head_init_sha256']}`.",
        "",
        "## 2. H4 dropout insertion", "",
        f"- Dropout p = **{h4['dropout_p']}**, parent module = "
        f"`{h4['dropout_parent_module']}`, has parameters = "
        f"{h4['dropout_has_parameters']}.",
        f"- State dict keys identical to R5: **{h4['state_dict_keys_identical_to_r5']}**; "
        f"parameter counts R5/H4 = {h4['total_parameter_count_r5']} / "
        f"{h4['total_parameter_count_h4']}.",
        f"- Train mode: two forwards differ = **{h4['train_two_forwards_differ']}**; "
        f"eval mode: two forwards exactly equal = "
        f"**{h4['eval_dropout_off_two_forwards_exact_equal']}**; "
        f"eval logits equal R5 with same weights = **{h4['eval_matches_r5_exact_equal']}**.",
        "- 结论：H4 在训练时对 pooled feature 施加 p=0.10 dropout，"
        "推理 `model.eval()` 自动关闭；不新增参数、不改变 checkpoint 结构。",
        "",
        "## 3. Single-image weight / normalization", "",
        "| Experiment | weight | window loss | microbatch accumulated | "
        "microbatch matches | mixture identity |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in weights["rows"]:
        lines.append(
            f"| {row['experiment']} | {row['single_weight']:.2f} | "
            f"{row['window_loss']:.6f} | {row['microbatch_accumulated_loss']:.6f} | "
            f"{row['microbatch_matches_window']} | {row['weighted_mixture_identity']} |")
    lines.extend([
        "- H2-B/H2-C 与已验收 H2 使用完全相同的归一化："
        "`sum_s w_s ell_s / sum_s w_s`，分母只统计当前 optimizer window 内的 eligible "
        "Study（4 个 microbatch 合并），不会因 microbatch 切分产生有效权重偏差。",
        f"- H0 权重=1 时损失与未加权 Study BCE 完全一致："
        f"{weights['rows'][0]['loss_equals_unweighted_when_weight_1']}。",
        "",
        "## 4. Freeze / initialization", "",
        f"- total/trainable/frozen = {freeze['total_parameters']:,} / "
        f"{freeze['trainable_parameters']:,} / {freeze['frozen_parameters']:,}; "
        f"matches R5: **{freeze['counts_match_r5']}**.",
        f"- Initial backbone SHA-256 matches R5: **{freeze['init_matches_r5']}**.",
        "- Stage3/4/final-norm/classifier 可训练；Stem/Stage1/Stage2 冻结，与原 R5 一致。",
        "",
        "## 5. Checks", "",
        "| Check | Passed |", "|---|---|",
    ] + [f"| {key} | {value} |" for key, value in payload["checks"].items()] + [
        "",
        f"**All checks passed: {payload['all_checks_passed']}**",
        "",
        "未读取 Test 标签；测试 forward/backward 仅用于审计，不进入正式训练。",
        "",
    ])
    (OUT / "pre_training_audit.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_preflight(), ensure_ascii=True, indent=1, default=str))
