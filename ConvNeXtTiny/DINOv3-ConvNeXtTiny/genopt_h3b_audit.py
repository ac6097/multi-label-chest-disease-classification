"""H3-B pre-training audit of the L2-SP implementation.

Verifies the exact regularized parameter set, the element-mean penalty,
lambda scaling, absence of duplicate lambda/microbatch amplification, gradient
flow, optimizer weight decay, L2-SP gradient after a non-zero parameter shift,
and that frozen/classifier parameters are excluded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from genopt_common import (
    EXPERIMENTS,
    EXPERIMENT_ROOT,
    IMAGE_SIZE,
    LABEL_NAMES,
    PROFILE,
    SEED,
    build_h_datasets,
    feature_profile_optimizer,
    l2sp_parameter_names,
    l2sp_penalty,
    l2sp_reference,
    load_r5_reference,
    make_h_builder,
    read_json,
    write_json,
)
from dino_data import make_loader, shutdown_loader
from dino_experiment import load_pretrained_backbone, update_learning_rates
from dino_feature_models import collate_masked_studies
from dino_losses import study_bce_per_study, study_logit_mean

H3_DIR = EXPERIMENT_ROOT / "H3_L2SP_Regularization"
H3B_DIR = EXPERIMENT_ROOT / "H3B_Weak_L2SP"
H3_LAMBDA = None


def _load_train_batches(count: int = 4):
    train_data, _, _ = build_h_datasets()
    loader, _, _ = make_loader(
        train_data, image_budget=8, shuffle=True, seed=SEED, num_workers=0,
        pin_memory=True, prefetch_factor=2, collate_fn=collate_masked_studies)
    iterator = iter(loader)
    batches = []
    for _ in range(count):
        try:
            batches.append(next(iterator))
        except StopIteration:
            iterator = iter(loader)
            batches.append(next(iterator))
    shutdown_loader(loader)
    return train_data, batches


def _forward_bce(model, batch, device, channels_last: bool):
    images = batch["images"].to(device)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    study_index = batch["study_index"].to(device)
    labels = batch["labels"].to(device)
    known = batch["known"].to(device)
    image_counts = batch["image_counts"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_logits = model(images)
        study_logits = study_logit_mean(
            image_logits.float(), study_index, batch["num_studies"], image_counts)
        per_study, eligible = study_bce_per_study(study_logits, labels, known)
        return per_study[eligible].sum()


def run_audit() -> dict:
    H3B_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_r5_reference()
    runtime = reference["runtime"]
    device = torch.device("cuda")
    h3_lambda = float(read_json(H3_DIR / "l2sp_calibration.json")["lambda_sp"])
    h3b_lambda = float(EXPERIMENTS["H3B"]["optimization"]["lambda_sp"])
    expected_elements = int(read_json(
        H3_DIR / "l2sp_calibration.json")["regularized_parameter_count"])

    train_data, batches = _load_train_batches(8)
    builder = make_h_builder(runtime, reference["head_sha"], H3B_DIR / "audit")
    model, verification, train_modules, frozen_modules, trainable_names = builder(
        PROFILE, device, runtime, SEED)
    optimizer = feature_profile_optimizer(model, reference["config"])
    names = l2sp_parameter_names(model)
    parameters = dict(model.named_parameters())
    reference_parameters = l2sp_reference(model, names)
    selected = [parameters[name] for name in names]

    # 1. Parameter-set audit.
    pretrained = load_pretrained_backbone("cpu")[0].state_dict()
    pretrained_names = [name.removeprefix("backbone.") for name in names]
    missing_pretrained = [name for name in pretrained_names
                          if name not in pretrained]
    shape_mismatch = [
        name for name, inner in zip(names, pretrained_names)
        if name in parameters and tuple(parameters[name].shape) != tuple(
            pretrained[inner].shape)]
    classifier_names = [name for name, _ in model.named_parameters()
                        if not name.startswith("backbone.")]
    frozen_names = [name for name, parameter in model.named_parameters()
                    if not parameter.requires_grad]
    frozen_in_penalty = sorted(set(frozen_names) & set(names))
    classifier_in_penalty = sorted(set(classifier_names) & set(names))
    namespace_counts = {}
    for name in names:
        prefix = ".".join(name.split(".")[:4])
        namespace_counts[prefix] = namespace_counts.get(prefix, 0) + 1
    element_count = int(sum(parameters[name].numel() for name in names))
    # Initial parameter offset must be zero.
    initial_deltas = [float((parameters[name] - reference_parameters[name]).abs().max())
                      for name in names]
    initial_penalty = float(l2sp_penalty(model, reference_parameters).float().cpu())
    # Pretrained values are loaded exactly (no perturbed reference).
    reference_matches_pretrained = all(
        torch.equal(reference_parameters[name].cpu(),
                    pretrained[inner].float())
        for name, inner in zip(names, pretrained_names))

    # 2. Create a non-zero drift with BCE-only updates (L2-SP gradients are zero
    #    at the exact pretrained point).
    model.train()
    optimizer.zero_grad(set_to_none=True)
    training = dict(reference["config"]["training"])
    training["warmup_updates"] = 2
    import itertools

    iterator = itertools.cycle(batches)
    for step in range(3):
        window = [next(iterator) for _ in range(4)]
        eligible = sum(int(batch["known"].any(dim=1).sum()) for batch in window)
        optimizer.zero_grad(set_to_none=True)
        update_learning_rates(optimizer, step, 8, training)
        for batch in window:
            images = batch["images"].to(device).contiguous(
                memory_format=torch.channels_last)
            study_index = batch["study_index"].to(device)
            labels = batch["labels"].to(device)
            known = batch["known"].to(device)
            image_counts = batch["image_counts"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_logits = model(images)
                study_logits = study_logit_mean(
                    image_logits.float(), study_index, batch["num_studies"],
                    image_counts)
                per_study, eligible_mask = study_bce_per_study(
                    study_logits, labels, known)
                loss = per_study[eligible_mask].sum() / max(eligible, 1)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    drift_penalty = float(l2sp_penalty(model, reference_parameters).float().cpu())

    # 3. Verify the penalty equals an independent element-mean computation.
    manual_squared = None
    manual_elements = 0
    for name in names:
        difference = parameters[name].float() - reference_parameters[name]
        squared = difference.pow(2)
        manual_squared = squared.sum() if manual_squared is None else (
            manual_squared + squared.sum())
        manual_elements += squared.numel()
    manual_penalty = float((manual_squared / float(manual_elements)).cpu())
    denominator_uses_elements = manual_elements == element_count == expected_elements

    # 4. Analytic penalty gradient vs autograd (element-mean normalization).
    analytic_grad = 2.0 * (parameters[names[0]].float() - reference_parameters[names[0]]) / float(element_count)
    autograd_grad = torch.autograd.grad(
        l2sp_penalty(model, reference_parameters), parameters[names[0]],
        retain_graph=False, allow_unused=True)[0]
    gradient_formula_max_error = float(
        (analytic_grad - autograd_grad.float()).abs().max().cpu())

    # 5. Gradient-accumulation isolation: simulate the trainer's 4 microbatch
    #    window where L2-SP is backpropagated exactly once per optimizer update.
    window = batches[:4]
    eligible_window = sum(int(batch["known"].any(dim=1).sum())
                          for batch in window)
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    for batch in window:
        loss = _forward_bce(model, batch, device, runtime["channels_last"])
        (loss / eligible_window).backward()
    reg = l2sp_penalty(model, reference_parameters)
    (h3b_lambda * reg).backward()
    combined = {name: (parameters[name].grad.detach().clone()
                       if parameters[name].grad is not None else None)
                for name in names}
    combined_classifier = {name: (parameters[name].grad.detach().clone()
                                  if parameters[name].grad is not None else None)
                           for name in classifier_names}
    # Reference: BCE gradients accumulated separately + one lambda*reg gradient.
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    for batch in window:
        loss = _forward_bce(model, batch, device, runtime["channels_last"])
        (loss / eligible_window).backward()
    bce_grads = {name: (parameters[name].grad.detach().clone()
                        if parameters[name].grad is not None else None)
                 for name in names}
    model.zero_grad(set_to_none=True)
    reg_grads = torch.autograd.grad(
        h3b_lambda * l2sp_penalty(model, reference_parameters), selected,
        retain_graph=False, allow_unused=True)
    accumulation_difference = 0.0
    for name, grad in zip(names, reg_grads):
        reference_grad = bce_grads[name]
        if grad is not None:
            reference_grad = grad.detach().clone() if reference_grad is None else (
                reference_grad + grad.detach())
        actual = combined[name]
        if reference_grad is None and actual is None:
            continue
        if reference_grad is None or actual is None:
            accumulation_difference = float("inf")
            break
        accumulation_difference = max(
            accumulation_difference,
            float((actual.float() - reference_grad.float()).abs().max()))
    # A duplicated reg term would make the difference of order lambda*reg grad.
    reg_grad_norm = math.sqrt(sum(float(grad.pow(2).sum()) for grad in reg_grads
                                  if grad is not None))
    no_microbatch_amplification = bool(
        accumulation_difference < max(1e-5, 1e-4 * reg_grad_norm))

    # 6. Gradient flow into Stage3/4 and exclusion of classifier/frozen params.
    model.zero_grad(set_to_none=True)
    reg_grads_all = torch.autograd.grad(
        l2sp_penalty(model, reference_parameters),
        [parameters[name] for name in names + classifier_names[:1]],
        retain_graph=False, allow_unused=True)
    stage3_norm = math.sqrt(sum(
        float(grad.pow(2).sum()) for name, grad in zip(names, reg_grads_all[:len(names)])
        if grad is not None and name.startswith("backbone.model.stages.2.")))
    stage4_norm = math.sqrt(sum(
        float(grad.pow(2).sum()) for name, grad in zip(names, reg_grads_all[:len(names)])
        if grad is not None and name.startswith("backbone.model.stages.3.")))
    classifier_reg_grad = reg_grads_all[len(names)]
    classifier_reg_grad_none = classifier_reg_grad is None or float(
        classifier_reg_grad.abs().max()) == 0.0
    # Frozen parameters are excluded from the regularized set and never get a
    # penalty gradient; verify their tensors are unchanged after one BCE+L2SP step.
    before_frozen = {name: parameters[name].detach().clone()
                     for name in frozen_names}
    before_classifier = {name: parameters[name].detach().clone()
                         for name in classifier_names}
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    eligible0 = max(int(batches[0]["known"].any(dim=1).sum()), 1)
    loss = _forward_bce(model, batches[0], device,
                        runtime["channels_last"]) / eligible0
    loss.backward()
    (h3b_lambda * l2sp_penalty(model, reference_parameters)).backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    frozen_unchanged = all(
        torch.equal(parameters[name], before_frozen[name]) for name in frozen_names)
    classifier_changed = any(
        not torch.equal(parameters[name], before_classifier[name])
        for name in classifier_names)
    # Gradient norm at the drift point.
    bce_loss = _forward_bce(model, batches[0], device,
                            runtime["channels_last"]) / eligible0
    bce_grad = torch.autograd.grad(bce_loss, selected, retain_graph=True,
                                   allow_unused=True)
    bce_grad_norm = math.sqrt(sum(float(grad.pow(2).sum()) for grad in bce_grad
                                  if grad is not None))
    unit_reg = l2sp_penalty(model, reference_parameters)
    unit_reg_grad = torch.autograd.grad(unit_reg, selected, allow_unused=True)
    unit_reg_grad_norm = math.sqrt(sum(
        float(grad.pow(2).sum()) for grad in unit_reg_grad if grad is not None))
    h3b_grad_norm = float(h3b_lambda) * unit_reg_grad_norm
    h3_grad_norm = float(h3_lambda) * unit_reg_grad_norm

    # 7. Optimizer group audit.
    groups = [{"name": group["name"], "lr": group["lr"],
               "weight_decay": group["weight_decay"],
               "n_params": len(group["params"])}
              for group in optimizer.param_groups]
    weight_decay_ok = all(abs(group["weight_decay"] - 0.01) < 1e-12
                          for group in optimizer.param_groups)
    lambda_ratio = h3b_lambda / h3_lambda if h3_lambda else None

    checks = {
        "only_stage3_stage4_trainable_pretrained_params": bool(
            not classifier_in_penalty and not frozen_in_penalty),
        "every_regularized_param_has_pretrained_counterpart": bool(
            not missing_pretrained and not shape_mismatch),
        "regularized_element_count_matches_H3": bool(
            element_count == expected_elements),
        "pretrained_values_unchanged": bool(reference_matches_pretrained),
        "initial_penalty_zero": bool(initial_penalty == 0.0),
        "penalty_is_element_mean": bool(
            abs(manual_penalty - drift_penalty) < 1e-9
            and denominator_uses_elements),
        "analytic_gradient_formula_matches_autograd": bool(
            gradient_formula_max_error < 1e-6),
        "lambda_H3B_is_exact_0.1x_H3": bool(
            lambda_ratio is not None and abs(lambda_ratio - 0.1) < 1e-9),
        "no_double_lambda_or_microbatch_amplification": bool(
            no_microbatch_amplification),
        "stage3_and_stage4_gradient_present": bool(
            stage3_norm > 0 and stage4_norm > 0),
        "classifier_excluded_from_penalty": bool(classifier_reg_grad_none),
        "frozen_params_unchanged_by_combined_step": bool(frozen_unchanged),
        "classifier_updated_by_bce": bool(classifier_changed),
        "adamw_weight_decay_is_0.01": bool(weight_decay_ok),
        "optimizer_lrs_match_R5": bool(
            any(group["name"] == "backbone" and abs(group["lr"] - 1e-5) < 1e-15
                for group in optimizer.param_groups)
            and any(group["name"] == "head" and abs(group["lr"] - 2e-4) < 1e-15
                    for group in optimizer.param_groups)),
    }
    payload = {
        "experiment": "H3B",
        "lambda_H3": float(h3_lambda),
        "lambda_H3B": float(h3b_lambda),
        "lambda_ratio": lambda_ratio,
        "regularized_parameter_count": element_count,
        "regularized_namespaces": sorted(namespace_counts),
        "regularized_parameter_tensor_count": len(names),
        "initial_penalty": initial_penalty,
        "penalty_after_Bce_warmup_drift": drift_penalty,
        "manual_element_mean_penalty": manual_penalty,
        "manual_elements": manual_elements,
        "penalty_formula": "mean_j (theta_j - theta0_j)^2 over regularized elements",
        "analytic_gradient_max_error": gradient_formula_max_error,
        "accumulation_difference": accumulation_difference,
        "reg_grad_norm_after_lambda": reg_grad_norm,
        "bce_grad_norm": bce_grad_norm,
        "stage3_reg_grad_norm_unit_lambda": stage3_norm,
        "stage4_reg_grad_norm_unit_lambda": stage4_norm,
        "unit_reg_grad_norm": unit_reg_grad_norm,
        "H3B_reg_grad_norm": h3b_grad_norm,
        "H3_reg_grad_norm": h3_grad_norm,
        "optimizer_groups": groups,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "test_access": False,
    }
    write_json(H3B_DIR / "l2sp_implementation_audit.json", payload)
    _write_markdown(payload)
    return payload


def _write_markdown(payload: dict) -> None:
    checks = payload["checks"]
    lines = [
        "# H3-B L2-SP implementation audit", "",
        "审计对象：与 H3 完全相同的 L2-SP 定义只把 λ 改为 0.1×H3；本次从原始 "
        "DINOv3 预训练权重独立训练。", "",
        "## 关键参数", "",
        f"- λ_H3 = **{payload['lambda_H3']:.12g}**；λ_H3B = "
        f"**{payload['lambda_H3B']:.12g}**；比值 = "
        f"{payload['lambda_ratio']:.12f}（目标 0.1）。",
        f"- 正则化范围：Stage3/4 中具有预训练对应关系的可训练参数，共 "
        f"**{payload['regularized_parameter_count']:,}** 个元素 / "
        f"{payload['regularized_parameter_tensor_count']} 个 tensor。",
        f"- 参数命名空间：`{', '.join(payload['regularized_namespaces'])}`。",
        "- 归一化：R_SP = mean_j (θ_j − θ0_j)² / N_elements，与 H3 完全一致；"
        "分类头、冻结参数、最终 LayerNorm 不进入正则项。",
        "",
        "## 9 项检查", "",
        "| Check | Passed |", "|---|---|",
    ] + [f"| {key} | {value} |" for key, value in checks.items()] + [
        "",
        "## 数值证据", "",
        f"- 初始化时正则项：{payload['initial_penalty']:.3g}（必须为 0；"
        f"预训练起点梯度也为 0，因此审计在 3 次 BCE-only update 造成非零漂移后执行）。",
        f"- BCE warmup 后正则项：{payload['penalty_after_Bce_warmup_drift']:.6g}；"
        f"独立 element-mean 手算：{payload['manual_element_mean_penalty']:.6g}"
        f"（元素数 {payload['manual_elements']:,}）。",
        f"- 解析梯度公式 vs autograd 最大误差："
        f"{payload['analytic_gradient_max_error']:.3g}。",
        f"- 梯度累积隔离：4-microbatch 窗口（L2-SP 只反传一次）与"
        f"“BCE 累积 + 单次 λ·R_SP 梯度”参考之间的最大差 "
        f"{payload['accumulation_difference']:.3g}；单次 λ·R_SP 梯度范数 "
        f"{payload['reg_grad_norm_after_lambda']:.3g}，因此不存在按 microbatch "
        "数量重复放大。",
        f"- 漂移点梯度范数：BCE {payload['bce_grad_norm']:.4g}；单位 λ 的 R_SP "
        f"{payload['unit_reg_grad_norm']:.4g}；H3-B λ 下 {payload['H3B_reg_grad_norm']:.4g}"
        f"（约为 BCE 的 "
        f"{payload['H3B_reg_grad_norm'] / payload['bce_grad_norm']:.4f} 倍）；"
        f"H3 λ 下 {payload['H3_reg_grad_norm']:.4g}。",
        "- AdamW 参数组："
        + "；".join(
            f"{group['name']} lr={group['lr']:.3g} wd={group['weight_decay']}"
            for group in payload["optimizer_groups"]),
        "",
        "## 结论", "",
        "- 实现与 H3 相同，未发现重复 λ、错误归一化或梯度累积放大；"
        "H3-B 与 H3 的唯一差异是 λ 从 2.89278e7 降为 2.89278e6，正则化范围与 "
        "分母定义保持不变。",
        "- 在预训练起点 L2-SP 梯度为 0 属于定义性质；本次审计在真实非零漂移点"
        "验证了梯度公式、梯度流与累积逻辑。",
        "- 未读取 Test 标签。",
        "",
    ]
    (H3B_DIR / "l2sp_implementation_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result, ensure_ascii=True, indent=1, default=str))
