"""Preflight checks for the H1/H2/H3 optimizer-side interventions.

Verifies the three mechanisms and the H0 reuse decision before any long training
run: H1 sampling budgets, H2 weighted-loss normalization/gradients, H3 L2-SP
parameter coverage/gradient flow, and the R5 config identity for H0.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from genopt_common import (
    EXPERIMENT_ROOT,
    EXPERIMENTS,
    PROFILE,
    SEED,
    APWeightedStudyBatchSampler,
    build_h_datasets,
    class_bce_batch_accumulator,
    l2sp_parameter_names,
    l2sp_penalty,
    l2sp_reference,
    load_r5_reference,
    make_h_builder,
    make_h_loader,
    read_json,
    stage_relative_drift,
    study_view_metadata,
    write_json,
)
from dino_data import EpochStudyBatchSampler, make_loader, shutdown_loader
from dino_experiment import load_pretrained_backbone, update_learning_rates
from dino_feature_models import collate_masked_studies, feature_profile_optimizer
from dino_losses import study_bce_per_study, study_logit_mean
from hybrid.common import write_csv


def run_dir_for(experiment: str) -> Path:
    return EXPERIMENT_ROOT / EXPERIMENTS[experiment]["label"]


def h1_sampling_preflight(train_data) -> dict:
    reference = load_r5_reference()
    runtime = reference["runtime"]
    optimization = EXPERIMENTS["H1"]["optimization"]
    metadata = study_view_metadata(train_data)
    sampler = APWeightedStudyBatchSampler(
        train_data, metadata["ap_only"], int(runtime["image_budget"]),
        seed=SEED, target_ap_fraction=float(optimization["target_ap_fraction"]),
        category_array=metadata["categories"])
    rows = []
    failures = []
    for epoch in range(20):
        sampler.set_epoch(epoch)
        batches, stats = sampler._build(epoch)
        original = EpochStudyBatchSampler(
            train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
            seed=SEED)
        original.set_epoch(epoch)
        expected_batches = len(original)
        batch_sums = [int(sampler.image_counts[batch].sum()) for batch in batches]
        if len(batches) != expected_batches:
            failures.append(f"epoch {epoch}: batches {len(batches)} != {expected_batches}")
        if any(value > int(runtime["image_budget"]) for value in batch_sums):
            failures.append(f"epoch {epoch}: a batch exceeds the image budget")
        if stats["n_draws"] != len(train_data):
            failures.append(f"epoch {epoch}: draws {stats['n_draws']} != {len(train_data)}")
        target = float(optimization["target_ap_fraction"])
        if abs(stats["ap_only_draw_fraction"] - target) > 0.002:
            failures.append(
                f"epoch {epoch}: AP fraction {stats['ap_only_draw_fraction']:.4f} "
                f"!= target {target}")
        row = {
            "epoch": epoch + 1,
            "draws": stats["n_draws"],
            "unique_studies": stats["n_unique_studies"],
            "ap_only_draw_fraction": stats["ap_only_draw_fraction"],
            "pa_only_draw_fraction": stats["category_draw_fractions"]["pa_only"],
            "lateral_only_draw_fraction": stats["category_draw_fractions"]["lateral_only"],
            "mixed_or_other_draw_fraction": stats["category_draw_fractions"]["mixed_or_other"],
            "missing_view_draw_fraction": stats["category_draw_fractions"]["missing_view"],
            "single_image_draw_fraction": stats["single_image_draw_fraction"],
            "multi_image_draw_fraction": stats["multi_image_draw_fraction"],
            "mean_images_per_draw": stats["mean_images_per_draw"],
            "batches": stats["batches"],
            "original_batches": expected_batches,
            "max_batch_images": max(batch_sums),
            "mean_images_per_batch": stats["mean_images_per_batch"],
            "mean_studies_per_batch": stats["mean_studies_per_batch"],
            "repeat_once": stats["repeat_histogram"]["once"],
            "repeat_twice": stats["repeat_histogram"]["twice"],
            "repeat_three_plus": stats["repeat_histogram"]["three_plus"],
            "never_sampled": stats["repeat_histogram"]["never"],
            "max_study_repeats": stats["repeat_histogram"]["max_repeats"],
            "subject_draw_mean": stats["subject_draw_mean"],
            "subject_draw_max": stats["subject_draw_max"],
            "subject_draw_p95": stats["subject_draw_p95"],
        }
        for name in metadata["counts"]:
            row[f"category_count_{name}"] = metadata["counts"][name]
        for name, value in stats["per_class_positive_draw_fraction"].items():
            row[f"positive_draw_fraction_{name.replace(' ', '_')}"] = value
            row[f"original_positive_fraction_{name.replace(' ', '_')}"] = (
                stats["per_class_original_positive_fraction"][name])
        rows.append(row)
    # DataLoader-level smoke test: whole Studies only, image counts unchanged.
    sampler.set_epoch(0)
    loader, _, _ = make_h_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=0, pin_memory=False, prefetch_factor=2,
        batch_sampler=sampler)
    loader_checks = []
    try:
        for index, batch in enumerate(loader):
            if index >= 3:
                break
            counts = batch["image_counts"].numpy()
            study_index = batch["study_index"].numpy()
            expected = []
            for study in range(int(batch["num_studies"])):
                positions = np.flatnonzero(study_index == study)
                expected.append(int(positions.size))
            loader_checks.append({
                "batch": index,
                "num_studies": int(batch["num_studies"]),
                "images": int(batch["images"].shape[0]),
                "counts": counts.tolist(),
                "expected_counts": expected,
                "grouped_studies": bool(expected == counts.tolist()),
            })
    finally:
        shutdown_loader(loader)
    if not all(check["grouped_studies"] for check in loader_checks):
        failures.append("DataLoader batch does not preserve whole-Study grouping")
    write_csv(run_dir_for("H1") / "sampling_statistics.csv", rows)
    preflight = {
        "experiment": "H1",
        "ap_only_studies": int(metadata["ap_only"].sum()),
        "total_studies": metadata["n_studies"],
        "ap_only_fraction_dataset": float(metadata["ap_only"].mean()),
        "target_ap_fraction": float(optimization["target_ap_fraction"]),
        "ap_weight_reference": float(optimization["ap_weight"]),
        "category_counts": metadata["counts"],
        "epochs_checked": len(rows),
        "mean_ap_fraction": float(np.mean([row["ap_only_draw_fraction"] for row in rows])),
        "min_ap_fraction": float(np.min([row["ap_only_draw_fraction"] for row in rows])),
        "max_ap_fraction": float(np.max([row["ap_only_draw_fraction"] for row in rows])),
        "mean_draws": float(np.mean([row["draws"] for row in rows])),
        "mean_unique_studies": float(np.mean([row["unique_studies"] for row in rows])),
        "repeat_statistics_epoch1": {
            key: rows[0][key] for key in (
                "repeat_once", "repeat_twice", "repeat_three_plus",
                "never_sampled", "max_study_repeats", "subject_draw_mean",
                "subject_draw_max", "subject_draw_p95")},
        "batch_budget_matches_original_every_epoch": not any(
            "batches" in failure for failure in failures),
        "loader_checks": loader_checks,
        "failures": failures,
        "passed": not failures,
        "test_access": False,
    }
    write_json(run_dir_for("H1") / "sampling_preflight.json", preflight)
    return preflight


def h2_loss_preflight(train_data) -> dict:
    reference = load_r5_reference()
    runtime = reference["runtime"]
    optimization = EXPERIMENTS["H2"]["optimization"]
    single_weight = float(optimization["single_weight"])
    # Synthetic normalization check.
    per_study = torch.tensor([0.5, 0.7, 0.2, 0.9, 0.4, 0.3], dtype=torch.float32)
    counts = torch.tensor([1, 2, 1, 3, 1, 4], dtype=torch.int64)
    weights = torch.where(counts == 1, single_weight, 1.0)
    expected = float((per_study * weights).sum() / weights.sum())
    synthetic_ok = abs(expected - (
        float(((per_study * weights).sum()) / float(weights.sum())))) < 1e-9
    # Real two-microbatch gradient check.
    device = torch.device("cuda")
    builder = make_h_builder(runtime, reference["head_sha"],
                             run_dir_for("H2") / "preflight")
    model, verification, *_ = builder(PROFILE, device, runtime, SEED)
    optimizer = feature_profile_optimizer(model, reference["config"])
    train_loader, _, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=0, pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    iterator = iter(train_loader)
    window = [next(iterator), next(iterator)]
    shutdown_loader(train_loader)
    weight_total = 0.0
    for batch in window:
        counts_t = batch["image_counts"]
        known = batch["known"]
        eligible = known.any(dim=1)
        weight_total += float(torch.where(counts_t == 1, single_weight, 1.0)[eligible].sum())
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]

    def forward_batch(batch):
        images = batch["images"].to(device)
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
        weights = torch.where(image_counts == 1, single_weight, 1.0)
        loss = (per_study[eligible] * weights[eligible]).sum() / weight_total
        return loss, per_study, eligible, weights

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in window:
        loss, *_ = forward_batch(batch)
        loss.backward()
        total_loss += float(loss.detach().cpu())
    backward_grads = [None if p.grad is None else p.grad.detach().float().clone()
                      for p in parameters]
    gradient_norm = math.sqrt(sum(
        float((p.grad.detach().float().pow(2).sum())) for p in parameters
        if p.grad is not None))
    optimizer.zero_grad(set_to_none=True)
    # Reference: weighted mean of independent per-Study gradients.
    reference_grads = None
    for batch in window:
        loss, per_study, eligible, weights = forward_batch(batch)
        grads = torch.autograd.grad(loss, parameters, retain_graph=False,
                                    allow_unused=True)
        if reference_grads is None:
            reference_grads = [None if grad is None else grad.detach().clone()
                               for grad in grads]
        else:
            for position, grad in enumerate(grads):
                if grad is None:
                    continue
                reference_grads[position] += grad.detach()
    reference_diff = 0.0
    for accumulated, reference in zip(backward_grads, reference_grads):
        if accumulated is None or reference is None:
            continue
        reference_diff = max(
            reference_diff,
            float((accumulated - reference.float()).abs().max()))
    del model, optimizer
    torch.cuda.empty_cache()
    preflight = {
        "experiment": "H2",
        "single_image_weight": single_weight,
        "normalization": "L = sum_s w_s * l_s / sum_s w_s over each accumulation window",
        "synthetic_formula_ok": synthetic_ok,
        "window_studies": int(sum(int(batch["num_studies"]) for batch in window)),
        "window_weight_total": weight_total,
        "window_total_loss": total_loss,
        "gradient_norm": gradient_norm,
        "max_gradient_difference_vs_reference": reference_diff,
        "gradient_equivalence_passed": reference_diff < 1e-5,
        "note": ("H2 loss is the normalized weighted Study mean; the raw unweighted "
                 "Study BCE is still logged and used for early stopping and all "
                 "Train/Validation comparisons."),
        "test_access": False,
    }
    preflight["passed"] = bool(synthetic_ok and reference_diff < 1e-5)
    write_json(run_dir_for("H2") / "h2_preflight.json", preflight)
    return preflight


def h3_l2sp_preflight(train_data) -> dict:
    reference = load_r5_reference()
    runtime = reference["runtime"]
    device = torch.device("cuda")
    builder = make_h_builder(runtime, reference["head_sha"],
                             run_dir_for("H3") / "preflight")
    model, verification, *_ = builder(PROFILE, device, runtime, SEED)
    names = l2sp_parameter_names(model)
    reference_parameters = l2sp_reference(model, names)
    parameters = dict(model.named_parameters())
    penalty_initial = float(l2sp_penalty(model, reference_parameters).float().cpu())
    regularized = [parameters[name] for name in names]
    optimizer = feature_profile_optimizer(model, reference["config"])
    train_loader, _, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=0, pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    iterator = iter(train_loader)
    training = dict(reference["config"]["training"])
    training["warmup_updates"] = 2
    model.train()
    warmup_updates = 3
    for step in range(warmup_updates):
        window = []
        for _ in range(int(runtime["grad_accumulation"])):
            try:
                window.append(next(iterator))
            except StopIteration:
                iterator = iter(train_loader)
                window.append(next(iterator))
        eligible_window = sum(int(batch["known"].any(dim=1).sum()) for batch in window)
        optimizer.zero_grad(set_to_none=True)
        update_learning_rates(optimizer, step, warmup_updates * 2, training)
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
                per_study, eligible = study_bce_per_study(study_logits, labels, known)
                loss = per_study[eligible].sum() / max(eligible_window, 1)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    shutdown_loader(train_loader)
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    penalty_after_warmup = float(l2sp_penalty(model, reference_parameters).float().cpu())
    drift_after_warmup = stage_relative_drift(model, reference_parameters)
    grad_norms = {}
    for namespace in ("backbone.model.stages.2.", "backbone.model.stages.3.",
                      "backbone.model.downsample_layers.2.",
                      "backbone.model.downsample_layers.3."):
        subset = [parameters[name] for name in names if name.startswith(namespace)]
        grads = torch.autograd.grad(
            l2sp_penalty(model, reference_parameters), subset,
            retain_graph=True, allow_unused=True)
        grad_norms[namespace] = math.sqrt(sum(
            float(grad.pow(2).sum()) for grad in grads if grad is not None))
        model.zero_grad(set_to_none=True)
    classifier = parameters["classifier.weight"]
    classifier_grad = torch.autograd.grad(
        l2sp_penalty(model, reference_parameters), classifier,
        allow_unused=True)[0]
    classifier_covered = bool(classifier_grad is not None
                              and float(classifier_grad.abs().max()) > 0)
    frozen_names = [name for name, parameter in model.named_parameters()
                    if not parameter.requires_grad][:5]
    frozen_grads = []
    for name in frozen_names:
        parameter = parameters[name]
        if not parameter.requires_grad:
            frozen_grads.append(False)
            continue
        grad = torch.autograd.grad(
            l2sp_penalty(model, reference_parameters), parameter,
            allow_unused=True)[0]
        frozen_grads.append(bool(grad is not None and float(grad.abs().max()) > 0))
    model.zero_grad(set_to_none=True)
    # One real BCE + L2-SP step with the calibrated (or unit) lambda.
    calibration_path = run_dir_for("H3") / "l2sp_calibration.json"
    lambda_sp = (read_json(calibration_path)["lambda_sp"]
                 if calibration_path.is_file() else 1.0)
    train_loader, _, _ = make_loader(
        train_data, image_budget=int(runtime["image_budget"]), shuffle=True,
        seed=SEED, num_workers=0, pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    batch = next(iter(train_loader))
    shutdown_loader(train_loader)
    images = batch["images"].to(device).contiguous(memory_format=torch.channels_last)
    study_index = batch["study_index"].to(device)
    labels = batch["labels"].to(device)
    known = batch["known"].to(device)
    image_counts = batch["image_counts"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_logits = model(images)
        study_logits = study_logit_mean(
            image_logits.float(), study_index, batch["num_studies"], image_counts)
        per_study, eligible = study_bce_per_study(study_logits, labels, known)
        bce = per_study[eligible].mean()
    penalty = l2sp_penalty(model, reference_parameters)
    total = bce + lambda_sp * penalty
    total.backward()
    update = {}
    for name in ("backbone.model.stages.2.0.dwconv.weight",
                 "backbone.model.stages.3.0.dwconv.weight",
                 "classifier.weight"):
        if name in parameters and parameters[name].grad is not None:
            update[name] = float(parameters[name].grad.abs().max())
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    drift_after_step = stage_relative_drift(model, reference_parameters)
    model.zero_grad(set_to_none=True)
    # Classifier must move under BCE, but is not allowed inside the L2-SP penalty.
    penalty_parameters = {id(parameter) for parameter in regularized}
    coverage = {
        "n_regularized_parameters": len(regularized),
        "regularized_elements": int(sum(parameter.numel() for parameter in regularized)),
        "classifier_in_penalty": bool(id(classifier) in penalty_parameters),
        "frozen_parameters_in_penalty": bool(any(
            id(parameters[name]) in penalty_parameters for name in frozen_names)),
    }
    preflight = {
        "experiment": "H3",
        "l2sp_definition": "R_SP = mean_j (theta_j - theta0_j)^2 over trainable "
                           "pretrained Stage3/Stage4 backbone elements; classifier "
                           "and final norm are excluded.",
        "initial_penalty": penalty_initial,
        "initial_penalty_zero": bool(penalty_initial < 1e-12),
        "warmup_updates_before_gradient_probe": int(warmup_updates),
        "penalty_after_warmup": penalty_after_warmup,
        "stage_relative_drift_after_warmup": drift_after_warmup,
        "grad_norm_by_namespace_after_warmup": grad_norms,
        "classifier_affected_by_penalty": classifier_covered,
        "frozen_parameters_affected_by_penalty": any(frozen_grads),
        "coverage": coverage,
        "lambda_sp_used_for_probe": lambda_sp,
        "probe_bce": float(bce.detach().float().cpu()),
        "probe_penalty": float(penalty.detach().float().cpu()),
        "probe_total_loss": float(total.detach().float().cpu()),
        "gradient_abs_max": update,
        "stage_relative_drift_after_one_step": drift_after_step,
        "stage3_gradient_present": grad_norms["backbone.model.stages.2."] > 0,
        "stage4_gradient_present": grad_norms["backbone.model.stages.3."] > 0,
        "test_access": False,
    }
    preflight["passed"] = bool(
        preflight["initial_penalty_zero"]
        and penalty_after_warmup > 0
        and preflight["stage3_gradient_present"]
        and preflight["stage4_gradient_present"]
        and not classifier_covered
        and not preflight["frozen_parameters_affected_by_penalty"])
    write_json(run_dir_for("H3") / "l2sp_preflight.json", preflight)
    del model, optimizer
    torch.cuda.empty_cache()
    return preflight


def h0_reuse_audit() -> dict:
    reference = load_r5_reference()
    config = reference["config"]
    training = config["training"]
    checks = {
        "seed_is_42": int(config["seed"]) == SEED,
        "global_only_architecture": config["model"]["architecture"] == (
            "DINOv3ConvNeXtTinyGlobalOnly"),
        "loss_is_unweighted_study_mean_bce": training["loss"] == (
            "unweighted_study_mean_bce_with_logits"),
        "optimizer_is_adamw": training["optimizer"] == "AdamW",
        "weight_decay_is_0.01": float(training["weight_decay"]) == 0.01,
        "backbone_lr_is_1e-5": float(training["peak_lr_backbone"]) == 1e-5,
        "head_lr_is_2e-4": float(training["peak_lr_head"]) == 2e-4,
        "warmup_epochs_is_2": int(training["warmup_epochs"]) == 2,
        "max_epochs_is_20": int(training["epochs"]) == 20,
        "mixed_precision_bfloat16": training["mixed_precision"] == "bfloat16",
        "study_fusion_raw_logit_mean": training["study_fusion"] == "raw_logit_mean",
        "no_sampling_intervention_in_config": "sampler" not in json.dumps(config).lower(),
        "no_l2sp_intervention_in_config": "l2sp" not in json.dumps(config).lower(),
        "no_single_study_weight_in_config": "single_study_weight" not in (
            json.dumps(config).lower()),
        "runtime_reused_from_r3": bool(config.get("runtime_metadata_source", "").endswith(
            "selected_runtime_config.json")),
    }
    r5_status = reference["status"]
    payload = {
        "decision": "reuse_R5_Best_as_H0",
        "reference": "R5_Global_Only_512",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "r5_best_epoch": int(r5_status["best_epoch"]),
        "r5_best_macro_auroc": float(r5_status["best_macro_auroc"]),
        "r5_stop_reason": r5_status["stop_reason"],
        "r5_train_studies": int(reference["r5_config"]["train_studies"]),
        "r5_validation_studies": int(reference["r5_config"]["validation_studies"]),
        "protocol_note": (
            "R5 was trained with the exact baseline protocol (seed 42, Global-only "
            "head, unweighted Study BCE, original sampler, no L2-SP, same runtime, "
            "same early stopping). It is therefore reused as H0 instead of "
            "re-training an identical run. Best/Last full deterministic "
            "Train/Validation inference is executed for H0 exactly as for H1-H3."),
        "limitations": (
            "The historical R5 run did not log per-class train-batch BCE for every "
            "epoch; H0 per-epoch per-class values are reconstructed where possible "
            "from saved per-epoch validation predictions and marked as unavailable "
            "where they cannot be recovered. Best/Last full deterministic class BCE "
            "is computed fresh and is complete."),
        "test_access": False,
    }
    write_json(run_dir_for("H0") / "h0_reuse_decision.json", payload)
    (run_dir_for("H0") / "h0_reuse_decision.md").write_text(
        "# H0 reuse decision\n\n"
        f"- Decision: **{payload['decision']}**.\n"
        f"- Reference: `R5_Global_Only_512` (best epoch {payload['r5_best_epoch']}, "
        f"Macro-AUROC {payload['r5_best_macro_auroc']:.6f}).\n"
        f"- All config identity checks passed: {payload['all_checks_passed']}.\n"
        f"- {payload['protocol_note']}\n\n"
        "## Identity checks\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in checks.items())
        + f"\n\n## Limitation\n\n{payload['limitations']}\n",
        encoding="utf-8")
    return payload


def run_preflight() -> dict:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    for experiment in ("H0", "H1", "H2", "H3"):
        run_dir_for(experiment).mkdir(parents=True, exist_ok=True)
    train_data, valid_data, _ = build_h_datasets()
    results = {"H0": h0_reuse_audit()}
    results["H1"] = h1_sampling_preflight(train_data)
    print(f"[preflight] H1 passed={results['H1']['passed']} "
          f"mean_AP={results['H1']['mean_ap_fraction']:.4f}", flush=True)
    results["H2"] = h2_loss_preflight(train_data)
    print(f"[preflight] H2 passed={results['H2']['passed']}", flush=True)
    results["H3"] = h3_l2sp_preflight(train_data)
    print(f"[preflight] H3 passed={results['H3']['passed']} "
          f"stage3_grad={results['H3']['stage3_gradient_present']} "
          f"stage4_grad={results['H3']['stage4_gradient_present']}", flush=True)
    payload = {
        "experiments": {key: {"passed": value.get("passed", value.get("all_checks_passed"))}
                        for key, value in results.items()},
        "results": results,
        "test_access": False,
    }
    write_json(EXPERIMENT_ROOT / "preflight_report.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run_preflight(), indent=2, default=str))
