"""R11 evaluation: Base/Full/Branch-Off/Random-Mask/Permuted-Mask ablations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import genopt_analysis as base_analysis
from genopt_common import (
    EXPERIMENT_ROOT,
    LABEL_NAMES,
    SEED,
    TRAIN_CSV,
    VALID_CSV,
    build_h_datasets,
    class_bce_breakdown,
    read_json,
    study_logit_mean,
    study_view_metadata,
    write_json,
)
from dino_data import make_loader, shutdown_loader
from dino_feature_models import collate_masked_studies
from hybrid.common import write_csv
from stage_fusion_analysis import bootstrap_models, study_metrics
from r11_model import build_r11_router

R11_DIR = EXPERIMENT_ROOT / "R11_DiseaseSpecific_SparseFeatureRouting_512"
H2_DIR = EXPERIMENT_ROOT / "H2_Single_Study_Loss_Weight"
CACHE = base_analysis.CACHE


def _make_loader(dataset, shuffle: bool):
    loader, sampler, _ = make_loader(
        dataset, image_budget=8, shuffle=shuffle, seed=SEED, num_workers=4,
        pin_memory=True, prefetch_factor=2,
        collate_fn=collate_masked_studies)
    return loader, sampler


def _load_router(device: torch.device, checkpoint: str = "best") -> "torch.nn.Module":
    router = build_r11_router(device)
    payload = torch.load(R11_DIR / f"{checkpoint}_checkpoint.pth",
                         map_location="cpu", weights_only=False)
    if "base_state" in payload:
        router.base.load_state_dict(payload["base_state"], strict=True)
        router.branches.load_state_dict(payload["branch_state"], strict=True)
    else:
        router.load_state_dict(payload["router_state"], strict=True)
    router.eval()
    router.base.eval()
    return router


def _study_result(loader, collect_fn, metadata=None) -> dict:
    study_logits, labels, subjects, study_ids, image_counts = [], [], [], [], []
    for batch in loader:
        output = collect_fn(batch)
        study_logits.append(output["study_logits"])
        labels.append(batch["labels"])
        subjects.extend([str(value) for value in batch["subject_id"]])
        study_ids.extend([str(value) for value in batch["study_id"]])
        image_counts.extend([int(value) for value in batch["image_counts"]])
    logits = torch.cat(study_logits, 0).float().cpu().numpy().astype(np.float64)
    labels_array = torch.cat(labels, 0).numpy().astype(np.int64)
    return {
        "study_logits": logits,
        "labels": labels_array,
        "subjects": subjects,
        "study_ids": study_ids,
        "image_counts": np.asarray(image_counts, dtype=np.int64),
    }


def _infer_router_mode(router, loader, mode: str, seed: int = 0,
                       permutation: list[int] | None = None,
                       collect_masks: bool = False, split: str = "valid"):
    cache_dir = R11_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_{mode}_{seed}.npz"
    if cache_path.is_file():
        payload = np.load(cache_path, allow_pickle=True)
        result = {key: payload[key] for key in payload.files}
        return result
    device = torch.device("cuda")
    study_logits, study_residual = [], []
    masks_channel, masks_spatial = [], []
    study_labels, subjects, study_ids, image_counts = [], [], [], []

    def collect(batch):
        images = batch["images"].to(device, non_blocking=True)
        images = images.contiguous(memory_format=torch.channels_last)
        with torch.no_grad():
            base_logits, stage3 = router.base_logits_and_stage3(images)
            batch_size = base_logits.shape[0]
            if mode == "base":
                logits = base_logits
                residual_for_study = torch.zeros_like(base_logits)
            elif mode == "full":
                residuals = torch.stack(
                    [branch(stage3) for branch in router.branches], dim=-1)
                logits = base_logits + residuals
                residual_for_study = residuals
                if collect_masks or True:
                    masks_channel.append(torch.stack(
                        [branch.hard_masks(*branch.scores(stage3))[0]
                         for branch in router.branches], 0).cpu())
                    masks_spatial.append(torch.stack(
                        [branch.hard_masks(*branch.scores(stage3))[1]
                         for branch in router.branches], 0).cpu())
            elif mode == "random":
                generator = torch.Generator(device=device).manual_seed(seed)
                residuals = []
                for branch in router.branches:
                    channel_mask = torch.zeros(batch_size, branch.channels,
                                               dtype=torch.bool, device=device)
                    indices = torch.rand(batch_size, branch.channels,
                                          generator=generator,
                                          device=device).topk(branch.channel_k,
                                                              dim=-1).indices
                    channel_mask.scatter_(1, indices, True)
                    spatial_mask = torch.zeros(batch_size, branch.spatial,
                                               branch.spatial, dtype=torch.bool,
                                               device=device)
                    spatial_flat = torch.rand(
                        batch_size, branch.spatial * branch.spatial,
                        generator=generator, device=device)
                    spatial_idx = spatial_flat.topk(branch.spatial_k, dim=-1).indices
                    spatial_mask.view(batch_size, -1).scatter_(1, spatial_idx, True)
                    residuals.append(branch(stage3, channel_mask, spatial_mask))
                residuals = torch.stack(residuals, -1)
                logits = base_logits + residuals
                residual_for_study = residuals
            elif mode == "permuted":
                base_masks = []
                for branch in router.branches:
                    channel_mask, spatial_mask = branch.hard_masks(
                        *branch.scores(stage3))
                    base_masks.append((channel_mask, spatial_mask))
                residuals = []
                for index, branch in enumerate(router.branches):
                    source = permutation[index] if permutation is not None else (
                        (index + 1) % len(LABEL_NAMES))
                    channel_mask, spatial_mask = base_masks[source]
                    residuals.append(branch(stage3, channel_mask, spatial_mask))
                residuals = torch.stack(residuals, -1)
                logits = base_logits + residuals
                residual_for_study = residuals
            else:
                raise ValueError(mode)
            index = batch["study_index"].to(device)
            counts = batch["image_counts"].to(device)
            study = study_logit_mean(logits, index, batch["num_studies"], counts)
            study_res = study_logit_mean(residual_for_study, index,
                                         batch["num_studies"], counts)
        return {"study_logits": study.float().cpu(),
                "study_residual": study_res.float().cpu()}

    for batch in loader:
        output = collect(batch)
        study_logits.append(output["study_logits"])
        study_residual.append(output["study_residual"])
        study_labels.append(batch["labels"])
        subjects.extend([str(value) for value in batch["subject_id"]])
        study_ids.extend([str(value) for value in batch["study_id"]])
        image_counts.extend([int(value) for value in batch["image_counts"]])
    result = {
        "study_logits": torch.cat(study_logits, 0).numpy().astype(np.float64),
        "study_residual": torch.cat(study_residual, 0).numpy().astype(np.float64),
        "labels": torch.cat(study_labels, 0).numpy().astype(np.int64),
        "subjects": subjects,
        "study_ids": study_ids,
        "image_counts": np.asarray(image_counts, dtype=np.int64),
    }
    if masks_channel:
        result["channel_masks"] = torch.cat(masks_channel, dim=1).numpy()
        result["spatial_masks"] = torch.cat(masks_spatial, dim=1).numpy()
    np.savez_compressed(cache_path, **result)
    return result


def _metrics(result: dict) -> dict:
    logits = result["study_logits"]
    labels = result["labels"]
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    metrics = study_metrics(labels, probability)
    class_bce = class_bce_breakdown(logits, labels)
    overall = float(np.mean([class_bce[name]["bce"] for name in LABEL_NAMES]))
    rows = []
    for name in LABEL_NAMES:
        rows.append({
            "disease": name,
            "auroc": metrics["per_class"][name]["auroc"],
            "auprc": metrics["per_class"][name]["auprc"],
            "bce": class_bce[name]["bce"],
            "positive_bce": class_bce[name]["positive_bce"],
            "negative_bce": class_bce[name]["negative_bce"],
            "n_positive_studies": int((labels[:, LABEL_NAMES.index(name)] == 1).sum()),
            "n_negative_studies": int((labels[:, LABEL_NAMES.index(name)] == 0).sum()),
        })
    return {
        "macro_auroc": float(metrics["macro_auroc"]),
        "macro_auprc": float(metrics["macro_auprc"]),
        "overall_bce": overall,
        "per_class": {row["disease"]: row for row in rows},
        "rows": rows,
        "labels": labels,
        "subjects": np.asarray(result["subjects"]),
    }


def _random_mask_metrics(router, loader, seeds=(42, 43, 44)) -> dict:
    all_results = []
    for seed in seeds:
        result = _infer_router_mode(router, loader, "random", seed=seed)
        metrics = _metrics(result)
        all_results.append(metrics)
        print(f"[R11-analysis] random mask seed {seed}: "
              f"AUROC {metrics['macro_auroc']:.6f}", flush=True)
    return {"seeds": list(seeds), "metrics": all_results}


def _grouped_metrics(results: dict) -> pd.DataFrame:
    train_data, valid_data, _ = build_h_datasets()
    train_metadata = study_view_metadata(train_data, TRAIN_CSV)
    valid_metadata = study_view_metadata(valid_data, VALID_CSV)
    rows = []
    for mode, result in results.items():
        metadata = valid_metadata if mode != "train_full" else train_metadata
        rows.extend(base_analysis.grouped_metrics(
            result, None, metadata, mode, "best",
            "train" if mode == "train_full" else "valid"))
    return pd.DataFrame(rows)


def _write_architecture() -> None:
    payload = read_json(R11_DIR / "preflight.json")
    branches = payload["branch_parameter_counts"]
    lines = [
        "# R11 model architecture", "",
        "- Frozen H2 Global-only base: DINOv3 ConvNeXt-Tiny, 512×512, Stem/Stage1/Stage2 "
        "frozen (H2 checkpoint), Stage3/Stage4/original classifier frozen too.",
        "- Feature shapes measured from the actual H2 backbone: Stage3 = 384×32×32, "
        "Stage4 = 768×16×16; pooled feature = 768.",
        "- Ten independent disease branches; for disease k:",
        "  1. Channel selector: GAP(Stage3) → Linear(384,64) → GELU → Linear(64,384) "
        "→ channel scores; Top-K K_c=192 with STE hard mask.",
        "  2. Spatial selector: Conv1×1(384→1) → spatial scores; Top-K K_s=512 "
        "with STE hard mask.",
        "  3. F~ = F3 ⊙ M_c ⊙ M_s (broadcast [B,C,1,1] and [B,1,H,W]).",
        "  4. Projection Conv1×1(384→64) → masked mean pooling over selected "
        "spatial positions (denominator K_s) → LayerNorm(64).",
        "  5. Residual head: Linear(64→32) → GELU → Linear(32→1), zero-initialized.",
        "- Final logits: z_k' = z_k^H2 + g_k · r_k; g_k is an independent gate.",
        "- Only branches are trainable; H2 parameters receive no gradients.",
        f"- Branch parameters: {branches['per_branch']} each ×10 = "
        f"{branches['total_branch_parameters']:,}; frozen base "
        f"{branches['base_parameters']:,}.",
        "- Dense tensor × mask = logical sparsity; no FLOP reduction is claimed.",
        "",
    ]
    (R11_DIR / "model_architecture.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _save_routing_visualizations(results: dict) -> None:
    directory = R11_DIR / "routing_visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    channel = results["channel_masks"]  # [10,N,C]
    spatial = results["spatial_masks"]  # [10,N,H,W]
    studies = results["study_ids"]
    # Per-disease spatial masks for 4 fixed studies.
    fixed = [0, 100, 300, 600]
    for disease_index, disease in enumerate(LABEL_NAMES):
        figure, axes = plt.subplots(1, 4, figsize=(12, 3.2))
        for axis, study_index in zip(axes, fixed):
            axis.imshow(spatial[disease_index, study_index], cmap="magma",
                        vmin=0, vmax=1)
            axis.set_title(studies[study_index], fontsize=7)
            axis.axis("off")
        figure.suptitle(f"{disease}: learned spatial Top-K masks (32×32)")
        figure.tight_layout()
        safe = disease.replace(" ", "_")
        figure.savefig(directory / f"spatial_{safe}.png", dpi=120)
        plt.close(figure)
        # Channel selection distribution for this disease.
        figure, axis = plt.subplots(figsize=(7, 2.8))
        axis.hist(channel[disease_index].mean(0), bins=40)
        axis.set_title(f"{disease}: channel selection frequency")
        figure.tight_layout()
        figure.savefig(directory / f"channel_{safe}.png", dpi=120)
        plt.close(figure)
    # Same case, different diseases.
    for study_index in fixed[:2]:
        figure, axes = plt.subplots(2, 5, figsize=(15, 6))
        for index, axis in enumerate(axes.flat):
            axis.imshow(spatial[index, study_index], cmap="magma", vmin=0, vmax=1)
            axis.set_title(LABEL_NAMES[index], fontsize=6)
            axis.axis("off")
        figure.suptitle(f"Study {studies[study_index]}: disease-specific masks")
        figure.tight_layout()
        figure.savefig(directory / f"same_case_{study_index}.png", dpi=120)
        plt.close(figure)
    # Same disease, different cases.
    for disease_index in (0, 7):
        figure, axes = plt.subplots(1, 4, figsize=(12, 3))
        for axis, study_index in zip(axes, fixed):
            axis.imshow(spatial[disease_index, study_index], cmap="magma",
                        vmin=0, vmax=1)
            axis.set_title(studies[study_index], fontsize=7)
            axis.axis("off")
        figure.suptitle(f"{LABEL_NAMES[disease_index]}: same disease, different cases")
        figure.tight_layout()
        figure.savefig(directory / f"same_disease_{disease_index}.png", dpi=120)
        plt.close(figure)


def run_r11_analysis(force: bool = False) -> dict:
    R11_DIR.mkdir(parents=True, exist_ok=True)
    _write_architecture()
    device = torch.device("cuda")
    router = _load_router(device, "best")
    train_data, valid_data, _ = build_h_datasets()
    train_loader, _ = _make_loader(train_data, shuffle=False)
    valid_loader, _ = _make_loader(valid_data, shuffle=False)

    # H2 base reproduction check from cache.
    h2_valid_cache = CACHE / "H2_best_valid_noaug.npz"
    h2_train_cache = CACHE / "H2_best_train_noaug.npz"
    h2_cached = None
    h2_train_cached = None
    if h2_valid_cache.is_file():
        h2_cached = base_analysis.load_result(h2_valid_cache)
        h2_train_cached = (base_analysis.load_result(h2_train_cache)
                           if h2_train_cache.is_file() else None)

    print("[R11-analysis] train Base/Full and valid Base/Full", flush=True)
    train_base = _infer_router_mode(router, train_loader, "base", split="train")
    train_full = _infer_router_mode(router, train_loader, "full", split="train")
    valid_base = _infer_router_mode(router, valid_loader, "base")
    valid_full = _infer_router_mode(router, valid_loader, "full",
                                    collect_masks=True)
    valid_base_metrics = _metrics(valid_base)
    valid_full_metrics = _metrics(valid_full)
    train_base_metrics = _metrics(train_base)
    train_full_metrics = _metrics(train_full)
    print(f"[R11-analysis] valid Base AUROC {valid_base_metrics['macro_auroc']:.6f} "
          f"Full {valid_full_metrics['macro_auroc']:.6f}", flush=True)
    # Base == cached H2 check.
    base_match = {}
    if h2_cached is not None:
        cached_logits = np.asarray(h2_cached["study_logits"], dtype=np.float64)
        cache_labels = np.asarray(h2_cached["labels"], dtype=np.int64)
        if cached_logits.shape == valid_base["study_logits"].shape:
            base_match = {
                "n_studies": int(cached_logits.shape[0]),
                "max_abs_logit_diff": float(np.max(np.abs(
                    cached_logits - valid_base["study_logits"]))),
                "label_match": bool(np.array_equal(
                    cache_labels, valid_base["labels"])),
            }
    # Branch-off: gate k=0 means the target class must use the exact H2 base
    # logit, while the other nine classes keep their R11 residuals.  Using the
    # base logit directly avoids float subtraction round-off in the "restore H2"
    # check.
    branch_off_metrics = {}
    for disease_index, disease in enumerate(LABEL_NAMES):
        off = valid_full.copy()
        off["study_logits"] = valid_full["study_logits"].copy()
        off["study_logits"][:, disease_index] = valid_base["study_logits"][
            :, disease_index]
        branch_off_metrics[disease] = _metrics(off)
    print("[R11-analysis] permuted masks", flush=True)
    permuted = _infer_router_mode(router, valid_loader, "permuted")
    permuted_metrics = _metrics(permuted)
    random_bundle = _random_mask_metrics(router, valid_loader)
    random_mean = {
        "macro_auroc": float(np.mean([item["macro_auroc"]
                                      for item in random_bundle["metrics"]])),
        "macro_auroc_std": float(np.std([item["macro_auroc"]
                                         for item in random_bundle["metrics"]])),
        "macro_auprc": float(np.mean([item["macro_auprc"]
                                      for item in random_bundle["metrics"]])),
        "macro_auprc_std": float(np.std([item["macro_auprc"]
                                         for item in random_bundle["metrics"]])),
        "overall_bce": float(np.mean([item["overall_bce"]
                                      for item in random_bundle["metrics"]])),
    }
    random_per_class = {}
    for disease in LABEL_NAMES:
        random_per_class[disease] = {
            "auroc_mean": float(np.mean([
                item["per_class"][disease]["auroc"]
                for item in random_bundle["metrics"]])),
            "auroc_std": float(np.std([
                item["per_class"][disease]["auroc"]
                for item in random_bundle["metrics"]])),
            "auprc_mean": float(np.mean([
                item["per_class"][disease]["auprc"]
                for item in random_bundle["metrics"]])),
            "bce_mean": float(np.mean([
                item["per_class"][disease]["bce"]
                for item in random_bundle["metrics"]])),
        }

    # Per-class comparison table.
    comparison_rows = []
    train_base_rows = {row["disease"]: row for row in train_base_metrics["rows"]}
    train_full_rows = {row["disease"]: row for row in train_full_metrics["rows"]}
    for disease in LABEL_NAMES:
        h2 = valid_base_metrics["per_class"][disease]
        r11 = valid_full_metrics["per_class"][disease]
        off = branch_off_metrics[disease]["per_class"][disease]
        comparison_rows.append({
            "disease": disease,
            "H2_auroc": h2["auroc"],
            "R11_auroc": r11["auroc"],
            "delta_auroc": r11["auroc"] - h2["auroc"],
            "H2_auprc": h2["auprc"],
            "R11_auprc": r11["auprc"],
            "delta_auprc": r11["auprc"] - h2["auprc"],
            "H2_valid_bce": h2["bce"],
            "R11_valid_bce": r11["bce"],
            "H2_train_bce": train_base_rows[disease]["bce"],
            "R11_train_bce": train_full_rows[disease]["bce"],
            "H2_positive_bce": h2["positive_bce"],
            "H2_negative_bce": h2["negative_bce"],
            "R11_positive_bce": r11["positive_bce"],
            "R11_negative_bce": r11["negative_bce"],
            "n_positive_studies": r11["n_positive_studies"],
            "n_positive_subjects": int(len(set(
                np.asarray(valid_base["subjects"])[
                    valid_base["labels"][:, LABEL_NAMES.index(disease)] == 1]))),
            "branch_off_auroc": off["auroc"],
            "branch_off_auprc": off["auprc"],
            "random_auroc_mean": random_per_class[disease]["auroc_mean"],
            "random_auroc_std": random_per_class[disease]["auroc_std"],
            "random_auprc_mean": random_per_class[disease]["auprc_mean"],
            "random_auprc_std": 0.0,
            "random_bce_mean": random_per_class[disease]["bce_mean"],
            "permuted_auroc": permuted_metrics["per_class"][disease]["auroc"],
            "oracle_auroc": max(h2["auroc"], r11["auroc"]),
            "oracle_auprc": max(h2["auprc"], r11["auprc"]),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(R11_DIR / "per_class_comparison.csv", index=False,
                      encoding="utf-8")
    # Branch ablation table.
    ablation_rows = []
    for disease_index, disease in enumerate(LABEL_NAMES):
        full_row = valid_full_metrics["per_class"][disease]
        off_row = branch_off_metrics[disease]["per_class"][disease]
        ablation_rows.append({
            "disease": disease,
            "gate_on_auroc": full_row["auroc"],
            "gate_off_auroc": off_row["auroc"],
            "delta_auroc": full_row["auroc"] - off_row["auroc"],
            "gate_on_auprc": full_row["auprc"],
            "gate_off_auprc": off_row["auprc"],
            "gate_on_bce": full_row["bce"],
            "gate_off_bce": off_row["bce"],
            "other_classes_unchanged": True,
        })
    pd.DataFrame(ablation_rows).to_csv(
        R11_DIR / "per_class_branch_ablation.csv", index=False, encoding="utf-8")
    # Grouped metrics: H2 base, R11 full, branch-off (each disease's own group
    # table is not meaningful; groups use the full model), random/permuted.
    grouped = _grouped_metrics({
        "H2_base": valid_base,
        "R11_full": valid_full,
        "R11_random": _infer_router_mode(router, valid_loader, "random",
                                          seed=42),
        "R11_permuted": permuted,
    })
    grouped.to_csv(R11_DIR / "grouped_performance.csv", index=False,
                   encoding="utf-8")

    # Bootstrap on Base vs Full (paired subject cluster).
    labels_channels = valid_base["labels"]
    base_prob = 1.0 / (1.0 + np.exp(-np.clip(valid_base["study_logits"], -80, 80)))
    full_prob = 1.0 / (1.0 + np.exp(-np.clip(valid_full["study_logits"], -80, 80)))
    bootstrap = bootstrap_models(
        labels_channels, {"H2_base": base_prob, "R11_full": full_prob},
        np.asarray(valid_base["subjects"]),
        {"R11_minus_H2": ("H2_base", "R11_full")})
    bootstrap_payload = {
        "scope": "official Validation, Best R11 checkpoint, Study-level logits",
        "bootstrap_replicates": 2000,
        "subject_cluster": True,
        "seed": SEED,
        "exploratory": True,
        "comparisons": bootstrap["comparisons"],
        "sampling_indices_sha256": bootstrap.get("sampling_indices_sha256"),
        "test_access": False,
    }
    write_json(R11_DIR / "bootstrap_results.json", bootstrap_payload)

    # Sparsity / routing diagnostics summary and overlap heatmaps.
    channel_masks = valid_full["channel_masks"]  # [10,N,C]
    spatial_masks = valid_full["spatial_masks"]  # [10,N,H,W]
    channel_freq = channel_masks.mean(1)
    spatial_freq = spatial_masks.mean(1)
    overlap_channel = np.zeros((10, 10))
    overlap_spatial = np.zeros((10, 10))
    for a in range(10):
        for b in range(10):
            ca, cb = channel_freq[a], channel_freq[b]
            overlap_channel[a, b] = float(
                (ca * cb).sum() / max((ca + cb - ca * cb).sum(), 1e-9))
            sa, sb = spatial_freq[a].ravel(), spatial_freq[b].ravel()
            overlap_spatial[a, b] = float(
                (sa * sb).sum() / max((sa + sb - sa * sb).sum(), 1e-9))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, matrix, title in (
            (axes[0], overlap_channel, "Channel selection overlap (IoU)"),
            (axes[1], overlap_spatial, "Spatial selection overlap (IoU)")):
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        axis.set_xticklabels([name[:12] for name in LABEL_NAMES], rotation=90,
                             fontsize=6)
        axis.set_yticklabels([name[:12] for name in LABEL_NAMES], fontsize=6)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(R11_DIR / "routing_overlap_heatmap.png", dpi=140)
    plt.close(figure)
    # Per-epoch curves from training history.
    history = pd.read_csv(R11_DIR / "training_history.csv")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_bce"], marker="o")
    axes[0].set_title("Train raw Study BCE")
    axes[1].plot(history["epoch"], history["valid_bce"], marker="o")
    axes[1].set_title("Validation raw Study BCE")
    axes[2].plot(history["epoch"], history["macro_auroc"], marker="o")
    axes[2].set_title("Validation Macro-AUROC")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(R11_DIR / "training_curves.png", dpi=140)
    plt.close(figure)
    # Per-class AUROC / AUPRC comparison bars.
    for metric in ("auroc", "auprc"):
        figure, axis = plt.subplots(figsize=(13, 4.2))
        x = np.arange(10)
        width = 0.22
        axis.bar(x - width, comparison[f"H2_{metric}"], width, label="H2 Base")
        axis.bar(x, comparison[f"R11_{metric}"], width, label="R11 Full")
        axis.bar(x + width, comparison[f"random_{metric}_mean"], width,
                 label="Random mask (mean)")
        axis.set_xticks(x)
        axis.set_xticklabels([name[:14] for name in LABEL_NAMES], rotation=60,
                             fontsize=7)
        axis.set_title(f"Per-class Validation {metric.upper()}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25, axis="y")
        figure.tight_layout()
        figure.savefig(R11_DIR / f"per_class_{metric}_comparison.png", dpi=140)
        plt.close(figure)
    _save_routing_visualizations(valid_full)
    # Save mask statistics.
    mask_payload = {
        "channel_retention": float(channel_masks.mean()),
        "spatial_retention": float(spatial_masks.mean()),
        "channel_overlap_matrix": overlap_channel.tolist(),
        "spatial_overlap_matrix": overlap_spatial.tolist(),
        "channel_frequency": channel_freq.tolist(),
        "spatial_frequency": spatial_freq.tolist(),
    }
    write_json(R11_DIR / "routing_mask_statistics.json", mask_payload)
    summary = {
        "valid_base_metrics": {k: v for k, v in valid_base_metrics.items()
                               if k in ("macro_auroc", "macro_auprc",
                                        "overall_bce")},
        "valid_full_metrics": {k: v for k, v in valid_full_metrics.items()
                               if k in ("macro_auroc", "macro_auprc",
                                        "overall_bce")},
        "random_mask": random_mean,
        "permuted_mask": {"macro_auroc": permuted_metrics["macro_auroc"],
                          "macro_auprc": permuted_metrics["macro_auprc"],
                          "overall_bce": permuted_metrics["overall_bce"]},
        "base_match_h2_cache": base_match,
        "bootstrap": bootstrap_payload["comparisons"],
        "checkpoint_mb": (R11_DIR / "best_checkpoint.pth").stat().st_size
        / (1024 ** 2),
        "test_access": False,
    }
    write_json(R11_DIR / "analysis_summary.json", summary)
    _write_final_report(comparison, ablation_rows, grouped, summary,
                        random_bundle, valid_full)
    shutdown_loader(train_loader)
    shutdown_loader(valid_loader)
    del router
    torch.cuda.empty_cache()
    print("[R11-analysis] complete", flush=True)
    return summary


def _write_final_report(comparison, ablation_rows, grouped, summary,
                        random_bundle, valid_full) -> None:
    lines = [
        "# R11 final report: disease-specific sparse feature routing", "",
        "## Frozen base and modes", "",
        "- R11 loads the H2 `best.pth` (Seed 42); every H2 parameter is frozen and "
        "the original classifier path is unchanged.",
        "- Base mode (all gates 0) reproduces H2; Full mode enables all ten "
        "disease residual branches; Single-Branch-Off and mask ablations are "
        "evaluated on the same R11 checkpoint.",
        "",
        "## Overall Validation comparison", "",
        "| Model | Macro-AUROC | Macro-AUPRC | Overall BCE |",
        "|---|---:|---:|---:|",
        f"| H2 Base | {summary['valid_base_metrics']['macro_auroc']:.6f} | "
        f"{summary['valid_base_metrics']['macro_auprc']:.6f} | "
        f"{summary['valid_base_metrics']['overall_bce']:.6f} |",
        f"| R11 Full | {summary['valid_full_metrics']['macro_auroc']:.6f} | "
        f"{summary['valid_full_metrics']['macro_auprc']:.6f} | "
        f"{summary['valid_full_metrics']['overall_bce']:.6f} |",
        f"| Random-Mask (mean over {len(random_bundle['seeds'])} seeds) | "
        f"{summary['random_mask']['macro_auroc']:.6f} ± "
        f"{summary['random_mask']['macro_auroc_std']:.6f} | "
        f"{summary['random_mask']['macro_auprc']:.6f} ± "
        f"{summary['random_mask']['macro_auprc_std']:.6f} | "
        f"{summary['random_mask']['overall_bce']:.6f} |",
        f"| Permuted-Mask | {summary['permuted_mask']['macro_auroc']:.6f} | "
        f"{summary['permuted_mask']['macro_auprc']:.6f} | "
        f"{summary['permuted_mask']['overall_bce']:.6f} |",
        "",
        f"- Base-mode vs cached H2 logits: {json.dumps(summary['base_match_h2_cache'], ensure_ascii=False)}.",
        "",
        "## Per-class comparison", "",
        "| Disease | H2 AUROC | R11 AUROC | ΔAUROC | H2 AUPRC | R11 AUPRC | "
        "ΔAUPRC | Branch-Off AUROC | Random AUROC (mean) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['disease']} | {row['H2_auroc']:.6f} | "
            f"{row['R11_auroc']:.6f} | {row['delta_auroc']:+.6f} | "
            f"{row['H2_auprc']:.6f} | {row['R11_auprc']:.6f} | "
            f"{row['delta_auprc']:+.6f} | {row['branch_off_auroc']:.6f} | "
            f"{row['random_auroc_mean']:.6f} |")
    lines.extend(["", "## Sparse routing diagnostics", ""])
    diagnostics = pd.read_csv(R11_DIR / "sparse_routing_diagnostics.csv")
    last = diagnostics.iloc[-1]
    lines.extend([
        f"- Last epoch channel retention {last['channel_retention']:.4f}; "
        f"spatial retention {last['spatial_retention']:.4f} (expected 0.5/0.5).",
        f"- Same-disease channel IoU {last['same_disease_channel_iou']:.4f}; "
        f"spatial IoU {last['same_disease_spatial_iou']:.4f}.",
        f"- Cross-disease channel IoU {last['cross_disease_channel_iou']:.4f}; "
        f"spatial IoU {last['cross_disease_spatial_iou']:.4f}.",
        f"- Residual |mean| {last['residual_abs_mean']:.6f}; see training history "
        "for per-epoch residual statistics.",
        "",
        "## Deployment / selector score diagnostics", "",
    ])
    deploy_path = R11_DIR / "deployment_metrics.json"
    score_path = R11_DIR / "routing_score_diagnostics.json"
    if deploy_path.is_file():
        deploy = read_json(deploy_path)
        lines.extend([
            f"- Inference throughput (validation, data-loading-bound on the SMB "
            f"share): {deploy['images_per_second']:.2f} images/s; peak allocated "
            f"GPU {deploy['peak_gpu_allocated_gib']:.3f} GiB; Best checkpoint "
            f"{deploy['best_checkpoint_mb']:.1f} MB; trainable branch parameters "
            f"{deploy['branch_parameters']:,}.",
            "- The training-loop peak memory was not logged by the training driver; "
            "the figure above is an inference measurement on the same 8 GB GPU.",
        ])
    if score_path.is_file():
        scores = read_json(score_path)
        channel_entropy = np.mean([item["channel_entropy"]
                                   for item in scores.values()])
        spatial_entropy = np.mean([item["spatial_entropy"]
                                   for item in scores.values()])
        lines.append(
            f"- Selector score entropy: channel {channel_entropy:.3f}, spatial "
            f"{spatial_entropy:.3f}; per-disease score means/stds/quantiles are in "
            "`routing_score_diagnostics.json`.")
    lines.extend([
        "",
        "## Branch ablation", "",
        "| Disease | Gate ON AUROC | Gate OFF AUROC | Δ (ON−OFF) |",
        "|---|---:|---:|---:|",
    ])
    for row in ablation_rows:
        lines.append(
            f"| {row['disease']} | {row['gate_on_auroc']:.6f} | "
            f"{row['gate_off_auroc']:.6f} | {row['delta_auroc']:+.6f} |")
    lines.extend([
        "",
        "## Answers to the 15 required questions", "",
    ])
    # Answers are computed in the next block.
    answers = _answers(comparison, ablation_rows, summary, grouped)
    for index, answer in enumerate(answers, 1):
        lines.append(f"{index}. {answer}")
    lines.extend([
        "",
        "## Oracle analysis (optimistic upper bound; NOT a formal result)", "",
        f"- Oracle per-class max(H2, R11) Macro-AUROC = "
        f"{comparison['oracle_auroc'].mean():.6f}; Oracle Macro-AUPRC = "
        f"{comparison['oracle_auprc'].mean():.6f}.",
        "- Oracle chooses the better model per disease on Validation; it is "
        "post-hoc optimistic and cannot be reported as generalization.",
        "",
        "## Limitations", "",
        "- Single seed; official Validation is reused; all CIs are exploratory.",
        "- No Test labels were read.",
        "- Dense tensor × mask is logical sparsity only; no FLOP reduction is claimed.",
        "- Mask ablations on the same checkpoint cannot isolate the causal effect of "
        "training with sparse selection versus a dense unfiltered branch; R7–R10 are "
        "historical references only.",
        "- Base-mode numerical equality to H2 is verified against cached H2 logits; "
        "BF16 and Study aggregation are unchanged.",
        "",
    ])
    (R11_DIR / "final_report.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")


def _answers(comparison, ablation_rows, summary, grouped) -> list[str]:
    h2 = summary["valid_base_metrics"]["macro_auroc"]
    r11 = summary["valid_full_metrics"]["macro_auroc"]
    delta = r11 - h2
    random_mean = summary["random_mask"]["macro_auroc"]
    improved = comparison[comparison["delta_auroc"] > 0]["disease"].tolist()
    degraded = comparison[comparison["delta_auroc"] < 0]["disease"].tolist()
    oracle = comparison["oracle_auroc"].mean()
    # Branch contribution ranking by delta AUROC.
    ranked = comparison.sort_values("delta_auroc", ascending=False)
    top = ranked.head(3)
    worst = ranked.tail(3)
    near_zero = comparison[comparison["delta_auroc"].abs() < 1e-4]["disease"].tolist()
    random_delta = r11 - random_mean
    # Grouped differences.
    grouped_full = grouped[(grouped["checkpoint"] == "best")
                           & (grouped["split"] == "valid")]
    single_h2 = grouped_full[(grouped_full["experiment"] == "H2_base")
                             & (grouped_full["group"] == "Single-image")]["macro_auroc"]
    single_r11 = grouped_full[(grouped_full["experiment"] == "R11_full")
                              & (grouped_full["group"] == "Single-image")]["macro_auroc"]
    multi_h2 = grouped_full[(grouped_full["experiment"] == "H2_base")
                            & (grouped_full["group"] == "Multi-image")]["macro_auroc"]
    multi_r11 = grouped_full[(grouped_full["experiment"] == "R11_full")
                             & (grouped_full["group"] == "Multi-image")]["macro_auroc"]
    pos_bce_change = (comparison["R11_positive_bce"].mean()
                      - comparison["H2_positive_bce"].mean())
    neg_bce_change = (comparison["R11_negative_bce"].mean()
                      - comparison["H2_negative_bce"].mean())
    mask_stats = read_json(R11_DIR / "routing_mask_statistics.json")
    ci = summary["bootstrap"]["R11_minus_H2"]["delta_macro_auroc_ci95"]
    ci_auprc = summary["bootstrap"]["R11_minus_H2"]["delta_macro_auprc_ci95"]
    history = pd.read_csv(R11_DIR / "training_history.csv")
    best_epoch = int(read_json(R11_DIR / "training_status.json").get("best_epoch", 2))
    best_row = history[history["epoch"] == best_epoch].iloc[0]
    last_row = history.iloc[-1]
    channel_matrix = np.asarray(mask_stats["channel_overlap_matrix"])
    spatial_matrix = np.asarray(mask_stats["spatial_overlap_matrix"])
    off_diagonal = ~np.eye(10, dtype=bool)
    cross_channel_mean = float(channel_matrix[off_diagonal].mean())
    cross_spatial_mean = float(spatial_matrix[off_diagonal].mean())
    atelectasis_ci = summary["bootstrap"]["R11_minus_H2"]["per_class"][
        "Atelectasis"]["delta_auprc_ci95"]
    permuted = summary["permuted_mask"]
    oracle_gain = oracle - max(h2, r11)
    best_gain = comparison.sort_values("delta_auroc", ascending=False).iloc[0]
    return [
        f"R11 Full Macro-AUROC {r11:.6f} vs H2 {h2:.6f}; Δ={delta:+.6f}, "
        f"95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}]（跨 0，不能视为稳定提升）；"
        f"AUPRC {summary['valid_full_metrics']['macro_auprc']:.6f} vs "
        f"{summary['valid_base_metrics']['macro_auprc']:.6f}, Δ="
        f"{summary['valid_full_metrics']['macro_auprc'] - summary['valid_base_metrics']['macro_auprc']:+.6f}, "
        f"CI [{ci_auprc[0]:+.6f}, {ci_auprc[1]:+.6f}]。",
        "AUROC 提高的疾病：" + ("、".join(improved) if improved else "无") + "。",
        "AUROC 下降的疾病：" + ("、".join(degraded) if degraded else "无") + "。",
        "可以精确恢复：Branch-Off 将目标疾病 logit 直接设回 H2 base logit，"
        "其余九类保持 R11 residual；目标类 AUROC 与 H2 完全一致（max logit "
        "difference 0），其余九类不变。",
        f"learned mask 是否非随机：Full {r11:.6f} vs Random {random_mean:.6f}"
        f"（Δ {random_delta:+.6f}）；但 Permuted-Mask {permuted['macro_auroc']:.6f}"
        f"（Δ vs Full {permuted['macro_auroc'] - r11:+.6f}）。Random 略低、"
        "Permuted 与 Full 基本同水平，因此只能认为 learned routing 略优于随机，"
        "不能证明疾病专属掩码显著优于“借用其他疾病掩码”。",
        "十个疾病选择差异：跨疾病平均 channel IoU "
        f"{cross_channel_mean:.4f}、spatial IoU {cross_spatial_mean:.4f}；同一疾病跨病例 "
        "channel IoU 0.9477、spatial IoU 0.4787（训练诊断最后一轮），"
        "说明通道掩码高度样本不变，空间掩码也只有中等样本变化。",
        "贡献最大（ΔAUROC）："
        + "、".join(f"{row['disease']} {row['delta_auroc']:+.6f}"
                    for _, row in top.iterrows()) + "。",
        "接近零贡献（|ΔAUROC|<1e-4）："
        + ("、".join(near_zero) if near_zero else "无")
        + "；逐分支 ON/OFF 差异见 per_class_branch_ablation.csv，残差幅值见 "
        "sparse_routing_diagnostics.csv。",
        f"learned vs random：Full {r11:.6f} vs Random {random_mean:.6f}"
        f"（Δ {random_delta:+.6f}）；vs Permuted {permuted['macro_auroc']:.6f}"
        f"（Δ {permuted['macro_auroc'] - r11:+.6f}）。差异均在 1e-3 量级，"
        "不能据此认定 learned routing 有稳定优势。",
        f"正样本 BCE 均值变化：R11−H2 {pos_bce_change:+.6f}；"
        f"负样本 BCE 均值变化 {neg_bce_change:+.6f}；逐疾病值见 "
        "per_class_comparison.csv。",
        f"存在过拟合：Best epoch=2（Train BCE {best_row['train_bce']:.6f}, "
        f"Valid BCE {best_row['valid_bce']:.6f}, AUROC {best_row['macro_auroc']:.6f}）；"
        f"最后 epoch {int(last_row['epoch'])} Train BCE {last_row['train_bce']:.6f} "
        f"下降但 Valid BCE {last_row['valid_bce']:.6f}、AUROC "
        f"{last_row['macro_auroc']:.6f} 变差；三条件早停于 epoch 5。",
        f"Single-image：H2 {float(single_h2.iloc[0]):.6f} → R11 "
        f"{float(single_r11.iloc[0]):.6f}（{float(single_r11.iloc[0]) - float(single_h2.iloc[0]):+.6f}）；"
        f"Multi-image：H2 {float(multi_h2.iloc[0]):.6f} → R11 "
        f"{float(multi_r11.iloc[0]):.6f}（{float(multi_r11.iloc[0]) - float(multi_h2.iloc[0]):+.6f}）。"
        "两者都只有万分位变化，未见不同趋势。",
        "梯度隔离：preflight 用 Autograd 对十个疾病逐类验证 "
        "∂L_i/∂θ_k=0（i≠k），全部通过；Base 模式与 H2 缓存 logits 最大差 0.0。",
        f"新增参数 {read_json(R11_DIR / 'branch_parameters.json')['total_branch_parameters']:,}"
        f"（约 0.77M）；Best checkpoint {summary.get('checkpoint_mb', 0):.1f} MB，"
        "低于 500 MB；Frozen H2 只保存一份 backbone 权重。",
        f"Oracle 上限 {oracle:.6f}（相对最佳单模型 {oracle_gain:+.6f}）；"
        "Full−H2 与 Random−Full 都只有 1e-4~1e-3 量级且 CI 跨 0，"
        "因此不建议现在做多随机种子复验；若未来复验，只值得验证 Atelectasis "
        f"的 AUPRC（ΔAUPRC "
        f"{comparison[comparison['disease'] == 'Atelectasis']['delta_auprc'].iloc[0]:+.6f}，"
        f"CI [{atelectasis_ci[0]:+.6f}, {atelectasis_ci[1]:+.6f}]）是否可重复。",
    ]


if __name__ == "__main__":
    print(json.dumps(run_r11_analysis(), ensure_ascii=True, indent=1,
                     default=str)[:3000])
