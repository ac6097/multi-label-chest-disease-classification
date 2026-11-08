"""Shared infrastructure for H1/H2/H3 generalization-optimization experiments.

Reuses the authoritative R5 Global-only configuration (config.json / freeze
config / selected runtime), the same DINOv3 pretrained initialization, the same
Study-level BCE, optimizer, LR schedule and early stopping.  Only the three
declared interventions are implemented, each behind an explicit optimization
config block.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
for _path in (str(PROJECT_DIR), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
if str(PROJECT_DIR) in sys.path:
    sys.path.remove(str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from dino_data import EpochStudyBatchSampler, read_data_config  # noqa: E402
from dino_experiment import load_pretrained_backbone  # noqa: E402
from dino_feature_models import (  # noqa: E402
    build_masked_datasets,
    collate_masked_studies,
    feature_profile_optimizer,
)
from dino_freeze_experiment import (  # noqa: E402
    checkpoint_payload,
    head_hash,
    profile_layout,
    set_profile_training_mode,
    state_rng,
)
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from dino_model import DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from hybrid.common import seed_everything, write_csv  # noqa: E402

WORK_DIR = HERE
EXPERIMENT_ROOT = WORK_DIR / "outputs" / "experiments" / "DINOv3_Generalization_Optimization"
R5_DIR = WORK_DIR / "outputs" / "experiments" / "DINOv3_Feature_Optimization" / "R5_Global_Only_512"
FEATURE_CONFIG_PATH = WORK_DIR / "configs" / "dino_feature_optimization.json"
DATA_CONFIG_PATH = PROJECT_DIR / "configs" / "hybrid_data.yaml"
TRAIN_CSV = PROJECT_DIR.parent / "cxr_first" / "train.csv"
VALID_CSV = PROJECT_DIR.parent / "cxr_first" / "valid.csv"
EXPECTED_BACKBONE_SHA256 = (
    "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2")
EXPECTED_HEAD_SHA256 = (
    "40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9")
PROFILE = "p2"
IMAGE_SIZE = 512
SEED = 42
LABEL_NAMES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity",
    "Pleural Effusion", "No Finding",
]

EXPERIMENTS = {
    "H0": {
        "label": "H0_R5_Baseline",
        "optimization": {"mode": "baseline"},
        "description": "R5 Global-only baseline (reused if config-identical).",
    },
    "H1": {
        "label": "H1_AP_Study_Sampling",
        "optimization": {
            "mode": "ap_sampling",
            "target_ap_fraction": 0.55,
            "ap_weight": 1.20,
        },
        "description": "Study-level weighted sampling toward AP-only Studies (target 55%).",
    },
    "H2": {
        "label": "H2_Single_Study_Loss_Weight",
        "optimization": {"mode": "single_study_weight", "single_weight": 1.25},
        "description": "Single-image Study BCE weight 1.25 with normalized weighted Study mean.",
    },
    "H3": {
        "label": "H3_L2SP_Regularization",
        "optimization": {"mode": "l2sp"},
        "description": "L2-SP penalty on trainable Stage3/4 pretrained parameters.",
    },
    "H3B": {
        "label": "H3B_Weak_L2SP",
        "optimization": {
            "mode": "l2sp",
            "lambda_sp": 0.1 * 28_927_799.425823808,
            "lambda_source": "0.1 x H3 calibrated lambda (2.89278e6)",
            "parameter_normalization": "mean squared deviation over regularized "
                                      "pretrained parameter elements (same as H3)",
        },
        "description": "Weak L2-SP (0.1 x H3 lambda) on the identical Stage3/4 "
                       "pretrained parameter set.",
    },
    "H2B": {
        "label": "H2B_SingleWeight_1p10",
        "optimization": {"mode": "single_study_weight", "single_weight": 1.10},
        "description": "Single-image Study BCE weight 1.10 with normalized "
                       "weighted Study mean (same implementation as H2).",
    },
    "H2C": {
        "label": "H2C_SingleWeight_1p50",
        "optimization": {"mode": "single_study_weight", "single_weight": 1.50},
        "description": "Single-image Study BCE weight 1.50 with normalized "
                       "weighted Study mean (same implementation as H2).",
    },
    "H4": {
        "label": "H4_Head_Dropout_0p10",
        "optimization": {"mode": "head_dropout", "head_dropout": 0.10},
        "description": "Dropout p=0.10 on the final pooled feature immediately "
                       "before the Global-only linear classifier; no L2-SP, no "
                       "sampler or other head changes.",
    },
}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")


def load_r5_reference() -> dict:
    config = read_json(R5_DIR / "config.json")
    status = read_json(R5_DIR / "training_status.json")
    runtime = config.get("selected_runtime")
    if runtime is None:
        runtime = read_json(R5_DIR / "selected_runtime_config.json")["selected_runtime"]
    payload = torch.load(R5_DIR / "best.pth", map_location="cpu", weights_only=False)
    head_sha = payload["run_context"]["global_classifier_init_sha256"]
    if head_sha != EXPECTED_HEAD_SHA256:
        raise RuntimeError("R5 head initialization SHA differs from the recorded R3 head")
    if payload["initial_weights_sha256"] != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError("R5 initial backbone SHA differs from the pinned weights")
    if int(config["experiment_config"]["seed"]) != SEED:
        raise RuntimeError("R5 seed differs from H-series seed")
    return {
        "config": config["experiment_config"],
        "freeze": config["freeze_config"],
        "runtime": runtime,
        "status": status,
        "head_sha": head_sha,
        "r5_config": config,
        "initialization_sha256": config["initialization_sha256"],
        "initialization_repository": config["initialization_repo"],
        "initialization_revision": config["initialization_revision"],
    }


def make_h_builder(runtime: dict, expected_head_sha: str, verification_dir: Path,
                   head_dropout: float = 0.0):
    """Identical initialization to R5; H4 may add parameter-free head dropout."""

    def builder(profile, device, _runtime, seed):
        if profile != PROFILE or int(seed) != SEED:
            raise RuntimeError(f"H-series must use profile={PROFILE}, seed={SEED}")
        seed_everything(int(seed))
        backbone, verification = load_pretrained_backbone("cpu")
        if verification["weights_sha256"] != EXPECTED_BACKBONE_SHA256:
            raise RuntimeError("H-series did not load the pinned official DINOv3 Tiny weights")
        model = DINOv3StudyClassifier(
            backbone, num_labels=NUM_LABELS, dropout=float(head_dropout)).to(device)
        trainable_modules, frozen_modules, trainable_names = profile_layout(model, profile)
        model.backbone.set_gradient_checkpointing(
            bool(runtime["gradient_checkpointing"]))
        if runtime["channels_last"]:
            model.to(memory_format=torch.channels_last)
        actual_head_sha = head_hash(model)
        if actual_head_sha != expected_head_sha:
            raise RuntimeError(
                f"H-series initial Global Head differs from R3: "
                f"expected={expected_head_sha}, actual={actual_head_sha}")
        verification["global_classifier_init_sha256"] = actual_head_sha
        verification["model_variant"] = "DINOv3 ConvNeXt-Tiny Global-only"
        verification["loaded_finetuned_checkpoint"] = False
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        if (total, trainable, frozen) != (27827818, 26592778, 1235040):
            raise RuntimeError(
                f"Unexpected Global-only parameter counts: {total}/{trainable}/{frozen}")
        verification_dir.mkdir(parents=True, exist_ok=True)
        (verification_dir / "initialization_verification.md").write_text(
            "\n".join([
                "# H-series initialization verification", "",
                "- Backbone: official DINOv3 ConvNeXt-Tiny, pinned local revision.",
                f"- Initial weight SHA-256: `{verification['weights_sha256']}`.",
                f"- Initial Global Head SHA-256: `{actual_head_sha}`; matches R3/R5.",
                "- No supervised-finetuned checkpoint is used; each experiment starts from "
                "the original pretrained weights.",
                f"- Parameters total/trainable/frozen: {total:,}/{trainable:,}/{frozen:,}.",
                f"- Seed: {seed}; runtime: budget={runtime['image_budget']}, "
                f"accumulation={runtime['grad_accumulation']}, "
                f"channels_last={runtime['channels_last']}.",
                "",
            ]) + "\n", encoding="utf-8")
        return model, verification, trainable_modules, frozen_modules, trainable_names

    return builder


def build_h_datasets():
    return build_masked_datasets(DATA_CONFIG_PATH, SEED, IMAGE_SIZE)


def r5_batch_forward(model, images, _batch):
    return model(images)


# ---------------------------------------------------------------------------
# AP-only / view metadata
# ---------------------------------------------------------------------------

def study_view_metadata(dataset, csv_path: Path | None = None) -> dict:
    """Classify every Study by its image-level ViewPosition metadata."""
    csv_path = TRAIN_CSV if csv_path is None else Path(csv_path)
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    for column in ("Subject_id", "Study_id", "ViewPosition", "Dicom_id"):
        if column not in frame.columns:
            raise RuntimeError(f"Train CSV is missing {column}")
    frame["ViewPosition"] = frame["ViewPosition"].str.strip().str.upper()
    grouped = frame.groupby(["Subject_id", "Study_id"], sort=False)
    categories = {}
    view_sets = {}
    for (subject, study), group in grouped:
        views = [value for value in group["ViewPosition"] if value]
        n_images = len(group)
        key = (str(subject), str(study))
        view_sets[key] = sorted(set(views))
        if len(views) != n_images:
            categories[key] = "missing_view"
        elif all(value == "AP" for value in views):
            categories[key] = "ap_only"
        elif all(value == "PA" for value in views):
            categories[key] = "pa_only"
        elif all(value in {"LATERAL", "LL", "PA LLD"} for value in views):
            categories[key] = "lateral_only"
        else:
            categories[key] = "mixed_or_other"
    ordered = [categories[key] for key in dataset.study_keys]
    view_order = [view_sets[key] for key in dataset.study_keys]
    category_array = np.asarray(ordered, dtype=object)
    ap_only = category_array == "ap_only"
    return {
        "categories": category_array,
        "ap_only": ap_only,
        "view_sets": view_order,
        "counts": {
            name: int((category_array == name).sum()) for name in (
                "ap_only", "pa_only", "lateral_only", "mixed_or_other",
                "missing_view")},
        "n_studies": len(ordered),
    }


# ---------------------------------------------------------------------------
# H1 weighted Study sampler
# ---------------------------------------------------------------------------

class APWeightedStudyBatchSampler(Sampler):
    """Study-level AP-weighted sampler with fixed epoch draw/batch budgets.

    Each epoch draws exactly ``len(dataset)`` whole Studies (with replacement)
    using exact AP-only quota (target fraction), then repacks them into exactly
    the same number of image-budget batches as the original R5 sampler for that
    epoch.  Study contents never change; no image-level over-sampling.
    """

    def __init__(self, dataset, ap_only: np.ndarray, image_budget: int,
                 seed: int, target_ap_fraction: float = 0.55,
                 category_array: np.ndarray | None = None):
        self.dataset = dataset
        self.image_counts = np.asarray(dataset.image_counts, dtype=np.int64)
        self.ap_indices = np.flatnonzero(np.asarray(ap_only, dtype=bool))
        self.other_indices = np.flatnonzero(~np.asarray(ap_only, dtype=bool))
        if self.ap_indices.size == 0 or self.other_indices.size == 0:
            raise RuntimeError("AP-only and non-AP-only Studies are both required")
        self.category_array = (None if category_array is None
                               else np.asarray(category_array, dtype=object))
        self.image_budget = max(1, int(image_budget))
        self.seed = int(seed)
        self.target_ap_fraction = float(target_ap_fraction)
        self.epoch = 0
        self._cache = {}

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _original_batch_count(self, epoch: int) -> int:
        original = EpochStudyBatchSampler(
            self.dataset, image_budget=self.image_budget, shuffle=True,
            seed=self.seed)
        original.set_epoch(epoch)
        return len(original)

    def _pack(self, draws: np.ndarray, batch_count: int,
              rng: np.random.Generator):
        items = [(int(index), int(self.image_counts[index])) for index in draws]
        order = rng.permutation(len(items))
        ordered = [items[int(position)] for position in order]
        ordered.sort(key=lambda item: -item[1])
        bins: list[list[int]] = []
        sizes: list[int] = []
        for index, size in ordered:
            placed = False
            for position in range(len(bins)):
                if sizes[position] + size <= self.image_budget:
                    bins[position].append(index)
                    sizes[position] += size
                    placed = True
                    break
            if not placed:
                bins.append([index])
                sizes.append(size)
        if len(bins) > batch_count:
            raise RuntimeError(
                f"AP sampler needs {len(bins)} batches but the epoch budget allows "
                f"{batch_count}; weighted sample is too image-heavy")
        while len(bins) < batch_count:
            candidates = [position for position in range(len(bins))
                          if len(bins[position]) >= 2]
            if not candidates:
                raise RuntimeError("Cannot split batches to reach the fixed epoch budget")
            position = max(candidates, key=lambda value: len(bins[value]))
            group = bins[position]
            middle = max(1, len(group) // 2)
            left, right = group[:middle], group[middle:]
            bins[position] = left
            bins.append(right)
        order = rng.permutation(len(bins))
        return [bins[int(position)] for position in order]

    def _build(self, epoch: int):
        if epoch in self._cache:
            return self._cache[epoch]
        total = len(self.dataset)
        n_ap = int(round(self.target_ap_fraction * total))
        n_other = total - n_ap
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + epoch * 9_176 + 355_423) % (2 ** 32))
        ap_draws = rng.choice(self.ap_indices, size=n_ap, replace=True)
        other_draws = rng.choice(self.other_indices, size=n_other, replace=True)
        draws = np.concatenate([ap_draws, other_draws])
        rng.shuffle(draws)
        batch_count = self._original_batch_count(epoch)
        batches = self._pack(draws, batch_count, rng)
        stats = self._statistics(epoch, draws, batches)
        self._cache[epoch] = (batches, stats)
        return self._cache[epoch]

    def _statistics(self, epoch: int, draws: np.ndarray, batches) -> dict:
        dataset = self.dataset
        base = dataset.dataset.base
        targets = np.stack([
            base.studies[int(index)]["target"].numpy()
            for index in range(len(dataset))]).astype(np.float32)
        subjects = np.asarray([str(base.studies[int(index)]["subject_id"])
                               for index in range(len(dataset))], dtype=object)
        image_counts = self.image_counts
        ap_only = np.zeros(len(dataset), dtype=bool)
        ap_only[self.ap_indices] = True
        draw_targets = targets[draws]
        appearance = np.bincount(draws, minlength=len(dataset))
        seen = appearance > 0
        repeat_hist = {
            "once": int(np.sum(appearance == 1)),
            "twice": int(np.sum(appearance == 2)),
            "three_plus": int(np.sum(appearance >= 3)),
            "never": int(np.sum(appearance == 0)),
            "max_repeats": int(appearance.max()) if appearance.size else 0,
        }
        subject_counts = pd.Series(subjects[draws]).value_counts()
        category = self.category_array
        stats = {
            "epoch": int(epoch),
            "n_draws": int(len(draws)),
            "n_unique_studies": int(seen.sum()),
            "ap_only_draw_fraction": float(ap_only[draws].mean()),
            "single_image_draw_fraction": float(
                (image_counts[draws] == 1).mean()),
            "multi_image_draw_fraction": float(
                (image_counts[draws] >= 2).mean()),
            "mean_images_per_draw": float(image_counts[draws].mean()),
            "batches": int(len(batches)),
            "mean_studies_per_batch": float(np.mean([len(batch) for batch in batches])),
            "mean_images_per_batch": float(np.mean(
                [float(image_counts[batch].sum()) for batch in batches])),
            "repeat_histogram": repeat_hist,
            "subject_draw_mean": float(subject_counts.mean()),
            "subject_draw_max": int(subject_counts.max()),
            "subject_draw_p95": float(np.quantile(subject_counts.to_numpy(), 0.95)),
            "per_class_positive_draw_fraction": {
                name: float(draw_targets[:, index].mean())
                for index, name in enumerate(LABEL_NAMES)},
            "per_class_original_positive_fraction": {
                name: float(targets[:, index].mean())
                for index, name in enumerate(LABEL_NAMES)},
        }
        if category is not None:
            stats["category_draw_fractions"] = {
                name: float(np.mean(category[draws] == name))
                for name in ("ap_only", "pa_only", "lateral_only",
                             "mixed_or_other", "missing_view")}
        return stats

    def __iter__(self):
        batches, _ = self._build(self.epoch)
        for batch in batches:
            yield [(int(index), self.epoch) for index in batch]

    def __len__(self):
        batches, _ = self._build(self.epoch)
        return len(batches)

    def statistics(self) -> dict:
        _, stats = self._build(self.epoch)
        return stats


def make_h_loader(dataset, image_budget: int, shuffle: bool, seed: int,
                  num_workers: int, pin_memory: bool, prefetch_factor: int,
                  batch_sampler=None):
    """DataLoader matching the R5 loader, optionally with a custom sampler."""
    from hybrid.common import seed_worker

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if batch_sampler is None:
        batch_sampler = EpochStudyBatchSampler(
            dataset, image_budget=image_budget, shuffle=shuffle, seed=seed)
    loader_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "collate_fn": collate_masked_studies,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)
    return loader, batch_sampler, generator


# ---------------------------------------------------------------------------
# Loss / metric utilities
# ---------------------------------------------------------------------------

def class_bce_breakdown(logits: np.ndarray, targets: np.ndarray,
                        known: np.ndarray | None = None) -> dict:
    """Mean / positive / negative BCE per class, Study-level (no weights)."""
    logits_t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    targets_t = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_t, targets_t, reduction="none").numpy()
    if known is None:
        known = np.ones_like(targets, dtype=bool)
    known = np.asarray(known, dtype=bool)
    rows = {}
    for index, name in enumerate(LABEL_NAMES):
        mask = known[:, index]
        values = elementwise[mask, index]
        truth = targets[mask, index] > 0.5
        rows[name] = {
            "bce": float(values.mean()) if values.size else float("nan"),
            "positive_bce": float(values[truth].mean()) if truth.any() else float("nan"),
            "negative_bce": float(values[~truth].mean()) if (~truth).any() else float("nan"),
            "n_positive": int(truth.sum()),
            "n_negative": int((~truth).sum()),
        }
    return rows


def class_bce_batch_accumulator():
    return {
        name: {"sum": 0.0, "count": 0, "pos_sum": 0.0, "pos_count": 0,
               "neg_sum": 0.0, "neg_count": 0}
        for name in LABEL_NAMES}


def accumulate_class_bce(accumulator: dict, study_logits: torch.Tensor,
                         labels: torch.Tensor, known: torch.Tensor,
                         row_mask: torch.Tensor | None = None) -> None:
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        study_logits.float(), labels.float(), reduction="none").detach()
    known_bool = known.detach().bool()
    labels_bool = labels.detach().bool()
    if row_mask is not None:
        elementwise = elementwise[row_mask]
        known_bool = known_bool[row_mask]
        labels_bool = labels_bool[row_mask]
    for index, name in enumerate(LABEL_NAMES):
        entry = accumulator[name]
        mask = known_bool[:, index]
        values = elementwise[mask, index]
        if values.numel() == 0:
            continue
        truth = labels_bool[mask, index]
        entry["sum"] += float(values.sum().cpu())
        entry["count"] += int(values.numel())
        if bool(truth.any()):
            entry["pos_sum"] += float(values[truth].sum().cpu())
            entry["pos_count"] += int(truth.sum())
        negative = ~truth
        if bool(negative.any()):
            entry["neg_sum"] += float(values[negative].sum().cpu())
            entry["neg_count"] += int(negative.sum())


def finalize_class_bce(accumulator: dict) -> dict:
    rows = {}
    for name in LABEL_NAMES:
        entry = accumulator[name]
        rows[name] = {
            "bce": entry["sum"] / entry["count"] if entry["count"] else float("nan"),
            "positive_bce": (entry["pos_sum"] / entry["pos_count"]
                             if entry["pos_count"] else float("nan")),
            "negative_bce": (entry["neg_sum"] / entry["neg_count"]
                             if entry["neg_count"] else float("nan")),
            "n_positive": int(entry["pos_count"]),
            "n_negative": int(entry["neg_count"]),
        }
    return rows


def merge_class_bce(train_rows: dict, valid_rows: dict) -> list[dict]:
    rows = []
    for name in LABEL_NAMES:
        train = train_rows[name]
        valid = valid_rows[name]
        rows.append({
            "disease": name,
            "train_bce": train["bce"],
            "valid_bce": valid["bce"],
            "loss_gap": valid["bce"] - train["bce"],
            "train_positive_bce": train["positive_bce"],
            "train_negative_bce": train["negative_bce"],
            "valid_positive_bce": valid["positive_bce"],
            "valid_negative_bce": valid["negative_bce"],
            "train_n_positive": train["n_positive"],
            "train_n_negative": train["n_negative"],
            "valid_n_positive": valid["n_positive"],
            "valid_n_negative": valid["n_negative"],
        })
    rows.append({
        "disease": "Overall BCE",
        "train_bce": float(np.mean([train_rows[name]["bce"] for name in LABEL_NAMES])),
        "valid_bce": float(np.mean([valid_rows[name]["bce"] for name in LABEL_NAMES])),
        "loss_gap": float(np.mean([valid_rows[name]["bce"] - train_rows[name]["bce"]
                                   for name in LABEL_NAMES])),
        "train_positive_bce": float("nan"),
        "train_negative_bce": float("nan"),
        "valid_positive_bce": float("nan"),
        "valid_negative_bce": float("nan"),
        "train_n_positive": int(sum(train_rows[name]["n_positive"] for name in LABEL_NAMES)),
        "train_n_negative": int(sum(train_rows[name]["n_negative"] for name in LABEL_NAMES)),
        "valid_n_positive": int(sum(valid_rows[name]["n_positive"] for name in LABEL_NAMES)),
        "valid_n_negative": int(sum(valid_rows[name]["n_negative"] for name in LABEL_NAMES)),
    })
    return rows


# ---------------------------------------------------------------------------
# L2-SP parameter selection
# ---------------------------------------------------------------------------

L2SP_NAMESPACES = (
    "backbone.model.downsample_layers.2.",
    "backbone.model.stages.2.",
    "backbone.model.downsample_layers.3.",
    "backbone.model.stages.3.",
)


def l2sp_parameter_names(model) -> list[str]:
    return [name for name, _ in model.named_parameters()
            if name.startswith(L2SP_NAMESPACES)]


def l2sp_reference(model, names: list[str]) -> dict[str, torch.Tensor]:
    """Original pretrained values for the regularized trainable parameters."""
    state = load_pretrained_backbone("cpu")[0].state_dict()
    reference = {}
    for name in names:
        inner = name.removeprefix("backbone.")
        if inner not in state:
            raise RuntimeError(f"L2-SP parameter has no pretrained counterpart: {name}")
        parameter = dict(model.named_parameters())[name]
        reference[name] = state[inner].detach().float().to(parameter.device).clone()
    return reference


def l2sp_penalty(model, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    elements = 0
    parameters = dict(model.named_parameters())
    for name, target in reference.items():
        difference = parameters[name].float() - target
        squared = difference.pow(2)
        total = squared.sum() if total is None else total + squared.sum()
        elements += squared.numel()
    if total is None:
        raise RuntimeError("L2-SP reference contains no parameters")
    return total / float(elements)


def stage_relative_drift(model, reference: dict[str, torch.Tensor]) -> dict:
    parameters = dict(model.named_parameters())
    result = {}
    for stage, prefix in (("stage3", "backbone.model.stages.2."),
                          ("stage4", "backbone.model.stages.3.")):
        squared_delta = 0.0
        squared_reference = 0.0
        for name, target in reference.items():
            if not name.startswith(prefix):
                continue
            difference = parameters[name].float() - target
            squared_delta += float(difference.pow(2).sum())
            squared_reference += float(target.pow(2).sum())
        result[stage] = {
            "absolute_drift_l2": math.sqrt(squared_delta),
            "reference_l2": math.sqrt(squared_reference),
            "relative_drift": math.sqrt(squared_delta)
            / (math.sqrt(squared_reference) + 1e-12),
        }
    return result


# ---------------------------------------------------------------------------
# Checkpoint/model loading for evaluation
# ---------------------------------------------------------------------------

def load_h_checkpoint_model(run_dir: Path, checkpoint: str, runtime: dict,
                            head_sha: str = EXPECTED_HEAD_SHA256):
    device = torch.device("cuda")
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    head_dropout = 0.0
    if config_path.is_file():
        config_payload = read_json(config_path)
        head_dropout = float(config_payload.get("optimization", {}).get(
            "head_dropout", 0.0))
    builder = make_h_builder(runtime, head_sha, run_dir / "verification_scratch",
                             head_dropout=head_dropout)
    model, verification, train_modules, frozen_modules, trainable_names = builder(
        PROFILE, device, runtime, SEED)
    payload = torch.load(run_dir / f"{checkpoint}.pth", map_location="cpu",
                         weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def state_dict_sha(state: dict) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous().float()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
