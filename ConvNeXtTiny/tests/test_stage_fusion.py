"""Required structural / numerical tests for R8, R9, R10.

Run from ConvNeXtTiny:

    python -m tests.test_stage_fusion
    python -m unittest tests.test_stage_fusion -v
"""

from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

DINO_DIR = Path(__file__).resolve().parents[1] / "DINOv3-ConvNeXtTiny"
PROJECT_DIR = DINO_DIR.parent
for path in (DINO_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dino_analysis import LABEL_NAMES  # noqa: E402
from dino_experiment import load_pretrained_backbone  # noqa: E402
from dino_freeze_experiment import head_hash  # noqa: E402
from dino_losses import study_bce_per_study, study_logit_mean  # noqa: E402
from dino_model import DINOv3ConvNeXtBackbone, DINOv3StudyClassifier, NUM_LABELS  # noqa: E402
from hybrid.common import seed_everything  # noqa: E402
from stage_fusion_experiment import (  # noqa: E402
    EXPECTED_HEAD_SHA,
    EXPECTED_NEW_PARAMETERS,
    EXPECTED_WEIGHT_SHA256,
    GROUP_NAMES,
    IMAGE_SIZE,
    OUTPUT_ROOT,
    R5_FROZEN_PARAMETERS,
    R5_TOTAL_PARAMETERS,
    SEED,
    freeze_layout,
    make_optimizer,
)
from stage_fusion_models import (  # noqa: E402
    ALPHA_INIT,
    NUM_HEADS,
    THETA_INIT,
    build_model,
)

SMALL_INPUT = 64  # multiple of 32; keeps CPU tests fast while exercising all stages


def random_backbone():
    seed_everything(SEED)
    backbone = DINOv3ConvNeXtBackbone()
    # The upstream Meta ConvNeXt implementation does not call init_weights() in
    # __init__, so its custom LayerNorm weight/bias are uninitialized memory.
    # Production always strict-loads the pinned pretrained weights, but these
    # structural tests use a random backbone, so initialise those layers
    # deterministically here to keep the tests reproducible.
    with torch.no_grad():
        for module in backbone.model.modules():
            if module.__class__.__name__ == "LayerNorm" and hasattr(
                    module, "init_weights"):
                module.init_weights()
    return backbone


def pretrained_pair(experiment: str):
    seed_everything(SEED)
    r5_backbone, r5_verification = load_pretrained_backbone("cpu")
    r5_model = DINOv3StudyClassifier(r5_backbone, num_labels=NUM_LABELS).cpu().eval()
    seed_everything(SEED)
    candidate_backbone, candidate_verification = load_pretrained_backbone("cpu")
    candidate = build_model(experiment, candidate_backbone).cpu().eval()
    return r5_model, candidate, r5_verification, candidate_verification


def synthetic_batch(images_per_study=(1, 2), size=SMALL_INPUT, seed=7):
    generator = torch.Generator().manual_seed(seed)
    counts = list(images_per_study)
    images = torch.randn(sum(counts), 3, size, size, generator=generator)
    study_index = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts))
    labels = (torch.rand(len(counts), NUM_LABELS, generator=generator) > 0.6).float()
    known = torch.ones_like(labels)
    batch = {
        "images": images,
        "study_index": study_index,
        "image_counts": torch.tensor(counts),
        "num_studies": len(counts),
        "labels": labels,
        "known": known,
    }
    return batch


def study_loss(model, batch):
    image_logits = model(batch["images"])
    study_logits = study_logit_mean(
        image_logits.float(), batch["study_index"], batch["num_studies"],
        batch["image_counts"])
    per_study, eligible = study_bce_per_study(
        study_logits, batch["labels"], batch["known"])
    return per_study[eligible].mean()


def gradient_of(model, name: str) -> torch.Tensor | None:
    for parameter_name, parameter in model.named_parameters():
        if parameter_name == name:
            return parameter.grad
    raise KeyError(name)


class TestBackboneAndShapes(unittest.TestCase):
    def test_pretrained_weights_are_pinned(self):
        _, verification = load_pretrained_backbone("cpu")
        self.assertEqual(verification["weights_sha256"], EXPECTED_WEIGHT_SHA256)
        self.assertEqual(verification["depths"], [3, 3, 9, 3])
        self.assertEqual(verification["dims"], [96, 192, 384, 768])

    def test_stage_shapes(self):
        seed_everything(SEED)
        backbone = DINOv3ConvNeXtBackbone()
        backbone.eval()
        with torch.inference_mode():
            pooled, stage3, stage4 = backbone.forward_features_with_stage3(
                torch.zeros(1, 3, SMALL_INPUT, SMALL_INPUT))
        self.assertEqual(backbone.model.norm.normalized_shape[-1], 768)
        self.assertEqual(tuple(stage3.shape[1:]), (384, SMALL_INPUT // 16,
                                                   SMALL_INPUT // 16))
        self.assertEqual(tuple(stage4.shape[1:]), (768, SMALL_INPUT // 32,
                                                   SMALL_INPUT // 32))
        self.assertEqual(tuple(pooled.shape), (1, 768))


class TestParameterContract(unittest.TestCase):
    def test_freeze_layout_parameter_counts(self):
        allowed_prefixes = (
            "backbone.model.downsample_layers.2.",
            "backbone.model.downsample_layers.3.",
            "backbone.model.stages.2.", "backbone.model.stages.3.",
            "backbone.model.norm.", "backbone.model.norms.3.",
            "classifier.", "stage3_branch.", "query_attention.", "fusion_alpha.")
        for experiment in ("R8", "R9", "R10"):
            with self.subTest(experiment=experiment):
                model = build_model(experiment, random_backbone())
                trainable_modules, frozen_modules, names = freeze_layout(
                    experiment, model)
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters()
                                if p.requires_grad)
                self.assertEqual(total, R5_TOTAL_PARAMETERS
                                 + EXPECTED_NEW_PARAMETERS[experiment])
                self.assertEqual(total - trainable, R5_FROZEN_PARAMETERS)
                self.assertEqual(len(names),
                                 len([n for n, p in model.named_parameters()
                                      if p.requires_grad]))
                for name in names:
                    self.assertTrue(name.startswith(allowed_prefixes), name)

    def test_head_hash_matches_r5(self):
        for experiment in ("R8", "R9", "R10"):
            with self.subTest(experiment=experiment):
                seed_everything(SEED)
                backbone, _ = load_pretrained_backbone("cpu")
                model = build_model(experiment, backbone)
                self.assertEqual(head_hash(model), EXPECTED_HEAD_SHA)

    def test_optimizer_covers_every_parameter_once(self):
        for experiment in ("R8", "R9", "R10"):
            with self.subTest(experiment=experiment):
                model = build_model(experiment, random_backbone())
                freeze_layout(experiment, model)
                config = {
                    "training": {"peak_lr_backbone": 1e-5, "peak_lr_head": 2e-4,
                                 "weight_decay": 0.01},
                    "freeze": {"training": {"fused_adamw": False}},
                }
                optimizer = make_optimizer(experiment)(model, config)
                identifiers = [id(p) for group in optimizer.param_groups
                               for p in group["params"]]
                expected = {id(p) for p in model.parameters() if p.requires_grad}
                self.assertEqual(len(identifiers), len(set(identifiers)))
                self.assertEqual(set(identifiers), expected)
                alpha_groups = [group for group in optimizer.param_groups
                                if any(p is model.fusion_alpha.theta
                                       for p in group["params"])]
                self.assertEqual(len(alpha_groups), 1)
                self.assertEqual(float(alpha_groups[0]["weight_decay"]), 0.0)
                self.assertEqual(float(alpha_groups[0]["lr"]), 2e-4)


class TestInitializationEquivalence(unittest.TestCase):
    def test_initial_logits_equal_r5(self):
        torch.manual_seed(0)
        images = torch.randn(3, 3, SMALL_INPUT, SMALL_INPUT)
        for experiment in ("R8", "R9", "R10"):
            with self.subTest(experiment=experiment):
                r5_model, candidate, verification, _ = pretrained_pair(experiment)
                self.assertEqual(verification["weights_sha256"], EXPECTED_WEIGHT_SHA256)
                with torch.inference_mode():
                    r5_logits = r5_model(images)
                    components = candidate.forward_components(images)
                error = float((r5_logits - components["logits"]).abs().max())
                self.assertLessEqual(error, 1e-6)
                alpha = components["alpha"]
                self.assertTrue(torch.allclose(
                    alpha, torch.full_like(alpha, ALPHA_INIT), atol=1e-6))
                self.assertAlmostEqual(float(THETA_INIT), math.log(0.1 / 0.9), places=6)
                if experiment == "R10":
                    self.assertEqual(float(components["conditioning"].abs().sum()), 0.0)
                del r5_model, candidate


class TestStudyAggregation(unittest.TestCase):
    def test_study_logit_mean_matches_manual_mean(self):
        generator = torch.Generator().manual_seed(1)
        logits = torch.randn(5, NUM_LABELS, generator=generator)
        study_index = torch.tensor([0, 0, 1, 2, 2])
        counts = torch.tensor([2, 1, 2])
        study_logits = study_logit_mean(logits, study_index, 3, counts)
        expected = torch.stack([
            logits[study_index == index].mean(dim=0) for index in range(3)])
        self.assertTrue(torch.allclose(study_logits, expected, atol=1e-6))


class TestBranchSemantics(unittest.TestCase):
    def test_branch_off_equals_r5_path(self):
        batch = synthetic_batch()
        for experiment in ("R8", "R9", "R10"):
            with self.subTest(experiment=experiment):
                model = build_model(experiment, random_backbone()).eval()
                with torch.inference_mode():
                    components = model.forward_components(
                        batch["images"], branch_off=True)
                    pooled, stage4 = model.backbone.forward_features(batch["images"])
                    expected = model.classifier(pooled)
                self.assertTrue(torch.allclose(
                    components["logits"], expected, atol=1e-6))
                self.assertTrue(torch.allclose(
                    components["stage4_logits"], expected, atol=1e-6))

    def test_single_disease_alpha_override_changes_only_that_disease(self):
        batch = synthetic_batch()
        model = build_model("R8", random_backbone()).eval()
        with torch.no_grad():
            model.stage3_branch.head.weight.normal_(std=0.1)
            model.stage3_branch.head.bias.normal_(std=0.1)
        full_alpha = model.fusion_alpha.alpha().detach().clone()
        index = 3
        override = full_alpha.clone()
        override[index] = 0.0
        with torch.inference_mode():
            full = model.forward_components(batch["images"])["logits"]
            ablated = model.forward_components(
                batch["images"], alpha_override=override)["logits"]
        delta = (full - ablated).abs()
        self.assertGreater(float(delta[:, index].max()), 0.0)
        others = [k for k in range(NUM_LABELS) if k != index]
        self.assertEqual(float(delta[:, others].max()), 0.0)

    def test_attention_softmax_normalisation_and_query_order(self):
        batch = synthetic_batch()
        model = build_model("R9", random_backbone()).eval()
        with torch.inference_mode():
            components = model.forward_components(
                batch["images"], return_attention_weights=True)
        weights = components["attention_weights"]
        self.assertEqual(weights.shape[0], batch["images"].shape[0])
        self.assertEqual(weights.shape[1], NUM_HEADS)
        self.assertEqual(weights.shape[2], NUM_LABELS)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(
            sums, torch.ones_like(sums), atol=1e-5))
        queries = model.query_attention.build_queries(
            batch["images"].shape[0])[0]
        for index in range(NUM_LABELS):
            self.assertTrue(torch.allclose(
                queries[0, index],
                model.query_attention.disease_queries[index], atol=1e-6))

    def test_uniform_attention_ignores_disease_identity(self):
        batch = synthetic_batch()
        model = build_model("R9", random_backbone()).eval()
        with torch.inference_mode():
            components = model.forward_components(
                batch["images"], attention_mode="uniform")
        attended = components["attended"]
        for index in range(1, NUM_LABELS):
            self.assertTrue(torch.allclose(
                attended[:, 0], attended[:, index], atol=1e-6))

    def test_r10_conditioning_zero_at_initialisation_and_gradient_later(self):
        batch = synthetic_batch()
        model = build_model("R10", random_backbone()).eval()
        with torch.inference_mode():
            components = model.forward_components(batch["images"])
            conditioning_sum = float(components["conditioning"].abs().sum())
            queries = components["queries"].clone()
            disease_queries = model.query_attention.disease_queries.unsqueeze(0)
        self.assertEqual(conditioning_sum, 0.0)
        self.assertEqual(float((queries - disease_queries).abs().sum()), 0.0)


class TestDelayedGradientContract(unittest.TestCase):
    def _build(self, experiment: str):
        model = build_model(experiment, random_backbone())
        freeze_layout(experiment, model)
        config = {
            "training": {"peak_lr_backbone": 1e-5, "peak_lr_head": 2e-4,
                         "weight_decay": 0.01},
            "freeze": {"training": {"fused_adamw": False}},
        }
        optimizer = make_optimizer(experiment)(model, config)
        return model, optimizer, synthetic_batch()

    def _step(self, model, optimizer, batch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        study_loss(model, batch).backward()
        snapshot = {
            name: (None if parameter.grad is None
                   else float(parameter.grad.detach().abs().max()))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        optimizer.step()
        return snapshot

    @staticmethod
    def _grad_max(model, name: str) -> float:
        parameters = dict(model.named_parameters())
        parameter = parameters[name]
        if parameter.grad is None:
            return 0.0
        return float(parameter.grad.detach().abs().max())

    def _auxiliary_isolation(self, model, batch):
        model.zero_grad(set_to_none=True)
        components = model.forward_components(batch["images"])
        auxiliary_study = study_logit_mean(
            components["stage3_logits"].float(), batch["study_index"],
            batch["num_studies"], batch["image_counts"])
        per_study, eligible = study_bce_per_study(
            auxiliary_study, batch["labels"], batch["known"])
        per_study[eligible].mean().backward()
        result = {
            "stage3_head_grad": (
                self._grad_max(model, "stage3_branch.head.weight")
                if hasattr(model, "stage3_branch")
                else self._grad_max(model, "query_attention.aux_head_weight")),
            "stage3_backbone_grad": self._grad_max(
                model, "backbone.model.stages.2.0.dwconv.weight"),
            "classifier_grad": self._grad_max(model, "classifier.weight"),
        }
        model.zero_grad(set_to_none=True)
        return result

    def test_r8_delayed_gradient(self):
        model, optimizer, batch = self._build("R8")
        first = self._step(model, optimizer, batch)
        self.assertGreater(first["stage3_branch.head.weight"], 0.0)
        self.assertEqual(first["fusion_alpha.theta"], 0.0)
        second = self._step(model, optimizer, batch)
        self.assertGreater(second["fusion_alpha.theta"], 0.0)
        isolation = self._auxiliary_isolation(model, batch)
        self.assertGreater(isolation["stage3_backbone_grad"], 0.0)
        self.assertEqual(isolation["classifier_grad"], 0.0)

    def test_r9_delayed_gradient(self):
        model, optimizer, batch = self._build("R9")
        first = self._step(model, optimizer, batch)
        self.assertGreater(first["query_attention.aux_head_weight"], 0.0)
        self.assertEqual(first["query_attention.key_proj.weight"], 0.0)
        self.assertEqual(first["query_attention.value_proj.weight"], 0.0)
        self.assertEqual(first["query_attention.disease_queries"], 0.0)
        self.assertEqual(first["fusion_alpha.theta"], 0.0)
        second = self._step(model, optimizer, batch)
        for name in ("query_attention.key_proj.weight",
                     "query_attention.value_proj.weight",
                     "query_attention.disease_queries",
                     "fusion_alpha.theta"):
            self.assertGreater(second[name], 0.0, name)
        isolation = self._auxiliary_isolation(model, batch)
        self.assertGreater(isolation["stage3_backbone_grad"], 0.0)
        self.assertEqual(isolation["classifier_grad"], 0.0)

    def test_r10_conditioning_gradient_delay(self):
        model, optimizer, batch = self._build("R10")
        first = self._step(model, optimizer, batch)
        self.assertEqual(first["query_attention.condition_proj.weight"], 0.0)
        self.assertGreater(first["query_attention.aux_head_weight"], 0.0)
        second = self._step(model, optimizer, batch)
        self.assertGreater(second["query_attention.condition_proj.weight"], 0.0)
        third = self._step(model, optimizer, batch)
        self.assertGreater(third["query_attention.condition_proj.weight"], 0.0)


class TestCheckpointAndInference(unittest.TestCase):
    def test_checkpoint_save_and_strict_reload(self):
        batch = synthetic_batch()
        for experiment in ("R8", "R9", "R10"):
            with self.subTest(experiment=experiment):
                model = build_model(experiment, random_backbone()).eval()
                with torch.inference_mode():
                    expected = model(batch["images"])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "best.pth"
                    torch.save({"model_state": model.state_dict()}, path)
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    fresh = build_model(experiment, random_backbone()).eval()
                    result = fresh.load_state_dict(payload["model_state"], strict=True)
                    self.assertEqual(result.missing_keys, [])
                    self.assertEqual(result.unexpected_keys, [])
                    with torch.inference_mode():
                        actual = fresh(batch["images"])
                self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_cpu_gpu_inference_consistency(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 runtime is not available")
        batch = synthetic_batch()
        for experiment in ("R8", "R9", "R10"):
            with self.subTest(experiment=experiment):
                model = build_model(experiment, random_backbone()).eval()
                with torch.inference_mode():
                    cpu_logits = model(batch["images"].float())
                device = torch.device("cuda")
                model.to(device).eval()
                images = batch["images"].to(device)
                with torch.inference_mode(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16):
                    gpu_logits = model(images)
                cpu_probability = torch.sigmoid(cpu_logits.float())
                gpu_probability = torch.sigmoid(gpu_logits.float().cpu())
                self.assertLessEqual(
                    float((cpu_probability - gpu_probability).abs().max()), 1e-2)


class TestPackageAndLabels(unittest.TestCase):
    def test_label_order_is_official(self):
        self.assertEqual(LABEL_NAMES, [
            "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
            "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
            "Lung Opacity", "Pleural Effusion", "No Finding"])

    def test_no_shared_residual_module_in_new_models(self):
        for experiment in ("R8", "R9", "R10"):
            model = build_model(experiment, random_backbone())
            self.assertFalse(hasattr(model, "transfer"))
            self.assertFalse(any("r7" in name for name in model.state_dict()))

    def test_submission_package_size_limit(self):
        zips = list(OUTPUT_ROOT.glob("*/submission_export/*.zip"))
        if not zips:
            self.skipTest("No submission package has been exported yet")
        for path in zips:
            self.assertLess(path.stat().st_size, 500_000_000, str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
