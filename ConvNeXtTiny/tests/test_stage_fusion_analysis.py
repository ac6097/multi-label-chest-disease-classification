"""Fast synthetic tests for the unified analysis/statistics/export helpers.

These do not load checkpoints or the NAS dataset; they guard the pure numerical
and file-writing paths that would otherwise only fail after a multi-hour run.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

DINO_DIR = Path(__file__).resolve().parents[1] / "DINOv3-ConvNeXtTiny"
PROJECT_DIR = DINO_DIR.parent
for path in (DINO_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage_fusion_analysis as analysis  # noqa: E402
from dino_analysis import LABEL_NAMES  # noqa: E402


def synthetic_labels(studies: int = 120, seed: int = 3):
    generator = np.random.default_rng(seed)
    labels = (generator.random((studies, 10)) > 0.65).astype(np.uint8)
    # Guarantee both polarities for every class.
    labels[0, :] = 1
    labels[1, :] = 0
    return labels


def synthetic_subjects(studies: int = 120, subjects: int = 40):
    return np.asarray([f"subject_{index % subjects:03d}" for index in range(studies)],
                      dtype=object)


class TestStudyMetrics(unittest.TestCase):
    def test_matches_sklearn_and_counts(self):
        labels = synthetic_labels()
        generator = np.random.default_rng(11)
        probabilities = generator.random(labels.shape) * 0.4 + labels * 0.4
        metrics = analysis.study_metrics(labels, probabilities)
        for index, name in enumerate(LABEL_NAMES):
            expected = roc_auc_score(labels[:, index], probabilities[:, index])
            self.assertAlmostEqual(metrics["per_class"][name]["auroc"], expected,
                                   places=10)
            self.assertEqual(metrics["per_class"][name]["n_positive"],
                             int(labels[:, index].sum()))
            self.assertEqual(metrics["per_class"][name]["n_negative"],
                             int((labels[:, index] == 0).sum()))
        macro = np.mean([metrics["per_class"][name]["auroc"]
                         for name in LABEL_NAMES])
        self.assertAlmostEqual(metrics["macro_auroc"], macro, places=10)


class TestSubjectBootstrap(unittest.TestCase):
    def setUp(self):
        self._replicates = analysis.BOOTSTRAP_REPLICATES
        analysis.BOOTSTRAP_REPLICATES = 80

    def tearDown(self):
        analysis.BOOTSTRAP_REPLICATES = self._replicates

    def test_draws_keep_subjects_together_and_are_deterministic(self):
        subjects = synthetic_subjects()
        draws, digest = analysis.make_subject_draws(subjects)
        draws_again, digest_again = analysis.make_subject_draws(subjects)
        self.assertEqual(digest, digest_again)
        self.assertEqual(len(draws), 80)
        positions = {subject: np.flatnonzero(subjects == subject)
                     for subject in sorted(set(subjects))}
        for indices in draws[:5]:
            # Every Subject is either fully present (possibly multiple times) or
            # fully absent; no partial Subject may leak into a resample.
            for subject, expected in positions.items():
                included = np.isin(expected, indices).sum()
                self.assertIn(int(included), (0, len(expected)))
            counts = [int(np.isin(indices, positions[subject]).sum()
                          // len(positions[subject])) for subject in positions]
            self.assertEqual(int(sum(counts)), len(set(subjects)))

    def test_bootstrap_models_shapes_and_comparisons(self):
        labels = synthetic_labels()
        subjects = synthetic_subjects()
        generator = np.random.default_rng(5)
        probabilities = {
            "R5_baseline": generator.random(labels.shape) * 0.4 + labels * 0.3,
            "R8_Full": generator.random(labels.shape) * 0.4 + labels * 0.35,
            "R8_BranchOff": generator.random(labels.shape) * 0.4 + labels * 0.3,
        }
        comparisons = {
            "R8_Full_minus_R5": ("R5_baseline", "R8_Full"),
            "R8_Full_minus_R8_BranchOff": ("R8_BranchOff", "R8_Full"),
        }
        payload = analysis.bootstrap_models(labels, probabilities, subjects,
                                            comparisons)
        self.assertEqual(payload["replicates"], 80)
        self.assertEqual(payload["aligned_validation_studies"], labels.shape[0])
        for mode in probabilities:
            entry = payload["models"][mode]
            self.assertTrue(np.isfinite(entry["macro_auroc"]))
            for name in LABEL_NAMES:
                ci = entry["per_class"][name]["auroc_ci95"]
                self.assertEqual(len(ci), 2)
                self.assertLessEqual(ci[0], ci[1])
        for name in comparisons:
            comparison = payload["comparisons"][name]
            self.assertEqual(len(comparison["delta_macro_auroc_ci95"]), 2)
            self.assertLessEqual(comparison["delta_macro_auroc_ci95"][0],
                                 comparison["delta_macro_auroc_ci95"][1])
            self.assertAlmostEqual(
                comparison["delta_macro_auroc"],
                (payload["models"][comparison["right"]]["macro_auroc"]
                 - payload["models"][comparison["left"]]["macro_auroc"]),
                places=6)
        skipped = analysis.bootstrap_models(
            labels, probabilities, subjects,
            {"missing": ("R5_baseline", "R99")})
        self.assertEqual(skipped["comparisons"]["missing"]["status"],
                         "skipped_missing_model")


class TestModeDerivationAndFrames(unittest.TestCase):
    def test_per_disease_modes_only_change_that_disease(self):
        generator = torch.Generator().manual_seed(0)
        stage4 = torch.randn(4, 10, generator=generator).numpy().astype(np.float64)
        stage3 = torch.randn(4, 10, generator=generator).numpy().astype(np.float64)
        alpha = np.full(10, 0.1)
        modes = analysis._derived_per_disease_modes(stage4, stage3, alpha, "R8")
        full = stage4 + alpha[None, :] * stage3
        self.assertEqual(len(modes), 10)
        for index, disease in enumerate(LABEL_NAMES):
            delta = full - modes[disease]
            self.assertTrue(np.all(delta[:, index] != 0))
            others = [k for k in range(10) if k != index]
            self.assertEqual(float(np.abs(delta[:, others]).max()), 0.0)

    def test_prediction_frame_columns(self):
        result = {
            "subjects": np.asarray(["s0", "s1"], dtype=object),
            "studies": np.asarray(["t0", "t1"], dtype=object),
            "labels": np.zeros((2, 10), dtype=np.uint8),
            "kinds": {"full": np.zeros((2, 10), dtype=np.float64)},
        }
        frame = analysis._predictions_frame(result, "R8")
        self.assertEqual(frame.columns[:2].tolist(), ["Subject_id", "Study_id"])
        self.assertIn("target_Pneumonia", frame.columns)
        self.assertIn("logit_R8_Pneumonia", frame.columns)
        self.assertIn("prob_R8_Pneumonia", frame.columns)

    def test_verdict_thresholds(self):
        positive = {"delta_macro_auroc": 0.01,
                    "delta_macro_auroc_ci95": [0.002, 0.02]}
        self.assertIn("正向", analysis._verdict(positive))
        crossing = {"delta_macro_auroc": 0.01,
                    "delta_macro_auroc_ci95": [-0.002, 0.02]}
        self.assertIn("无法确认", analysis._verdict(crossing))
        negative = {"delta_macro_auroc": -0.01,
                    "delta_macro_auroc_ci95": [-0.02, -0.002]}
        self.assertIn("负向", analysis._verdict(negative))


class TestPreprocessAndTemplate(unittest.TestCase):
    def test_preprocess_paths_shape_and_normalisation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.jpg"
            Image.new("L", (320, 160), color=200).save(path)
            tensor = analysis._preprocess_paths([str(path)], 512)
        self.assertEqual(tuple(tensor.shape), (1, 3, 512, 512))
        self.assertTrue(torch.isfinite(tensor).all())
        # 320x160 keeps aspect ratio: resized to 512x256, padded top/bottom by 128.
        top_row = tensor[0, :, 0, :]
        bottom_row = tensor[0, :, -1, :]
        middle_row = tensor[0, :, 256, :]
        self.assertTrue(torch.allclose(top_row, bottom_row, atol=1e-6))
        self.assertFalse(torch.allclose(top_row, middle_row, atol=1e-6))

    def test_infer_template_compiles(self):
        compile(analysis._INFER_TEMPLATE, "<infer_template>", "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
