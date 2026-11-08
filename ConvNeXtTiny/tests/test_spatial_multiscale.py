import unittest

import torch

from model.spatial_multiscale import (
    SpatialMultiscaleHead,
    SpatialPyramidPooling,
    Stage3Stage4SpatialFusion,
)
from train import build_lr_schedule_table
from utils.metrics import multilabel_metrics


class SpatialMultiscaleModuleTests(unittest.TestCase):
    def test_spp_concatenates_global_and_four_regional_averages(self):
        feature_map = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        pooled = SpatialPyramidPooling(1)(feature_map)
        expected = torch.tensor([[7.5, 2.5, 4.5, 10.5, 12.5]])
        self.assertEqual(tuple(pooled.shape), (1, 5))
        self.assertTrue(torch.equal(pooled, expected))

    def test_stage3_stage4_fusion_aligns_stage4_and_backpropagates(self):
        torch.manual_seed(4)
        fusion = Stage3Stage4SpatialFusion(384, 768, projected_channels=256)
        stage3 = torch.randn(2, 384, 20, 20, requires_grad=True)
        stage4 = torch.randn(2, 768, 10, 10, requires_grad=True)
        fused = fusion(stage3, stage4)
        self.assertEqual(tuple(fused.shape), (2, 256, 20, 20))
        fused.square().mean().backward()
        for tensor in (stage3, stage4):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)
        for parameter in fusion.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_all_four_feature_pooling_modes_have_expected_classifier_width(self):
        torch.manual_seed(5)
        maps = {
            "stage3": torch.randn(2, 384, 20, 20),
            "stage4": torch.randn(2, 768, 10, 10),
        }
        cases = (
            ("stage4", "gap", 768),
            ("stage4", "spp", 3840),
            ("stage3_stage4", "gap", 256),
            ("stage3_stage4", "spp", 1280),
        )
        for feature_mode, pooling_mode, expected_width in cases:
            with self.subTest(feature_mode=feature_mode, pooling_mode=pooling_mode):
                head = SpatialMultiscaleHead(
                    384, 768, 10, feature_mode=feature_mode, pooling_mode=pooling_mode)
                feature_map = head.feature_map(maps)
                logits = head(maps)
                self.assertEqual(head.embedding_width, expected_width)
                self.assertEqual(tuple(logits.shape), (2, 10))
                self.assertEqual(feature_map.shape[1], 256 if feature_mode == "stage3_stage4" else 768)
                logits.square().mean().backward()
                self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

    def test_metrics_include_threshold_confusion_counts(self):
        targets = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1]]).numpy()
        probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.1], [0.1, 0.4]]).numpy()
        known = torch.ones_like(torch.tensor(targets), dtype=torch.bool).numpy()
        metrics = multilabel_metrics(targets, probabilities, known)
        first = metrics["per_label"][0]
        self.assertEqual((first["tp"], first["fp"], first["tn"], first["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(first["precision"], 0.5)
        self.assertAlmostEqual(first["recall"], 0.5)
        self.assertAlmostEqual(first["f1"], 0.5)

    def test_twenty_epoch_optimizer_update_schedule_reaches_lr_floor(self):
        config = {
            "epochs": 20,
            "warmup_epochs": 2,
            "cosine_end_epoch": 20,
            "learning_rate_backbone": 1e-5,
            "learning_rate_head": 2e-4,
            "min_lr_backbone": 5e-7,
            "min_lr_head": 1e-5,
            "scheduler_step_unit": "optimizer_update",
        }
        updates = [1005 + (epoch % 4) for epoch in range(20)]
        _spec, rows = build_lr_schedule_table(config, updates)
        self.assertEqual(len(rows), 20)
        self.assertAlmostEqual(rows[-1]["backbone_lr_last_used"], 5e-7, places=14)
        self.assertAlmostEqual(rows[-1]["head_lr_last_used"], 1e-5, places=12)
        self.assertAlmostEqual(rows[-1]["backbone_lr_next"], 5e-7, places=14)
        self.assertAlmostEqual(rows[-1]["head_lr_next"], 1e-5, places=12)
        for row in rows:
            self.assertAlmostEqual(row["head_lr_start"] / row["backbone_lr_start"], 20.0, places=10)


if __name__ == "__main__":
    unittest.main()
