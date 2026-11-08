import unittest

import torch
from torch import nn

from model.dsap_rpb import (
    DiseaseSpecificAttentionPooling,
    RelativePositionBias2D,
    sinusoidal_position_encoding_2d,
)
from train import build_lr_schedule_table, build_lr_scheduler


class LrScheduleTests(unittest.TestCase):
    def test_cosine_reaches_nonzero_floor_at_epoch_20_and_stays_flat(self):
        config = {
            "epochs": 40,
            "warmup_epochs": 2,
            "cosine_end_epoch": 20,
            "learning_rate_backbone": 1e-5,
            "learning_rate_head": 2e-4,
            "min_lr_backbone": 5e-7,
            "min_lr_head": 1e-5,
            "scheduler_step_unit": "optimizer_update",
        }
        updates_by_epoch = [2, 3] + [1] * 18 + [4] * 20
        spec, table = build_lr_schedule_table(config, updates_by_epoch)
        rows = {row["epoch"]: row for row in table}
        self.assertEqual(spec["warmup_steps"], 5)
        self.assertEqual(spec["cosine_steps"], sum(updates_by_epoch[2:20]))
        self.assertAlmostEqual(rows[2]["backbone_lr_last_used"], 1e-5, places=12)
        self.assertAlmostEqual(rows[2]["head_lr_last_used"], 2e-4, places=12)
        self.assertAlmostEqual(rows[3]["backbone_lr_start"], 1e-5, places=12)
        self.assertAlmostEqual(rows[20]["backbone_lr_last_used"], 5e-7, places=15)
        self.assertAlmostEqual(rows[20]["head_lr_last_used"], 1e-5, places=12)
        for epoch in range(21, 41):
            self.assertAlmostEqual(rows[epoch]["backbone_lr_start"], 5e-7, places=15)
            self.assertAlmostEqual(rows[epoch]["head_lr_start"], 1e-5, places=12)
            self.assertAlmostEqual(rows[epoch]["backbone_lr_last_used"], 5e-7, places=15)
            self.assertAlmostEqual(rows[epoch]["head_lr_last_used"], 1e-5, places=12)
        for epoch in range(3, 21):
            self.assertLessEqual(rows[epoch]["backbone_lr_last_used"],
                                 rows[epoch]["backbone_lr_start"] + 1e-15)

        backbone = nn.Parameter(torch.zeros(()))
        head = nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD([
            {"params": [backbone], "lr": config["learning_rate_backbone"]},
            {"params": [head], "lr": config["learning_rate_head"]},
        ])
        scheduler, _ = build_lr_scheduler(optimizer, config, updates_by_epoch)
        actual = {}
        for epoch in range(1, 41):
            first = [group["lr"] for group in optimizer.param_groups]
            last = None
            for _ in range(updates_by_epoch[epoch - 1]):
                last = [group["lr"] for group in optimizer.param_groups]
                optimizer.step()
                scheduler.step()
            actual[epoch] = (first, last)
        self.assertAlmostEqual(actual[20][1][0], 5e-7, places=15)
        self.assertAlmostEqual(actual[20][1][1], 1e-5, places=12)
        self.assertAlmostEqual(actual[21][0][0], 5e-7, places=15)
        self.assertAlmostEqual(actual[21][0][1], 1e-5, places=12)
        self.assertAlmostEqual(actual[40][1][0], 5e-7, places=15)
        self.assertAlmostEqual(actual[40][1][1], 1e-5, places=12)


class RelativePositionBiasTests(unittest.TestCase):
    def test_two_dimensional_indices_and_bias_gradient(self):
        module = RelativePositionBias2D(num_heads=4, max_height=5, max_width=6)
        index = module.relative_position_index(3, 4, torch.device("cpu"))
        token_count = 3 * 4
        self.assertEqual(tuple(index.shape), (token_count, token_count))
        self.assertTrue(torch.equal(index.diagonal(), index.new_full((token_count,),
                                                                    (5 - 1) * 11 + (6 - 1))))
        self.assertNotEqual(int(index[0, 1]), int(index[1, 0]))
        bias = module(3, 4)
        self.assertEqual(tuple(bias.shape), (1, 4, token_count, token_count))
        resized_bias = module(7, 9)
        self.assertEqual(tuple(resized_bias.shape), (1, 4, 63, 63))
        bias.square().sum().backward()
        self.assertIsNotNone(module.bias_table.grad)
        self.assertTrue(torch.isfinite(module.bias_table.grad).all())
        self.assertGreater(float(module.bias_table.grad.abs().sum()), 0.0)

    def test_dynamic_non_square_position_encoding(self):
        encoding = sinusoidal_position_encoding_2d(3, 5, 16, torch.device("cpu"))
        self.assertEqual(tuple(encoding.shape), (1, 15, 16))
        self.assertFalse(torch.equal(encoding[:, 0], encoding[:, 5]))
        self.assertFalse(torch.equal(encoding[:, 0], encoding[:, 1]))


class DiseaseAttentionTests(unittest.TestCase):
    def test_attention_outputs_and_all_head_gradients(self):
        torch.manual_seed(7)
        head = DiseaseSpecificAttentionPooling(
            in_channels=12, num_labels=10, dimension=16, num_heads=4,
            max_relative_height=5, max_relative_width=6,
        )
        feature_map = torch.randn(2, 12, 3, 5, requires_grad=True)
        logits, diagnostics = head(feature_map, return_attention=True)
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertEqual(tuple(diagnostics["disease_attention"].shape), (2, 10, 3, 5))
        self.assertEqual(tuple(diagnostics["spatial_attention"].shape), (2, 4, 15, 15))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(diagnostics["disease_attention"]).all())
        self.assertTrue(torch.allclose(
            diagnostics["disease_attention"].sum(dim=(2, 3)),
            torch.ones(2, 10), atol=1e-6,
        ))
        self.assertTrue(torch.allclose(diagnostics["alpha"], torch.full((10,), 0.1), atol=1e-6))
        logits.square().mean().backward()
        parameters = [
            head.disease_queries,
            head.spatial_attention.relative_position_bias.bias_table,
            head.spatial_attention.q_projection.weight,
            head.spatial_attention.k_projection.weight,
            head.spatial_attention.v_projection.weight,
            head.spatial_attention.position_scale,
            head.alpha_logits,
            head.classifier_weight,
        ]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(feature_map.grad).all())


if __name__ == "__main__":
    unittest.main()
