import unittest

import torch
from PIL import Image, ImageDraw

from datasets.hybrid import build_hybrid_transform
from model.hybrid_pretraining import (mask_to_pixels, mask_to_vit_grid,
                                      masked_pixel_mse, normalized_latents,
                                      sample_spatial_mask, sample_spatial_mask_for_ids,
                                      unpatchify_vit_pixels)


class HybridMaskTests(unittest.TestCase):
    def test_train_and_validation_transforms_return_raw_square_rgb(self):
        source = Image.new("RGB", (2544, 3056), color=(255, 255, 255))
        ImageDraw.Draw(source).rectangle((800, 500, 1700, 2600), fill=(0, 0, 0))
        train = build_hybrid_transform(320, train=True)(source)
        valid = build_hybrid_transform(320, train=False)(source)
        for tensor in (train, valid):
            self.assertEqual(tuple(tensor.shape), (3, 320, 320))
            self.assertGreaterEqual(float(tensor.min()), 0.0)
            self.assertLessEqual(float(tensor.max()), 1.0)

    def test_mask_has_exact_60_percent_at_each_resolution(self):
        generator = torch.Generator().manual_seed(42)
        mask = sample_spatial_mask(3, torch.device("cpu"), generator)
        self.assertEqual(tuple(mask.shape), (3, 10, 10))
        self.assertTrue(torch.equal(mask.sum(dim=(1, 2)), torch.tensor([60, 60, 60])))
        vit_mask = mask_to_vit_grid(mask)
        self.assertEqual(tuple(vit_mask.shape), (3, 20, 20))
        self.assertTrue(torch.equal(vit_mask.sum(dim=(1, 2)), torch.tensor([240, 240, 240])))
        pixel_mask = mask_to_pixels(mask, 32)
        self.assertTrue(torch.equal(pixel_mask.sum(dim=(1, 2)), torch.tensor([60 * 32 * 32] * 3)))

    def test_h1_h2_masks_are_keyed_to_image_identity(self):
        subjects = ["s1", "s2"]
        studies = ["st1", "st2"]
        images = ["im1", "im2"]
        first = sample_spatial_mask_for_ids(subjects, studies, images, 3, 42, torch.device("cpu"))
        repeated = sample_spatial_mask_for_ids(subjects, studies, images, 3, 42, torch.device("cpu"))
        reordered = sample_spatial_mask_for_ids(subjects[::-1], studies[::-1], images[::-1],
                                               3, 42, torch.device("cpu"))
        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.equal(first, reordered.flip(0)))
        self.assertTrue(torch.equal(first.sum(dim=(1, 2)), torch.tensor([60, 60])))

    def test_pixel_loss_uses_only_masked_pixels(self):
        target = torch.zeros((1, 3, 320, 320))
        mask = torch.zeros((1, 10, 10), dtype=torch.bool)
        mask[0, 0, 0] = True
        prediction = target.clone()
        prediction[:, :, 32:, :] = 1
        self.assertEqual(float(masked_pixel_mse(prediction, target, mask, 32)), 0.0)
        prediction[:, :, :32, :32] = 1
        self.assertAlmostEqual(float(masked_pixel_mse(prediction, target, mask, 32)), 1.0, places=6)

    def test_vit_patch_decoder_unpatchifies_to_320_square(self):
        patches = torch.randn(2, 400, 3 * 16 * 16)
        images = unpatchify_vit_pixels(patches)
        self.assertEqual(tuple(images.shape), (2, 3, 320, 320))

    def test_latent_normalization_preserves_spatial_shapes_and_gradients(self):
        feature_map = torch.randn(2, 384, 20, 20, requires_grad=True)
        normalized = normalized_latents(feature_map)
        self.assertEqual(tuple(normalized.shape), tuple(feature_map.shape))
        normalized.square().mean().backward()
        self.assertIsNotNone(feature_map.grad)
        tokens = torch.randn(2, 400, 384, requires_grad=True)
        normalized_tokens = normalized_latents(tokens)
        self.assertEqual(tuple(normalized_tokens.shape), tuple(tokens.shape))
        normalized_tokens.square().mean().backward()
        self.assertIsNotNone(tokens.grad)


if __name__ == "__main__":
    unittest.main()
