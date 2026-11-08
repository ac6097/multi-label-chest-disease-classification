import unittest

from PIL import Image

from datasets.transforms import ResizePad


class ResizePadTests(unittest.TestCase):
    def test_preserves_aspect_ratio_and_pads_on_the_short_axis(self):
        cases = (
            # PIL size is (width, height); expected padding is (left, top, right, bottom).
            ((2544, 3056), (27, 0, 27, 0)),
            ((3056, 2544), (0, 27, 0, 27)),
            ((2021, 2021), (0, 0, 0, 0)),
        )
        transform = ResizePad(320)

        for original_size, expected_padding in cases:
            with self.subTest(original_size=original_size):
                image = Image.new("L", original_size, color=255)
                result = transform(image)

                self.assertEqual(result.size, (320, 320))
                bounds = result.getbbox()
                self.assertIsNotNone(bounds)
                left, top, right, bottom = bounds
                actual_padding = (left, top, 320 - right, 320 - bottom)
                self.assertEqual(actual_padding, expected_padding)

                resized_width = right - left
                resized_height = bottom - top
                self.assertAlmostEqual(
                    resized_width / resized_height,
                    original_size[0] / original_size[1],
                    delta=0.01,
                )


if __name__ == "__main__":
    unittest.main()
