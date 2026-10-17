from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class ResizePad:
    """Resize without changing aspect ratio, then pad to a square canvas."""
    def __init__(self, size: int, fill: int = 0):
        self.size, self.fill = size, fill

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = T.functional.resize(image, new_size, InterpolationMode.BILINEAR)
        resized_width, resized_height = image.size
        pad_w, pad_h = self.size - resized_width, self.size - resized_height
        left, top = pad_w // 2, pad_h // 2
        return T.functional.pad(image, [left, top, pad_w - left, pad_h - top], fill=self.fill)


def build_transform(size: int = 320, train: bool = False):
    ops = [ResizePad(size), T.Grayscale(num_output_channels=3)]
    if train:
        ops.append(T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05)], p=0.5))
        ops.append(T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03), fill=0))
    ops.extend([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return T.Compose(ops)
