from __future__ import annotations

import random
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


class ResizeLongestSide:
    """
    Resize image so that the longest side becomes target_size,
    preserving aspect ratio.
    """

    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w >= h:
            new_w = self.target_size
            new_h = int(round(h * (self.target_size / w)))
        else:
            new_h = self.target_size
            new_w = int(round(w * (self.target_size / h)))

        return img.resize((new_w, new_h), resample=Image.BILINEAR)


class PadToSquare:
    """
    Symmetric padding to make the image square.
    """

    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        pad_w = self.size - w
        pad_h = self.size - h

        if pad_w < 0 or pad_h < 0:
            raise ValueError(
                f"Image is larger than target square: got {(w, h)}, target {self.size}"
            )

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return TF.pad(img, padding=(left, top, right, bottom), fill=self.fill)


class ThermalPercentileNormalize:
    """
    Percentile-based thermal normalization per image.

    Steps:
    1. Convert image to numpy
    2. If RGB, use one channel (assuming grayscale-like RGB thermal export)
    3. Compute p_low and p_high
    4. Clip to [p_low, p_high]
    5. Rescale to [0, 255]
    6. Return as 3-channel PIL image for compatibility with ImageNet backbones
    """

    def __init__(self, p_low: float = 5.0, p_high: float = 95.0):
        self.p_low = p_low
        self.p_high = p_high

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)

        # If image is RGB but visually grayscale, keep first channel
        if arr.ndim == 3:
            arr = arr[..., 0]

        p1 = np.percentile(arr, self.p_low)
        p2 = np.percentile(arr, self.p_high)

        if p2 <= p1:
            # fallback: avoid division by zero
            arr_norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = np.clip(arr, p1, p2)
            arr = (arr - p1) / (p2 - p1)
            arr = arr * 255.0
            arr_norm = arr.astype(np.uint8)

        # Back to PIL, then to RGB for DenseNet/EfficientNet
        img_out = Image.fromarray(arr_norm, mode="L").convert("RGB")
        return img_out


class ThermalDrift:
    """
    Simulate small global thermal drift by shifting brightness.
    """

    def __init__(self, max_shift: float = 0.08, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr = np.array(img).astype(np.float32) / 255.0
        shift = random.uniform(-self.max_shift, self.max_shift)
        arr = np.clip(arr + shift, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)


class ThermalSensorNoise:
    """
    Add mild Gaussian-like sensor noise.
    """

    def __init__(self, noise_std: float = 0.02, p: float = 0.5):
        self.noise_std = noise_std
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, self.noise_std, size=arr.shape)
        arr = np.clip(arr + noise, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)


class SmallCutout:
    """
    Apply a small square cutout to simulate mild occlusion.
    """

    def __init__(self, size: int = 16, p: float = 0.3, fill_value: int = 0):
        self.size = size
        self.p = p
        self.fill_value = fill_value

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr = np.array(img).copy()
        h, w = arr.shape[:2]

        cut = min(self.size, h, w)
        if cut <= 0:
            return img

        y = random.randint(0, max(0, h - cut))
        x = random.randint(0, max(0, w - cut))

        arr[y:y + cut, x:x + cut] = self.fill_value
        return Image.fromarray(arr)


def get_transforms_thermal(mode: str, image_size: int = 224):
    """
    Thermal-aware transforms.

    mode:
    - 'train': percentile normalization + thermal augmentations
    - 'val' or 'test': percentile normalization only, deterministic
    """

    base = [
        ResizeLongestSide(image_size),
        PadToSquare(image_size, fill=0),
        ThermalPercentileNormalize(p_low=5.0, p_high=95.0),
    ]

    if mode == "train":
        return transforms.Compose(
            base
            + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                ThermalDrift(max_shift=0.08, p=0.5),
                ThermalSensorNoise(noise_std=0.02, p=0.5),
                SmallCutout(size=16, p=0.3, fill_value=0),
                transforms.ToTensor(),
            ]
        )
    elif mode in {"val", "test"}:
        return transforms.Compose(
            base
            + [
                transforms.ToTensor(),
            ]
        )
    else:
        raise ValueError("mode must be one of: 'train', 'val', 'test'")