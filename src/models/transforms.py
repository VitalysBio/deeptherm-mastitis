from PIL import Image
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF


class ResizeLongestSide:
    """
    Resize image so that the longest side becomes target_size,
    keeping aspect ratio.
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
    Pad image to (size x size) with symmetric padding.
    """
    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        pad_w = self.size - w
        pad_h = self.size - h
        if pad_w < 0 or pad_h < 0:
            raise ValueError(f"Image is larger than target square: got {(w,h)} target {self.size}")

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return TF.pad(img, padding=(left, top, right, bottom), fill=self.fill)


def get_transforms(mode: str, image_size: int = 224):

    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    base = [
        ResizeLongestSide(image_size),
        PadToSquare(image_size, fill=0),
    ]

    if mode == "train":
        return transforms.Compose(
            base + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ToTensor(),
                imagenet_norm,
            ]
        )
    else:
        return transforms.Compose(
            base + [
                transforms.ToTensor(),
                imagenet_norm,
            ]
        )