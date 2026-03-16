from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


@dataclass
class DatasetConfig:
    project_root: Path
    csv_path: Path
    image_view: str           # "crop" or "full"
    split: str                # "train" or "test"
    fold: Optional[int] = None
    mode: str = "train"       # "train", "val", "test"
    swap_lr: bool = False     # set True if image left/right is reversed vs metadata
    center_margin: int = 8    # pixels removed around the center split after preprocessing
    half_size: int = 224      # each half resized to half_size x half_size


class TIDSHalvesMultitaskDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, transform=None):
        self.cfg = cfg
        self.transform = transform

        df = pd.read_csv(cfg.csv_path)

        if cfg.image_view not in {"crop", "full"}:
            raise ValueError("image_view must be 'crop' or 'full'")

        path_col = "crop_path" if cfg.image_view == "crop" else "full_path"
        if path_col not in df.columns:
            raise ValueError(f"Missing column in splits CSV: {path_col}")

        # Global split
        df = df[df["split"] == cfg.split].copy()

        # Train/val split inside training pool
        if cfg.split == "train":
            if cfg.fold is None:
                raise ValueError("fold must be provided when split='train'")
            if cfg.mode not in {"train", "val"}:
                raise ValueError("mode must be 'train' or 'val' when split='train'")

            if cfg.mode == "train":
                df = df[df["cv_fold"] != cfg.fold]
            else:
                df = df[df["cv_fold"] == cfg.fold]

        if cfg.split == "test":
            cfg.mode = "test"

        required_cols = ["label", "l_scc_class", "r_scc_class", path_col]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing required column in splits CSV: {c}")

        df = df[df[path_col].notna()].copy()
        df = df[df["l_scc_class"].notna() & df["r_scc_class"].notna()].copy()

        self.df = df.reset_index(drop=True)
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        r = self.df.iloc[idx]

        img_path = self.cfg.project_root / str(r[self.path_col])
        img = Image.open(img_path).convert("RGB")

        if self.transform is None:
            raise ValueError("Transform must be provided for TIDSHalvesMultitaskDataset")

        # Full-image preprocessing first
        img = self.transform(img)  # tensor [C,H,W]

        _, h, w = img.shape
        mid = w // 2
        margin = self.cfg.center_margin // 2

        left_end = max(1, mid - margin)
        right_start = min(w - 1, mid + margin)

        img_left = img[:, :, :left_end]
        img_right = img[:, :, right_start:]

        # Resize each half to square for branch encoder
        img_left = TF.resize(img_left, [self.cfg.half_size, self.cfg.half_size], antialias=True)
        img_right = TF.resize(img_right, [self.cfg.half_size, self.cfg.half_size], antialias=True)

        y_bin = torch.tensor(float(r["label"]), dtype=torch.float32)

        # Convert from 1..5 to 0..4 for CrossEntropyLoss
        left_cls = int(r["l_scc_class"]) - 1
        right_cls = int(r["r_scc_class"]) - 1

        if self.cfg.swap_lr:
            left_cls, right_cls = right_cls, left_cls

        targets = {
            "y_bin": y_bin,
            "y_left": torch.tensor(left_cls, dtype=torch.long),
            "y_right": torch.tensor(right_cls, dtype=torch.long),
            "id": torch.tensor(int(r["id"]), dtype=torch.long),
        }

        return img_left, img_right, targets