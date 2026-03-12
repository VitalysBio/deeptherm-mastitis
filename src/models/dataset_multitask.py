from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    project_root: Path
    csv_path: Path
    image_view: str  # "crop" or "full"
    split: str       # "train" or "test"
    fold: Optional[int] = None
    mode: str = "train"  # "train", "val", "test"


class TIDSMastitisMultitaskDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        transform=None,
    ):
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

        # Train/val selection inside train pool
        if cfg.split == "train":
            if cfg.fold is None:
                raise ValueError("fold must be provided when split='train'")
            if cfg.mode not in {"train", "val"}:
                raise ValueError("mode must be 'train' or 'val' when split='train'")

            if cfg.mode == "train":
                df = df[df["cv_fold"] != cfg.fold]
            else:
                df = df[df["cv_fold"] == cfg.fold]

        # For test split, fold is ignored
        if cfg.split == "test":
            cfg.mode = "test"

        # Keep only valid paths and non-missing multitask targets
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        r = self.df.iloc[idx]

        img_path = self.cfg.project_root / str(r[self.path_col])
        img = Image.open(img_path).convert("RGB")

        # Binary label for BCE
        y_bin = torch.tensor(float(r["label"]), dtype=torch.float32)

        # Convert SCC classes from 1..5 to 0..4 for CrossEntropyLoss
        y_left = torch.tensor(int(r["l_scc_class"]) - 1, dtype=torch.long)
        y_right = torch.tensor(int(r["r_scc_class"]) - 1, dtype=torch.long)

        targets = {
            "y_bin": y_bin,
            "y_left": y_left,
            "y_right": y_right,
            "id": torch.tensor(int(r["id"]), dtype=torch.long),
        }

        if self.transform:
            img = self.transform(img)

        return img, targets