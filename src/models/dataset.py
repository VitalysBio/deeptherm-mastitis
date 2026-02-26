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
    fold: Optional[int] = None  # 0..4 when split="train"
    mode: str = "train"  # "train", "val", "test"


class TIDSMastitisDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        transform=None,
        target_transform=None,
    ):
        self.cfg = cfg
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(cfg.csv_path)

        if cfg.image_view not in {"crop", "full"}:
            raise ValueError("image_view must be 'crop' or 'full'")

        path_col = "crop_path" if cfg.image_view == "crop" else "full_path"
        if path_col not in df.columns:
            raise ValueError(f"Missing column in splits CSV: {path_col}")

        # Filter by global split
        df = df[df["split"] == cfg.split].copy()

        # For training pool we need a fold and mode
        if cfg.split == "train":
            if cfg.fold is None:
                raise ValueError("fold must be provided when split='train'")
            if cfg.mode not in {"train", "val"}:
                raise ValueError("mode must be 'train' or 'val' when split='train'")

            if cfg.mode == "train":
                df = df[df["cv_fold"] != cfg.fold]
            else:
                df = df[df["cv_fold"] == cfg.fold]

        # For test set, ignore fold
        if cfg.split == "test":
            cfg.mode = "test"

        # Keep only rows with an existing path
        df = df[df[path_col].notna()].copy()

        self.df = df.reset_index(drop=True)
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        r = self.df.iloc[idx]

        img_path = self.cfg.project_root / str(r[self.path_col])
        img = Image.open(img_path).convert("RGB")

        y_bin = int(r["label"])

        # Optional multitask targets
        y = {
            "y_bin": y_bin,
            "l_scc_class": int(r["l_scc_class"]) if pd.notna(r.get("l_scc_class")) else None,
            "r_scc_class": int(r["r_scc_class"]) if pd.notna(r.get("r_scc_class")) else None,
            "l_scc": float(r["l_scc"]) if pd.notna(r.get("l_scc")) else None,
            "r_scc": float(r["r_scc"]) if pd.notna(r.get("r_scc")) else None,
            "id": int(r["id"]),
        }

        if self.transform:
            img = self.transform(img)

        # Convert binary label to tensor for BCE
        y["y_bin"] = torch.tensor(y["y_bin"], dtype=torch.float32)

        return img, y