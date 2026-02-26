from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dataset import DatasetConfig, TIDSMastitisDataset
from src.models.transforms import get_transforms

from datetime import datetime
import json


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device: torch.device):
    # DenseNet-121 baseline, pretrained if available
    try:
        from torchvision.models import densenet121, DenseNet121_Weights

        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)
    except Exception:
        # Fallback if weights enum not available
        from torchvision.models import densenet121

        model = densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)  # binary logit
    model = model.to(device)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []
    all_ids = []

    for x, y in loader:
        x = x.to(device)
        y_bin = y["y_bin"].to(device)  # float32 0/1

        logits = model(x).squeeze(1)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y_bin.detach().cpu().numpy())
        all_ids.append(y["id"].detach().cpu().numpy())

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_y).astype(int)
    ids = np.concatenate(all_ids)

    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    # Metrics
    out = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # AUCs require both classes present
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
        out["pr_auc"] = float(average_precision_score(y_true, probs))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"], out["fp"], out["fn"], out["tp"] = [int(x) for x in cm.ravel()]

    return out, pd.DataFrame({"id": ids, "y_true": y_true, "prob": probs})


def train_one_fold(
    project_root: Path,
    csv_path: Path,
    fold: int,
    image_view: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TIDSMastitisDataset(
        DatasetConfig(project_root, csv_path, image_view=image_view, split="train", fold=fold, mode="train"),
        transform=get_transforms("train"),
    )
    val_ds = TIDSMastitisDataset(
        DatasetConfig(project_root, csv_path, image_view=image_view, split="train", fold=fold, mode="val"),
        transform=get_transforms("val"),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    threshold=1e-3,
    min_lr=1e-6,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"densenet121_{image_view}_fold{fold}_{timestamp}"

    run_dir = project_root / "reports" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "fold": fold,
        "image_view": image_view,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    history = []
    best_f1 = -1.0
    best_path = run_dir / "best_model.pt"

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Saving to: {run_dir}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y_bin = y["y_bin"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y_bin)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        val_metrics, val_preds = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "seconds": round(time.time() - t0, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"Prec {val_metrics['precision']:.4f} | Rec {val_metrics['recall']:.4f} | "
            f"PR-AUC {val_metrics['pr_auc']:.4f} | ROC-AUC {val_metrics['roc_auc']:.4f}"
        )

        # Save best model by F1 (baseline comparable to paper table)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fold": fold,
                    "image_view": image_view,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            val_preds.to_csv(run_dir / "best_val_predictions.csv", index=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(run_dir / "history.csv", index=False)

    print(f"Best F1: {best_f1:.4f}")
    print(f"Best model saved: {best_path}")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    ap.add_argument("--epochs", type=int, default=5)  # CPU-friendly
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing splits.csv at: {csv_path}")

    train_one_fold(
        project_root=project_root,
        csv_path=csv_path,
        fold=args.fold,
        image_view=args.image_view,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()