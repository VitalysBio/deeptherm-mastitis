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

def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_block(model):
    # Freeze everything
    for p in model.features.parameters():
        p.requires_grad = False

    # unfreeze last denseblock + final norm
    for name, p in model.features.named_parameters():
        if name.startswith("denseblock4") or name.startswith("norm5"):
            p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

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

def run_phase(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    scheduler,
    max_epochs,
    patience,
    run_dir,
    phase_name,
):

    history = []
    best_f1 = -1.0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
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

        if scheduler is not None:
            scheduler.step(val_metrics["f1"])

        row = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "seconds": round(time.time() - t0, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            f"[{phase_name}] Epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"Prec {val_metrics['precision']:.4f} | "
            f"Rec {val_metrics['recall']:.4f} | "
            f"PR-AUC {val_metrics['pr_auc']:.4f} | "
            f"ROC-AUC {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["f1"]
            no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "phase": phase_name,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                run_dir / f"best_{phase_name}.pt",
            )

            val_preds.to_csv(
                run_dir / f"best_{phase_name}_val_predictions.csv", index=False
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{phase_name}] Early stopping triggered.")
            break

    return history

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(device)
    criterion = nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"densenet121_{image_view}_fold{fold}_{timestamp}"
    run_dir = project_root / "reports" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Saving to: {run_dir}")

    history = []

    # =====================
    # PHASE 1 — HEAD ONLY
    # =====================
    freeze_backbone(model)

    optimizer1 = torch.optim.AdamW(trainable_params(model), lr=3e-4)

    hist1 = run_phase(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer1,
        scheduler=None,
        max_epochs=8,
        patience=3,
        run_dir=run_dir,
        phase_name="head",
    )

    history.extend(hist1)

    # =====================
    # PHASE 2 — FINE TUNE
    # =====================
    unfreeze_last_block(model)

    optimizer2 = torch.optim.AdamW(trainable_params(model), lr=2e-5)

    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-3,
        min_lr=1e-6,
    )

    hist2 = run_phase(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer2,
        scheduler2,
        max_epochs=15,
        patience=4,
        run_dir=run_dir,
        phase_name="finetune",
    )

    history.extend(hist2)

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    print("Training finished.")
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