from __future__ import annotations

import argparse
from pathlib import Path
import random
import time
import json
from datetime import datetime

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

from src.models.dataset_halves_multitask import DatasetConfig, TIDSHalvesMultitaskDataset
from src.models.transforms_halves import get_transforms_halves
from src.models.halves_multitask_densenet import (
    DenseNet121HalvesMultitask,
    freeze_backbone,
    unfreeze_last_block,
    trainable_params,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_logits = []
    all_y = []

    all_left_true = []
    all_left_pred = []

    all_right_true = []
    all_right_pred = []

    for x_left, x_right, y in loader:
        x_left = x_left.to(device)
        x_right = x_right.to(device)

        outputs = model(x_left, x_right)

        logits_bin = outputs["logits_bin"]
        logits_left = outputs["logits_left"]
        logits_right = outputs["logits_right"]

        y_bin = y["y_bin"].to(device)
        y_left = y["y_left"].to(device)
        y_right = y["y_right"].to(device)

        all_logits.append(logits_bin.detach().cpu().numpy())
        all_y.append(y_bin.detach().cpu().numpy())

        all_left_pred.append(torch.argmax(logits_left, dim=1).detach().cpu().numpy())
        all_left_true.append(y_left.detach().cpu().numpy())

        all_right_pred.append(torch.argmax(logits_right, dim=1).detach().cpu().numpy())
        all_right_true.append(y_right.detach().cpu().numpy())

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_y).astype(int)
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    left_pred = np.concatenate(all_left_pred)
    left_true = np.concatenate(all_left_true)

    right_pred = np.concatenate(all_right_pred)
    right_true = np.concatenate(all_right_true)

    out = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
        out["pr_auc"] = float(average_precision_score(y_true, probs))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = int(tn), int(fp), int(fn), int(tp)

    out["left_acc"] = float(accuracy_score(left_true, left_pred))
    out["right_acc"] = float(accuracy_score(right_true, right_pred))

    return out


def run_phase(
    model,
    train_loader,
    val_loader,
    device,
    criterion_bin,
    criterion_cls,
    optimizer,
    scheduler,
    max_epochs,
    patience,
    phase_name,
    lambda_bin=1.0,
    lambda_left=0.5,
    lambda_right=0.5,
):
    history = []
    best_f1 = -1.0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        n_batches = 0

        for x_left, x_right, y in train_loader:
            x_left = x_left.to(device)
            x_right = x_right.to(device)

            y_bin = y["y_bin"].to(device)
            y_left = y["y_left"].to(device)
            y_right = y["y_right"].to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(x_left, x_right)

            loss_bin = criterion_bin(outputs["logits_bin"], y_bin)
            loss_left = criterion_cls(outputs["logits_left"], y_left)
            loss_right = criterion_cls(outputs["logits_right"], y_right)

            loss = (
                lambda_bin * loss_bin
                + lambda_left * loss_left
                + lambda_right * loss_right
            )

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)

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
            f"LeftAcc {val_metrics['left_acc']:.4f} | "
            f"RightAcc {val_metrics['right_acc']:.4f}"
        )

        if val_metrics["f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["f1"]
            no_improve = 0
            best_state = {
                "model_state": model.state_dict(),
                "phase": phase_name,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{phase_name}] Early stopping triggered.")
            break

    return history, best_state


def train_one_fold(project_root, csv_path, fold, image_view, batch_size, seed, swap_lr=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TIDSHalvesMultitaskDataset(
        DatasetConfig(
            project_root=project_root,
            csv_path=csv_path,
            image_view=image_view,
            split="train",
            fold=fold,
            mode="train",
            swap_lr=swap_lr,
        ),
        transform=get_transforms_halves("train"),
    )
    val_ds = TIDSHalvesMultitaskDataset(
        DatasetConfig(
            project_root=project_root,
            csv_path=csv_path,
            image_view=image_view,
            split="train",
            fold=fold,
            mode="val",
            swap_lr=swap_lr,
        ),
        transform=get_transforms_halves("val"),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = DenseNet121HalvesMultitask(pretrained=True).to(device)

    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_cls = nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"halves_multitask_densenet121_{image_view}_fold{fold}_{timestamp}"
    run_dir = project_root / "reports" / "runs_halves_multitask" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "fold": fold,
                "image_view": image_view,
                "batch_size": batch_size,
                "seed": seed,
                "swap_lr": swap_lr,
            },
            f,
            indent=4,
        )

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Saving to: {run_dir}")

    history = []

    # Phase 1
    freeze_backbone(model)
    optimizer1 = torch.optim.AdamW(trainable_params(model), lr=3e-4)
    hist1, best_head = run_phase(
        model, train_loader, val_loader, device,
        criterion_bin, criterion_cls,
        optimizer1, None,
        max_epochs=8, patience=3,
        phase_name="head"
    )
    history.extend(hist1)
    torch.save(best_head, run_dir / "best_head.pt")

    # Phase 2
    unfreeze_last_block(model)
    optimizer2 = torch.optim.AdamW(trainable_params(model), lr=2e-5)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="max", factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6
    )
    hist2, best_finetune = run_phase(
        model, train_loader, val_loader, device,
        criterion_bin, criterion_cls,
        optimizer2, scheduler2,
        max_epochs=15, patience=4,
        phase_name="finetune"
    )
    history.extend(hist2)
    torch.save(best_finetune, run_dir / "best_finetune.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print("Training finished.")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--swap_lr", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"

    train_one_fold(
        project_root=project_root,
        csv_path=csv_path,
        fold=args.fold,
        image_view=args.image_view,
        batch_size=args.batch_size,
        seed=args.seed,
        swap_lr=args.swap_lr,
    )


if __name__ == "__main__":
    main()