from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

from src.models.dataset import DatasetConfig, TIDSMastitisDataset
from src.models.transforms_thermal import get_transforms_thermal



# Utilities


def parse_fold_from_name(name: str):
    m = re.search(r"fold(\d+)", name)
    return int(m.group(1)) if m else None


def build_model(device: torch.device):
    from torchvision.models import densenet121, DenseNet121_Weights
    try:
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    except Exception:
        model = densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model.to(device)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_ids, all_y, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(1).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

        all_probs.append(probs)
        all_ids.append(y["id"].cpu().numpy())
        all_y.append(y["y_bin"].cpu().numpy())

    ids = np.concatenate(all_ids)
    y_true = np.concatenate(all_y).astype(int)
    probs = np.concatenate(all_probs)

    return ids, y_true, probs


def compute_metrics(y_true, probs, thr=0.5):
    y_pred = (probs >= thr).astype(int)

    out = {}
    out["acc"] = accuracy_score(y_true, y_pred)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["mcc"] = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, probs)
        out["pr_auc"] = average_precision_score(y_true, probs)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = tn, fp, fn, tp

    return out


# Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--image_view", type=str, default="crop")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="test_results")
    ap.add_argument("--phase", type=str, required=True, choices=["head","finetune"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"

    test_ds = TIDSMastitisDataset(
        DatasetConfig(project_root, csv_path, image_view=args.image_view, split="test", fold=None, mode="test"),
        transform=get_transforms_thermal("test"),
    )

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    runs_dir = Path(args.runs_dir)
    run_folders = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    per_fold_metrics = []
    prob_table = None

    print(f"Device: {device}")
    print(f"Test size: {len(test_ds)}")

    for run in run_folders:
        if args.phase == "finetune":
            ckpt_path = run / "best_finetune.pt"
        elif args.phase == "head":
            ckpt_path = run / "best_head.pt"
        else:
            raise ValueError("phase must be 'head' or 'finetune'")
        
        if not ckpt_path.exists():
            continue

        fold = parse_fold_from_name(run.name)

        model = build_model(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        ids, y_true, probs = predict_probs(model, test_loader, device)
        metrics = compute_metrics(y_true, probs, thr=args.thr)

        metrics["fold"] = fold
        per_fold_metrics.append(metrics)

        df_probs = pd.DataFrame({
            "id": ids,
            "y_true": y_true,
            f"prob_fold{fold}": probs
        })

        if prob_table is None:
            prob_table = df_probs
        else:
            prob_table = prob_table.merge(df_probs, on=["id","y_true"])

    # Save per-fold metrics
    per_fold_df = pd.DataFrame(per_fold_metrics).sort_values("fold")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_df.to_csv(out_dir / "test_per_fold.csv", index=False)

    print("\nPer-fold test metrics:")
    print(per_fold_df[["fold","f1","precision","recall","pr_auc","roc_auc","mcc","acc"]])

    # Ensemble
    prob_cols = [c for c in prob_table.columns if c.startswith("prob_fold")]
    prob_table["prob_ensemble"] = prob_table[prob_cols].mean(axis=1)

    ens_metrics = compute_metrics(
        prob_table["y_true"].values,
        prob_table["prob_ensemble"].values,
        thr=args.thr
    )

    print("\nEnsemble test metrics:")
    for k,v in ens_metrics.items():
        if k not in ["tn","fp","fn","tp"]:
            print(f"{k}: {v:.4f}")

    prob_table.to_csv(out_dir / "test_ensemble_predictions.csv", index=False)


if __name__ == "__main__":
    main()