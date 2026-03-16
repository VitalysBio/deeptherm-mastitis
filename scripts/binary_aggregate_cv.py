from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(df):
    y_true = df["y_true"].values
    probs = df["prob"].values
    y_pred = (probs >= 0.5).astype(int)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)
        metrics["pr_auc"] = average_precision_score(y_true, probs)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


def parse_fold(run_name: str):
    m = re.search(r"fold(\d+)", run_name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--phase", type=str, required=True, choices=["head", "finetune"])
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    phase = args.phase

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No runs found in: {runs_dir}")

    # quedarse con la corrida más reciente por fold
    latest_by_fold = {}

    for run in run_dirs:
        fold = parse_fold(run.name)
        if fold is None:
            continue

        pred_file = run / f"best_{phase}_val_predictions.csv"
        if not pred_file.exists():
            continue

        if fold not in latest_by_fold:
            latest_by_fold[fold] = run
        else:
            if run.stat().st_mtime > latest_by_fold[fold].stat().st_mtime:
                latest_by_fold[fold] = run

    if not latest_by_fold:
        raise FileNotFoundError(f"No runs with best_{phase}_val_predictions.csv found in {runs_dir}")

    all_preds = []
    fold_metrics = []

    for fold in sorted(latest_by_fold.keys()):
        run = latest_by_fold[fold]
        pred_file = run / f"best_{phase}_val_predictions.csv"

        df = pd.read_csv(pred_file)
        metrics = compute_metrics(df)
        metrics["fold"] = fold
        metrics["phase"] = phase
        metrics["run"] = run.name

        fold_metrics.append(metrics)
        all_preds.append(df)

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(args.out_csv, index=False)

    print("\nPer-fold metrics\n")
    print(fold_df)

    print("\nMean ± Std\n")
    for col in ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]:
        mean = fold_df[col].mean()
        std = fold_df[col].std()
        print(f"{col:10s}: {mean:.4f} ± {std:.4f}")

    all_preds = pd.concat(all_preds, ignore_index=True)
    global_metrics = compute_metrics(all_preds)

    print("\nGlobal metrics (all folds pooled)\n")
    for k, v in global_metrics.items():
        print(f"{k:10s}: {v:.4f}")


if __name__ == "__main__":
    main()