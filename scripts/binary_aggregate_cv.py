from pathlib import Path
import pandas as pd
import numpy as np

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


def main():

    runs_dir = Path("reports/runs_halves_binary_twoenc")

    all_preds = []
    fold_metrics = []

    for run in sorted(runs_dir.glob("halves_binary_twoenc_densenet121_crop_fold*")):

        pred_file = run / "best_head_val_predictions.csv"

        if not pred_file.exists():
            pred_file = run / "best_finetune_val_predictions.csv"

        if not pred_file.exists():
            continue

        df = pd.read_csv(pred_file)

        metrics = compute_metrics(df)
        metrics["run"] = run.name

        fold_metrics.append(metrics)

        all_preds.append(df)

    fold_df = pd.DataFrame(fold_metrics)

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