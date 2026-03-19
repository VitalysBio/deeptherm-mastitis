from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = tn, fp, fn, tp

    return out


def format_mean_sd(mean_val, sd_val, digits=3):
    return f"{mean_val:.{digits}f} ± {sd_val:.{digits}f}"


def build_main_table(per_fold_df, ens_metrics, digits=3):
    metrics_to_show = ["f1", "precision", "recall", "pr_auc", "roc_auc", "mcc", "acc"]

    rows = []

    per_fold_df = per_fold_df.sort_values("fold").reset_index(drop=True)

    for _, row in per_fold_df.iterrows():
        rows.append(
            {
                "Model": f"Fold {int(row['fold'])}",
                "F1": f"{row['f1']:.{digits}f}",
                "Precision": f"{row['precision']:.{digits}f}",
                "Recall": f"{row['recall']:.{digits}f}",
                "PR-AUC": f"{row['pr_auc']:.{digits}f}",
                "ROC-AUC": f"{row['roc_auc']:.{digits}f}",
                "MCC": f"{row['mcc']:.{digits}f}",
                "Accuracy": f"{row['acc']:.{digits}f}",
            }
        )

    mean_vals = per_fold_df[metrics_to_show].mean()
    sd_vals = per_fold_df[metrics_to_show].std()

    rows.append(
        {
            "Model": "Mean ± SD",
            "F1": format_mean_sd(mean_vals["f1"], sd_vals["f1"], digits),
            "Precision": format_mean_sd(mean_vals["precision"], sd_vals["precision"], digits),
            "Recall": format_mean_sd(mean_vals["recall"], sd_vals["recall"], digits),
            "PR-AUC": format_mean_sd(mean_vals["pr_auc"], sd_vals["pr_auc"], digits),
            "ROC-AUC": format_mean_sd(mean_vals["roc_auc"], sd_vals["roc_auc"], digits),
            "MCC": format_mean_sd(mean_vals["mcc"], sd_vals["mcc"], digits),
            "Accuracy": format_mean_sd(mean_vals["acc"], sd_vals["acc"], digits),
        }
    )

    rows.append(
        {
            "Model": "Ensemble",
            "F1": f"{ens_metrics['f1']:.{digits}f}",
            "Precision": f"{ens_metrics['precision']:.{digits}f}",
            "Recall": f"{ens_metrics['recall']:.{digits}f}",
            "PR-AUC": f"{ens_metrics['pr_auc']:.{digits}f}",
            "ROC-AUC": f"{ens_metrics['roc_auc']:.{digits}f}",
            "MCC": f"{ens_metrics['mcc']:.{digits}f}",
            "Accuracy": f"{ens_metrics['acc']:.{digits}f}",
        }
    )

    return pd.DataFrame(rows)


def save_table_figure(table_df, out_path, title="Test set performance across folds and ensemble"):
    fig_height = 0.75 * len(table_df) + 1.5
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    n_cols = len(table_df.columns)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.12)
        else:
            cell.set_height(0.10)

    mean_row_idx = len(table_df) - 1
    ensemble_row_idx = len(table_df)

    
    mean_table_row = list(table_df["Model"]).index("Mean ± SD") + 1
    ensemble_table_row = list(table_df["Model"]).index("Ensemble") + 1

    for col in range(n_cols):
        table[(mean_table_row, col)].set_text_props(weight="bold")
        table[(ensemble_table_row, col)].set_text_props(weight="bold")

    plt.title(title, fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_figure(y_true, probs, thr, out_path, title="Ensemble confusion matrix"):
    y_pred = (probs >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, weight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Carpeta donde están test_per_fold.csv y test_ensemble_predictions.csv")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Carpeta de salida para tabla y figuras")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--digits", type=int, default=3)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_path = input_dir / "test_per_fold.csv"
    ensemble_pred_path = input_dir / "test_ensemble_predictions.csv"

    if not per_fold_path.exists():
        raise FileNotFoundError(f"No se encontró: {per_fold_path}")
    if not ensemble_pred_path.exists():
        raise FileNotFoundError(f"No se encontró: {ensemble_pred_path}")

    per_fold_df = pd.read_csv(per_fold_path)
    ensemble_df = pd.read_csv(ensemble_pred_path)

    required_per_fold_cols = {"fold", "f1", "precision", "recall", "pr_auc", "roc_auc", "mcc", "acc"}
    required_ens_cols = {"y_true", "prob_ensemble"}

    if not required_per_fold_cols.issubset(per_fold_df.columns):
        missing = required_per_fold_cols - set(per_fold_df.columns)
        raise ValueError(f"Faltan columnas en test_per_fold.csv: {missing}")

    if not required_ens_cols.issubset(ensemble_df.columns):
        missing = required_ens_cols - set(ensemble_df.columns)
        raise ValueError(f"Faltan columnas en test_ensemble_predictions.csv: {missing}")

    ens_metrics = compute_metrics(
        ensemble_df["y_true"].values,
        ensemble_df["prob_ensemble"].values,
        thr=args.thr
    )

    table_df = build_main_table(per_fold_df, ens_metrics, digits=args.digits)

    # Guardar tabla en CSV y Excel
    table_df.to_csv(out_dir / "test_results_main_table.csv", index=False)
    try:
        table_df.to_excel(out_dir / "test_results_main_table.xlsx", index=False)
    except Exception as e:
        print(f"No se pudo guardar Excel: {e}")

    # Guardar figura de tabla
    save_table_figure(
        table_df,
        out_dir / "test_results_main_table.png",
        title="Test set performance across folds, mean ± SD, and ensemble"
    )

    # Guardar matriz de confusión
    cm = save_confusion_matrix_figure(
        ensemble_df["y_true"].values,
        ensemble_df["prob_ensemble"].values,
        thr=args.thr,
        out_path=out_dir / "ensemble_confusion_matrix.png",
        title=f"Ensemble confusion matrix, threshold = {args.thr}"
    )

    # Guardar matriz como CSV
    cm_df = pd.DataFrame(
        cm,
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"]
    )
    cm_df.to_csv(out_dir / "ensemble_confusion_matrix.csv")

    print("\nTabla principal:")
    print(table_df.to_string(index=False))

    print("\nMatriz de confusión del ensemble:")
    print(cm_df)

    print(f"\nArchivos guardados en: {out_dir}")


if __name__ == "__main__":
    main()