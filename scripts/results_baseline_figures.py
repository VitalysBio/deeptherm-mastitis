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


# Metrics

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



# Build main table

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



# Web styled table figure

def save_table_figure(table_df, out_path, title=None):
    fig_height = 0.62 * len(table_df) + 1.0
    fig, ax = plt.subplots(figsize=(12.5, fig_height))

    # Fondo transparente
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.45)

    n_rows = len(table_df)
    n_cols = len(table_df.columns)


    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor("black")
        cell.set_facecolor((1, 1, 1, 0))


    for col in range(n_cols):
        cell = table[(0, col)]
        cell.visible_edges = "TB"
        cell.set_linewidth(1.2)
        cell.set_text_props(weight="bold", color="black")
        cell.set_height(0.12)

 
    for row in range(1, n_rows + 1):
        for col in range(n_cols):
            cell = table[(row, col)]
            cell.visible_edges = "B"
            cell.set_linewidth(0.7)
            cell.set_height(0.105)

  
    mean_row = list(table_df["Model"]).index("Mean ± SD") + 1
    for col in range(n_cols):
        table[(mean_row, col)].set_text_props(weight="bold")

   
    ensemble_row = list(table_df["Model"]).index("Ensemble") + 1
    for col in range(n_cols):
        table[(ensemble_row, col)].set_text_props(weight="bold")

  
    recall_col = list(table_df.columns).index("Recall")
    table[(ensemble_row, recall_col)].set_text_props(weight="bold")

    if title:
        plt.title(title, fontsize=13, weight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.03
    )
    plt.close()



# Web styled confusion matrix

def save_confusion_matrix_figure(
    y_true,
    probs,
    thr,
    out_path,
    title=None,
    class_names=("Healthy", "SCM"),
):
    y_pred = (probs >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

   
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    im = ax.imshow(cm, interpolation="nearest")

    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels([f"Pred {class_names[0]}", f"Pred {class_names[1]}"], fontsize=11)
    ax.set_yticklabels([f"True {class_names[0]}", f"True {class_names[1]}"], fontsize=11)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, weight="bold", pad=10)

    
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                color="white" if cm[i, j] > threshold else "black",
            )

   
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.03
    )
    plt.close()

    return cm



# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Carpeta donde están test_per_fold.csv y test_ensemble_predictions.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Carpeta de salida para tabla y figuras",
    )
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--digits", type=int, default=3)

    
    ap.add_argument("--show_titles", action="store_true")

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

   
    table_df.to_csv(out_dir / "test_results_main_table.csv", index=False)
    try:
        table_df.to_excel(out_dir / "test_results_main_table.xlsx", index=False)
    except Exception as e:
        print(f"No se pudo guardar Excel: {e}")

    
    save_table_figure(
        table_df=table_df,
        out_path=out_dir / "test_results_main_table.png",
        title="Test set performance" if args.show_titles else None,
    )

    
    cm = save_confusion_matrix_figure(
        y_true=ensemble_df["y_true"].values,
        probs=ensemble_df["prob_ensemble"].values,
        thr=args.thr,
        out_path=out_dir / "ensemble_confusion_matrix.png",
        title=f"Ensemble confusion matrix, threshold={args.thr}" if args.show_titles else None,
        class_names=("Healthy", "SCM"),
    )

    
    cm_df = pd.DataFrame(
        cm,
        index=["True Healthy", "True SCM"],
        columns=["Pred Healthy", "Pred SCM"]
    )
    cm_df.to_csv(out_dir / "ensemble_confusion_matrix.csv")

    print("\nTabla principal:")
    print(table_df.to_string(index=False))

    print("\nMatriz de confusión del ensemble:")
    print(cm_df)

    print(f"\nArchivos guardados en: {out_dir}")


if __name__ == "__main__":
    main()