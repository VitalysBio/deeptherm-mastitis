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
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.models.dataset import DatasetConfig, TIDSMastitisDataset
from src.models.transforms import get_transforms


def parse_fold_from_name(name: str) -> int | None:
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
        logits = model(x).squeeze(1).detach().cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

        all_probs.append(probs)
        all_ids.append(y["id"].detach().cpu().numpy())
        all_y.append(y["y_bin"].detach().cpu().numpy())

    ids = np.concatenate(all_ids)
    y_true = np.concatenate(all_y).astype(int)
    probs = np.concatenate(all_probs).astype(float)
    return ids, y_true, probs


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float):
    y_pred = (probs >= thr).astype(int)
    out = {
        "thr": float(thr),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
        out["pr_auc"] = float(average_precision_score(y_true, probs))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = int(tn), int(fp), int(fn), int(tp)
    return out


def choose_threshold_screening(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_recall: float,
    min_precision: float | None,
    grid_size: int = 400,
):
    # Threshold grid: include extremes
    thresholds = np.linspace(0.0, 1.0, grid_size)

    best = None
    candidates = []

    for thr in thresholds:
        y_pred = (probs >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if rec + 1e-12 < target_recall:
            continue
        if min_precision is not None and prec + 1e-12 < min_precision:
            continue

        candidates.append((thr, prec, rec, f1))

    if not candidates:
        # Fallback: choose threshold that maximizes recall (and then precision) even if target not reached
        best_thr = 0.0
        best_tuple = (-1.0, -1.0, -1.0)  # rec, prec, f1
        for thr in thresholds:
            y_pred = (probs >= thr).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            tup = (rec, prec, f1)
            if tup > best_tuple:
                best_tuple = tup
                best_thr = float(thr)
        return best_thr, {"note": "fallback_max_recall", "target_recall": target_recall, "min_precision": min_precision}

    # Among candidates meeting recall (and min_precision), pick highest precision, tie-breaker F1, then higher threshold
    candidates.sort(key=lambda x: (x[1], x[3], x[0]), reverse=True)
    thr, prec, rec, f1 = candidates[0]
    return float(thr), {"note": "meets_constraints", "target_recall": target_recall, "min_precision": min_precision, "prec": float(prec), "rec": float(rec), "f1": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--phase", type=str, required=True, choices=["head", "finetune"])
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--target_recall", type=float, default=0.80)
    ap.add_argument("--min_precision", type=float, default=0.50)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"

    # Test dataloader
    test_ds = TIDSMastitisDataset(
        DatasetConfig(project_root, csv_path, image_view=args.image_view, split="test", fold=None, mode="test"),
        transform=get_transforms("test"),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows_thr = []
    rows_test = []
    prob_table = None
    thresholds = []

    # Loop runs (folds)
    for run in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        fold = parse_fold_from_name(run.name)
        if fold is None:
            continue

        val_pred_path = run / f"best_{args.phase}_val_predictions.csv"
        ckpt_path = run / f"best_{args.phase}.pt"

        if not val_pred_path.exists() or not ckpt_path.exists():
            continue

        # Load validation predictions
        vdf = pd.read_csv(val_pred_path)
        if not {"y_true", "prob"}.issubset(set(vdf.columns)):
            raise ValueError(f"{val_pred_path} must have columns y_true and prob")

        yv = vdf["y_true"].astype(int).values
        pv = vdf["prob"].astype(float).values

        # Choose threshold for screening
        thr, info = choose_threshold_screening(
            y_true=yv,
            probs=pv,
            target_recall=args.target_recall,
            min_precision=args.min_precision if args.min_precision is not None else None,
        )
        thresholds.append(thr)

        rows_thr.append(
            {
                "fold": fold,
                "run_name": run.name,
                "phase": args.phase,
                "thr": thr,
                "target_recall": args.target_recall,
                "min_precision": args.min_precision,
                "note": info.get("note", ""),
                "val_prec": info.get("prec", np.nan),
                "val_rec": info.get("rec", np.nan),
                "val_f1": info.get("f1", np.nan),
            }
        )

        # Predict test with model
        model = build_model(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        ids, yt, pt = predict_probs(model, test_loader, device)

        # Metrics with fold-specific threshold
        m = compute_metrics(yt, pt, thr=thr)
        m["fold"] = fold
        m["run_name"] = run.name
        m["phase"] = args.phase
        rows_test.append(m)

        df_probs = pd.DataFrame({"id": ids, "y_true": yt, f"prob_fold{fold}": pt})
        prob_table = df_probs if prob_table is None else prob_table.merge(df_probs, on=["id", "y_true"], how="inner")

    if not rows_test:
        raise RuntimeError("No folds were evaluated. Check runs_dir and file names in each run folder.")

    # Save thresholds + per-fold test metrics
    thr_df = pd.DataFrame(rows_thr).sort_values("fold")
    test_df = pd.DataFrame(rows_test).sort_values("fold")

    thr_df.to_csv(out_dir / f"thresholds_{args.phase}.csv", index=False)
    test_df.to_csv(out_dir / f"test_per_fold_{args.phase}_thr_optimized.csv", index=False)

    # Ensemble probs
    prob_cols = [c for c in prob_table.columns if c.startswith("prob_fold")]
    prob_table["prob_ensemble"] = prob_table[prob_cols].mean(axis=1)

    # Ensemble threshold: use median of fold thresholds (robust)
    thr_ens = float(np.median(np.array(thresholds)))
    ens_metrics = compute_metrics(prob_table["y_true"].values.astype(int), prob_table["prob_ensemble"].values, thr=thr_ens)

    prob_table.to_csv(out_dir / f"test_ensemble_predictions_{args.phase}.csv", index=False)

    # Print summary
    print(f"Device: {device}")
    print(f"Folds evaluated: {len(test_df)}")
    print(f"Ensemble threshold (median of folds): {thr_ens:.4f}")

    print("\nPer-fold thresholds (from validation):")
    print(thr_df[["fold", "thr", "note", "val_prec", "val_rec", "val_f1"]].to_string(index=False))

    print("\nPer-fold test metrics using fold-specific thresholds:")
    print(test_df[["fold", "thr", "f1", "precision", "recall", "pr_auc", "roc_auc", "mcc", "acc"]].to_string(index=False))

    print("\nEnsemble test metrics using ensemble threshold:")
    for k, v in ens_metrics.items():
        if k in {"tn", "fp", "fn", "tp"}:
            continue
        print(f"{k}: {v:.4f}")
    print(f"Confusion: tn={ens_metrics['tn']} fp={ens_metrics['fp']} fn={ens_metrics['fn']} tp={ens_metrics['tp']}")


if __name__ == "__main__":
    main()