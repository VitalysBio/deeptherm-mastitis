from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
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

from src.models.dataset_halves_binary import DatasetConfig, TIDSHalvesBinaryDataset
from src.models.transforms_halves import get_transforms_halves
from src.models.halves_binary_densenet_twoenc import DenseNet121HalvesBinaryTwoEnc


def parse_fold_from_name(name: str):
    m = re.search(r"fold(\d+)", name)
    return int(m.group(1)) if m else None


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()

    all_ids = []
    all_y = []
    all_probs = []

    for x_left, x_right, y in loader:
        x_left = x_left.to(device)
        x_right = x_right.to(device)

        logits = model(x_left, x_right)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_ids.append(y["id"].cpu().numpy())
        all_y.append(y["y_bin"].cpu().numpy())
        all_probs.append(probs)

    ids = np.concatenate(all_ids)
    y_true = np.concatenate(all_y).astype(int)
    probs = np.concatenate(all_probs).astype(float)

    return ids, y_true, probs


def compute_metrics(y_true, probs, thr=0.5):
    y_pred = (probs >= thr).astype(int)

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

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--phase", type=str, required=True, choices=["head", "finetune"])
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"

    test_ds = TIDSHalvesBinaryDataset(
        DatasetConfig(
            project_root=project_root,
            csv_path=csv_path,
            image_view=args.image_view,
            split="test",
            fold=None,
            mode="test",
        ),
        transform=get_transforms_halves("test"),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    per_fold_rows = []
    prob_table = None

    run_folders = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    for run in run_folders:
        fold = parse_fold_from_name(run.name)
        if fold is None:
            continue

        ckpt_path = run / f"best_{args.phase}.pt"
        if not ckpt_path.exists():
            continue

        model = DenseNet121HalvesBinaryTwoEnc(pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        ids, y_true, probs = predict_probs(model, test_loader, device)
        metrics = compute_metrics(y_true, probs, thr=args.thr)
        metrics["fold"] = fold
        metrics["run_name"] = run.name
        metrics["phase"] = args.phase
        per_fold_rows.append(metrics)

        df_probs = pd.DataFrame({
            "id": ids,
            "y_true": y_true,
            f"prob_fold{fold}": probs,
        })
        prob_table = df_probs if prob_table is None else prob_table.merge(df_probs, on=["id", "y_true"])

    if not per_fold_rows:
        raise RuntimeError("No checkpoints found. Check runs_dir and phase.")

    per_fold_df = pd.DataFrame(per_fold_rows).sort_values("fold")
    per_fold_df.to_csv(out_dir / f"test_per_fold_{args.phase}.csv", index=False)

    print("Per-fold test metrics:")
    print(per_fold_df[["fold", "f1", "precision", "recall", "pr_auc", "roc_auc", "mcc", "acc"]].to_string(index=False))

    prob_cols = [c for c in prob_table.columns if c.startswith("prob_fold")]
    prob_table["prob_ensemble"] = prob_table[prob_cols].mean(axis=1)

    ens = compute_metrics(prob_table["y_true"].values, prob_table["prob_ensemble"].values, thr=args.thr)
    prob_table.to_csv(out_dir / f"test_ensemble_probs_{args.phase}.csv", index=False)

    print("\nEnsemble test metrics:")
    print(f"acc: {ens['acc']:.4f}")
    print(f"f1: {ens['f1']:.4f}")
    print(f"precision: {ens['precision']:.4f}")
    print(f"recall: {ens['recall']:.4f}")
    print(f"mcc: {ens['mcc']:.4f}")
    print(f"roc_auc: {ens['roc_auc']:.4f}")
    print(f"pr_auc: {ens['pr_auc']:.4f}")
    print(f"Confusion: tn={ens['tn']} fp={ens['fp']} fn={ens['fn']} tp={ens['tp']}")


if __name__ == "__main__":
    main()