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

from src.models.dataset_multitask import DatasetConfig, TIDSMastitisMultitaskDataset
from src.models.transforms import get_transforms
from src.models.multitask_densenet import DenseNet121Multitask


def parse_fold_from_name(name: str):
    m = re.search(r"fold(\d+)", name)
    return int(m.group(1)) if m else None


@torch.no_grad()
def predict_multitask(model, loader, device):
    model.eval()

    all_ids = []
    all_y_bin = []
    all_prob_bin = []

    all_left_true = []
    all_left_pred = []

    all_right_true = []
    all_right_pred = []

    for x, y in loader:
        x = x.to(device)

        outputs = model(x)

        logits_bin = outputs["logits_bin"]
        logits_left = outputs["logits_left"]
        logits_right = outputs["logits_right"]

        probs_bin = torch.sigmoid(logits_bin).cpu().numpy()
        pred_left = torch.argmax(logits_left, dim=1).cpu().numpy()
        pred_right = torch.argmax(logits_right, dim=1).cpu().numpy()

        all_ids.append(y["id"].cpu().numpy())
        all_y_bin.append(y["y_bin"].cpu().numpy())
        all_prob_bin.append(probs_bin)

        all_left_true.append(y["y_left"].cpu().numpy())
        all_left_pred.append(pred_left)

        all_right_true.append(y["y_right"].cpu().numpy())
        all_right_pred.append(pred_right)

    ids = np.concatenate(all_ids)
    y_bin = np.concatenate(all_y_bin).astype(int)
    prob_bin = np.concatenate(all_prob_bin).astype(float)

    left_true = np.concatenate(all_left_true).astype(int)
    left_pred = np.concatenate(all_left_pred).astype(int)

    right_true = np.concatenate(all_right_true).astype(int)
    right_pred = np.concatenate(all_right_pred).astype(int)

    return ids, y_bin, prob_bin, left_true, left_pred, right_true, right_pred


def compute_binary_metrics(y_true, probs, thr=0.5):
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
    out["thr"] = float(thr)
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

    test_ds = TIDSMastitisMultitaskDataset(
        DatasetConfig(
            project_root=project_root,
            csv_path=csv_path,
            image_view=args.image_view,
            split="test",
            fold=None,
            mode="test",
        ),
        transform=get_transforms("test"),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    per_fold_rows = []
    prob_table = None
    left_vote_table = None
    right_vote_table = None

    run_folders = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    for run in run_folders:
        fold = parse_fold_from_name(run.name)
        if fold is None:
            continue

        ckpt_path = run / f"best_{args.phase}.pt"
        if not ckpt_path.exists():
            continue

        model = DenseNet121Multitask(pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        ids, y_bin, prob_bin, left_true, left_pred, right_true, right_pred = predict_multitask(
            model, test_loader, device
        )

        metrics = compute_binary_metrics(y_bin, prob_bin, thr=args.thr)
        metrics["left_acc"] = float(accuracy_score(left_true, left_pred))
        metrics["right_acc"] = float(accuracy_score(right_true, right_pred))
        metrics["fold"] = fold
        metrics["run_name"] = run.name
        metrics["phase"] = args.phase
        per_fold_rows.append(metrics)

        df_prob = pd.DataFrame({
            "id": ids,
            "y_true": y_bin,
            f"prob_fold{fold}": prob_bin,
        })
        prob_table = df_prob if prob_table is None else prob_table.merge(df_prob, on=["id", "y_true"])

        df_left = pd.DataFrame({
            "id": ids,
            "y_left_true": left_true,
            f"left_pred_fold{fold}": left_pred,
        })
        left_vote_table = df_left if left_vote_table is None else left_vote_table.merge(df_left, on=["id", "y_left_true"])

        df_right = pd.DataFrame({
            "id": ids,
            "y_right_true": right_true,
            f"right_pred_fold{fold}": right_pred,
        })
        right_vote_table = df_right if right_vote_table is None else right_vote_table.merge(df_right, on=["id", "y_right_true"])

    if not per_fold_rows:
        raise RuntimeError("No multitask checkpoints found. Check runs_dir and phase.")

    per_fold_df = pd.DataFrame(per_fold_rows).sort_values("fold")
    per_fold_df.to_csv(out_dir / f"test_per_fold_{args.phase}.csv", index=False)

    print("Per-fold test metrics:")
    print(
        per_fold_df[
            ["fold", "f1", "precision", "recall", "pr_auc", "roc_auc", "mcc", "acc", "left_acc", "right_acc"]
        ].to_string(index=False)
    )

    # Ensemble for binary task: mean probability
    prob_cols = [c for c in prob_table.columns if c.startswith("prob_fold")]
    prob_table["prob_ensemble"] = prob_table[prob_cols].mean(axis=1)
    ens_bin = compute_binary_metrics(prob_table["y_true"].values, prob_table["prob_ensemble"].values, thr=args.thr)

    # Ensemble for left/right tasks: majority vote
    left_cols = [c for c in left_vote_table.columns if c.startswith("left_pred_fold")]
    right_cols = [c for c in right_vote_table.columns if c.startswith("right_pred_fold")]

    left_vote_table["left_pred_ensemble"] = left_vote_table[left_cols].mode(axis=1)[0]
    right_vote_table["right_pred_ensemble"] = right_vote_table[right_cols].mode(axis=1)[0]

    ens_left_acc = float(accuracy_score(left_vote_table["y_left_true"], left_vote_table["left_pred_ensemble"]))
    ens_right_acc = float(accuracy_score(right_vote_table["y_right_true"], right_vote_table["right_pred_ensemble"]))

    prob_table.to_csv(out_dir / f"test_ensemble_probs_{args.phase}.csv", index=False)
    left_vote_table.to_csv(out_dir / f"test_left_preds_{args.phase}.csv", index=False)
    right_vote_table.to_csv(out_dir / f"test_right_preds_{args.phase}.csv", index=False)

    print("\nEnsemble test metrics:")
    print(f"acc: {ens_bin['acc']:.4f}")
    print(f"f1: {ens_bin['f1']:.4f}")
    print(f"precision: {ens_bin['precision']:.4f}")
    print(f"recall: {ens_bin['recall']:.4f}")
    print(f"mcc: {ens_bin['mcc']:.4f}")
    print(f"roc_auc: {ens_bin['roc_auc']:.4f}")
    print(f"pr_auc: {ens_bin['pr_auc']:.4f}")
    print(f"left_acc: {ens_left_acc:.4f}")
    print(f"right_acc: {ens_right_acc:.4f}")
    print(
        f"Confusion: tn={ens_bin['tn']} fp={ens_bin['fp']} fn={ens_bin['fn']} tp={ens_bin['tp']}"
    )


if __name__ == "__main__":
    main()