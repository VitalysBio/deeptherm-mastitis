from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def parse_fold_from_name(name: str) -> int | None:
    m = re.search(r"fold(\d+)", name)
    return int(m.group(1)) if m else None


def load_history(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "history.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing history.csv in {run_dir}")
    df = pd.read_csv(p)

    if "phase" not in df.columns:
        raise ValueError(f"'phase' column not found in {p}")

    df["phase"] = df["phase"].astype(str).str.strip().str.lower()

    numeric_cols = [
        "epoch",
        "train_loss",
        "acc",
        "f1",
        "precision",
        "recall",
        "mcc",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "seconds",
        "lr",
        "left_acc",
        "right_acc",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def pick_best(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Columns: {list(df.columns)}")
    df2 = df.dropna(subset=[metric])
    if df2.empty:
        raise ValueError(f"No valid values for metric '{metric}' after NaN filtering.")
    idx = df2[metric].idxmax()
    return df2.loc[idx]


def summarize_phase(run_dir: Path, df: pd.DataFrame, phase: str, metric: str) -> dict | None:
    sub = df[df["phase"] == phase].copy()
    if sub.empty:
        return None

    best = pick_best(sub, metric)

    row = {
        "run_name": run_dir.name,
        "fold": parse_fold_from_name(run_dir.name),
        "phase": phase,
        "best_metric": metric,
        "best_epoch": int(best["epoch"]) if pd.notna(best["epoch"]) else None,
        "has_best_head_pt": (run_dir / "best_head.pt").exists(),
        "has_best_finetune_pt": (run_dir / "best_finetune.pt").exists(),
    }

    for col in [
        "train_loss",
        "acc",
        "f1",
        "precision",
        "recall",
        "mcc",
        "roc_auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "seconds",
        "lr",
        "left_acc",
        "right_acc",
    ]:
        if col in best.index:
            row[col] = best[col]

    return row


def print_mean_sd(df: pd.DataFrame, title: str):
    print(f"\n{title}")
    for m in ["f1", "precision", "recall", "pr_auc", "roc_auc", "mcc", "acc", "left_acc", "right_acc"]:
        if m in df.columns and df[m].notna().any():
            mean = pd.to_numeric(df[m], errors="coerce").mean()
            sd = pd.to_numeric(df[m], errors="coerce").std()
            print(f"{m:>9}: {mean:.4f} ± {sd:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--select_metric", type=str, default="f1")
    ap.add_argument("--out_csv", type=str, default="cv_summary_by_phase.csv")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    run_folders = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    if not run_folders:
        raise FileNotFoundError(f"No run folders found inside: {runs_dir}")

    rows = []
    for run in run_folders:
        hist_path = run / "history.csv"
        if not hist_path.exists():
            print(f"Skipping (no history.csv): {run.name}")
            continue

        df = load_history(run)

        for phase in ["head", "finetune"]:
            r = summarize_phase(run, df, phase=phase, metric=args.select_metric)
            if r is not None:
                rows.append(r)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No rows produced. Check your runs_dir and history.csv files.")

    out_df = out_df.sort_values(["phase", "fold", "run_name"], na_position="last")

    out_path = Path(args.out_csv)
    out_df.to_csv(out_path, index=False)

    print(f"Saved summary -> {out_path.resolve()}")

    cols_show = [
        c for c in [
            "phase", "fold", "best_epoch", "f1", "precision", "recall",
            "pr_auc", "roc_auc", "mcc", "acc", "left_acc", "right_acc", "lr"
        ] if c in out_df.columns
    ]
    print("\nBest per fold (by phase):")
    print(out_df[cols_show].to_string(index=False))

    print_mean_sd(out_df[out_df["phase"] == "head"], title="HEAD phase mean ± SD")
    print_mean_sd(out_df[out_df["phase"] == "finetune"], title="FINETUNE phase mean ± SD")


if __name__ == "__main__":
    main()