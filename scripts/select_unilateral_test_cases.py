from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    path_col = "crop_path" if args.image_view == "crop" else "full_path"

    required = {"id", "split", "label", "l_scc_class", "r_scc_class", path_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[df["split"] == "test"].copy()
    df = df[df[path_col].notna()].copy()
    df = df[df["l_scc_class"].notna() & df["r_scc_class"].notna()].copy()

    left_healthy = df["l_scc_class"].isin([1, 2])
    left_sick = df["l_scc_class"].isin([3, 4, 5])
    right_healthy = df["r_scc_class"].isin([1, 2])
    right_sick = df["r_scc_class"].isin([3, 4, 5])

    unilateral = df[(left_healthy & right_sick) | (left_sick & right_healthy)].copy()

    unilateral["pattern"] = unilateral.apply(
        lambda r: "L_healthy_R_sick" if r["l_scc_class"] in [1, 2] else "L_sick_R_healthy",
        axis=1
    )

    unilateral = unilateral.sort_values(["pattern", "id"]).reset_index(drop=True)
    unilateral.to_csv(args.out_csv, index=False)

    print(f"Total unilateral test cases: {len(unilateral)}")
    print(unilateral[["id", "label", "l_scc_class", "r_scc_class", "pattern"]].head(20).to_string(index=False))
    print(f"\nSaved to: {args.out_csv}")


if __name__ == "__main__":
    main()