from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_image_as_gray(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32)

    # If RGB but thermal-like grayscale, use first channel
    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr


def mean_intensity_halves(arr: np.ndarray, center_margin: int = 8):
    h, w = arr.shape
    mid = w // 2
    margin = center_margin // 2

    left = arr[:, : max(1, mid - margin)]
    right = arr[:, min(w - 1, mid + margin) :]

    return float(left.mean()), float(right.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True, help="Path to splits.csv or manifest.csv")
    ap.add_argument("--project_root", type=str, default=".", help="Project root for relative paths")
    ap.add_argument("--image_view", type=str, default="crop", choices=["crop", "full"])
    ap.add_argument("--min_class_diff", type=int, default=2, help="Minimum abs(L-R) class difference")
    ap.add_argument("--max_images", type=int, default=50, help="Max images to inspect")
    ap.add_argument("--center_margin", type=int, default=8, help="Ignore a few pixels around center split")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    project_root = Path(args.project_root)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    path_col = "crop_path" if args.image_view == "crop" else "full_path"
    required = {"l_scc_class", "r_scc_class", path_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Use only rows with valid labels and paths
    df = df[
        df["l_scc_class"].notna()
        & df["r_scc_class"].notna()
        & df[path_col].notna()
    ].copy()

    df["class_diff"] = (df["l_scc_class"] - df["r_scc_class"]).abs()
    df = df[df["class_diff"] >= args.min_class_diff].copy()

    if "split" in df.columns:
        # Optional: just inspect test or all
        # Uncomment if you want only test:
        # df = df[df["split"] == "test"].copy()
        pass

    df = df.head(args.max_images).copy()

    if df.empty:
        raise RuntimeError("No images found after filtering. Try lowering --min_class_diff.")

    results = []

    for _, r in df.iterrows():
        img_path = project_root / str(r[path_col])
        if not img_path.exists():
            continue

        arr = load_image_as_gray(img_path)
        left_mean, right_mean = mean_intensity_halves(arr, center_margin=args.center_margin)

        l_cls = int(r["l_scc_class"])
        r_cls = int(r["r_scc_class"])

        hotter_side = "left" if left_mean > right_mean else "right"
        higher_scc_side = "left" if l_cls > r_cls else "right"

        results.append(
            {
                "id": int(r["id"]) if "id" in r else None,
                "image_path": str(r[path_col]),
                "l_scc_class": l_cls,
                "r_scc_class": r_cls,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "hotter_side": hotter_side,
                "higher_scc_side": higher_scc_side,
                "match": hotter_side == higher_scc_side,
            }
        )

    res_df = pd.DataFrame(results)

    if res_df.empty:
        raise RuntimeError("No valid image paths were processed.")

    match_rate = res_df["match"].mean()

    print("Orientation check summary")
    print("-" * 40)
    print(f"Images analyzed: {len(res_df)}")
    print(f"Match rate (hotter side == higher SCC side): {match_rate:.3f}")
    print()

    print("Sample rows:")
    print(
        res_df[
            ["id", "l_scc_class", "r_scc_class", "left_mean", "right_mean", "hotter_side", "higher_scc_side", "match"]
        ].head(15).to_string(index=False)
    )

    out_csv = csv_path.parent / f"orientation_check_{args.image_view}.csv"
    res_df.to_csv(out_csv, index=False)
    print()
    print(f"Saved detailed results to: {out_csv}")

    # Heuristic interpretation
    print()
    if match_rate >= 0.65:
        print("Interpretation: likely NO swap needed (image left ~ metadata left).")
    elif match_rate <= 0.35:
        print("Interpretation: likely swap needed (image left ~ metadata right).")
    else:
        print("Interpretation: inconclusive. Visual inspection recommended.")
        print("Possible reasons: weak thermal asymmetry, noisy images, grayscale scale inversion, or imperfect centering.")


if __name__ == "__main__":
    main()