from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "TIDS Dataset"

FULL_DIR = RAW_DIR / "TIDS_full_images"
CROP_DIR = RAW_DIR / "TIDS_cropped"

META_PATH = RAW_DIR / "ID_Labels.csv"

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"


def find_image(path_root: Path, label_folder: str, image_id: int) -> Path | None:
    p = path_root / label_folder / f"{image_id}.jpg"
    if p.exists():
        return p
    p = path_root / label_folder / f"{image_id}.png"
    if p.exists():
        return p
    return None


def resolve_paths_for_id(image_id: int, label: int):
    label_folder = "SCM" if int(label) == 1 else "healthy"

    full_path = find_image(FULL_DIR, label_folder, image_id)
    crop_path = find_image(CROP_DIR, label_folder, image_id)

    return full_path, crop_path


if __name__ == "__main__":
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    df = pd.read_csv(META_PATH)

    required_cols = {"ID", "L SCC", "R SCC", "L SCC class", "R SCC class", "Date", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")

    rows = []
    missing_full = 0
    missing_crop = 0

    for _, r in df.iterrows():
        image_id = int(r["ID"])
        label = int(r["label"])

        full_path, crop_path = resolve_paths_for_id(image_id, label)

        if full_path is None:
            missing_full += 1
        if crop_path is None:
            missing_crop += 1

        rows.append(
            {
                "id": image_id,
                "label": label,
                "date": str(r["Date"]),
                "l_scc": int(r["L SCC"]) if pd.notna(r["L SCC"]) else None,
                "r_scc": int(r["R SCC"]) if pd.notna(r["R SCC"]) else None,
                "l_scc_class": int(r["L SCC class"]) if pd.notna(r["L SCC class"]) else None,
                "r_scc_class": int(r["R SCC class"]) if pd.notna(r["R SCC class"]) else None,
                "full_path": str(full_path.relative_to(PROJECT_ROOT)) if full_path else None,
                "crop_path": str(crop_path.relative_to(PROJECT_ROOT)) if crop_path else None,
            }
        )

    out_df = pd.DataFrame(rows)

    out_dir = OUT_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print("Manifest created")
    print(f"Rows: {out_df.shape[0]}")
    print(f"Missing full images: {missing_full}")
    print(f"Missing cropped images: {missing_crop}")
    print(f"Saved to: {OUT_PATH}")