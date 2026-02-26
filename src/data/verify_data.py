from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
META_EXT = {".csv", ".xlsx", ".xls", ".json", ".tsv"}


def detect_label_from_path(p: Path):
    parts = [x.lower() for x in p.parts]
    if "healthy" in parts:
        return "healthy"
    if "scm" in parts or "mastitis" in parts:
        return "scm"
    return "unknown"


def detect_view_from_path(p: Path):
    s = str(p).lower()
    if "full" in s:
        return "full"
    if "crop" in s:
        return "cropped"
    return "unknown"


if __name__ == "__main__":
    print(f"Looking into: {DATA_DIR}")

    all_files = [p for p in DATA_DIR.rglob("*") if p.is_file()]
    print(f"Total files found: {len(all_files)}")

    image_files = [p for p in all_files if p.suffix.lower() in IMG_EXT]
    meta_files = [p for p in all_files if p.suffix.lower() in META_EXT]

    print(f"Total image files: {len(image_files)}")
    print(f"Metadata-like files: {len(meta_files)}")

    if meta_files:
        print("\nMetadata files detected:")
        for p in meta_files[:30]:
            print(" -", p.relative_to(DATA_DIR))
        if len(meta_files) > 30:
            print(f" ... ({len(meta_files)-30} more)")

    labels = Counter()
    views = Counter()
    for p in image_files:
        labels[detect_label_from_path(p)] += 1
        views[detect_view_from_path(p)] += 1

    print("\nImage counts by label:")
    for k, v in labels.items():
        print(f" - {k}: {v}")

    print("\nImage counts by view type:")
    for k, v in views.items():
        print(f" - {k}: {v}")

    print("\nSample images:")
    for p in image_files[:10]:
        print(" -", p.relative_to(DATA_DIR))