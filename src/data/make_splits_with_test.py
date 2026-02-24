from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_PATH = OUT_DIR / "splits.csv"
TEST_IDS_PATH = OUT_DIR / "test_ids.csv"
TRAIN_CV_IDS_PATH = OUT_DIR / "train_cv_ids.csv"

TEST_SIZE = 0.20
N_SPLITS = 5
SEED = 42


def print_distribution(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(df["label"].value_counts().sort_index())
    props = df["label"].value_counts(normalize=True).sort_index()
    print("\nProportions")
    print(props.round(3))


def sanity_check_cv(train_df: pd.DataFrame) -> None:
    ctab = train_df.groupby(["cv_fold", "label"]).size().unstack(fill_value=0)
    print("\nCounts per CV fold and label")
    print(ctab)

    if (ctab == 0).any().any():
        raise RuntimeError(
            "At least one CV fold contains 0 samples for a class. "
            "Consider reducing N_SPLITS."
        )

    props = ctab.div(ctab.sum(axis=1), axis=0)
    print("\nProportions per CV fold")
    print(props.round(3))


if __name__ == "__main__":
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    required = {"id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df = df.copy()
    df["id"] = df["id"].astype(int)
    df["label"] = df["label"].astype(int)

    print_distribution(df, "Full dataset distribution")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df[["id"]], df["label"]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df["split"] = "train"
    test_df["split"] = "test"

    print_distribution(train_df, "Train CV pool distribution")
    print_distribution(test_df, "Test distribution")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    train_df["cv_fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df[["id"]], train_df["label"])):
        train_df.loc[train_df.index[val_idx], "cv_fold"] = fold_idx

    if (train_df["cv_fold"] < 0).any():
        raise RuntimeError("Some training rows did not receive a CV fold assignment")

    sanity_check_cv(train_df)

    test_df["cv_fold"] = -1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits_df = pd.concat([train_df, test_df], axis=0).sort_values("id")
    splits_df.to_csv(SPLITS_PATH, index=False)

    test_df[["id", "label"]].sort_values("id").to_csv(TEST_IDS_PATH, index=False)
    train_df[["id", "label", "cv_fold"]].sort_values(["cv_fold", "id"]).to_csv(TRAIN_CV_IDS_PATH, index=False)

    print("\nSaved files")
    print(f"1. {SPLITS_PATH}")
    print(f"2. {TEST_IDS_PATH}")
    print(f"3. {TRAIN_CV_IDS_PATH}")