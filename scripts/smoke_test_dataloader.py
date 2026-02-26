from pathlib import Path
from torch.utils.data import DataLoader

from src.models.dataset import DatasetConfig, TIDSMastitisDataset
from src.models.transforms import get_transforms


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "processed" / "splits.csv"

    fold = 0

    train_ds = TIDSMastitisDataset(
        DatasetConfig(project_root, csv_path, image_view="crop", split="train", fold=fold, mode="train"),
        transform=get_transforms("train"),
    )
    val_ds = TIDSMastitisDataset(
        DatasetConfig(project_root, csv_path, image_view="crop", split="train", fold=fold, mode="val"),
        transform=get_transforms("val"),
    )
    test_ds = TIDSMastitisDataset(
        DatasetConfig(project_root, csv_path, image_view="crop", split="test", fold=None, mode="test"),
        transform=get_transforms("test"),
    )

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Test size:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    x, y = next(iter(train_loader))
    print("Batch image tensor:", x.shape)
    print("Batch y_bin:", y["y_bin"][:5])
    print("Batch ids:", y["id"][:5])