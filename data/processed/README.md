This folder contains reproducible artifacts derived from the raw dataset.

- manifest.csv: metadata + relative paths to full/cropped images
- splits.csv: train/test split + CV folds for the training pool
- test_ids.csv: fixed hold-out test IDs
- train_cv_ids.csv: CV fold assignment for the training pool

Raw images are not stored in the repository. Use src/data/download_data.py to fetch them from Zenodo.