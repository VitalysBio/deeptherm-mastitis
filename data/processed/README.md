# Executive Summary — Dataset Splitting Strategy

To ensure rigorous and reproducible evaluation on a relatively small dataset (n = 418 animals), we implemented a two-stage splitting strategy designed to prevent data leakage, maintain class balance, and provide stable performance estimates.

1. Hold-Out Test Set (20%)

A fixed 20% of the dataset is reserved as an untouched test set.

Stratified by binary label (healthy vs SCM)

Never used for model selection or hyperparameter tuning

Used only once for final performance reporting

This guarantees an unbiased estimate of real-world generalization.

2. Stratified 5-Fold Cross-Validation (Within 80%)

The remaining 80% of the data forms the training pool.

Inside this pool:

5-fold Stratified Cross-Validation is applied

Each fold preserves class proportions

Each sample appears in validation exactly once

Performance is reported as:

Mean ± Standard Deviation across folds

This provides robustness and stability in model comparison.

3. Why Stratify Only on Binary Label?

Although the dataset includes ordinal severity classes (SCC class 1–5), exploratory analysis showed:

Healthy animals (label = 0) correspond to SCC classes 1–2

SCM animals (label = 1) correspond to SCC classes 3–5

Severity is therefore nested within the binary label and perfectly correlated with it.

For this reason:

Splitting is stratified only on the binary label

Severity distribution is naturally preserved

No artificial subgroup fragmentation is introduced

4. Data Leakage Prevention

The pipeline guarantees:

No animal appears in both training and test sets

No validation fold shares samples with its corresponding training subset

Data augmentation is applied only to training samples

Validation and test sets remain unaltered

Final Structure

Full Dataset
→ 20% Hold-Out Test (final evaluation)
→ 80% Training Pool
  → 5-Fold Stratified Cross-Validation

Raw images are not stored in the repository. Use src/data/download_data.py to fetch them from Zenodo.