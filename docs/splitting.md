# Splitting Methods

DPNet provides scaffold-aware, random, and perimeter split helpers in
`dpnet.preprocess`. The package CLI uses scaffold splitting by default through
`dpnet process`.

## Preprocessing Pipeline

`dpnet process <task>` performs:

1. Load `database/**/<task>/task_meta.json`.
2. Read `raw/<task>.csv`.
3. Canonicalize SMILES with `molvs.standardize_smiles`.
4. Drop rows with invalid canonicalized SMILES.
5. Deduplicate by SMILES and optional extra deduplication columns.
6. Generate Murcko scaffolds with RDKit.
7. Split into `train`, `valid`, and `test`.
8. Export normalized processed CSV files.

The CLI uses `task_meta.split_method`, or `--split-method` when provided. Valid
methods are `scaffold`, `random`, and `perimeter`; missing metadata defaults to
`scaffold`. For classification tasks, `dpnet process` automatically picks the
first non-regression label from `task_meta.json` as `stratify_col`.

## Label Stratification

All split helpers accept an optional `stratify_col` argument:

```python
splits = scaffold_split_df(df, stratify_col="label", strict=True)
splits = perimeter_split_df(df, stratify_col="label")
folds = random_split_df_5fold(df, stratify_col="label")
```

When `stratify_col` is provided:

- Random holdout helpers use scikit-learn stratified random splitting.
- Random K-fold uses `StratifiedKFold`.
- Scaffold helpers preserve scaffold groups and use `StratifiedGroupKFold`
  where the group count and class counts make that possible.
- Perimeter holdout helpers use soft label quotas: they prioritize perimeter
  candidates from under-filled classes, then fill any remaining rows by
  perimeter rank.
- If stratification is impossible for a local split, DPNet logs a warning and
  falls back to the corresponding ordinary split.
- If `stratify_col` is not present in the dataframe, DPNet raises `ValueError`.

This fallback is intentional for molecular datasets: small scaffold groups,
rare classes, or highly imbalanced labels often cannot satisfy strict
stratified splitting constraints.

## Scaffold Split With Test

Function:

```python
from dpnet.preprocess import scaffold_split_df

splits = scaffold_split_df(
    df,
    scaffold_col="scaffold",
    stratify_col="label",
    valid_size=0.1,
    random_state=42,
    strict=True,
)
```

Returns:

```python
{
    "train": train_df,
    "valid": valid_df,
    "test": test_df,
}
```

### Strict Mode

When `strict=True`:

- Rows are partitioned into empty-scaffold and non-empty-scaffold groups.
- If `stratify_col` is provided and feasible, non-empty scaffold groups are
  split with `StratifiedGroupKFold` to select `test`.
- Otherwise, non-empty scaffold groups are sorted by group size in descending
  order and every 10th scaffold group is assigned to `test`.
- Empty-scaffold rows are randomly split, with `valid_size` used as the test
  fraction for the empty-scaffold subset.
- Remaining non-test scaffold groups are split into train and valid within each
  scaffold group.
- Remaining empty-scaffold rows are also split into train and valid.

This mode is intended to keep the test set scaffold-aware while avoiding a fully
random molecule-level test split.

### Non-Strict Mode

When `strict=False`:

- All rows are grouped by scaffold.
- If `stratify_col` is provided and feasible, scaffold groups are split with
  `StratifiedGroupKFold` to select `test`.
- Otherwise, scaffold groups are shuffled with `random_state` and 10% of
  scaffold groups are assigned to `test`.
- Remaining rows are split into train and valid with `valid_size`.

This mode is scaffold-level for test assignment but less constrained than strict
mode.

## Perimeter Split

Function:

```python
from dpnet.preprocess import perimeter_split_df

splits = perimeter_split_df(
    df,
    smiles_col="smiles",
    valid_size=0.1,
    test_size=0.1,
    stratify_col="label",
    random_state=42,
)
```

Perimeter split is an extrapolation-oriented split for small molecules. DPNet
computes RDKit Morgan fingerprints, computes pairwise Jaccard distance
(equivalent to Tanimoto distance for binary fingerprints), and chooses high-rank
perimeter molecules for the holdout set.

Behavior:

- `test` is selected from the boundary of Morgan fingerprint space.
- `train` and `valid` are split from the remaining rows with the existing
  random split primitive.
- `stratify_col` uses soft label quotas for the perimeter holdout. If exact
  label quotas cannot be satisfied, DPNet logs a warning and fills by perimeter
  rank.
- The default maximum size is 10000 samples because exact pairwise distance
  computation is O(n^2).

`dpnet process` handles oversize perimeter requests differently from the Python
helper. If a processed dataset has more than 10000 rows and perimeter split is
requested, the CLI logs a warning and falls back to scaffold split. The Python
helper raises `ValueError` so library callers can decide their own fallback.

Current project scale note: the largest datasets under `database` are the CYP
inhibitor tasks, with up to 23040 raw rows, so they will fall back to scaffold
when perimeter is requested. `distribution/mrp1` has 9920 raw rows and remains
under the default perimeter limit before preprocessing.

## Perimeter Split Without Test

Function:

```python
from dpnet.preprocess import perimeter_split_df_no_test

splits = perimeter_split_df_no_test(
    df,
    smiles_col="smiles",
    valid_size=0.1,
    stratify_col="label",
    random_state=42,
)
```

Returns `train` and `valid` only. Perimeter split does not provide a 5-fold
helper because boundary-selection methods are not naturally fold-based. For
robust benchmark estimates, run repeated holdout with multiple seeds and report
mean and standard deviation.

## Scaffold Split Without Test

Function:

```python
from dpnet.preprocess import scaffold_split_df_no_test

splits = scaffold_split_df_no_test(
    df,
    stratify_col="label",
    valid_size=0.1,
    random_state=42,
)
```

Returns `train` and `valid` only. Non-empty scaffolds are shuffled, a scaffold
level validation set is selected, and empty-scaffold rows are split randomly.
If `stratify_col` is provided and feasible, the non-empty scaffold validation
selection uses group-aware stratification.

## Scaffold 5-Fold Split

Function:

```python
from dpnet.preprocess import scaffold_split_df_5fold

folds = scaffold_split_df_5fold(
    df,
    n_splits=5,
    stratify_col="label",
    random_state=42,
)
```

Returns a list of fold dictionaries. Non-empty scaffolds are shuffled and
assigned round-robin to validation folds. Empty-scaffold rows are randomly split
per fold. If `stratify_col` is provided and feasible, non-empty scaffold folds
use group-aware stratification.

## Random Splits

DPNet also includes random split helpers when scaffold-aware splitting is not
needed.

Train, valid, and test:

```python
from dpnet.preprocess import random_split_df

splits = random_split_df(
    df,
    valid_size=0.1,
    test_size=0.1,
    stratify_col="label",
    random_state=42,
)
```

Train and valid only:

```python
from dpnet.preprocess import random_split_df_no_test

splits = random_split_df_no_test(
    df,
    valid_size=0.1,
    stratify_col="label",
    random_state=42,
)
```

K-fold:

```python
from dpnet.preprocess import random_split_df_5fold

folds = random_split_df_5fold(df, n_splits=5, stratify_col="label", random_state=42)
```

If `stratify_col` is provided, random split helpers use stratification where the
underlying scikit-learn splitter supports it and fall back to ordinary random
splitting otherwise.
