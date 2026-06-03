import random
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from loguru import logger

from molvs import standardize_smiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import pdist
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)

from .features import MorganFeaturizer


PERIMETER_MAX_SAMPLES = 10000


def report_df(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    scaffold_col: str = "scaffold",
) -> dict:
    # Nulls
    null_smiles = int(df[smiles_col].isna().sum()) if smiles_col in df.columns else None
    null_scaffold = (
        int(df[scaffold_col].isna().sum()) if scaffold_col in df.columns else None
    )
    null_total = int(df.isna().sum().sum())

    if null_smiles and null_smiles > 0:
        logger.warning(f"Null values found in '{smiles_col}': {null_smiles}")
    if null_scaffold and null_scaffold > 0:
        logger.warning(f"Null values found in '{scaffold_col}': {null_scaffold}")

    # Duplicates (by smiles)
    dup_smiles = (
        int(df.duplicated(subset=[smiles_col]).sum())
        if smiles_col in df.columns
        else None
    )
    if dup_smiles and dup_smiles > 0:
        logger.warning(f"Duplicate '{smiles_col}' found: {dup_smiles}")

    # Empty scaffold = NaN or ""
    if scaffold_col in df.columns:
        empty_scaffold_mask = df[scaffold_col].isna() | (df[scaffold_col] == "")
        num_empty_scaffolds = int(empty_scaffold_mask.sum())
        if num_empty_scaffolds > 0:
            logger.info(
                f"Empty scaffolds found in '{scaffold_col}': {num_empty_scaffolds}"
            )
    else:
        num_empty_scaffolds = None

    return {
        "num_samples": int(len(df)),
        "num_unique_smiles": (
            int(df[smiles_col].nunique()) if smiles_col in df.columns else None
        ),
        "num_unique_scaffolds": (
            int(df[scaffold_col].nunique()) if scaffold_col in df.columns else None
        ),
        "num_empty_scaffolds": num_empty_scaffolds,
        "num_duplicates_smiles": dup_smiles,
        "num_null_smiles": null_smiles,
        "num_null_scaffold": null_scaffold,
        "num_null_total": null_total,
    }


# =====================================================
# Scaffold & SMILES utilities
# =====================================================


def generate_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception as e:
        logger.error(f"Error generating scaffold for {smiles}: {e}")
        return None


def generate_scaffold_df(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold",
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    logger.info(f"Generating scaffolds for {len(df)} samples")
    tqdm.pandas(desc="Generating scaffolds")
    df[scaffold_col] = df[smiles_col].progress_apply(generate_scaffold)
    df = df[df[scaffold_col].notna()].reset_index(drop=True)

    logger.info(f"Unique scaffolds: {df[scaffold_col].nunique()}")
    logger.info(f"Empty scaffolds: {(df[scaffold_col] == '').sum()}")
    return df


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        return standardize_smiles(smiles)
    except Exception as e:
        logger.error(f"Error canonicalizing smiles {smiles}: {e}")
        return None


def canonicalize_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    tqdm.pandas(desc=f"Canonicalizing {column}")
    df[column] = df[column].progress_apply(canonicalize_smiles)
    return df.dropna(subset=[column]).reset_index(drop=True)


def deduplicate_df(df: pd.DataFrame, columns: List[str]):
    before = len(df)
    duplicates = df[df.duplicated(subset=columns)]
    df = df.drop_duplicates(subset=columns).reset_index(drop=True)
    logger.info(f"Deduplicated: {before} → {len(df)}")
    return df, duplicates


# =====================================================
# Core split primitives
# =====================================================


def split_train_val(
    df: pd.DataFrame,
    valid_size: float,
    stratify_col: Optional[str],
    random_state: int,
    context: str = "train/valid split",
):
    n = len(df)

    # ---- edge cases ----
    if n == 0:
        return df, df

    if n == 1:
        # no split possible: put sample into train
        return df.reset_index(drop=True), df.iloc[0:0]

    # ---- normal case ----
    _validate_stratify_col(df, stratify_col)
    stratify = df[stratify_col] if stratify_col else None

    try:
        train_df, val_df = train_test_split(
            df,
            test_size=valid_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as e:
        if stratify_col is None:
            raise
        _warn_stratify_fallback(context, stratify_col, e)
        train_df, val_df = train_test_split(
            df,
            test_size=valid_size,
            random_state=random_state,
            stratify=None,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _validate_stratify_col(df: pd.DataFrame, stratify_col: Optional[str]) -> None:
    if stratify_col is not None and stratify_col not in df.columns:
        raise ValueError(
            f"stratify_col '{stratify_col}' not found in columns: {list(df.columns)}"
        )


def _warn_stratify_fallback(context: str, stratify_col: str, error: Exception) -> None:
    logger.warning(
        f"Stratified split fallback for {context} using '{stratify_col}': {error}"
    )


def _n_splits_from_fraction(fraction: float) -> int:
    if fraction <= 0 or fraction >= 1:
        return 2
    return max(2, int(round(1.0 / fraction)))


def _group_values(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col not in df.columns:
        raise ValueError(
            f"group_col '{group_col}' not found in columns: {list(df.columns)}"
        )
    return df[group_col].fillna("").astype(str)


def _groupby_dict(df: pd.DataFrame, group_col: str) -> Dict[str, pd.DataFrame]:
    groups = _group_values(df, group_col)
    return {
        group: df.loc[indexes].reset_index(drop=True)
        for group, indexes in groups.groupby(groups, sort=False).groups.items()
    }


def _concat_or_empty(
    parts: List[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if parts:
        return pd.concat(parts, ignore_index=True)
    return template.iloc[0:0].reset_index(drop=True)


def _validate_perimeter_size(df: pd.DataFrame, max_samples: int) -> None:
    if len(df) > max_samples:
        raise ValueError(
            "Perimeter split requires pairwise molecular distances and is limited "
            f"to {max_samples} samples; got {len(df)} samples"
        )


def _holdout_count(n_samples: int, holdout_size: float) -> int:
    if n_samples <= 1:
        return 0
    if holdout_size <= 0:
        return 0
    if holdout_size >= 1:
        return n_samples - 1
    return min(n_samples - 1, max(1, int(round(n_samples * holdout_size))))


def _condensed_pair(condensed_index: int, n_samples: int) -> tuple[int, int]:
    i = int(
        n_samples
        - 2
        - np.floor(
            np.sqrt(-8 * condensed_index + 4 * n_samples * (n_samples - 1) - 7) / 2
            - 0.5
        )
    )
    j = int(
        condensed_index
        + i
        + 1
        - n_samples * (n_samples - 1) // 2
        + (n_samples - i) * ((n_samples - i) - 1) // 2
    )
    return i, j


def _perimeter_ranked_indices(fingerprints: np.ndarray) -> List[int]:
    n_samples = len(fingerprints)
    if n_samples == 0:
        return []
    if n_samples == 1:
        return [0]

    distances = pdist(fingerprints.astype(bool), metric="jaccard")
    ranked_pairs = np.argsort(distances)[::-1]
    selected = set()
    ranked = []

    for condensed_index in ranked_pairs:
        i, j = _condensed_pair(int(condensed_index), n_samples)
        if i in selected or j in selected:
            continue
        selected.add(i)
        selected.add(j)
        ranked.extend([i, j])
        if len(ranked) >= n_samples:
            break

    ranked.extend(i for i in range(n_samples) if i not in selected)
    return ranked


def _stratified_holdout_quotas(
    labels: pd.Series,
    holdout_count: int,
) -> dict[object, int]:
    counts = labels.value_counts(dropna=False)
    if holdout_count <= 0 or counts.empty:
        return {}

    raw = counts / len(labels) * holdout_count
    quotas = raw.astype(int).to_dict()
    remaining = holdout_count - sum(quotas.values())

    for label in (raw - np.floor(raw)).sort_values(ascending=False).index:
        if remaining <= 0:
            break
        quotas[label] += 1
        remaining -= 1

    return quotas


def _select_perimeter_holdout_indices(
    df: pd.DataFrame,
    ranked_indices: List[int],
    holdout_count: int,
    stratify_col: Optional[str],
    context: str,
) -> List[int]:
    if holdout_count <= 0:
        return []
    if stratify_col is None:
        return ranked_indices[:holdout_count]

    _validate_stratify_col(df, stratify_col)
    labels = df[stratify_col].reset_index(drop=True)
    quotas = _stratified_holdout_quotas(labels, holdout_count)
    selected = []
    selected_set = set()
    selected_counts = {label: 0 for label in quotas}

    for row_idx in ranked_indices:
        label = labels.iloc[row_idx]
        if selected_counts.get(label, 0) >= quotas.get(label, 0):
            continue
        selected.append(row_idx)
        selected_set.add(row_idx)
        selected_counts[label] = selected_counts.get(label, 0) + 1
        if len(selected) >= holdout_count:
            break

    if len(selected) < holdout_count:
        logger.warning(
            f"Soft stratified perimeter split could not satisfy all quotas for "
            f"{context} using '{stratify_col}'; filling remaining samples by "
            "perimeter rank"
        )
        for row_idx in ranked_indices:
            if row_idx in selected_set:
                continue
            selected.append(row_idx)
            selected_set.add(row_idx)
            if len(selected) >= holdout_count:
                break

    return selected[:holdout_count]


def _perimeter_holdout(
    df: pd.DataFrame,
    smiles_col: str,
    holdout_size: float,
    stratify_col: Optional[str],
    max_samples: int,
    context: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if smiles_col not in df.columns:
        raise ValueError(f"smiles_col '{smiles_col}' not found in columns: {list(df.columns)}")
    _validate_stratify_col(df, stratify_col)
    _validate_perimeter_size(df, max_samples)

    n_samples = len(df)
    holdout_count = _holdout_count(n_samples, holdout_size)
    if holdout_count == 0:
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)

    featurizer = MorganFeaturizer()
    fingerprints = featurizer.transform(df[smiles_col].tolist())
    ranked_indices = _perimeter_ranked_indices(fingerprints)
    holdout_indices = _select_perimeter_holdout_indices(
        df,
        ranked_indices,
        holdout_count,
        stratify_col,
        context,
    )
    holdout_set = set(holdout_indices)
    train_indices = [i for i in range(n_samples) if i not in holdout_set]

    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[holdout_indices].reset_index(drop=True),
    )


def _stratified_group_holdout(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: Optional[str],
    n_splits: int,
    random_state: int,
    context: str,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    if stratify_col is None or len(df) == 0:
        return None

    _validate_stratify_col(df, stratify_col)
    groups = _group_values(df, group_col)

    if groups.nunique(dropna=False) < n_splits:
        return None

    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        train_idx, holdout_idx = next(
            splitter.split(df, df[stratify_col], groups=groups)
        )
    except (TypeError, ValueError) as e:
        _warn_stratify_fallback(context, stratify_col, e)
        return None

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[holdout_idx].reset_index(drop=True),
    )


def _stratified_group_folds(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: Optional[str],
    n_splits: int,
    random_state: int,
    context: str,
) -> Optional[List[Dict[str, pd.DataFrame]]]:
    if stratify_col is None or len(df) == 0:
        return None

    _validate_stratify_col(df, stratify_col)
    groups = _group_values(df, group_col)

    if groups.nunique(dropna=False) < n_splits:
        return None

    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return [
            {
                "train": df.iloc[train_idx].reset_index(drop=True),
                "valid": df.iloc[valid_idx].reset_index(drop=True),
            }
            for train_idx, valid_idx in splitter.split(
                df, df[stratify_col], groups=groups
            )
        ]
    except (TypeError, ValueError) as e:
        _warn_stratify_fallback(context, stratify_col, e)
        return None


def partition_by_scaffold(
    df: pd.DataFrame,
    scaffold_col: str,
    min_count: int,
):
    """
    Returns:
        empty_df
        large_scaffold_dfs: dict[scaffold, df]
        small_scaffold_df
    """
    empty_mask = df[scaffold_col].isna() | (df[scaffold_col] == "")
    df_empty = df[empty_mask].reset_index(drop=True)
    df_nonempty = df[~empty_mask].reset_index(drop=True)

    counts = df_nonempty[scaffold_col].value_counts()

    large_scaffolds = counts[counts >= min_count].index.tolist()
    small_scaffolds = counts[counts < min_count].index.tolist()

    large_groups = {
        sc: df_nonempty[df_nonempty[scaffold_col] == sc].reset_index(drop=True)
        for sc in large_scaffolds
    }

    df_small = df_nonempty[df_nonempty[scaffold_col].isin(small_scaffolds)].reset_index(
        drop=True
    )

    return df_empty, large_groups, df_small


# =====================================================
# 1. Scaffold split with TEST
# =====================================================


def scaffold_split_df(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold",
    stratify_col: Optional[str] = None,
    valid_size: float = 0.1,
    random_state: int = 42,
    strict: bool = False,
) -> Dict[str, pd.DataFrame]:

    rng = random.Random(random_state)
    _validate_stratify_col(df, stratify_col)

    # =====================================================
    # split empty / non-empty
    # =====================================================
    empty_mask = df[scaffold_col].isna() | (df[scaffold_col] == "")
    df_empty = df[empty_mask].reset_index(drop=True)
    df_nonempty = df[~empty_mask].reset_index(drop=True)

    train_parts, val_parts = [], []

    # =====================================================
    # STRICT MODE
    # =====================================================
    if strict:
        # ---------- 1. group non-empty by scaffold ----------
        grouped_holdout = _stratified_group_holdout(
            df_nonempty,
            scaffold_col,
            stratify_col,
            _n_splits_from_fraction(valid_size),
            random_state,
            "strict scaffold test split",
        )

        if grouped_holdout is not None:
            remain_nonempty, test_nonempty = grouped_holdout
            scaffold_groups = _groupby_dict(remain_nonempty, scaffold_col)
        else:
            scaffold_groups = _groupby_dict(df_nonempty, scaffold_col)

            # ---------- 2. sort scaffolds by size (descending) ----------
            scaffold_sizes = {sc: len(sc_df) for sc, sc_df in scaffold_groups.items()}
            sorted_scaffolds = [
                sc
                for sc, _ in sorted(
                    scaffold_sizes.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]

            # ---------- 3. every 10th scaffold (i = 9, 19, 29, ...) -> test ----------
            test_scaffolds = set(
                sorted_scaffolds[i + 9]
                for i in range(0, len(sorted_scaffolds) - 9, 10)
            )

            # ---------- 4. build test (non-empty) ----------
            test_nonempty = (
                pd.concat(
                    [scaffold_groups[sc] for sc in test_scaffolds],
                    ignore_index=True,
                )
                if test_scaffolds
                else df.iloc[0:0]
            )

            scaffold_groups = {
                sc: sc_df
                for sc, sc_df in scaffold_groups.items()
                if sc not in test_scaffolds
            }

        # ---------- 5. empty scaffold → random 10% to test ----------
        empty_train, empty_test = split_train_val(
            df_empty,
            valid_size,
            stratify_col,
            random_state,
            "strict empty-scaffold test split",
        )

        # ---------- 6. final test ----------
        test_df = pd.concat([test_nonempty, empty_test], ignore_index=True)

        # ---------- 7. remaining non-empty → train / valid ----------
        for sc, sc_df in scaffold_groups.items():
            tr, va = split_train_val(
                sc_df,
                valid_size,
                stratify_col,
                rng.randint(0, 2**31 - 1),
                f"strict scaffold train/valid split for scaffold {sc}",
            )
            train_parts.append(tr)
            val_parts.append(va)

        # ---------- 8. remaining empty → train / valid ----------
        if len(empty_train) > 0:
            tr, va = split_train_val(
                empty_train,
                valid_size,
                stratify_col,
                random_state,
                "strict empty-scaffold train/valid split",
            )
            train_parts.append(tr)
            val_parts.append(va)

        return {
            "train": _concat_or_empty(train_parts, df),
            "valid": _concat_or_empty(val_parts, df),
            "test": test_df.reset_index(drop=True),
        }

    # =====================================================
    # NON-STRICT MODE
    # =====================================================

    grouped_holdout = _stratified_group_holdout(
        df,
        scaffold_col,
        stratify_col,
        _n_splits_from_fraction(0.1),
        random_state,
        "non-strict scaffold test split",
    )

    if grouped_holdout is not None:
        remain_df, test_df = grouped_holdout
    else:
        # ---------- 1. group all scaffolds (including empty) ----------
        scaffold_groups = _groupby_dict(df, scaffold_col)

        all_scaffolds = list(scaffold_groups.keys())
        rng.shuffle(all_scaffolds)

        # ---------- 2. scaffold-level random 10% -> test ----------
        n_test = max(1, int(0.1 * len(all_scaffolds)))
        test_scaffolds = set(all_scaffolds[:n_test])

        test_df = _concat_or_empty(
            [scaffold_groups[sc] for sc in test_scaffolds],
            df,
        )

        remain_df = _concat_or_empty(
            [scaffold_groups[sc] for sc in all_scaffolds if sc not in test_scaffolds],
            df,
        )

    # ---------- 3. remaining → train / valid ----------
    train_df, valid_df = split_train_val(
        remain_df,
        valid_size,
        stratify_col,
        random_state,
        "non-strict scaffold train/valid split",
    )

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


# =====================================================
# 2. Scaffold split NO TEST
# =====================================================


def scaffold_split_df_no_test(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold",
    stratify_col: Optional[str] = None,
    valid_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:

    rng = random.Random(random_state)
    _validate_stratify_col(df, stratify_col)

    # split empty / non-empty
    empty_mask = df[scaffold_col].isna() | (df[scaffold_col] == "")
    df_empty = df[empty_mask].reset_index(drop=True)
    df_nonempty = df[~empty_mask].reset_index(drop=True)

    grouped_holdout = _stratified_group_holdout(
        df_nonempty,
        scaffold_col,
        stratify_col,
        _n_splits_from_fraction(valid_size),
        random_state,
        "scaffold no-test validation split",
    )

    if grouped_holdout is not None:
        train_df, valid_df = grouped_holdout
    else:
        # group non-empty by scaffold
        scaffold_groups = _groupby_dict(df_nonempty, scaffold_col)

        scaffolds = list(scaffold_groups.keys())
        rng.shuffle(scaffolds)

        # scaffold-level train / valid split
        n_valid = max(1, int(valid_size * len(scaffolds)))
        valid_scaffolds = set(scaffolds[:n_valid])

        train_df = _concat_or_empty(
            [scaffold_groups[sc] for sc in scaffolds if sc not in valid_scaffolds],
            df,
        )
        valid_df = _concat_or_empty(
            [scaffold_groups[sc] for sc in valid_scaffolds],
            df,
        )

    # empty scaffold → random split
    if len(df_empty) > 0:
        empty_train, empty_valid = split_train_val(
            df_empty,
            valid_size,
            stratify_col,
            random_state,
            "scaffold no-test empty-scaffold split",
        )
        train_df = pd.concat([train_df, empty_train], ignore_index=True)
        valid_df = pd.concat([valid_df, empty_valid], ignore_index=True)

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
    }


# =====================================================
# 3. Scaffold 5-fold CV
# =====================================================


def scaffold_split_df_5fold(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold",
    n_splits: int = 5,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
) -> List[Dict[str, pd.DataFrame]]:

    rng = random.Random(random_state)
    _validate_stratify_col(df, stratify_col)

    # split empty / non-empty
    empty_mask = df[scaffold_col].isna() | (df[scaffold_col] == "")
    df_empty = df[empty_mask].reset_index(drop=True)
    df_nonempty = df[~empty_mask].reset_index(drop=True)

    stratified_folds = _stratified_group_folds(
        df_nonempty,
        scaffold_col,
        stratify_col,
        n_splits,
        random_state,
        "scaffold 5-fold split",
    )

    if stratified_folds is not None:
        results = stratified_folds
    else:
        # group non-empty by scaffold
        scaffold_groups = _groupby_dict(df_nonempty, scaffold_col)

        scaffolds = list(scaffold_groups.keys())
        rng.shuffle(scaffolds)

        folds = [[] for _ in range(n_splits)]
        for i, sc in enumerate(scaffolds):
            folds[i % n_splits].append(sc)

        results = []

        for i in range(n_splits):
            valid_scaffolds = set(folds[i])
            train_scaffolds = set(scaffolds) - valid_scaffolds

            train_df = _concat_or_empty(
                [scaffold_groups[sc] for sc in train_scaffolds],
                df,
            )
            valid_df = _concat_or_empty(
                [scaffold_groups[sc] for sc in valid_scaffolds],
                df,
            )

            results.append(
                {
                    "train": train_df.reset_index(drop=True),
                    "valid": valid_df.reset_index(drop=True),
                }
            )

    for i, fold in enumerate(results):
        # empty scaffold → random split per fold
        if len(df_empty) > 0:
            empty_train, empty_valid = split_train_val(
                df_empty,
                1.0 / n_splits,
                stratify_col,
                random_state + i,
                f"scaffold 5-fold empty-scaffold split for fold {i}",
            )
            fold["train"] = pd.concat(
                [fold["train"], empty_train], ignore_index=True
            ).reset_index(drop=True)
            fold["valid"] = pd.concat(
                [fold["valid"], empty_valid], ignore_index=True
            ).reset_index(drop=True)

    return results


# =====================================================
# 4. Perimeter split
# =====================================================


def perimeter_split_df(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    valid_size: float = 0.1,
    test_size: float = 0.1,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
    max_samples: int = PERIMETER_MAX_SAMPLES,
) -> Dict[str, pd.DataFrame]:
    """
    Perimeter split: train / valid / test.

    The test set is selected from molecules on the outskirts of the Morgan
    fingerprint space. Train/valid is then split from the remaining rows using
    the existing random split primitive, including stratified fallback.
    """

    remain_df, test_df = _perimeter_holdout(
        df,
        smiles_col,
        test_size,
        stratify_col,
        max_samples,
        "perimeter test split",
    )

    train_df, valid_df = split_train_val(
        remain_df,
        valid_size / (1 - test_size),
        stratify_col,
        random_state,
        "perimeter train/valid split",
    )

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def perimeter_split_df_no_test(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    valid_size: float = 0.1,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
    max_samples: int = PERIMETER_MAX_SAMPLES,
) -> Dict[str, pd.DataFrame]:
    """
    Perimeter split: train / valid only.
    """

    train_df, valid_df = _perimeter_holdout(
        df,
        smiles_col,
        valid_size,
        stratify_col,
        max_samples,
        "perimeter validation split",
    )

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
    }


def random_split_df(
    df: pd.DataFrame,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Fully random split: train / valid / test

    - No scaffold awareness
    - Optional stratification
    """

    _validate_stratify_col(df, stratify_col)

    # 1. split out test
    temp_df, test_df = split_train_val(
        df,
        test_size,
        stratify_col,
        random_state,
        "random test split",
    )

    # 2. split train / valid
    train_df, valid_df = split_train_val(
        temp_df,
        valid_size / (1 - test_size),
        stratify_col,
        random_state,
        "random train/valid split",
    )

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def random_split_df_no_test(
    df: pd.DataFrame,
    valid_size: float = 0.1,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Fully random split: train / valid only
    """

    _validate_stratify_col(df, stratify_col)
    train_df, valid_df = split_train_val(
        df,
        valid_size,
        stratify_col,
        random_state,
        "random no-test train/valid split",
    )

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
    }


def random_split_df_5fold(
    df: pd.DataFrame,
    n_splits: int = 5,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
) -> list[Dict[str, pd.DataFrame]]:
    """
    Fully random K-fold split

    - No scaffold awareness
    - Optional stratification (via StratifiedKFold)
    """

    _validate_stratify_col(df, stratify_col)
    folds = []
    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    split_iter = splitter.split(df)

    if stratify_col:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        try:
            split_iter = list(splitter.split(df, df[stratify_col]))
        except ValueError as e:
            _warn_stratify_fallback(
                "random 5-fold split",
                stratify_col,
                e,
            )
            split_iter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            ).split(df)

    for train_idx, val_idx in split_iter:
        folds.append(
            {
                "train": df.iloc[train_idx].reset_index(drop=True),
                "valid": df.iloc[val_idx].reset_index(drop=True),
            }
        )

    return folds
