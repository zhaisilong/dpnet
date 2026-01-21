import random
from typing import Dict, Optional, List

import pandas as pd
from tqdm.auto import tqdm
from loguru import logger

from molvs import standardize_smiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


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
):
    n = len(df)

    # ---- edge cases ----
    if n == 0:
        return df, df

    if n == 1:
        # no split possible: put sample into train
        return df.reset_index(drop=True), df.iloc[0:0]

    # ---- normal case ----
    stratify = df[stratify_col] if stratify_col else None

    train_df, val_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=stratify,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


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
        scaffold_groups = {
            sc: sc_df.reset_index(drop=True)
            for sc, sc_df in df_nonempty.groupby(scaffold_col)
        }

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

        # ---------- 3. every 10th scaffold (i = 9, 19, 29, ...) → test ----------
        test_scaffolds = set(
            sorted_scaffolds[i + 9] for i in range(0, len(sorted_scaffolds) - 9, 10)
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

        # ---------- 5. empty scaffold → random 10% to test ----------
        empty_train, empty_test = split_train_val(
            df_empty,
            valid_size,
            stratify_col,
            random_state,
        )

        # ---------- 6. final test ----------
        test_df = pd.concat([test_nonempty, empty_test], ignore_index=True)

        # ---------- 7. remaining non-empty → train / valid ----------
        for sc, sc_df in scaffold_groups.items():
            if sc in test_scaffolds:
                continue
            tr, va = split_train_val(
                sc_df,
                valid_size,
                stratify_col,
                rng.randint(0, 2**31 - 1),
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
            )
            train_parts.append(tr)
            val_parts.append(va)

        return {
            "train": pd.concat(train_parts, ignore_index=True),
            "valid": pd.concat(val_parts, ignore_index=True),
            "test": test_df.reset_index(drop=True),
        }

    # =====================================================
    # NON-STRICT MODE
    # =====================================================

    # ---------- 1. group all scaffolds (including empty) ----------
    scaffold_groups = {
        sc: sc_df.reset_index(drop=True) for sc, sc_df in df.groupby(scaffold_col)
    }

    all_scaffolds = list(scaffold_groups.keys())
    rng.shuffle(all_scaffolds)

    # ---------- 2. scaffold-level random 10% → test ----------
    n_test = max(1, int(0.1 * len(all_scaffolds)))
    test_scaffolds = set(all_scaffolds[:n_test])

    test_df = pd.concat(
        [scaffold_groups[sc] for sc in test_scaffolds],
        ignore_index=True,
    )

    remain_df = pd.concat(
        [scaffold_groups[sc] for sc in all_scaffolds if sc not in test_scaffolds],
        ignore_index=True,
    )

    # ---------- 3. remaining → train / valid ----------
    train_df, valid_df = split_train_val(
        remain_df,
        valid_size,
        stratify_col,
        random_state,
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

    # split empty / non-empty
    empty_mask = df[scaffold_col].isna() | (df[scaffold_col] == "")
    df_empty = df[empty_mask].reset_index(drop=True)
    df_nonempty = df[~empty_mask].reset_index(drop=True)

    # group non-empty by scaffold
    scaffold_groups = {
        sc: sc_df.reset_index(drop=True)
        for sc, sc_df in df_nonempty.groupby(scaffold_col)
    }

    scaffolds = list(scaffold_groups.keys())
    rng.shuffle(scaffolds)

    # scaffold-level train / valid split
    n_valid = max(1, int(valid_size * len(scaffolds)))
    valid_scaffolds = set(scaffolds[:n_valid])

    train_df = pd.concat(
        [scaffold_groups[sc] for sc in scaffolds if sc not in valid_scaffolds],
        ignore_index=True,
    )
    valid_df = pd.concat(
        [scaffold_groups[sc] for sc in valid_scaffolds],
        ignore_index=True,
    )

    # empty scaffold → random split
    if len(df_empty) > 0:
        empty_train, empty_valid = split_train_val(
            df_empty,
            valid_size,
            stratify_col,
            random_state,
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
    random_state: int = 42,
) -> List[Dict[str, pd.DataFrame]]:

    rng = random.Random(random_state)

    # split empty / non-empty
    empty_mask = df[scaffold_col].isna() | (df[scaffold_col] == "")
    df_empty = df[empty_mask].reset_index(drop=True)
    df_nonempty = df[~empty_mask].reset_index(drop=True)

    # group non-empty by scaffold
    scaffold_groups = {
        sc: sc_df.reset_index(drop=True)
        for sc, sc_df in df_nonempty.groupby(scaffold_col)
    }

    scaffolds = list(scaffold_groups.keys())
    rng.shuffle(scaffolds)

    folds = [[] for _ in range(n_splits)]
    for i, sc in enumerate(scaffolds):
        folds[i % n_splits].append(sc)

    results = []

    for i in range(n_splits):
        valid_scaffolds = set(folds[i])
        train_scaffolds = set(scaffolds) - valid_scaffolds

        train_df = pd.concat(
            [scaffold_groups[sc] for sc in train_scaffolds],
            ignore_index=True,
        )
        valid_df = pd.concat(
            [scaffold_groups[sc] for sc in valid_scaffolds],
            ignore_index=True,
        )

        # empty scaffold → random split per fold
        if len(df_empty) > 0:
            empty_train, empty_valid = split_train_val(
                df_empty,
                1.0 / n_splits,
                None,
                random_state + i,
            )
            train_df = pd.concat([train_df, empty_train], ignore_index=True)
            valid_df = pd.concat([valid_df, empty_valid], ignore_index=True)

        results.append(
            {
                "train": train_df.reset_index(drop=True),
                "valid": valid_df.reset_index(drop=True),
            }
        )

    return results


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

    stratify = df[stratify_col] if stratify_col else None

    # 1. split out test
    temp_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    stratify_temp = temp_df[stratify_col] if stratify_col else None

    # 2. split train / valid
    train_df, valid_df = train_test_split(
        temp_df,
        test_size=valid_size / (1 - test_size),
        random_state=random_state,
        stratify=stratify_temp,
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

    stratify = df[stratify_col] if stratify_col else None

    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=stratify,
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

    folds = []

    if stratify_col:
        kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        split_iter = kf.split(df, df[stratify_col])
    else:
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        split_iter = kf.split(df)

    for train_idx, val_idx in split_iter:
        folds.append(
            {
                "train": df.iloc[train_idx].reset_index(drop=True),
                "valid": df.iloc[val_idx].reset_index(drop=True),
            }
        )

    return folds
