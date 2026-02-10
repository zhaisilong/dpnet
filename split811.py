#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import pandas as pd
from pandas.errors import EmptyDataError
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# --------------------------
# IO / column utils
# --------------------------
def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None


def detect_smiles_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["smiles", "SMILES", "Smiles", "canonical_smiles", "mol", "smi"]:
        if c in df.columns:
            return c
    if df.shape[1] == 1:
        return df.columns[0]
    return None


# --------------------------
# Scaffold utils
# --------------------------
def compute_scaffold(smiles: str, include_chirality: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Returns:
      (is_valid_smiles, scaffold_str_or_None)
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        if sc is None or (isinstance(sc, str) and sc.strip() == ""):
            return True, None
        return True, sc
    except Exception:
        # SMILES valid but scaffold extraction failed -> treat as no_scaffold
        return True, None


# --------------------------
# Split helpers
# --------------------------
def split_within_indices(idxs: List[int], valid_frac: float, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Random split idxs into train/valid by valid_frac."""
    if not idxs:
        return [], []
    idxs = idxs[:]
    rng.shuffle(idxs)
    n = len(idxs)
    n_valid = int(round(n * valid_frac))
    if n_valid >= n:
        n_valid = n - 1 if n > 1 else n
    valid = idxs[:n_valid]
    train = idxs[n_valid:]
    return train, valid


def choose_test_scaffolds(
    scaffold_sizes: Dict[str, int],
    target_test: int,
    seed: int,
    max_overshoot_ratio: float = 1.2,
) -> List[str]:
    """
    Pick scaffolds so that sum(size) ~ target_test.
    Avoid super large scaffold in test when possible (size > target_test * max_overshoot_ratio).
    """
    rng = random.Random(seed)
    items = list(scaffold_sizes.items())  # (scaffold, size)

    # shuffle ties by size
    size_to_sc = defaultdict(list)
    for sc, sz in items:
        size_to_sc[sz].append(sc)

    sizes_desc = sorted(size_to_sc.keys(), reverse=True)
    ordered = []
    for sz in sizes_desc:
        arr = size_to_sc[sz]
        rng.shuffle(arr)
        for sc in arr:
            ordered.append((sc, sz))

    chosen = []
    cur = 0
    max_allowed = int(target_test * max_overshoot_ratio) if target_test > 0 else 0

    for sc, sz in ordered:
        if target_test > 0 and sz > max_allowed:
            continue
        if cur < target_test:
            chosen.append(sc)
            cur += sz
        else:
            break

    # fallback: if nothing chosen but candidates exist, pick closest-to-target scaffold
    if len(chosen) == 0 and len(ordered) > 0:
        best_sc, best_sz = ordered[0]
        best_dist = abs(best_sz - target_test)
        for sc, sz in ordered:
            dist = abs(sz - target_test)
            if dist < best_dist:
                best_sc, best_sz, best_dist = sc, sz, dist
        chosen = [best_sc]

    return chosen


# --------------------------
# Core per-task split811
# --------------------------
def make_split811_for_task(
    task_dir: str,
    include_chirality: bool,
    seed: int,
    # your updated rules:
    min_test_scaffold_size: int = 6,   # >5
    min_tv_scaffold_size: int = 10,    # >=10
    test_frac: float = 0.10,
    tv_valid_frac: float = 0.10,       # 9:1 => valid=0.1
) -> Optional[Dict]:
    task = os.path.basename(task_dir.rstrip("/"))
    raw_csv = os.path.join(task_dir, "raw", f"{task}.csv")
    if not os.path.isfile(raw_csv):
        return None

    df = safe_read_csv(raw_csv)
    if df is None:
        return None

    smiles_col = detect_smiles_col(df)
    if smiles_col is None:
        return None

    # --- compute scaffold and filter invalid smiles ---
    valid_rows = []
    scaffolds = []
    dropped_invalid = 0

    for i, s in enumerate(df[smiles_col].tolist()):
        ok, sc = compute_scaffold(s, include_chirality=include_chirality)
        if not ok:
            dropped_invalid += 1
            continue
        valid_rows.append(i)
        scaffolds.append(sc)

    df_valid = df.iloc[valid_rows].copy().reset_index(drop=True)
    df_valid["_scaffold"] = scaffolds

    n_total = int(len(df))
    n_valid = int(len(df_valid))

    # group indices by scaffold (None -> no_scaffold pool)
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    no_scaffold_indices: List[int] = []
    for idx, sc in enumerate(df_valid["_scaffold"].tolist()):
        if sc is None:
            no_scaffold_indices.append(idx)
        else:
            scaffold_to_indices[sc].append(idx)

    # split scaffolds by size
    # - eligible_for_test: size >= min_test_scaffold_size
    # - eligible_for_tv (per-scaffold 9:1): size >= min_tv_scaffold_size
    eligible_for_test = {sc: idxs for sc, idxs in scaffold_to_indices.items() if len(idxs) >= min_test_scaffold_size}
    eligible_for_tv = {sc: idxs for sc, idxs in scaffold_to_indices.items() if len(idxs) >= min_tv_scaffold_size}

    # anything scaffold with size < min_tv_scaffold_size will go to MIX (unless selected into TEST)
    # (this includes size 1..9)
    tv_small_scaffolds = {sc: idxs for sc, idxs in scaffold_to_indices.items() if len(idxs) < min_tv_scaffold_size}

    # --- choose test scaffolds from eligible_for_test ---
    target_test = int(round(test_frac * n_valid))
    test_scaffolds = choose_test_scaffolds(
        scaffold_sizes={sc: len(idxs) for sc, idxs in eligible_for_test.items()},
        target_test=target_test,
        seed=seed,
        max_overshoot_ratio=1.2,
    )
    test_scaffolds_set = set(test_scaffolds)

    # build test indices
    test_indices: List[int] = []
    for sc in test_scaffolds_set:
        test_indices.extend(scaffold_to_indices.get(sc, []))

    # remaining scaffolds (excluding test scaffolds)
    remain_scaffolds = {sc: idxs for sc, idxs in scaffold_to_indices.items() if sc not in test_scaffolds_set}

    rng = random.Random(seed)
    train_indices: List[int] = []
    valid_indices: List[int] = []

    # --- train/valid: for scaffolds with size >= min_tv_scaffold_size (and not in test), do per-scaffold 9:1 ---
    for sc, idxs in remain_scaffolds.items():
        if len(idxs) >= min_tv_scaffold_size:
            tr, va = split_within_indices(idxs, valid_frac=tv_valid_frac, rng=rng)
            train_indices.extend(tr)
            valid_indices.extend(va)

    # --- mix: (1) scaffolds with size < min_tv_scaffold_size AND not in test
    #          (2) no_scaffold molecules
    mix_indices: List[int] = []
    for sc, idxs in remain_scaffolds.items():
        if len(idxs) < min_tv_scaffold_size:
            mix_indices.extend(idxs)
    mix_indices.extend(no_scaffold_indices)

    mix_train, mix_valid = split_within_indices(mix_indices, valid_frac=tv_valid_frac, rng=rng)
    train_indices.extend(mix_train)
    valid_indices.extend(mix_valid)

    # build dfs
    df_train = df_valid.iloc[train_indices].drop(columns=["_scaffold"]).reset_index(drop=True)
    df_valid_out = df_valid.iloc[valid_indices].drop(columns=["_scaffold"]).reset_index(drop=True)
    df_test = df_valid.iloc[test_indices].drop(columns=["_scaffold"]).reset_index(drop=True)

    # --- write to split811 (clear existing files first) ---
    out_dir = os.path.join(task_dir, "split811")
    os.makedirs(out_dir, exist_ok=True)

    for fn in os.listdir(out_dir):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_valid_out.to_csv(os.path.join(out_dir, "valid.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # counts & ratios (relative to n_valid = after dropping invalid)
    train_n, valid_n, test_n = len(df_train), len(df_valid_out), len(df_test)

    def ratio(x: int) -> float:
        return (x / n_valid) if n_valid > 0 else 0.0

    summary = {
        "task": task,
        "raw_csv": raw_csv,
        "n_total": n_total,
        "dropped_invalid_smiles": int(dropped_invalid),
        "n_valid": n_valid,
        "train_n": int(train_n),
        "valid_n": int(valid_n),
        "test_n": int(test_n),
        "train_ratio": ratio(train_n),
        "valid_ratio": ratio(valid_n),
        "test_ratio": ratio(test_n),
        "min_test_scaffold_size": int(min_test_scaffold_size),
        "min_tv_scaffold_size": int(min_tv_scaffold_size),
        "n_scaffolds_total": int(len(scaffold_to_indices)),
        "n_test_scaffolds": int(len(test_scaffolds_set)),
        "n_no_scaffold_mols": int(len(no_scaffold_indices)),
        "target_test": int(target_test),
        "n_eligible_test_scaffolds": int(len(eligible_for_test)),
        "n_eligible_tv_scaffolds": int(len({sc: idxs for sc, idxs in eligible_for_tv.items() if sc not in test_scaffolds_set})),
        "n_mix_scaffolds_excluding_test": int(len({sc: idxs for sc, idxs in remain_scaffolds.items() if len(idxs) < min_tv_scaffold_size})),
    }
    return summary


# --------------------------
# Batch runner
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root dir containing task subdirs, e.g. .../database/classification")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_chirality", action="store_true", default=True)

    # your updated thresholds
    ap.add_argument("--min_test_scaffold_size", type=int, default=6, help=">5 => default 6")
    ap.add_argument("--min_tv_scaffold_size", type=int, default=10, help=">=10 => default 10")

    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--tv_valid_frac", type=float, default=0.10)  # 9:1 => valid 0.1
    ap.add_argument("--summary_name", type=str, default="split811_summary.csv")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root not found: {root}")

    subdirs = sorted(
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    print(f"[ROOT] {root}")
    print(f"[INFO] Found {len(subdirs)} tasks")

    rows = []
    for task_dir in subdirs:
        r = make_split811_for_task(
            task_dir=task_dir,
            include_chirality=args.include_chirality,
            seed=args.seed,
            min_test_scaffold_size=args.min_test_scaffold_size,
            min_tv_scaffold_size=args.min_tv_scaffold_size,
            test_frac=args.test_frac,
            tv_valid_frac=args.tv_valid_frac,
        )
        if r is None:
            continue
        rows.append(r)
        print(
            f" {r['task']}: n_valid={r['n_valid']} "
            f"train/valid/test={r['train_n']}/{r['valid_n']}/{r['test_n']} "
            f"ratios={r['train_ratio']:.3f}/{r['valid_ratio']:.3f}/{r['test_ratio']:.3f} "
            f"test_scaffolds={r['n_test_scaffolds']}"
        )

    df_sum = pd.DataFrame(rows)
    out_summary = os.path.join(root, args.summary_name)
    df_sum.to_csv(out_summary, index=False)
    print(f"[DONE] summary saved -> {out_summary}")


if __name__ == "__main__":
    main()