import click
import pandas as pd
from loguru import logger
from pathlib import Path
from .utils import find_task_root
from .preprocess import (
    canonicalize_df,
    deduplicate_df,
    generate_scaffold_df,
    scaffold_split_df,
    report_df,
)
from .parser import TaskMeta


@click.group()
def cli():
    pass


@cli.command()
@click.argument("task", type=str, default="bbbp")
@click.argument("meta_name", type=str, default="task_meta")
@click.option("--skip_check", is_flag=True, default=False)
@click.option("--root_dir", type=str)
def process(task: str, meta_name: str, skip_check: bool, root_dir: str):
    if root_dir:
        task_root = Path(root_dir)
    else:
        task_root = find_task_root(task)
    task_meta = TaskMeta.load(task_root / f"{meta_name}.json")
    click.echo(f"Task: {task}\nRoot: {task_root}\nMeta: {task_meta}")

    df = pd.read_csv(task_root / "raw" / f"{task}.csv")
    df = canonicalize_df(df, task_meta.smiles_col)

    df, duplicates = deduplicate_df(df, [task_meta.smiles_col])
    duplicates.to_csv(task_root / "duplicates.csv", index=False)

    if not task_meta.id_col:
        df = df.reset_index(drop=True)
        df["cid"] = df.index.map(lambda x: f"cid_{x}")
        id_col = "cid"
    else:
        id_col = task_meta.id_col

    # check if the labels are valid
    if not skip_check:
        for label in task_meta.iter_labels():
            if label.problem_type != "regression":
                if df[label.label_col].nunique() != label.num_classes:
                    raise ValueError(
                        f"Label {label.id} has {df[label.label_col].nunique()} values, but the meta specifies that it should have {label.num_classes}"
                    )

    df = generate_scaffold_df(df, smiles_col=task_meta.smiles_col)
    logger.info(f"strict test: {task_meta.strict_test}")
    df_dict = scaffold_split_df(
        df, strict=task_meta.strict_test, random_state=task_meta.seed
    )

    for split_name, df in df_dict.items():
        report = report_df(df, smiles_col=task_meta.smiles_col, scaffold_col="scaffold")
        logger.info(f"Report {split_name}:\n{report}")

    processed_dir = task_root / task_meta.processed_dir / task_meta.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    # here we rename
    select_cols = [id_col, "smiles"]

    n_labels = 0
    for label in task_meta.iter_labels():
        for split_name, df in df_dict.items():
            df.rename(
                columns={task_meta.smiles_col: "smiles", label.label_col: label.id},
                inplace=True,
            )
        select_cols.append(label.id)
        n_labels += 1

    logger.info(f"Selected {n_labels} labels")

    if task_meta.extra_cols:
        select_cols.extend(task_meta.extra_cols)

    for split_name, df in df_dict.items():
        df = df[select_cols]
        df.to_csv(processed_dir / f"{split_name}.csv", index=False)

    task_meta.save(processed_dir / f"{task_meta.name}.json")
