import click
from loguru import logger
from pathlib import Path
from .runner import BaselineRunConfig, run_baseline


@click.group()
def cli():
    pass


@cli.command()
@click.argument("task", type=str, default="bbbp")
@click.argument("meta_name", type=str, default="task_meta")
@click.option("--skip_check", is_flag=True, default=False)
@click.option("--root-dir", "--root_dir", type=str)
@click.option("--extra_deduplicate_cols", type=str, default=None)
@click.option(
    "--split-method",
    type=click.Choice(["scaffold", "random", "perimeter"]),
    default=None,
)
def process(
    task: str,
    meta_name: str,
    skip_check: bool,
    root_dir: str,
    extra_deduplicate_cols: str,
    split_method: str | None,
):
    import pandas as pd

    from .parser import TaskMeta
    from .preprocess import (
        canonicalize_df,
        deduplicate_df,
        generate_scaffold_df,
        PERIMETER_MAX_SAMPLES,
        perimeter_split_df,
        random_split_df,
        report_df,
        scaffold_split_df,
    )
    from .utils import find_task_root

    if root_dir:
        task_root = Path(root_dir)
    else:
        task_root = find_task_root(task)
    task_meta = TaskMeta.load(task_root / f"{meta_name}.json")
    click.echo(f"Task: {task}\nRoot: {task_root}\nMeta: {task_meta}")

    df = pd.read_csv(task_root / "raw" / f"{task}.csv")
    df = canonicalize_df(df, task_meta.smiles_col)

    deduplicate_cols = [task_meta.smiles_col]
    if isinstance(extra_deduplicate_cols, str):
        deduplicate_cols.extend(extra_deduplicate_cols.split(","))
    df, duplicates = deduplicate_df(df, deduplicate_cols)
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

    stratify_col = None
    for label in task_meta.iter_labels():
        if label.problem_type != "regression":
            stratify_col = label.label_col
            break

    if stratify_col:
        logger.info(f"Using stratify column for split: {stratify_col}")
    else:
        logger.info("No classification label found; split stratification disabled")

    df = generate_scaffold_df(df, smiles_col=task_meta.smiles_col)
    selected_split_method = split_method or task_meta.split_method or "scaffold"
    split_config = task_meta.split_config or {}
    logger.info(f"split method: {selected_split_method}")
    logger.info(f"strict test: {task_meta.strict_test}")

    if selected_split_method == "scaffold":
        df_dict = scaffold_split_df(
            df,
            strict=task_meta.strict_test,
            random_state=task_meta.seed,
            stratify_col=stratify_col,
        )
    elif selected_split_method == "random":
        df_dict = random_split_df(
            df,
            random_state=task_meta.seed,
            stratify_col=stratify_col,
        )
    elif selected_split_method == "perimeter":
        perimeter_max_samples = int(
            split_config.get("perimeter_max_samples", PERIMETER_MAX_SAMPLES)
        )
        if len(df) > perimeter_max_samples:
            logger.warning(
                f"Perimeter split requested for {len(df)} samples, exceeding "
                f"the limit of {perimeter_max_samples}; falling back to scaffold split"
            )
            selected_split_method = "scaffold"
            df_dict = scaffold_split_df(
                df,
                strict=task_meta.strict_test,
                random_state=task_meta.seed,
                stratify_col=stratify_col,
            )
        else:
            df_dict = perimeter_split_df(
                df,
                smiles_col=task_meta.smiles_col,
                random_state=task_meta.seed,
                stratify_col=stratify_col,
                max_samples=perimeter_max_samples,
            )
    else:
        raise ValueError(f"Unknown split method: {selected_split_method}")

    task_meta.split_method = selected_split_method

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

    if isinstance(task_meta.extra_cols, list):
        select_cols.extend(task_meta.extra_cols)
    elif isinstance(task_meta.extra_cols, str):
        select_cols.append(task_meta.extra_cols)

    for split_name, df in df_dict.items():
        df = df[select_cols]
        df.to_csv(processed_dir / f"{split_name}.csv", index=False)

    task_meta.save(processed_dir / f"{task_meta.name}.json")


@cli.command()
@click.argument("task", type=str, default="bbbp")
@click.option("--model", type=click.Choice(["rf"]), default="rf", show_default=True)
@click.option("--root-dir", type=click.Path(path_type=Path), default=None)
@click.option("--task-dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("runs"),
    show_default=True,
)
@click.option("--seed", type=int, default=None)
@click.option("--n-estimators", type=int, default=500, show_default=True)
@click.option("--n-jobs", type=int, default=4, show_default=True)
def run(
    task: str,
    model: str,
    root_dir: Path | None,
    task_dir: Path | None,
    output_dir: Path,
    seed: int | None,
    n_estimators: int,
    n_jobs: int,
):
    result = run_baseline(
        BaselineRunConfig(
            task=task,
            model=model,
            root_dir=root_dir,
            task_dir=task_dir,
            output_dir=output_dir,
            seed=seed,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )
    )
    click.echo(f"Run saved to {result.run_dir}")
    click.echo(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    cli()
