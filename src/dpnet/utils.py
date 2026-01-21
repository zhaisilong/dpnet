import os

from pathlib import Path
from loguru import logger

def find_project_root(
    start=None,
    markers=("configs", ".git", "pyproject.toml", ".gitignore", ".cursorignore"),
) -> os.PathLike:
    """查找项目根目录:
    1. 优先使用环境变量 PROJECT_ROOT
    2. 向上递归查找 markers
    3. 找不到则 raise
    """
    # 1. 优先从 env 获取
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)

    # 2. 向上查找
    here = os.path.abspath(start or os.getcwd())
    while here != "/":
        if any(os.path.exists(os.path.join(here, m)) for m in markers):
            return here
        here = os.path.dirname(here)

    # 3. 最终失败 → raise
    raise RuntimeError(
        f"项目根目录未找到，请设置环境变量 PROJECT_ROOT " f"或者确保包含 {markers} 之一"
    )


def find_task_root(task_name: str) -> Path:
    database_root = Path(find_project_root()) / "database"

    # 匹配 "database/task_name" 目录
    matches = list(database_root.rglob(task_name))

    filtered_matches = []
    for match in matches:
        if "processed" in str(match):
            logger.info(f"Skipping processed directory: {match}")
            continue
        else:
            filtered_matches.append(match)

    if not filtered_matches:
        raise RuntimeError(f"Task root not found: {database_root}/{task_name}")

    if len(filtered_matches) > 1:
        raise RuntimeError(
            f"Multiple task roots found for '{task_name}', please specify: {matches}"
        )

    task_root = filtered_matches[0]

    if not task_root.is_dir():
        raise RuntimeError(f"Task root exists but is not a directory: {task_root}")

    return task_root
