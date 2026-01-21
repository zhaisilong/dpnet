import json
import torch
import pandas as pd

from loguru import logger
from pathlib import Path
from .utils import find_task_root
from .parser import TaskMeta


class DPNetDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row.to_dict()


class DPNet:
    def __init__(self, task_dir: Path | str):
        self.task_dir = Path(task_dir)
        self.task_name = self.task_dir.name
        self.task_meta_path = self.task_dir / f"{self.task_name}.json"
        self.task_meta = TaskMeta.load(self.task_meta_path)

        self.datasets = self.init_data()

    def init_data(self) -> dict[str, DPNetDataset]:
        _datasets = {}
        for split_name in ["train", "valid", "test"]:
            _datasets[split_name] = DPNetDataset(
                pd.read_csv(self.task_dir / f"{split_name}.csv")
            )
        return _datasets
