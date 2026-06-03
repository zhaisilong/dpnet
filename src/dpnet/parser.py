import json
import tomllib
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
from .utils import find_project_root


def get_version() -> int:
    toml_path = Path(find_project_root()) / "pyproject.toml"

    with toml_path.open("rb") as f:
        data = tomllib.load(f)

    return int(data["project"]["version"].split(".")[0]) + 1


@dataclass
class LabelMeta:
    id: str
    label_col: str
    problem_type: Literal["binary", "multiclass", "regression"]
    num_classes: Optional[int] = None

    def __post_init__(self):
        if self.problem_type == "multiclass":
            assert self.num_classes is not None

    def to_dict(self):
        return {
            "id": self.id,
            "label_col": self.label_col,
            "problem_type": self.problem_type,
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(
            id=dict["id"],
            label_col=dict["label_col"],
            problem_type=dict["problem_type"],
            num_classes=dict["num_classes"],
        )


@dataclass
class TaskMeta:
    name: str
    id_col: Optional[str] = None
    seed: Optional[int] = None
    smiles_col: Optional[str] = None
    labels: List[LabelMeta] = field(default_factory=list)
    strict_test: Optional[bool] = None
    processed_dir: Optional[str] = None
    dialect: Optional[str] = None
    version: Optional[int] = None
    extra_cols: Optional[List[str]] = None
    split_method: Literal["scaffold", "random", "perimeter"] = "scaffold"
    split_config: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        raw_labels = data.get("labels")

        if raw_labels:
            labels = [
                LabelMeta.from_dict(label) if isinstance(label, dict) else label
                for label in raw_labels
            ]
        else:
            labels = [LabelMeta("label", "binary", 2)]

        return cls(
            name=data["name"],
            version=data.get("version") or get_version(),
            dialect=data.get("dialect", "dpnet"),
            processed_dir=data.get("processed_dir", "processed"),
            id_col=data.get("id_col", "cid"),
            smiles_col=data.get("smiles_col", "smiles"),
            strict_test=data.get("strict_test", True),
            labels=labels,
            seed=data.get("seed"),
            extra_cols=data.get("extra_cols"),
            split_method=data.get("split_method", "scaffold"),
            split_config=data.get("split_config") or {},
        )

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "dialect": self.dialect,
            "processed_dir": self.processed_dir,
            "id_col": self.id_col,
            "smiles_col": self.smiles_col,
            "strict_test": self.strict_test,
            "labels": [label.to_dict() for label in self.labels],
            "seed": self.seed,
            "extra_cols": self.extra_cols or [],
            "split_method": self.split_method,
            "split_config": self.split_config or {},
        }

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return repr(self.to_dict())

    def iter_labels(self):
        for label in self.labels:
            yield label

    def save(self, save_path: str | Path):
        save_path = Path(save_path)
        save_path.write_text(json.dumps(self.to_dict(), indent=4))

    @classmethod
    def load(cls, load_path: str | Path):
        load_path = Path(load_path)
        return cls.from_dict(json.loads(load_path.read_text()))
