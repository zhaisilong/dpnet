from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


class FeaturizationError(ValueError):
    """Raised when a molecule cannot be converted into features."""


@dataclass(frozen=True)
class MorganFingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class MorganFeaturizer:
    def __init__(self, config: MorganFingerprintConfig | None = None):
        self.config = config or MorganFingerprintConfig()
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.config.radius,
            fpSize=self.config.n_bits,
            includeChirality=self.config.include_chirality,
        )

    def transform(self, smiles: Iterable[str]) -> np.ndarray:
        rows = [self._featurize_one(smi, row_idx) for row_idx, smi in enumerate(smiles)]
        if not rows:
            return np.empty((0, self.config.n_bits), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

    def _featurize_one(self, smiles: str, row_idx: int) -> np.ndarray:
        if not isinstance(smiles, str) or not smiles.strip():
            raise FeaturizationError(f"Invalid SMILES at row {row_idx}: {smiles!r}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise FeaturizationError(f"Invalid SMILES at row {row_idx}: {smiles!r}")

        fp = self._generator.GetFingerprint(mol)
        arr = np.zeros((self.config.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
