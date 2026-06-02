import numpy as np
import pytest

from dpnet.features import FeaturizationError, MorganFeaturizer, MorganFingerprintConfig


def test_morgan_featurizer_is_deterministic():
    featurizer = MorganFeaturizer(MorganFingerprintConfig(n_bits=128))

    first = featurizer.transform(["CCO", "c1ccccc1"])
    second = featurizer.transform(["CCO", "c1ccccc1"])

    assert first.shape == (2, 128)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_morgan_featurizer_rejects_invalid_smiles():
    featurizer = MorganFeaturizer()

    with pytest.raises(FeaturizationError, match="Invalid SMILES"):
        featurizer.transform(["not-a-smiles"])
