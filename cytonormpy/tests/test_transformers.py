import pytest
import numpy as np
from cytonormpy._transformation._transformations import (LogicleTransformer,
                                                         AsinhTransformer,
                                                         LogTransformer,
                                                         HyperLogTransformer)


@pytest.fixture
def test_array() -> np.ndarray:
    return np.random.randint(1, 1000, 100).reshape(10, 10).astype(np.float64)


def test_logtransformer(test_array: np.ndarray):
    t = LogTransformer()
    transformed = t.transform(test_array)
    rev_transformed = t.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(test_array, rev_transformed)


def test_hyperlogtransformer(test_array: np.ndarray):
    t = HyperLogTransformer()
    transformed = t.transform(test_array)
    rev_transformed = t.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(test_array, rev_transformed)


def test_logicletransformer(test_array: np.ndarray):
    t = LogicleTransformer()
    transformed = t.transform(test_array)
    rev_transformed = t.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(test_array, rev_transformed)


def test_asinhtransformer(test_array: np.ndarray):
    t = AsinhTransformer()
    transformed = t.transform(test_array)
    rev_transformed = t.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(test_array, rev_transformed)


def test_logtransformer_channel_idxs(test_array: np.ndarray):
    t = LogTransformer(channel_indices = list(range(5)))
    transformed = t.transform(test_array)
    np.testing.assert_array_almost_equal(
        transformed[:, 5:],
        test_array[:, 5:]
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        transformed[:, :4],
        test_array[:, :4]
    )
    rev_transformed = t.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(test_array, rev_transformed)


def test_hyperlogtransformer_channel_idxs(test_array: np.ndarray):
    t = HyperLogTransformer(channel_indices = list(range(5)))
    transformed = t.transform(test_array)
    np.testing.assert_array_almost_equal(
        transformed[:, 5:],
        test_array[:, 5:]
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        transformed[:, :4],
        test_array[:, :4]
    )
    rev_transformed = t.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(test_array, rev_transformed)


def test_logicletransformer_channel_idxs(test_array: np.ndarray):
    t = LogicleTransformer(channel_indices = list(range(5)))
    transformed = t.transform(test_array)
    np.testing.assert_array_almost_equal(
        transformed[:, 5:],
        test_array[:, 5:]
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        transformed[:, :4],
        test_array[:, :4]
    )
    rev_transformed = t.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(test_array, rev_transformed)
