import pytest
import numpy as np

from cytonormpy._utils._utils import regularize_values


def test_regularize_values_unchanged_arrays():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5, 6])

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, x)
    assert np.array_equal(y_p, y)


def test_regularize_values_unchanged_arrays_unsorted():
    x = np.array([0, 2, 1, 3, 4, 5])
    y = np.array([1, 3, 2, 4, 5, 6])

    x_p, y_p = regularize_values(x, y)
    o = np.argsort(x)
    assert np.array_equal(x_p, x[o])
    assert np.array_equal(y_p, y[o])


def test_regularize_values():
    x = np.array([0, 0, 0, 1, 2, 3])
    y = np.array([0, 1, 2, 3, 4, 5])

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_regularize_values_reversed():
    x = np.array([3, 2, 1, 0, 0, 0])
    y = np.array([0, 1, 2, 3, 4, 5])

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([4, 2, 1, 0]))


def test_regularize_values_double_reversed():
    x = np.array([3, 2, 1, 0, 0, 0])
    y = np.array([5, 4, 3, 2, 1, 0])

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_regularize_values_multiple_doublets():
    x = np.array([0, 0, 0, 1, 1, 1, 2, 3])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 4, 6, 7]))


def test_regularize_values_neg_values():
    x = np.array([-1, -1, -1, 1, 2, 3])
    y = np.array([0, 1, 2, 3, 4, 5])

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([-1, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_regularize_values_float():
    x = np.array([0, 0, 0, 1, 2, 3]).astype(np.float64)
    y = np.array([0, 1, 2, 3, 4, 5]).astype(np.float64)

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]).astype(np.float64))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]).astype(np.float64))


def test_regularize_values_median():
    x = np.array([0, 0, 0, 1, 2, 3])
    y = np.array([0, 1, 2, 3, 4, 5])

    x_p, y_p = regularize_values(x, y, ties = np.median)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_regularize_values_shape_mismatch():
    x = np.array([0, 4, 2])
    y = np.array([0, 1, 1, 1])
    with pytest.raises(AssertionError):
        _, _ = regularize_values(x, y)


def test_regularize_values_nan():
    x = np.array([0, 0, 0, 1, 2, np.nan, np.nan, 3])
    y = np.array([0, 1, 2, 3, 4, np.nan, np.nan, 5])

    x_p, y_p = regularize_values(x, y, ties = np.median)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))
