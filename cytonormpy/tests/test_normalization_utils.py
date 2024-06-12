import pytest
import pandas as pd
import numpy as np

from cytonormpy._utils._utils import (_all_batches_have_reference)
from cytonormpy._normalization._utils import numba_quantiles  # Replace with the actual import path


def test_all_batches_have_reference():
    ref = ["control", "other", "control", "other", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )

    assert _all_batches_have_reference(df,
                                       "reference",
                                       "batch",
                                       ref_control_value = "control")


def test_all_batches_have_reference_ValueError():
    ref = ["control", "other", "control", "unknown", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    with pytest.raises(ValueError):
        _all_batches_have_reference(df,
                                    "reference",
                                    "batch",
                                    ref_control_value = "control")


def test_all_batches_have_reference_batch_only_controls():
    ref = ["control", "other", "control", "control", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    assert _all_batches_have_reference(df,
                                       "reference",
                                       "batch",
                                       ref_control_value = "control")


def test_all_batches_have_reference_batch_false():
    ref = ["control", "other", "other", "other", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    assert not _all_batches_have_reference(df,
                                           "reference",
                                           "batch",
                                           ref_control_value = "control")


def test_all_batches_have_reference_batch_wrong_control_value():
    ref = ["control", "other", "other", "other", "control", "other"]
    batch = ["1", "1", "2", "2", "3", "3"]

    df = pd.DataFrame(
        data = {"reference": ref, "batch": batch},
        index = pd.Index(list(range(len(ref))))
    )
    assert not _all_batches_have_reference(df,
                                           "reference",
                                           "batch",
                                           ref_control_value = "ref")


@pytest.mark.parametrize("data, q, expected_shape", [
    # Normal use-cases
    (np.array([3.0, 1.0, 4.0, 1.5, 2.0], dtype=np.float64), np.array([0.25, 0.5, 0.75], dtype=np.float64), (3,)),
    (np.linspace(0, 100, 1000, dtype=np.float64), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3,)),
    (np.random.rand(100), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3,)),
    
    # Edge cases
    (np.array([1.0], dtype=np.float64), np.array([0.5], dtype=np.float64), (1,)),
    (np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64), np.array([0.25, 0.5, 0.75], dtype=np.float64), (3,)),
    (np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64), (2,)),
    
    # Higher-dimensional arrays (2D with single column)
    (np.random.rand(15).reshape(-1, 1), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3, 1)),
    (np.linspace(1, 100, 10).reshape(-1, 1), np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64), (4, 1)),
    (np.array([[2], [3], [5], [8], [13]], dtype=np.float64), np.array([0.25, 0.5, 0.75], dtype=np.float64), (3, 1)),

    # Large arrays
    (np.random.rand(10000), np.array([0.01, 0.5, 0.99], dtype=np.float64), (3,)),
    (np.random.rand(10000).reshape(-1, 1), np.array([0.01, 0.5, 0.99], dtype=np.float64), (3, 1)),
    
])
def test_numba_quantiles(data, q, expected_shape):
    expected = np.quantile(data, q, axis=0, keepdims=True).reshape(expected_shape)
    result = numba_quantiles(data, q)
    
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape} vs {expected_shape}"
    
    assert np.array_equal(result, expected), f"Mismatch: {result} vs {expected}"


