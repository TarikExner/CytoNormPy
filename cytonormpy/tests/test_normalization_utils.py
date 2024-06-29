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
    # Normal use-cases for 1D arrays
    (np.array([3.0, 1.0, 4.0, 1.5, 2.0], dtype=np.float64), np.array([0.25, 0.5, 0.75], dtype=np.float64), (3,)),
    (np.linspace(0, 100, 1000, dtype=np.float64), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3,)),
    (np.random.rand(100), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3,)),
    
    # Edge cases for 1D arrays
    (np.array([1.0], dtype=np.float64), np.array([0.5], dtype=np.float64), (1,)),
    (np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64), np.array([0.25, 0.5, 0.75], dtype=np.float64), (3,)),
    (np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64), (2,)),
    
    # Large arrays
    (np.random.rand(10000), np.array([0.01, 0.5, 0.99], dtype=np.float64), (3,)),
])
def test_numba_quantiles_1d(data, q, expected_shape):
    # Convert data to 2D for np.quantile to keep comparison consistent
    data_2d = data[:, None]
    expected = np.quantile(data_2d, q, axis=0).flatten()  # np.quantile result for 1D should be flattened
    result = numba_quantiles(data, q)
    
    # Check if shapes match
    assert result.shape == expected_shape
    
    # Check if values match
    assert np.array_equal(result, expected)

def test_invalid_quantiles_1d():
    # Test invalid quantiles with 1D arrays
    with pytest.raises(ValueError):
        numba_quantiles(np.array([1.0, 2.0], dtype=np.float64), np.array([-0.1, 1.1], dtype=np.float64))
    with pytest.raises(ValueError):
        numba_quantiles(np.array([1.0, 2.0], dtype=np.float64), np.array([1.5], dtype=np.float64))


@pytest.mark.parametrize("data, q, expected_shape", [
    # Normal use-cases for 2D arrays
    (np.random.rand(10, 5), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3, 5)),
    (np.linspace(0, 100, 1000).reshape(200, 5), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3, 5)),
    (np.random.rand(100, 3), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3, 3)),
    
    # Edge cases for 2D arrays where second dimension is 1
    (np.random.rand(15, 1), np.array([0.1, 0.5, 0.9], dtype=np.float64), (3, 1)),
    (np.linspace(1, 100, 10).reshape(-1, 1), np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64), (4, 1)),
    (np.array([[2], [3], [5], [8], [13]], dtype=np.float64), np.array([0.25, 0.5, 0.75], dtype=np.float64), (3, 1)),

    # Large arrays
    (np.random.rand(10000, 10), np.array([0.01, 0.5, 0.99], dtype=np.float64), (3, 10)),
    
    # Empty arrays
    (np.array([[]], dtype=np.float64), np.array([0.5], dtype=np.float64), (1, 0)),
])
def test_numba_quantiles_2d(data, q, expected_shape):
    # Ensure comparison with np.quantile is consistent
    expected = np.quantile(data, q, axis=0, keepdims=True).reshape(expected_shape)
    result = numba_quantiles(data, q)
    
    # Check if shapes match
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape} vs {expected_shape}"
    
    # Check if values match
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-8), f"Mismatch: {result} vs {expected}"

def test_invalid_array_shape_2d():
    with pytest.raises(ValueError):
        numba_quantiles(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64), np.array([0.5], dtype=np.float64))

def test_invalid_quantiles_2d():
    with pytest.raises(ValueError):
        numba_quantiles(np.array([[1.0], [2.0]], dtype=np.float64), np.array([-0.1, 1.1], dtype=np.float64))
    with pytest.raises(ValueError):
        numba_quantiles(np.array([[1.0], [2.0]], dtype=np.float64), np.array([1.5], dtype=np.float64))

