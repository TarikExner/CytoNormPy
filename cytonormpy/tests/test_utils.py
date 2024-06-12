import pytest
import numpy as np

from cytonormpy._utils._utils import (regularize_values,
                                      numba_searchsorted,
                                      numba_unique_indices,
                                      _numba_mean,
                                      _numba_median)


def test_regularize_values_unchanged_arrays():
    x = np.array([0, 1, 2, 3, 4, 5], dtype = np.float64)
    y = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, x)
    assert np.array_equal(y_p, y)


def test_regularize_values_unchanged_arrays_unsorted():
    x = np.array([0, 2, 1, 3, 4, 5], dtype = np.float64)
    y = np.array([1, 3, 2, 4, 5, 6], dtype = np.float64)

    x_p, y_p = regularize_values(x, y)
    o = np.argsort(x)
    assert np.array_equal(x_p, x[o])
    assert np.array_equal(y_p, y[o])


def test_regularize_values():
    x = np.array([0, 0, 0, 1, 2, 3], dtype = np.float64)
    y = np.array([0, 1, 2, 3, 4, 5], dtype = np.float64)

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_regularize_values_reversed():
    x = np.array([3, 2, 1, 0, 0, 0], dtype = np.float64)
    y = np.array([0, 1, 2, 3, 4, 5], dtype = np.float64)

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([4, 2, 1, 0]))


def test_regularize_values_double_reversed():
    x = np.array([3, 2, 1, 0, 0, 0], dtype = np.float64)
    y = np.array([5, 4, 3, 2, 1, 0], dtype = np.float64)

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_regularize_values_multiple_doublets():
    x = np.array([0, 0, 0, 1, 1, 1, 2, 3], dtype = np.float64)
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype = np.float64)

    x_p, y_p = regularize_values(x, y)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 4, 6, 7]))


def test_regularize_values_neg_values():
    x = np.array([-1, -1, -1, 1, 2, 3], dtype = np.float64)
    y = np.array([0, 1, 2, 3, 4, 5], dtype = np.float64)

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
    x = np.array([0, 0, 0, 1, 2, 3], dtype = np.float64)
    y = np.array([0, 1, 2, 3, 4, 5], dtype = np.float64)

    x_p, y_p = regularize_values(x, y, ties = np.median)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_regularize_values_shape_mismatch():
    x = np.array([0, 4, 2], dtype = np.float64)
    y = np.array([0, 1, 1, 1], dtype = np.float64)
    with pytest.raises(AssertionError):
        _, _ = regularize_values(x, y)


def test_regularize_values_nan():
    x = np.array([0, 0, 0, 1, 2, np.nan, np.nan, 3], dtype = np.float64)
    y = np.array([0, 1, 2, 3, 4, np.nan, np.nan, 5], dtype = np.float64)

    x_p, y_p = regularize_values(x, y, ties = np.median)
    assert np.array_equal(x_p, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_p, np.array([1, 3, 4, 5]))


def test_single_value_insertion_left():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype = np.float64)
    values = np.array([25.0], dtype = np.float64)
    sorter = np.argsort(arr)
    side_left = 0  # 'left'

    expected = np.searchsorted(arr, values, side = 'left', sorter = sorter)
    result = numba_searchsorted(arr, values, side_left, sorter)
    assert np.array_equal(result, expected)


def test_multiple_values_insertion_right():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype = np.float64)
    values = np.array([5.0, 35.0, 45.0], dtype = np.float64)
    sorter = np.argsort(arr)
    side_right = 1  # 'right'

    expected = np.searchsorted(arr, values, side = 'right', sorter = sorter)
    result = numba_searchsorted(arr, values, side_right, sorter)
    assert np.array_equal(result, expected)


def test_edge_cases_left():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype = np.float64)
    values = np.array([0.0, 10.0, 50.0, 60.0], dtype = np.float64)
    sorter = np.argsort(arr)
    side_left = 0  # 'left'

    expected = np.searchsorted(arr, values, side = 'left', sorter = sorter)
    result = numba_searchsorted(arr, values, side_left, sorter)
    assert np.array_equal(result, expected)


def test_using_sorter():
    arr = np.array([50.0, 20.0, 10.0, 40.0, 30.0], dtype = np.float64)
    values = np.array([25.0, 5.0, 35.0, 45.0], dtype = np.float64)
    sorter = np.argsort(arr)
    side_left = 0  # 'left'

    expected = np.searchsorted(arr, values, side = 'left', sorter = sorter)
    result = numba_searchsorted(arr, values, side_left, sorter)
    assert np.array_equal(result, expected)


def test_unique_basic_case():
    arr = np.array([5.0, 3.0, 5.0, 2.0, 1.0, 3.0, 4.0], dtype = np.float64)
    expected_values, expected_indices = np.unique(arr, return_index=True)
    result_values, result_indices = numba_unique_indices(arr)
    assert np.array_equal(result_values, expected_values)
    assert np.array_equal(result_indices, expected_indices)


def test_unique_empty_array():
    arr = np.array([], dtype = np.float64)
    expected_values, expected_indices = np.unique(arr, return_index = True)
    result_values, result_indices = numba_unique_indices(arr)
    assert np.array_equal(result_values, expected_values)
    assert np.array_equal(result_indices, expected_indices)


def test_unique_all_same():
    arr = np.array([2.0, 2.0, 2.0, 2.0], dtype = np.float64)
    expected_values, expected_indices = np.unique(arr, return_index = True)
    result_values, result_indices = numba_unique_indices(arr)
    assert np.array_equal(result_values, expected_values)
    assert np.array_equal(result_indices, expected_indices)

def test_unique_sorted():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype = np.float64)
    expected_values, expected_indices = np.unique(arr, return_index = True)
    result_values, result_indices = numba_unique_indices(arr)
    assert np.array_equal(result_values, expected_values)
    assert np.array_equal(result_indices, expected_indices)

def test_unique_reverse_sorted():
    arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype = np.float64)
    expected_values, expected_indices = np.unique(arr, return_index = True)
    result_values, result_indices = numba_unique_indices(arr)
    assert np.array_equal(result_values, expected_values)
    assert np.array_equal(result_indices, expected_indices)


def test_empty_array_numba_mean():
    arr = np.array([], dtype=np.float64)
    with pytest.raises(ZeroDivisionError):
        _ = _numba_mean(arr)

def test_single_element_numba_mean():
    arr = np.array([42], dtype=np.float64)
    assert _numba_mean(arr) == np.mean(arr)

def test_positive_integers_numba_mean():
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_negative_integers_numba_mean():
    arr = np.array([-1, -2, -3, -4, -5], dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_mixed_integers_numba_mean():
    arr = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_large_numbers_numba_mean():
    arr = np.array([1e10, 1e10, 1e10, 1e10, 1e10], dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_small_numbers_numba_mean():
    arr = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10], dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_mixed_large_small_numbers_numba_mean():
    arr = np.array([1e10, 1e-10, -1e10, -1e-10], dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_nan_values_numba_mean():
    arr = np.array([1.0, 2.0, np.nan], dtype=np.float64)
    assert np.isnan(_numba_mean(arr))

def test_inf_values_numba_mean():
    arr = np.array([1.0, 2.0, np.inf], dtype=np.float64)
    assert np.isinf(_numba_mean(arr))

def test_large_array_numba_mean():
    arr = np.random.rand(1000000).astype(np.float64)
    assert np.isclose(_numba_mean(arr), np.mean(arr), rtol=1e-7)

def test_all_zeros_numba_mean():
    arr = np.zeros(1000, dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_all_ones_numba_mean():
    arr = np.ones(1000, dtype=np.float64)
    assert np.array_equal(_numba_mean(arr), np.mean(arr))

def test_random_values_numba_mean():
    arr = np.random.random(1000).astype(np.float64)
    assert np.isclose(_numba_mean(arr), np.mean(arr), rtol=1e-7)

def test_random_normal_distribution_numba_mean():
    arr = np.random.normal(0, 1, 1000).astype(np.float64)
    assert np.isclose(_numba_mean(arr), np.mean(arr), rtol=1e-7)

def test_random_uniform_distribution_numba_mean():
    arr = np.random.uniform(-100, 100, 1000).astype(np.float64)
    assert np.isclose(_numba_mean(arr), np.mean(arr), rtol=1e-7)

def test_single_element_numba_median():
    arr = np.array([42], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_positive_integers_numba_median():
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_negative_integers_numba_median():
    arr = np.array([-1, -2, -3, -4, -5], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_mixed_integers_numba_median():
    arr = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_large_numbers_numba_median():
    arr = np.array([1e10, 1e10, 1e10, 1e10, 1e10], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_small_numbers_numba_median():
    arr = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_mixed_large_small_numbers_numba_median():
    arr = np.array([1e10, 1e-10, -1e10, -1e-10], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_nan_values_numba_median():
    arr = np.array([1.0, 2.0, np.nan], dtype=np.float64)
    assert not np.array_equal(_numba_median(arr), np.median(arr))

def test_inf_values_numba_median():
    arr = np.array([1.0, 2.0, np.inf], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_large_array_numba_median():
    arr = np.random.rand(1000000).astype(np.float64)
    assert np.isclose(_numba_median(arr), np.median(arr), rtol=1e-7)

def test_all_zeros_numba_median():
    arr = np.zeros(1000, dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_all_ones_numba_median():
    arr = np.ones(1000, dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_random_values_numba_median():
    arr = np.random.random(1000).astype(np.float64)
    assert np.isclose(_numba_median(arr), np.median(arr), rtol=1e-7)

def test_random_normal_distribution_numba_median():
    arr = np.random.normal(0, 1, 1000).astype(np.float64)
    assert np.isclose(_numba_median(arr), np.median(arr), rtol=1e-7)

def test_random_uniform_distribution_numba_median():
    arr = np.random.uniform(-100, 100, 1000).astype(np.float64)
    assert np.isclose(_numba_median(arr), np.median(arr), rtol=1e-7)

def test_even_number_elements_numba_median():
    arr = np.array([1, 3, 3, 6, 7, 8, 9, 15], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_odd_number_elements_numba_median():
    arr = np.array([1, 3, 3, 6, 7, 8, 9], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_sorted_array_numba_median():
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_reverse_sorted_array_numba_median():
    arr = np.array([5, 4, 3, 2, 1], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))

def test_array_with_repeated_elements_numba_median():
    arr = np.array([1, 1, 1, 1, 1], dtype=np.float64)
    assert np.array_equal(_numba_median(arr), np.median(arr))
