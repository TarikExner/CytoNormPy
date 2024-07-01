import warnings
import numpy as np
import pandas as pd

from typing import Optional, Callable, Union

from numba import njit, float64, int32, int64
from numba.types import Tuple

@njit(float64[:](float64[:]))
def numba_diff(arr):
    result = np.empty(arr.size - 1, dtype=arr.dtype)
    for i in range(arr.size - 1):
        result[i] = arr[i + 1] - arr[i]
    return result


@njit(float64[:](float64[:], float64[:]))
def _select_interpolants_numba(x: np.ndarray,
                               y: np.ndarray):
    """\
    Modifies the tangents mi to ensure the monotonicity of the
    resulting Hermite Spline.

    This implementation follows
    https://en.wikipedia.org/wiki/Monotone_cubic_interpolation and
    https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/monoSpl.c
    """
    dy = numba_diff(y)
    dx = numba_diff(x)
    Sx = dy / dx
    m = np.array([Sx[0]] + list((Sx[1:] + Sx[:-1]) / 2) + [Sx[-1]])
    n = m.shape[0]
    # Fritsch Carlson algorithm
    for k in range(n - 1):
        Sk = Sx[k]
        k1 = k + 1
        if Sk == 0:
            m[k] = m[k1] = 0
        else:
            alpha = m[k] / Sk
            beta = m[k1] / Sk
            a2b3 = 2 * alpha + beta - 3
            ab23 = alpha + 2 * beta - 3

            if (a2b3 > 0) & \
               (ab23 > 0) & \
               (alpha * (a2b3 + ab23) < a2b3 * a2b3):
                tauS = 3 * Sk / np.sqrt(alpha**2 + beta**2)
                m[k] = tauS * alpha
                m[k1] = tauS * beta
    return m

@njit(float64(float64[:]))
def _numba_mean(arr) -> np.ndarray:
    """
    Calculate the mean of a float64 array.
    """
    return np.sum(arr) / arr.size


@njit(float64(float64[:]))
def _numba_median(arr):
    """
    Calculate the median of a float64 array.
    """
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    
    if n % 2 == 0:
        median = (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
    else:
        median = sorted_arr[n // 2]
    
    return median


@njit(int32[:](float64[:], float64[:], int32, int64[:]))
def numba_searchsorted(arr, values, side, sorter):
    """
    Numba-compatible searchsorted function for single and multiple values with 'left' and 'right' modes.
    
    Parameters
    ----------

    arr
        The sorted input array.
    values
        Array of values to insert.
    side
        Integer specifying insertion side (0 for 'left', 1 for 'right').
    sorter
        An index array that defines the sorting order of `arr`.

    Returns
    -------
    An array of indices where each value in `values` should be inserted.

    """
    def binary_search(arr, value, side, sorter):
        left, right = 0, sorter.size
        while left < right:
            mid = (left + right) // 2
            mid_value = arr[int(sorter[mid])]
            if (side == 0 and mid_value < value) or (side == 1 and mid_value <= value):
                left = mid + 1
            else:
                right = mid
        return left

    indices = np.empty(values.size, dtype=np.int32)
    for i in range(values.size):
        indices[i] = binary_search(arr, values[i], side, sorter)
    return indices

@njit((float64[:],))
def numba_unique_indices(arr):
    """
    Numba-compatible function to find unique elements and their original indices.
    
    Parameters
    ----------
    arr
        Input array from which to find unique elements.
    
    Returns
    -------
    unique_arr
        Array of unique elements.
    indices
        Array of indices in the original array where unique elements were found.

    """
    if arr.size == 0:
        return np.empty(0, dtype=arr.dtype), np.empty(0, dtype=np.intp)

    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]
    
    unique_values = []
    unique_indices = []
    
    previous_value = sorted_arr[0]
    unique_values.append(previous_value)
    unique_indices.append(sorted_indices[0])
    
    for i in range(1, sorted_arr.size):
        current_value = sorted_arr[i]
        if current_value != previous_value:
            unique_values.append(current_value)
            unique_indices.append(sorted_indices[i])
        previous_value = current_value
    
    unique_arr = np.array(unique_values, dtype=arr.dtype)
    indices = np.array(unique_indices, dtype=np.intp)
    
    return unique_arr, indices


@njit(Tuple((int32[:], int32[:]))(float64[:], float64[:], int64[:]))
def match(x: np.ndarray,
          y: np.ndarray,
          sorter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = numba_searchsorted(x, y, 0, sorter) # side = 0 means 'left'
    right = numba_searchsorted(x, y, 1, sorter) # side = 0 means 'right'
    return left, right


@njit(float64[:](float64[:], float64, float64, int32))
def _insert_to_array(y, b, e, ties):
    if ties == 0:
        y[b:e] = _numba_mean(y[b:e])
    elif ties == 1:
        y[b:e] = _numba_median(y[b:e])
    return y


@njit((float64[:], float64[:], int32, int32))
def _regularize(x: np.ndarray,
                y: np.ndarray,
                ties: int,
                nx: int):
    o = np.argsort(x)
    x = x[o]
    y = y[o]
    ux, idxs = numba_unique_indices(x)
    if ux.shape[0] < nx:
        # y = tapply(y, match(x, x), fun)
        ls, rs = match(x, x, sorter = np.argsort(x))
        matches = np.empty((ls.size, 2), dtype=np.int64)
        matches[:, 0] = ls
        matches[:, 1] = rs
        # matches = np.vstack([ls, rs]).T
        unique_matches_list = []
        for i in range(matches.shape[0]):
            is_unique = True
            for um in unique_matches_list:
                if matches[i, 0] == um[0] and matches[i, 1] == um[1]:
                    is_unique = False
                    break
            if is_unique:
                unique_matches_list.append((matches[i, 0], matches[i, 1]))
        
        unique_matches = np.empty((len(unique_matches_list), 2), dtype=np.int64)
        for i, (left, right) in enumerate(unique_matches_list):
            if left <= right:
                unique_matches[i, 0] = left
                unique_matches[i, 1] = right
            else:
                unique_matches[i, 0] = right
                unique_matches[i, 1] = left

        for row in unique_matches:
            if row[0] > row[1]:
                row[0], row[1] = row[1], row[0]

        for b, e in zip(unique_matches[:, 0],
                        unique_matches[:, 1]):
            y = _insert_to_array(y, b, e, ties)

        x = x[idxs]
        y = y[idxs]

        assert x.shape[0] == y.shape[0]

    return x, y

@njit(Tuple((float64[:], float64[:]))(float64[:], float64[:]))
def remove_nans_numba(x, y):
    """
    Remove NaNs from x and y in a Numba-compatible way.
    
    Parameters
    ----------
    x
        numpy array of type float64
    y
        numpy array of type float64
    
    Returns
    -------
    x_cleaned
        numpy array of type float64 without NaNs
    y_cleaned
        numpy array of type float64 without NaNs
    """
    isnan_mask = np.isnan(x) | np.isnan(y)
    
    x_cleaned = x[~isnan_mask]
    y_cleaned = y[~isnan_mask]
    
    return x_cleaned, y_cleaned


def regularize_values(x: np.ndarray,
                      y: np.ndarray,
                      ties: Optional[Union[str, int, Callable]] = np.mean
                      ) -> tuple[np.ndarray, np.ndarray]:
    """\
    Implementation of the R regularize.values function in python.
    """

    assert x.shape[0] == y.shape[0], "x and y length must be the same."
    x, y = remove_nans_numba(x, y)
    # if np.any(np.isnan(x)) or np.any(np.isnan(y)):
    #     x = x[~np.isnan(x)]
    #     y = y[~np.isnan(y)]

    nx = x.shape[0]

    if ties != "ordered":
        if ties == np.mean:
            ties = 0
        elif ties == np.median:
            ties = 1
        elif ties is None:
            ties = -1
        if ties == -1:
            warnings.warn(
                "Collapsing to unique 'x' values",
                UserWarning
            )
        assert not isinstance(ties, Callable)
        assert not isinstance(ties, str)
        x, y = _regularize(x, y, ties, nx)

    return x, y


def _all_batches_have_reference(df: pd.DataFrame,
                                reference: str,
                                batch: str,
                                ref_control_value: Optional[str]
                                ) -> bool:
    """
    Function checks if there are samples labeled ref_control_value
    for each batch.
    """
    _df: pd.DataFrame = pd.DataFrame(df[[reference, batch]].drop_duplicates())

    if len(_df[reference].unique()) != 2:
        raise ValueError(
            "Please make sure that there are only two values in "
            "the reference column. Have found "
            f"{_df[reference].unique().tolist()}"
        )

    # if both uniques are present in all batches, that's fine
    ref_per_batch = _df.groupby(batch, observed = True).nunique()
    if all(ref_per_batch[reference] == 2):
        return True

    # alternatively, batches might only contain controls
    one_refs = ref_per_batch[ref_per_batch[reference] == 1]
    one_ref_batches = one_refs.index.tolist()

    if all(
        _df.loc[
            _df[batch].isin(one_ref_batches), reference
        ] == ref_control_value
    ):
        return True

    return False


def _conclusive_reference_values(df: pd.DataFrame,
                                 reference: str) -> bool:
    """
    checks if there are no more than two values in the reference column.
    We allow the option that every sample is labeled as control.
    """
    if len(df[reference].unique()) > 2:
        return False
    return True
