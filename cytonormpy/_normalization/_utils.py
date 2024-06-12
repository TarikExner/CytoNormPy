import numpy as np
from numba import njit, float64

import numpy as np
from numba import njit, float64

@njit(float64[:](float64[:], float64[:]), cache=True)
def _numba_quantiles_1d(a, q):
    """
    Compute quantiles for a 1D numpy array.
    
    Parameters:
    a : numpy.ndarray
        Input 1D array of type np.float64.
    q : numpy.ndarray
        Quantiles to compute, should be in the range [0, 1].
    
    Returns:
    numpy.ndarray
        Computed quantiles for the input array.
    """
    sorted_a = np.sort(a)
    n = len(sorted_a)
    quantiles = np.empty(len(q), dtype=np.float64)
    
    for i in range(len(q)):
        position = q[i] * (n - 1)
        lower_index = int(np.floor(position))
        upper_index = int(np.ceil(position))
        
        if lower_index == upper_index:
            quantiles[i] = sorted_a[lower_index]
        else:
            lower_value = sorted_a[lower_index]
            upper_value = sorted_a[upper_index]
            quantiles[i] = lower_value + (upper_value - lower_value) * (position - lower_index)
    
    return quantiles

@njit(float64[:, :](float64[:, :], float64[:]), cache=True)
def _numba_quantiles_2d(a, q):
    """
    Compute quantiles for a 2D numpy array with the second dimension of size 1.
    
    Parameters:
    a : numpy.ndarray
        Input 2D array of shape (n, 1) with type np.float64.
    q : numpy.ndarray
        Quantiles to compute, should be in the range [0, 1].
    
    Returns:
    numpy.ndarray
        Computed quantiles for the input array in shape (n_quantiles, 1).
    """
    a_1d = np.ascontiguousarray(a[:, 0])  # Convert to contiguous 1D array
    quantiles = _numba_quantiles_1d(a_1d, q)
    return np.ascontiguousarray(quantiles).reshape(-1, 1)

def numba_quantiles(a, q):
    """
    Compute quantiles for a 1D or 2D numpy array.

    Parameters:
    a : numpy.ndarray
        Input 1D or 2D array of type np.float64.
    q : numpy.ndarray
        Quantiles to compute, should be in the range [0, 1].

    Returns:
    numpy.ndarray
        Computed quantiles for the input array.
        - If input is 1D, returns 1D array of shape (n_quantiles,).
        - If input is 2D with shape (n, 1), returns 2D array of shape (n_quantiles, 1).
    """
    if a.ndim == 1:
        return _numba_quantiles_1d(a, q)
    else: 
        return _numba_quantiles_2d(a, q)

