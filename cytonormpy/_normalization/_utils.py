import numpy as np
from numba import njit, float64

@njit(float64[:, :](float64[:, :], float64[:]), cache=True)
def numba_quantiles_2d(a, q):
    """
    Compute quantiles for a 2D numpy array along axis 0.
    
    Parameters:
    a : numpy.ndarray
        Input 2D array of type np.float64.
    q : numpy.ndarray
        Quantiles to compute, should be in the range [0, 1].

    Returns:
    numpy.ndarray
        Computed quantiles for the input array along axis 0.
        Output shape is (len(q), a.shape[1]).
    """
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError("Quantiles should be in the range [0, 1].")

    n_quantiles = len(q)
    n_columns = a.shape[1]
    quantiles = np.empty((n_quantiles, n_columns), dtype=np.float64)
    
    for col in range(n_columns):
        sorted_col = np.sort(a[:, col])
        n = len(sorted_col)
        for i in range(n_quantiles):
            position = q[i] * (n - 1)
            lower_index = int(np.floor(position))
            upper_index = int(np.ceil(position))
            
            if lower_index == upper_index:
                quantiles[i, col] = sorted_col[lower_index]
            else:
                lower_value = sorted_col[lower_index]
                upper_value = sorted_col[upper_index]
                quantiles[i, col] = lower_value + (upper_value - lower_value) * (position - lower_index)
    
    return quantiles

@njit(float64[:](float64[:], float64[:]), cache=True)
def numba_quantiles_1d(a, q):
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

    if np.any(q < 0) or np.any(q > 1):
        raise ValueError("Quantiles should be in the range [0, 1].")

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

def numba_quantiles(a, q):
    """
    Compute quantiles for a 1D or 2D numpy array along axis 0.
    
    Parameters:
    a : numpy.ndarray
        Input 1D or 2D array of type np.float64.
    q : numpy.ndarray
        Quantiles to compute, should be in the range [0, 1].

    Returns:
    numpy.ndarray
        Computed quantiles for the input array.
        - If input is 1D, returns 1D array of shape (len(q),).
        - If input is 2D, returns 2D array of shape (len(q), a.shape[1]).
    """
    if a.ndim == 1:
        return numba_quantiles_1d(a, q)
    elif a.ndim == 2:
        return numba_quantiles_2d(a, q)
    else:
        raise ValueError("Input array must be 1D or 2D.")
