# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def compute_mean(double[:] data):
    """Compute the mean of a 1D array."""
    cdef:
        int i
        int n = data.shape[0]
        double total = 0.0
    
    for i in range(n):
        total += data[i]
    
    return total / n

def compute_variance(double[:] data):
    """Compute the variance of a 1D array."""
    cdef:
        int i
        int n = data.shape[0]
        double mean = 0.0
        double variance = 0.0
        double diff
    
    # Calculate mean
    for i in range(n):
        mean += data[i]
    mean /= n
    
    # Calculate variance
    for i in range(n):
        diff = data[i] - mean
        variance += diff * diff
    
    return variance / n

def scale_array(double[:] data, double factor):
    """Scale all elements in array by a factor."""
    cdef:
        int i
        int n = data.shape[0]
    
    result = np.empty(n, dtype=np.float64)
    cdef double[:] result_view = result
    
    for i in range(n):
        result_view[i] = data[i] * factor
    
    return result

def matrix_row_sums(double[:, :] matrix):
    """Compute sum of each row in a 2D matrix."""
    cdef:
        int i, j
        int rows = matrix.shape[0]
        int cols = matrix.shape[1]
        double row_sum
    
    result = np.empty(rows, dtype=np.float64)
    cdef double[:] result_view = result
    
    for i in range(rows):
        row_sum = 0.0
        for j in range(cols):
            row_sum += matrix[i, j]
        result_view[i] = row_sum
    
    return result