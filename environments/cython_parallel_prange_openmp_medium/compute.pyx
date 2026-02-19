import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_row_sums(cnp.ndarray[cnp.float64_t, ndim=2] data):
    cdef int n_rows = data.shape[0]
    cdef int n_cols = data.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.zeros(n_rows, dtype=np.float64)
    cdef int i, j
    cdef double row_sum
    
    with nogil:
        for i in prange(n_rows, schedule='static'):
            row_sum = 0.0
            for j in range(n_cols):
                row_sum += data[i, j]
            result[i] = row_sum
    
    return result