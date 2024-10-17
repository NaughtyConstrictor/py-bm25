cimport numpy as cnp
import numpy as np


cdef cnp.ndarray[cnp.float64_t, ndim=1] arrayXd_to_numpy(ArrayXd arr):
    cdef int size = arr.size()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] np_array = np.empty(size, dtype=np.float64)

    cdef int i
    for i in range(size):
        np_array[i] = arr(i)

    return np_array

cdef cnp.ndarray[cnp.float64_t, ndim=2] arrayXXd_to_numpy(ArrayXXd arr):
    cdef int rows = arr.rows()
    cdef int cols = arr.cols()
    cdef cnp.ndarray[cnp.float64_t, ndim=2] np_array = np.empty((rows, cols), dtype=np.float64)

    cdef int i, j
    for i in range(rows):
        for j in range(cols):
            np_array[i, j] = arr(i, j)

    return np_array