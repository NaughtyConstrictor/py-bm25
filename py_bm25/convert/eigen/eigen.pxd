cimport numpy as cnp


cdef extern from "Eigen/Dense":
    cdef cppclass ArrayXXd "Eigen::ArrayXXd":
        ArrayXXd() except +
        double& operator()(int, int)
        int rows()
        int cols()

    cdef cppclass ArrayXd "Eigen::ArrayXd":
        ArrayXd() except +
        double& operator()(int)
        int size()
        


cdef cnp.ndarray[cnp.float64_t, ndim=1] arrayXd_to_numpy(ArrayXd)
cdef cnp.ndarray[cnp.float64_t, ndim=2] arrayXXd_to_numpy(ArrayXXd)
