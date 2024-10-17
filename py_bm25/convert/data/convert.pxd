from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


ctypedef vector[string] str_vec
ctypedef vector[str_vec] str_vec_vec


cdef str_vec encode(list[str])
cdef str_vec_vec encode_batch(list[list[str]])
cdef dict dict_float(unordered_map[string, double] &)
cdef dict dict_list(unordered_map[string, vector[int]] &)
cdef list list_dict(vector[unordered_map[string, int]] &)
