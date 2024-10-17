from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from py_bm25.convert.eigen cimport ArrayXd, ArrayXXd


cdef extern from "src/bm25.h" namespace "BM25":
    cdef cppclass BM25_:
        double get_k1() const
        double get_b() const
        void set_k1(double)
        double set_b(double)
        int get_corpus_size() const
        double get_avgdl() const
        bool get_invert_index() const
        vector[unordered_map[string, int]]& get_index()
        unordered_map[string, vector[int]]& get_inverted_index()
        unordered_map[string, double]& get_idfs()
        vector[int]& get_doc_len()
        ArrayXd get_scores(vector[string] &)
        ArrayXXd get_scores_batch(vector[vector[string]] &)

    cdef cppclass BM25Okapi_(BM25_):
        BM25Okapi_(const vector[vector[string]] &, double, double, double, bool) except +
        double get_epsilon() const
        double set_epsilon(double)

    cdef cppclass BM25L_(BM25_):
        BM25L_(const vector[vector[string]] &, double, double, double, bool) except +
        double get_delta() const
        double set_delta(double)

    cdef cppclass BM25Plus_(BM25_):
        BM25Plus_(const vector[vector[string]] &, double, double, double, bool) except +
        double get_delta() const
        double set_delta(double)
