import warnings
from libcpp cimport bool
from typing import Dict, List, TypeVar, Union

cimport numpy as cnp
import numpy as np

from py_bm25.bm25 cimport BM25_, BM25Okapi_, BM25L_, BM25Plus_
from py_bm25.convert.data cimport (
    str_vec, str_vec_vec, encode, encode_batch, dict_float, dict_list, list_dict
)
from py_bm25.convert.eigen cimport arrayXd_to_numpy, arrayXXd_to_numpy


V = TypeVar("V")
NormalIndexType = List[Dict[str, int]]
InvIndexType = Dict[str, List[int]]
IndexType = Union[NormalIndexType, InvIndexType]
Doc = List[str]
Query = List[str]


cdef class BM25:
    """
    Base class for BM25 ranking models.

    This class serves as a wrapper for the underlying C++ `BM25_` base class functionalities, which implements 
    core functionality common to all BM25 variants (`BM25Okapi`, `BM25L`, `BM25Plus`). The purpose of this 
    class is to provide a high-level interface and data descriptors for attributes and methods shared across 
    the different BM25 models.

    This class should not be instantiated directly. Use one of the subclasses (`BM25Okapi`, `BM25L`, `BM25Plus`) 
    that specialize the BM25 algorithm for different use cases.

    Attributes
    ----------
    k1 : float
        Free parameter k1.
    b : float
        Free parameter b.
    corpus_size : int
        The total number of documents in the corpus.
    avgdl : float
        The average document length across all documents in the corpus.
    idfs : Dict[str, float]
        Inverse document frequencies (IDFs) for the terms in the corpus.
    doc_len : List[int]
        The length (in number of tokens) for each document in the corpus.
    index : IndexType
        Term frequencies of documents in the corpus. This can be either a standard index or 
        an inverted index, depending on whether the `invert_index` flag is set.

    Raises
    ------
    TypeError
        If an attempt is made to instantiate this class directly.
    """
    cdef BM25_ * base_obj
    cdef list[int] _doc_len
    cdef dict[str, float] _idfs
    cdef object _index
    cdef bool _doc_len_cached
    cdef bool _idfs_cached
    cdef bool _index_cached


    def __cinit__(self, list corpus, *args, **kwargs) -> None:
        if type(self) is BM25:
            raise TypeError(
                "`BM25` is an abstract base class for the BM25 models (`BM25Okapi`, `BM25L`, `BM25Plus`). "
                "It cannot be instantiated directly. Please instantiate one of the subclasses. "
                "See help(BM25) for more info."
            )
        
        if len(corpus) == 0 or len(corpus[0]) == 0:
            warnings.warn(
                "The provided corpus is empty. Please provide a non-empty corpus for meaningful results."
            )
        self.base_obj = NULL
        self._doc_len_cached = False
        self._idfs_cached = False
        self._index_cached = False
        self._doc_len = None
        self._idfs = None
        self._index = None
    
    def __repr__(self) -> None:
        class_ = type(self)
        if class_ is BM25Okapi:
            param = "epsilon"
        else:
            param = "delta"

        return (
            f"{class_.__name__}("
            f"k1={self.k1}, "
            f"b={self.b}, "
            f"{param}={getattr(self, param)}, "
            f"invert_index={self.invert_index}"
            f")"
        )

    def __dealloc__(self):
        del self.base_obj
        del self

    @property
    def k1(self) -> float:
        """
        Get or set the value of the k1 parameter.

        Getter
        ------
        Returns
        -------
        float
            The current k1 value.

        Setter
        ------
        Parameters
        ----------
        k1 : float
            The new value to set for the k1 parameter.

        Examples
        --------
        Set the value of k1:

        >>> model.k1 = 1.5

        Get the current value of k1:
        
        >>> current_k1 = model.k1
        >>> print(current_k1)
        1.5
        """
        return self.base_obj.get_k1()

    @k1.setter
    def k1(self, k1: float) -> None:
        self.base_obj.set_k1(k1)

    @property
    def b(self) -> float:
        """
        Get or set the value of the b parameter.

        Getter
        ------
        Returns
        -------
        float
            The current b value.

        Setter
        ------
        Parameters
        ----------
        b : float
            The new value to set for the b parameter.

        Examples
        --------
        Set the value of b:
        
        >>> model.b = 0.75

        Get the current value of b:
        
        >>> current_b = model.b
        >>> print(current_b)
        0.75
        """
        return self.base_obj.get_b()

    @b.setter
    def b(self, b: float) -> None:
        self.base_obj.set_b(b)

    @property
    def corpus_size(self) -> int:
        """
        Get the number of documents in the corpus.

        Returns
        -------
        int
            The number of documents in the corpus.

        Examples
        --------
        Get the size of the corpus:

        >>> from bm25 import BM25Okapi
        >>> corpus = [["doc1"], ["doc2"], ["doc3"]]
        >>> model = BM25Okapi(corpus)
        >>> size = model.corpus_size
        >>> print(size)
        3
        """
        return self.base_obj.get_corpus_size()

    @property
    def avgdl(self) -> float:
        """
        Get the average document length of the corpus.

        Returns
        -------
        float
            The average document length.

        Examples
        --------
        Get the average document length:

        >>> from bm25 import BM25Okapi
        >>> corpus = [["t1", "t2"], ["t3"], ["t4", "t5", "t6"]]
        >>> model = BM25Okapi(corpus)
        >>> average_length = model.avgdl  # equivalent to sum(map(len, corpus)) / len(corpus)
        >>> print(average_length)
        2.0
        """
        return self.base_obj.get_avgdl()

    @property
    def invert_index(self) -> bool:
        """
        Get the status of the inverted index usage.

        Returns
        -------
        bool
            True if using an inverted index, False otherwise (normal index).

        Examples
        --------
        Check if inverted index is used:

        >>> is_inverted = model.invert_index
        >>> print(is_inverted)
        True
        """
        return self.base_obj.get_invert_index()

    cpdef list[int] _get_doc_len(self):
        """
        Retrieve the document lengths.

        Returns
        -------
        list[int]
            The lengths of documents as a list.
        """
        return list(self.base_obj.get_doc_len())

    @property
    def doc_len(self) -> List[int]:
        """
        Get the document lengths.

        Returns
        -------
        List[int]
            A list of document lengths.

        Examples
        --------
        Get the lengths of all documents:

        >>> from bm25 import BM25Okapi
        >>> corpus = [["t1", "t2"], ["t3"], ["t4", "t5", "t6"]]
        >>> model = BM25Okapi(corpus)
        >>> lengths = model.doc_len
        >>> print(lengths)
        [2, 1, 3]

        Notes
        -----
        This property caches the document lengths after the first retrieval. The returned list is mutable,
        and while users can modify it, such modifications will not affect the underlying C++ object.
        If the user even mutates the returned object and needs to access the original values, they can 
        call the `_get_doc_len` method.
        """
        if not self._doc_len_cached:
            self._doc_len = self._get_doc_len()
            self._doc_len_cached = True
        return self._doc_len

    cpdef dict[str, float] _get_idfs(self):
        """
        Retrieve the inverse document frequencies (IDFs).

        Returns
        -------
        dict[str, float]
            A dictionary with terms as keys and their IDF values as values.
        """
        return dict_float(self.base_obj.get_idfs())

    @property
    def idfs(self) -> Dict[str, float]:
        """
        Get the inverse document frequencies for terms.

        Returns
        -------
        Dict[str, float]
            A dictionary with terms as keys and their IDF values as values.

        Examples
        --------
        Get the inverse document frequencies:

        >>> idf_values = model.idfs
        >>> print(idf_values)
        {'term1': 1.2, 'term2': 0.8}

        Notes
        -----
        This property caches the IDF values after the first retrieval. The returned dictionary is mutable,
        and while users can modify it, such modifications will not affect the underlying C++ object.
        If the user even mutates the returned object and needs to access the original values, they can 
        call the `_get_idfs` method.
        """
        if not self._idfs_cached:
            self._idfs = self._get_idfs()
            self._idfs_cached = True
        return self._idfs

    cpdef object _get_index(self):
        """
        Retrieve the index of term frequencies.

        Returns
        -------
        object
            The normal index  or inverted index, depending on the `invert_index` flag.
        """
        if self.invert_index:
            return dict_list(self.base_obj.get_inverted_index())
        return list_dict(self.base_obj.get_index())
               
    @property
    def index(self) -> IndexType:
        """
        Get the index of term frequencies.

        Returns
        -------
        IndexType
            The index as a list of dictionaries (normal index) or an inverted index, 
            depending on the `invert_index` flag. 

        Examples
        --------
        Access the term frequency index:

        >>> from bm25 import BM25Okapi
        >>> import pprint

        >>> corpus = [
        ...     ["the", "cat", "in", "the", "hat"],
        ...     ["the", "quick", "brown", "fox"],
        ...     ["jumps", "over", "the", "lazy", "dog"]
        >>> ]
        >>> model = BM25Okapi(corpus)
        >>> model_inv = BM25Okapi(corpus, invert_index=True)

        >>> print("Normal Index:")
        >>> pprint.pprint(model.index)
        [{'cat': 1, 'hat': 1, 'in': 1, 'the': 2},
         {'brown': 1, 'fox': 1, 'quick': 1, 'the': 1},
         {'dog': 1, 'jumps': 1, 'lazy': 1, 'over': 1, 'the': 1}]

        >>> print("Inverted Index:")
        >>> pprint.pprint(model_inv.index)
        {'brown': [0, 1, 0],
         'cat': [1, 0, 0],
         'dog': [0, 0, 1],
         'fox': [0, 1, 0],
         'hat': [1, 0, 0],
         'in': [1, 0, 0],
         'jumps': [0, 0, 1],
         'lazy': [0, 0, 1],
         'over': [0, 0, 1],
         'quick': [0, 1, 0],
         'the': [2, 1, 1]}
        
        Notes
        -----
        This property caches the index after the first retrieval. The returned index is mutable, and while 
        users can modify it, such modifications will not affect the underlying C++ object.
        If the user even mutates the returned object and needs to access the original values, they can 
        call the `_get_index` method.

        For more information on the index types and their usage, please refer to the class docstring 
        using `help(type(self))` or `help(BM25)`.
        """
        if not self._index_cached:
            self._index = self._get_index()
            self._index_cached = True
        return self._index


    cdef cnp.ndarray[cnp.float64_t, ndim=1] _get_scores(self, list[str] query):
        """
        Compute the scores for a single of query.

        Parameters
        ----------
        query : list[str]
            The list of query terms.

        Returns
        -------
        cnp.ndarray[cnp.float64_t, ndim=1]
            The computed scores as a NumPy array.
        """
        cdef str_vec c_query = encode(query)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] scores = arrayXd_to_numpy(
            self.base_obj.get_scores(c_query)
        )
        return scores

    cdef cnp.ndarray[cnp.float64_t, ndim=2] _get_scores_batch(self, list[list[str]] queries):
        """
        Compute the scores for a batch of queries.

        Parameters
        ----------
        queries : list[list[str]]
            The list of queries.

        Returns
        -------
        cnp.ndarray[cnp.float64_t, ndim=2]
            The computed scores as a NumPy array.
        """
        cdef str_vec_vec c_queries = encode_batch(queries)         
        cdef cnp.ndarray[cnp.float64_t, ndim=2] scores = arrayXXd_to_numpy(
            self.base_obj.get_scores_batch(c_queries)
        )
        return scores

    def get_scores(self, queries: Union[Query, List[Query]]) -> np.ndarray:
        """
        Get the scores for either a single query or a batch of queries.

        Parameters
        ----------
        queries : Union[Query, List[Query]]
            The queries, either as a single list of terms or a batch of query lists.

        Returns
        -------
        np.ndarray
            The computed scores as a NumPy array.

        Examples
        --------
        >>> corpus = [
        ...     ["the", "cat", "in", "the", "hat"],
        ...     ["the", "quick", "brown", "fox"],
        ...     ["jumps", "over", "the", "lazy", "dog"]
        >>> ]
        >>> model = BM25Plus(corpus)
        >>> model_inv = BM25Plus(corpus, invert_index=True)
        >>> query = ["quick", "fox"]
        >>> model.get_scores(query)
        array([2.77258872, 5.73566064, 2.77258872])

        >>> model_inv.get_scores(query)
        array([2.77258872, 5.73566064, 2.77258872])
        """
        if len(queries) == 0:
            warnings.warn("Empty query")
            return np.zeros(self.corpus_size, dtype=np.float64)

        if isinstance(queries[0], str):
            return self._get_scores(queries)
        return self._get_scores_batch(queries)

    @staticmethod
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sort_partition(cnp.ndarray[cnp.float64_t, ndim=1] scores, int n):
        cdef cnp.ndarray[cnp.int64_t, ndim=1] top_n = np.argpartition(scores, n - 1)[:n]
        cdef cnp.ndarray[cnp.int64_t, ndim=1] tmp = np.argsort(scores[top_n])
        return top_n[tmp]

    @staticmethod
    cdef cnp.ndarray[cnp.int64_t, ndim=2] sort_partition2d(cnp.ndarray[cnp.float64_t, ndim=2] scores, int n):
        cdef cnp.ndarray[cnp.int64_t, ndim=2] col_idx = np.argpartition(scores, n - 1)[:, :n]
        cdef cnp.ndarray[cnp.int64_t, ndim=2] row_idx = np.arange(scores.shape[0])[:, np.newaxis]
        top_n = np.argsort(scores[row_idx, col_idx])
        return col_idx[row_idx, top_n]

    cdef list[V] _get_top_n(self, list[str] query, list[V] documents, int n):
        """
        Get the top N documents for a single query.

        Parameters
        ----------
        query : list[str]
            The list of query terms.
        documents : list[V]
            The list of documents to score.
        n : int
            The number of top documents to return.

        Returns
        -------
        list[V]
            The top N documents as a list.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] scores = -self._get_scores(query)
        cdef int size = scores.shape[0]
        cdef cnp.ndarray[cnp.int64_t, ndim=1] top_n
        n = min(self.corpus_size, n)

        if (size >= 100_000) and (n <= size * 0.5):
            top_n = BM25.sort_partition(scores, n)
        else:
            top_n = np.argsort(scores)[:n]

        return [documents[i] for i in top_n]

    cdef list[list[V]] _get_top_n_batch(self, list[list[str]] queries, list[V] documents, int n):
        """
        Get the top N documents for a batch of queries.

        Parameters
        ----------
        query : list[list[str]]
            The list of queries.
        documents : list[V]
            The list of document ids to score.
        n : int
            The number of top documents to return.

        Returns
        -------
        list[list[V]]
            The top N documents for each query as a list.
        """

        cdef cnp.ndarray[cnp.float64_t, ndim=2] scores = -self._get_scores_batch(queries)
        cdef cnp.ndarray[cnp.int64_t, ndim=2] top_n
        cdef int size = scores.shape[1]
        n = min(self.corpus_size, n)

        if (size >= 100_000) and (n <= size * 0.5):
            top_n = BM25.sort_partition2d(scores, n)
        else:
            top_n = np.argsort(scores)[:, :n]

        return [[documents[i] for i in top] for top in top_n]
    
    def get_top_n(
        self, queries: Union[Query, List[Query]], documents: List[V], n: int=5
    ) -> Union[List[V], List[List[V]]]:
        """
        Get the top N documents for either a single query or a batch of queries.

        Parameters
        ----------
        queries : Union[Query, List[Query]]
            The queries, either as a single list of terms or a batch of queries.
        documents : List[V]
            The list of documents to score.
        n : int, optional, default=5
            The number of top documents to return (default is 5).

        Returns
        -------
        Union[List[V], List[List[V]]]
            The top N documents as a list.

        Examples
        --------
        Get the top 3 documents for a single query:

        >>> from bm25 import BM25Okapi
        >>> corpus = [["doc1"], ["doc2"], ["doc3"]]
        >>> model = BM25Okapi(corpus)
        >>> documents = ["doc1", "doc2", "doc3"]
        >>> top_docs = model.get_top_n(["query_term"], documents, n=3)
        >>> print(top_docs)
        ['doc2', 'doc1']

        Get the top 3 documents for a batch of queries:

        >>> queries = [["query_term1"], ["query_term2"]]
        >>> top_docs_batch = model.get_top_n(queries, documents, n=3)
        >>> print(top_docs_batch)
        [['doc2', 'doc1'], ['doc3', 'doc1']]
        """
        if len(documents) != self.corpus_size:
            raise ValueError("The documents given don't match the index corpus!")

        if len(queries) == 0:
            warnings.warn("Empty query")
            return documents[:n]

        if isinstance(queries[0], str):
            return self._get_top_n(queries, documents, n)
        return self._get_top_n_batch(queries, documents, n)


cdef class BM25Okapi(BM25):
    """
    BM25Okapi ranking model.

    This class implements the BM25Okapi ranking function, a variant of the BM25 algorithm used for text retrieval.
    
    BM25Okapi(corpus: List[Doc], k1: float=1.5, b: float=0.75, epsilon: float=0.25, invert_index: bool=False)
    Parameters
    ----------
    (for the __new__ method; see Notes below)

    corpus : List[Doc]
        The collection of tokenized documents. Each document is represented as a list of tokens (strings).
    k1 : float, optional, default=1.5
        Free parameter k1.
    b : float, optional, default=0.75
        Free parameter b.
    epsilon : float, optional, default=0.25
        Free parameter epsilon.
    invert_index : bool, optional, default=False
        Whether to use the inverted index, which significantly speeds up query processing but increases memory usage.
    
    Attributes
    ----------
    k1 : float
        Current value of the k1 parameter.
    b : float
        Current value of the b parameter.
    epsilon : float
        Current value of the delta parameter.
    corpus_size : int
        The number of documents in the corpus.
    avgdl : float
        The average document length across the corpus.
    idfs : Dict[str, float]
        A dictionary containing the inverse document frequency (IDF) of each term in the corpus.
    doc_len : List[int]
        A list containing the length (number of tokens) of each document in the corpus.
    index : IndexType
        The term frequencies of documents in the corpus. Either a list of dictionaries (normal index)
        or a dictionary of lists (inverted index).

    Examples
    --------
    >>> docs_tokens = [['apple', 'banana', 'apple'], ['banana', 'fruit']]
    >>> bm25 = BM25Okapi(docs_tokens)
    >>> query = ['apple', 'banana']
    >>> bm25.get_scores(query)
    array([-0.12304571, -0.14738442])

    Notes
    -----
    No `__init__` method is needed because the model is fully initialized after the `__new__` method.

    The inverted index, if enabled, speeds up query processing but increases memory usage.
    Future versions may include optimizations for memory efficiency using sparse matrices.
    """
    cdef BM25Okapi_* obj

    def __cinit__(self, list corpus, double k1=1.5, double b=0.75, double epsilon=0.25, bool invert_index=False):
        cdef str_vec_vec c_corpus = encode_batch(corpus)
        self.obj = new BM25Okapi_(c_corpus, k1, b, epsilon, invert_index)
        self.base_obj = self.obj

    @property 
    def epsilon(self) -> float:
        return self.obj.get_epsilon()

    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        self.obj.set_epsilon(epsilon)


cdef class BM25L(BM25):
    """
    BM25L ranking model.

    This class implements the BM25L ranking function, a variant of the BM25 algorithm used for text retrieval.
    
    BM25L(corpus: List[Doc], k1: float=1.5, b: float=0.75, delta: float=0.5, invert_index: bool=False)
    Parameters
    ----------
    (for the __new__ method; see Notes below)

    corpus : List[Doc]
        The collection of tokenized documents. Each document is represented as a list of tokens (strings).
    k1 : float, optional, default=1.5
        Free parameter k1.
    b : float, optional, default=0.75
        Free parameter b.
    delta : float, optional, default=0.5
        Free parameter delta.
    invert_index : bool, optional, default=False
        Whether to use the inverted index, which significantly speeds up query processing but increases memory usage.
    
    Attributes
    ----------
    k1 : float
        Current value of the k1 parameter.
    b : float
        Current value of the b parameter.
    delta : float
        Current value of the delta parameter.
    corpus_size : int
        The number of documents in the corpus.
    avgdl : float
        The average document length across the corpus.
    idfs : Dict[str, float]
        A dictionary containing the inverse document frequency (IDF) of each term in the corpus.
    doc_len : List[int]
        A list containing the length (number of tokens) of each document in the corpus.
    index : IndexType
        The term frequencies of documents in the corpus. Either a list of dictionaries (normal index)
        or a dictionary of lists (inverted index).

    Examples
    --------
    >>> docs_tokens = [['apple', 'banana', 'apple'], ['banana', 'fruit']]
    >>> bm25 = BM25L(docs_tokens)
    >>> query = ['apple', 'banana']
    >>> bm25.get_scores(query)
    array([1.25524857, 0.67378015])

    Notes
    -----
    No `__init__` method is needed because the model is fully initialized after the `__new__` method.

    The inverted index, if enabled, speeds up query processing but increases memory usage.
    Future versions may include optimizations for memory efficiency using sparse matrices.
    """
    cdef BM25L_* obj

    def __cinit__(self, list corpus, double k1=1.5, double b=0.75, double delta=0.5, bool invert_index=False):
        cdef str_vec_vec c_corpus = encode_batch(corpus)
        self.obj = new BM25L_(c_corpus, k1, b, delta, invert_index)
        self.base_obj = self.obj
                
    @property 
    def delta(self) -> float:
        """
        Get or set the value of the delta parameter for BM25L.

        Getter
        ------
        Returns
        -------
        float
            The current delta value.

        Setter
        ------
        Parameters
        ----------
        delta : float
            The new value to set for the delta parameter.
        """
        return self.obj.get_delta()

    @delta.setter
    def delta(self, delta: float) -> None:
        self.obj.set_delta(delta)


cdef class BM25Plus(BM25):
    """
    BM25Plus ranking model.

    This class implements the BM25Plus ranking function, a variant of the BM25 algorithm used for text retrieval.
    
    BM25Plus(corpus: List[Doc], k1: float=1.5, b: float=0.75, delta: float=1.0, invert_index: bool=False)
    Parameters
    ----------
    (for the __new__ method; see Notes below)

    corpus : List[Doc]
        The collection of tokenized documents. Each document is represented as a list of tokens (strings).
    k1 : float, optional, default=1.5
        Free parameter k1.
    b : float, optional, default=0.75
        Free parameter b.
    delta : float, optional, default=1.0
        Free parameter delta.
    invert_index : bool, optional, default=False
        Whether to use the inverted index, which significantly speeds up query processing but increases memory usage.
    
    Attributes
    ----------
    k1 : float
        Current value of the k1 parameter.
    b : float
        Current value of the b parameter.
    delta : float
        Current value of the delta parameter.
    corpus_size : int
        The number of documents in the corpus.
    avgdl : float
        The average document length across the corpus.
    idfs : Dict[str, float]
        A dictionary containing the inverse document frequency (IDF) of each term in the corpus.
    doc_len : List[int]
        A list containing the length (number of tokens) of each document in the corpus.
    index : IndexType
        The term frequencies of documents in the corpus. Either a list of dictionaries (normal index)
        or a dictionary of lists (inverted index).

    Examples
    --------
    >>> docs_tokens = [['apple', 'banana', 'apple'], ['banana', 'fruit']]
    >>> bm25 = BM25Plus(docs_tokens)
    >>> query = ['apple', 'banana']
    >>> bm25.get_scores(query)
    array([3.3507111 , 1.94964345])

    Notes
    -----
    No `__init__` method is needed because the model is fully initialized after the `__new__` method.

    The inverted index, if enabled, speeds up query processing but increases memory usage.
    Future versions may include optimizations for memory efficiency using sparse matrices.
    """
    cdef BM25Plus_* obj

    def __cinit__(self, list corpus, double k1=1.5, double b=0.75, double delta=1.0, bool invert_index=False):
        cdef str_vec_vec c_corpus = encode_batch(corpus)
        self.obj = new BM25Plus_(c_corpus, k1, b, delta, invert_index)
        self.base_obj = self.obj
        
    @property 
    def delta(self) -> float:
        """
        Get or set the value of the delta parameter for BM25Plus.

        Getter
        ------
        Returns
        -------
        float
            The current delta value.

        Setter
        ------
        Parameters
        ----------
        delta : float
            The new value to set for the delta parameter.
        """
        return self.obj.get_delta()

    @delta.setter
    def delta(self, delta: float) -> None:
        self.obj.set_delta(delta)
