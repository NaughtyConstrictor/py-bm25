from libcpp.utility cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef str_vec encode(list[str] py_doc):
    """
    Encode a single document into a C++ compatible format.

    Parameters
    ----------
    py_doc : list[str]
        The document represented as a list of tokens (strings).

    Returns
    -------
    vector[string]
        The encoded document as a C++ vector of strings.

    Notes
    -----
    This function converts each token of the document into a C++ string
    and stores it in a C++ vector. The document is treated as a list of
    individual tokens, each of which is encoded as UTF-8.
    """
    cdef string c_token
    cdef size_t len_doc = len(py_doc)
    cdef str_vec c_doc = str_vec()
    c_doc.reserve(len_doc)

    for token in py_doc:
        c_token = token.encode("utf-8")
        c_doc.push_back(c_token)

    return c_doc


cdef str_vec_vec encode_batch(list[list[str]] py_docs):
    """
    Encode a batch of documents into a C++ compatible format.

    Parameters
    ----------
    py_docs : list[list[str]]
        The list of documents, where each document is represented as a
        list of tokens (strings).

    Returns
    -------
    vector[vector[string]]
        The encoded batch of documents as a C++ vector of vectors of strings.

    Notes
    -----
    This function converts each document, which is represented as a list
    of tokens, into a C++ vector of strings. The entire batch is stored
    in a C++ vector of vectors. Each token is encoded as UTF-8.
    """
    cdef size_t len_doc
    cdef str_vec_vec c_docs
    cdef str_vec c_doc
    cdef string c_token
    cdef size_t num_docs = len(py_docs)
    c_docs.reserve(num_docs)

    for doc in py_docs:
        len_doc = len(doc)
        c_doc = str_vec()
        c_doc.reserve(len_doc)
        for token in doc:
            c_token = token.encode("utf-8")
            c_doc.push_back(c_token)
        c_docs.push_back(c_doc)

    return c_docs


cdef dict dict_float(unordered_map[string, double] &c_mapping):
    """
    Convert a C++ unordered_map[string, double] to a Python dictionary with float values.

    Parameters
    ----------
    c_mapping : unordered_map[string, double]
        The C++ unordered_map with terms (strings) as keys and float values (e.g., IDF scores).

    Returns
    -------
    dict
        The converted dictionary with terms (strings) as keys and float values.

    Notes
    -----
    This function is used for converting the IDF scores from a C++ unordered_map to a Python dictionary.
    """
    cdef pair[string, double] item
    cdef dict py_mapping = {}
    cdef str key
    cdef double value
    for item in c_mapping:
        key = item.first.decode("utf-8")
        value = item.second
        py_mapping[key] = value
    return py_mapping



cdef dict dict_list(unordered_map[string, vector[int]] &c_mapping):
    """
    Convert a C++ unordered_map[string, vector[int]] to a Python dictionary with lists of integers.

    Parameters
    ----------
    c_mapping : unordered_map[string, vector[int]]
        The C++ unordered_map with terms (strings) as keys and lists of integers as values.

    Returns
    -------
    dict
        The converted dictionary with terms (strings) as keys and lists of integers as values.

    Notes
    -----
    This function is used for converting term frequency data (where each term is
    associated with a list of integer frequencies) from C++ unordered_map to a Python dictionary
    (for the inverted index; see help(BM25) for more info).
    """
    cdef pair[string, vector[int]] item
    cdef dict py_mapping = {}
    cdef str key
    cdef list value

    for item in c_mapping:
        key = item.first.decode("utf-8")
        value = item.second
        py_mapping[key] = value
    
    return py_mapping


cdef list list_dict(vector[unordered_map[string, int]] &c_freqs):
    """
    Convert a C++ vector of unordered_map[string, int] to a Python list of dictionaries.

    Parameters
    ----------
    c_freqs : vector[unordered_map[string, int]]
        The C++ vector of unordered_map, where each map represents term frequencies
        for a document with string keys and integer values.

    Returns
    -------
    list
        The converted list of dictionaries, where each dictionary represents term frequencies
        for a document, with string keys and integer values.

    Notes
    -----
    This function is used for converting document-term frequency data from C++ vector of 
    unordered_map to a Python list of dictionaries, where each dictionary holds the term frequencies 
    for one document (for the normal index; see help(BM25) for more info).
    """
    cdef unordered_map[string, int] doc_freq
    cdef pair[string, int] item
    cdef list py_freqs = []
    cdef dict py_freq_dict
    cdef str token
    cdef int freq

    for doc_freq in c_freqs:
        py_freq_dict = dict()
        for item in doc_freq:
            token = item.first.decode("utf-8")
            freq = item.second
            py_freq_dict[token] = freq
        py_freqs.append(py_freq_dict)

    return py_freqs

