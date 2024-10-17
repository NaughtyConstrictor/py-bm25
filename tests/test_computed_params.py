from operator import itemgetter

import numpy as np
import pytest
from data import cranfield_docs_tokens, sample_docs_tokens


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [sample_docs_tokens, cranfield_docs_tokens], ids=["sample", "cranfield"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_get_avgdl(bm25_factory, rank_bm25_factory, algorithm, corpus, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    rank_bm25_model = rank_bm25_factory(algorithm, corpus)
    assert bm25_model.avgdl == rank_bm25_model.avgdl


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [sample_docs_tokens, cranfield_docs_tokens], ids=["sample", "cranfield"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_get_idf(bm25_factory, rank_bm25_factory, algorithm, corpus, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    rank_bm25_model = rank_bm25_factory(algorithm, corpus)
    idfs = bm25_model.idfs
    expected = rank_bm25_model.idf
    assert idfs.keys() == expected.keys()
    idfs_scores = itemgetter(*expected.keys())(idfs)
    expected_scores = itemgetter(*expected.keys())(expected)
    assert np.allclose(idfs_scores, expected_scores, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [sample_docs_tokens, cranfield_docs_tokens], ids=["sample", "cranfield"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_get_doc_len(bm25_factory, rank_bm25_factory, algorithm, corpus, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    rank_bm25_model = rank_bm25_factory(algorithm, corpus)
    assert bm25_model.doc_len == rank_bm25_model.doc_len


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [sample_docs_tokens, cranfield_docs_tokens], ids=["sample", "cranfield"])
def test_get_index(bm25_factory, rank_bm25_factory, algorithm, corpus):
    bm25_model = bm25_factory(algorithm, corpus)
    rank_bm25_model = rank_bm25_factory(algorithm, corpus)
    assert bm25_model.index == rank_bm25_model.doc_freqs


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
def test_get_inverted_index(bm25_factory, algorithm):
    bm25_model = bm25_factory(algorithm, sample_docs_tokens, invert_index=True)
    expected = {
        "the": [2, 0, 1, 1, 2],
        "quick": [1, 0, 1, 1, 0],
        "brown": [1, 1, 0, 0, 1],
        "fox": [1, 1, 1, 1, 1],
        "jumps": [1, 0, 0, 1, 0],
        "over": [1, 1, 1, 1, 0],
        "lazy": [1, 1, 1, 1, 1],
        "dog": [1, 1, 1, 1, 1],
        "a": [0, 2, 1, 1, 0],
        "fast": [0, 1, 0, 0, 0],
        "leaps": [0, 1, 0, 0, 0],
        "is": [0, 0, 1, 0, 0],
        "jumped": [0, 0, 1, 0, 0],
        "by": [0, 0, 1, 0, 0],
        "swiftly": [0, 0, 0, 1, 0],
        "and": [0, 0, 0, 0, 1],
        "are": [0, 0, 0, 0, 1],
        "friends": [0, 0, 0, 0, 1]
    }
    assert bm25_model.index == expected