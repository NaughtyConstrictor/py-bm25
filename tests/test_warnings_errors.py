import re

import pytest
import py_bm25


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_empty_corpus(bm25_factory, algorithm, kwargs):
    with pytest.warns(
        UserWarning, 
        match="The provided corpus is empty. Please provide a non-empty corpus for meaningful results."
        ):
        bm25_factory(algorithm, corpus=[[]], **kwargs)


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_empty_query(bm25_factory, algorithm, kwargs):
    bm_25_model = bm25_factory(algorithm, corpus=[[""]], **kwargs)

    with pytest.warns(
        UserWarning, 
        match="Empty query"
        ):
        bm_25_model.get_scores([])


def test_instantiate_base_class_raises_error():
    with pytest.raises(
        TypeError, 
        match=re.escape(
            "`BM25` is an abstract base class for the BM25 models (`BM25Okapi`, `BM25L`, `BM25Plus`). "
            "It cannot be instantiated directly. Please instantiate one of the subclasses. "
            "See help(BM25) for more info."
        )):
        py_bm25.bm25.BM25([[""]])


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("documents", [[], ["a", "b", "c"], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_get_top_n_raises_value_error_when_corpus_size_doesnt_match(bm25_factory, algorithm, documents, kwargs):
    bm25_model = bm25_factory(algorithm, [[""]], **kwargs)
    with pytest.raises(ValueError, match="The documents given don't match the index corpus!"):
        bm25_model.get_top_n([""], documents, 2)