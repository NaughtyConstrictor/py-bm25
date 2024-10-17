import pytest
from data import cranfield_docs_tokens, sample_docs_tokens


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [[[]], sample_docs_tokens, cranfield_docs_tokens], ids=["empty", "sample", "cranfield"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_get_k1(bm25_factory, algorithm, corpus, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    assert bm25_model.k1 == 1.5


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [[[]], sample_docs_tokens, cranfield_docs_tokens], ids=["empty", "sample", "cranfield"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_get_b(bm25_factory, algorithm, corpus, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    assert bm25_model.b == 0.75


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("algorithm, param, expected", [
    ("BM25Okapi", "epsilon", 0.25), 
    ("BM25L", "delta", 0.5), 
    ("BM25Plus", "delta", 1.0)
])
@pytest.mark.parametrize("corpus", [[[]], sample_docs_tokens, cranfield_docs_tokens], ids=["empty", "sample", "cranfield"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_get_epsilon_delta(bm25_factory, algorithm, param, expected, corpus, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    assert getattr(bm25_model, param) == expected
