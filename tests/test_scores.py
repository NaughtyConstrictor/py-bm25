import random

import numpy as np
import pytest
from data import (
    cranfield_docs_tokens, 
    cranfield_queries_tokens,
    cranfield_scores,
    sample_docs_tokens,
    sample_queries_tokens,
    sample_scores
)


cranfield_queries_ids_sample = random.choices(range(len(cranfield_queries_tokens)), k=15)
cranfield_queries_tokens_sample = [cranfield_queries_tokens[id_] for id_ in cranfield_queries_ids_sample]

@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
@pytest.mark.parametrize("query", sample_queries_tokens, ids=[f"query_{i}" for i in range(len(sample_queries_tokens))])
def test_scores_sample(bm25_factory, rank_bm25_factory, algorithm, kwargs, query):
    bm25_model = bm25_factory(algorithm, sample_docs_tokens, **kwargs)
    rank_bm25_model = rank_bm25_factory(algorithm, sample_docs_tokens)
    assert np.allclose(
        bm25_model.get_scores(query), rank_bm25_model.get_scores(query), atol=1e-10, rtol=1e-9
    )


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
@pytest.mark.parametrize("query", cranfield_queries_tokens_sample, ids=cranfield_queries_ids_sample)
def test_scores_cranfield(bm25_factory, rank_bm25_factory, algorithm, kwargs, query):
    bm25_model = bm25_factory(algorithm, cranfield_docs_tokens, **kwargs)
    rank_bm25_model = rank_bm25_factory(algorithm, cranfield_docs_tokens)
    assert np.allclose(
        bm25_model.get_scores(query), rank_bm25_model.get_scores(query), atol=1e-10, rtol=1e-9
    )


@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_scores_batch_sample(bm25_factory, algorithm, kwargs):
    bm25_model = bm25_factory(algorithm, sample_docs_tokens, **kwargs)
    assert np.allclose(
        bm25_model.get_scores(sample_queries_tokens), sample_scores[algorithm], atol=1e-10, rtol=1e-9
    )


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_scores_batch_cranfield(bm25_factory, algorithm, kwargs):
    bm25_model = bm25_factory(algorithm, cranfield_docs_tokens, **kwargs)
    assert np.allclose(
        bm25_model.get_scores(cranfield_queries_tokens), cranfield_scores[algorithm], atol=1e-10, rtol=1e-9
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [[[""]], sample_docs_tokens, cranfield_docs_tokens], ids=["one", "sample", "cranfield"])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_scores_empty_query(bm25_factory, algorithm, corpus, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    scores = bm25_model.get_scores([])
    assert scores.shape == (len(corpus),)
    assert (scores == np.zeros(len(corpus), dtype=np.float64)).all()



@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["BM25Okapi", "BM25L", "BM25Plus"])
@pytest.mark.parametrize("corpus", [[[""]], sample_docs_tokens, cranfield_docs_tokens], ids=["one", "sample", "cranfield"])
@pytest.mark.parametrize("queries", [[[]], [[], [], []], [[], [], [], [], [], [], []]])
@pytest.mark.parametrize("kwargs", [{}, {"invert_index": True}], ids=["normal_index", "inverted_index"])
def test_scores_empty_query2(bm25_factory, algorithm, corpus, queries, kwargs):
    bm25_model = bm25_factory(algorithm, corpus, **kwargs)
    scores = bm25_model.get_scores(queries)
    assert scores.shape == (len(queries), len(corpus))
    assert (scores == np.zeros(len(corpus), dtype=np.float64)).all()
