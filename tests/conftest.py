import pytest
import rank_bm25

import py_bm25


@pytest.fixture
def bm25_factory():
    def _make(algorithm, corpus, **kwargs):
        return getattr(py_bm25, algorithm)(corpus, **kwargs)
    return _make


@pytest.fixture
def rank_bm25_factory():
    def _make(algorithm, corpus, **kwargs):
        return getattr(rank_bm25, algorithm)(corpus, **kwargs)
    return _make
