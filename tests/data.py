import ir_datasets
import numpy as np
import rank_bm25


# @inproceedings{macavaney:sigir2021-irds,
#   author = {MacAvaney, Sean and Yates, Andrew and Feldman, Sergey and Downey, Doug and Cohan, Arman and Goharian, Nazli},
#   title = {Simplified Data Wrangling with ir_datasets},
#   year = {2021},
#   booktitle = {SIGIR}
# }


dataset = ir_datasets.load("cranfield")
docs = list(dataset.docs)
queries = list(dataset.queries)
cranfield_docs_tokens = [doc.text.split() for doc in docs]
cranfield_queries_tokens = [query.text.split() for query in queries]
cranfield_docs_ids = [doc.doc_id for doc in docs]

sample_docs_tokens = [
    doc.split()
    for doc in [
        "the quick brown fox jumps over the lazy dog",
        "a fast brown fox leaps over a lazy dog",
        "the lazy dog is jumped over by a quick fox",
        "a quick fox swiftly jumps over the lazy dog",
        "the brown fox and the lazy dog are friends"
    ]
]
sample_queries_tokens = [
    query.split()
    for query in [
        "quick fox",
        "lazy dog",
        "brown",
        "jumps",
        "friends"
    ]
]
sample_docs_ids = [str(i) for i in range(len(sample_docs_tokens))]


def make_scores(corpus, queries):
    scores = {}
    for class_ in ("BM25Okapi", "BM25L", "BM25Plus"):
        model = getattr(rank_bm25, class_)(corpus)
        scores[class_] = np.array([model.get_scores(query) for query in queries])
    return scores

sample_scores = make_scores(sample_docs_tokens, sample_queries_tokens)
cranfield_scores = make_scores(cranfield_docs_tokens, cranfield_queries_tokens)
