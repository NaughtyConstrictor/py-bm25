# BM25: High-Performance BM25 Ranking Functions for Efficient Search

The `py_bm25` package offers efficient implementations of the BM25 ranking functions tailored for fast and efficient information retrieval. Built with Cython and C++, this package includes three variants: `BM25Okapi`, `BM25L`, and `BM25Plus`. It is designed to improve performance over the [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) package.

This package supports two types of indexes:

- **Normal Index (Default)**: A list of dictionaries where each dictionary contains the token frequencies for a corresponding document.

- **Inverted Index**: A dictionary where keys are tokens and values are lists of integers representing token frequencies across all documents. Each list is as long as the number of documents, and the dictionary size corresponds to the number of unique tokens across all documents.   
This index type offers significant speed improvements but consumes more memory. Due to its structure, the inverted index can become very sparse (containing many zeros), especially with large corpora. Future versions may include optimizations using sparse matrices to mitigate memory usage.

The package leverages [`OpenMP`](https://www.openmp.org) to enable true parallelism, which helps bypass Python’s Global Interpreter Lock (GIL) and enhances the performance of computations. This ensures that even with the normal index, which is less memory-intensive, the operations remain efficient.

Additionally, [`Eigen`](https://eigen.tuxfamily.org/index.php?title=Main_Page) is utilized to handle mathematical operations efficiently on the C++ level, further contributing to the package’s performance.

The algorithms in this package are based on the paper [Improvements to BM25 and Language Models Examined](https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf), which provides a comprehensive overview and benchmarking of various BM25 methods.

## Installation

### Prerequisites

Before installing, ensure the following are installed on your system:

- A **C++ compiler** supporting C++11 (e.g., `g++` or `clang`)
- **OpenMP**: Ensure that OpenMP is supported by your compiler (it's usually included by default on modern compilers like `g++` or `clang`).
- **Eigen**: You need to have Eigen (a C++ library for linear algebra) installed. By default, the package looks for Eigen in `/usr/include/eigen3`. If it's installed elsewhere, you must set the `EIGEN_PATH` environment variable to the appropriate path.

#### Example on Ubuntu:

```bash
sudo apt install g++ libomp-dev
sudo apt install libeigen3-dev
```

On other systems, follow the relevant steps to install OpenMP and Eigen.

### Installing from PyPI

You can install the `py_bm25` package directly using `pip`:

```bash
pip install py_bm25
```

### Installing from Source

To install `py_bm25` from source, follow these steps:

```bash
git clone https://github.com/NaughtyConstrictor/bm25.git
cd py_bm25
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
EIGEN_PATH=/your/eigen/path pip install .
```

In the above command, replace `/your/eigen/path` with the actual path to Eigen if it's not in the default location.

## Usage

### Importing the Package

```python
from py_bm25 import BM25Okapi, BM25L, BM25Plus
```

### Initializing the Models

```python
# Example corpus: list of documents where each document is a list of tokens
corpus = [
    ["the", "cat", "in", "the", "hat"],
    ["the", "quick", "brown", "fox"],
    ["jumps", "over", "the", "lazy", "dog"]
]

# Initialize with normal index (default)
bm25_okapi = BM25Okapi(corpus)

# Initialize with inverted index
bm25_okapi_inv = BM25Okapi(corpus, invert_index=True)

# Initialize BM25Okapi
bm25_okapi = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)

# Initialize BM25L
bm25_l = BM25L(corpus, k1=1.5, b=0.75, delta=0.5)

# Initialize BM25Plus
bm25_plus = BM25Plus(corpus, k1=1.5, b=0.75, delta=1)
```

### Computing Scores

```python
# Single query
query = ["quick", "fox"]
scores = bm25_okapi.get_scores(query)
# same results as: scores = bm25_okapi_inv.get_scores(query)
print("Scores:", scores)

# Batch of queries
queries = [["quick", "fox"], ["jumps", "over"]]
# same results as: scores = bm25_okapi_inv.get_scores(queries)
batch_scores = bm25_okapi.get_scores(queries)
print("Batch Scores:", batch_scores)
```

### Getting Top-N Documents

```python
documents = ["Doc1", "Doc2", "Doc3"] # number of documents must match the corpus size
top_n_docs = bm25_okapi.get_top_n(query, documents, n=2)
print("Top N Documents:", top_n_docs)
```

### Index Types Example

```python
import pprint

print("Normal Index:")
pprint.pprint(bm25_okapi.index)

print("Inverted Index:")
pprint.pprint(bm25_okapi_inv.index)
```

**Normal Index Output:**
```python
[{'cat': 1, 'hat': 1, 'in': 1, 'the': 2},
 {'brown': 1, 'fox': 1, 'quick': 1, 'the': 1},
 {'dog': 1, 'jumps': 1, 'lazy': 1, 'over': 1, 'the': 1}]
```

**Inverted Index Output:**
```python
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
```

## License

This package is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please submit issues and pull requests to the GitHub repository.
