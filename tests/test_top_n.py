import random
import numpy as np


def get_top_n(scores, ids, n):
    scores = -scores
    top_n = np.argpartition(scores, n - 1)[:n]
    tmp = np.argsort(scores[top_n])

    return [ids[i] for i in top_n[tmp]]


def get_top_n_batch(scores, ids, n):
    scores = -scores
    col_idx = np.argpartition(scores, n - 1)[:, :n]
    row_idx = np.arange(scores.shape[0])[:, np.newaxis]
    top_n = np.argsort(scores[row_idx, col_idx])
    
    return [[ids[i] for i in top] for top in col_idx[row_idx, top_n]]


def test_get_top_n_1():
    scores = np.array([
        0.79709087, 0.35765708, 0.7281863 , 0.25257506, 0.31291027,
        0.83307321, 0.03800933, 0.60563541, 0.26460682, 0.3938609
    ])
    ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    expected = ['f', 'a', 'c', 'h', 'j', 'b', 'e', 'i', 'd', 'g']
    assert get_top_n(scores, ids, 5) == expected[:5]


def test_get_top_n_2():
    scores = np.array([
        0.79709087, 0.35765708, 0.7281863 , 0.25257506, 0.31291027,
        0.83307321, 0.03800933, 0.60563541, 0.26460682, 0.3938609
    ])
    ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    expected = ['f', 'a', 'c', 'h', 'j', 'b', 'e', 'i', 'd', 'g']
    assert get_top_n(scores, ids, 10) == expected


def test_get_top_n_3():
    array = [random.random() for _ in range(7)]
    array += [1.5, 2.0, 2.5]
    random.shuffle(array)
    idx1 = array.index(2.5)
    idx2 = array.index(2.0)
    idx3 = array.index(1.5)
    ids = [f"id_{i}" for i in range(len(array))]
    expected = [f"id_{i}" for i in (idx1, idx2, idx3)]
    array = np.array(array)
    assert get_top_n(array, ids, 3) == expected


def test_get_top_n_3():
    array = [random.random() for _ in range(100)]
    max_vals = list(range(11, 1, -1))
    array += max_vals
    random.shuffle(array)
    max_idx = [array.index(val) for val in max_vals]
    ids = [f"id_{i}" for i in range(len(array))]
    expected = [f"id_{i}" for i in max_idx]
    array = np.array(array)
    assert get_top_n(array, ids, 10) == expected


def test_get_top_n_batch(): 
    array = [[random.random() for _ in range(100)] for _ in range(5)]
    max_vals = list(range(11, 1, -1))
    array = [arr + max_vals for arr in array]
    for arr in array:
        random.shuffle(arr)
    max_idx = [[arr.index(val) for val in max_vals] for arr in array]
    ids = [f"id_{i}" for i in range(len(array[0]))]
    expected = [[f"id_{i}" for i in idx] for idx in max_idx]
    array = np.array(array)
    assert get_top_n_batch(array, ids, 10) == expected
