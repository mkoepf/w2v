import numpy as np
from w2v.vectorize import vectorize_1_hot, training_matrix
from typing import List


def test_vectorize_1_hot():
    assert np.allclose(
        vectorize_1_hot('hello', ['i', 'say', 'hello', 'again']),
        np.array([0, 0, 1, 0])
    )

    assert np.allclose(
        vectorize_1_hot('i', ['i', 'say', 'hello', 'again']),
        np.array([1, 0, 0, 0])
    )

    assert np.allclose(
        vectorize_1_hot('outside', ['i', 'say', 'hello', 'again']),
        np.array([0, 0, 0, 0])
    )


def test_training_matrix():
    word_pairs: List[(str, str)] = [('a', 'b'), ('d', 'c')]
    vocabulary: List[str] = ['a', 'b', 'c', 'd']

    (x, y) = training_matrix(word_pairs, vocabulary)

    assert np.allclose(
        x,
        np.array([[1, 0, 0, 0],
                 [0, 0, 0, 1]])
    )
    assert np.allclose(
        y,
        np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0]])
    )
