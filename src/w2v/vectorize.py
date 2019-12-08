import numpy as np
from typing import List, Tuple


def vectorize_1_hot(word: str, vocabulary: List[str]) -> np.array:
    return np.fromiter((w == word for w in vocabulary), dtype=int)


def training_matrix(word_pairs: List[Tuple[str, str]], vocabulary: List[str]) -> Tuple[np.array, np.array]:

    in_words: List[str] = [x for (x, y) in word_pairs]
    out_words: List[str] = [y for (x, y) in word_pairs]

    return (
        (np.column_stack([vectorize_1_hot(w, vocabulary) for w in in_words])).transpose(),
        (np.column_stack([vectorize_1_hot(w, vocabulary) for w in out_words]).transpose())
    )


