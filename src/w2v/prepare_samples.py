# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = w2v.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

from typing import Tuple, List
from nltk import FreqDist


def take_window(wordlist: List[str], pos: int, width: int) -> List[str]:
    left_window: List[str] = wordlist[max(0, pos - width):pos]
    right_window: List[str] = wordlist[pos + 1:min(pos + 1 + width, len(wordlist))]
    return left_window + right_window


def wordlist_from_sentence(sentence: str) -> List[str]:
    # Here will be preprocessing

    return sentence.split(" ")


def wordlists_from_sentences(sentences: List[str]) -> List[List[str]]:
    return [wordlist_from_sentence(sentence) for sentence in sentences]


def samples_from_wordlist(wordlist: List[str], window_width: int) -> List[Tuple[str, str]]:

    windows: List[Tuple[str, List[str]]] = \
        [(w, take_window(wordlist, i, window_width)) for (i, w) in enumerate(wordlist)]

    return [(w, win_word) for (w, win) in windows for win_word in win]


def samples_from_wordlists(wordlists: List[List[str]], width: int) -> List[Tuple[str, str]]:
    return [sample for wl in wordlists for sample in samples_from_wordlist(wl, width)]


def samples_from_sentence(sentence: str, window_width: int) -> List[Tuple[str, str]]:
    words: List[str] = wordlist_from_sentence(sentence)
    return samples_from_wordlist(words, window_width)


def samples_from_sentences(sentences: List[str], window_width: int) -> List[Tuple[str, str]]:
    samples_list: List[List[Tuple[str, str]]] = \
        [samples_from_sentence(sentence, window_width) for sentence in sentences]

    # flatten
    return [item for sublist in samples_list for item in sublist]


def vocabulary_from_wordlists(wordlists: List[List[str]], max_vocabulary_size: int) -> List[str]:
    vocabulary_all: List[str] = [item for sublist in wordlists for item in sublist]

    fdist: FreqDist = FreqDist(vocabulary_all)

    vocabulary_most_common: List[Tuple[str, int]] = fdist.most_common(max_vocabulary_size)

    return [word for (word, freq) in vocabulary_most_common]


def filter_samples_by_vocabulary(samples: List[Tuple[str, str]], vocabulary: List[str]) -> List[Tuple[str, str]]:
    return [(word1, word2) for (word1, word2) in samples if word1 in vocabulary and word2 in vocabulary]
