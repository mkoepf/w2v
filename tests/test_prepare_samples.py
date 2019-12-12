from w2v.prepare_samples import samples_from_sentence, take_window, vocabulary_from_wordlists, samples_from_wordlists, \
    samples_from_sentences, wordlist_from_sentence, wordlists_from_sentences, filter_samples_by_vocabulary

from typing import List, Tuple


def test_take_window():
    assert take_window(['I'], 0, 1) == []
    assert take_window(['I'], 0, 42) == []
    assert take_window(['I'], 0, 0) == []
    assert take_window(['I'], 42, 1) == []

    assert take_window(['I', 'do'], 0, 0) == []
    assert take_window(['I', 'do'], 0, 1) == ['do']
    assert take_window(['I', 'do'], 0, 2) == ['do']
    assert take_window(['I', 'do'], 0, 42) == ['do']
    assert take_window(['I', 'do'], 1, 0) == []
    assert take_window(['I', 'do'], 1, 1) == ['I']
    assert take_window(['I', 'do'], 1, 2) == ['I']

    assert take_window(['I', 'do', 'not', 'solemnly', 'swear'], 2, 2) == ['I', 'do', 'solemnly', 'swear']
    assert take_window(['I', 'do', 'not', 'solemnly', 'swear'], 1, 2) == ['I', 'not', 'solemnly']
    assert take_window(['I', 'do', 'not', 'solemnly', 'swear'], 0, 2) == ['do', 'not']
    assert take_window(['I', 'do', 'not', 'solemnly', 'swear'], 4, 2) == ['not', 'solemnly']


def test_wordlist_from_sentence():
    assert wordlist_from_sentence('A sentence like no other.') == [
        'A', 'sentence', 'like', 'no', 'other.'
    ]
    assert wordlist_from_sentence("A nice little sentence, isn't it?") == [
        'A', 'nice', 'little', 'sentence,', "isn't", 'it?'
    ]


def test_wordlists_from_sentences():
    assert wordlists_from_sentences([
        'A sentence like no other.',
        "A nice little sentence, isn't it?"
    ]) == [
               ['A', 'sentence', 'like', 'no', 'other.'],
               ['A', 'nice', 'little', 'sentence,', "isn't", 'it?']
           ]


def test_samples_from_sentence():
    assert samples_from_sentence('I do', 1) == [
        ('I', 'do'),
        ('do', 'I')
    ]
    assert samples_from_sentence('I do not swear', 1) == [
        ('I', 'do'),
        ('do', 'I'),
        ('do', 'not'),
        ('not', 'do'),
        ('not', 'swear'),
        ('swear', 'not')
    ]
    assert samples_from_sentence('I do not swear', 2) == [
        ('I', 'do'),
        ('I', 'not'),
        ('do', 'I'),
        ('do', 'not'),
        ('do', 'swear'),
        ('not', 'I'),
        ('not', 'do'),
        ('not', 'swear'),
        ('swear', 'do'),
        ('swear', 'not')
    ]


def test_samples_from_sentences():
    assert samples_from_sentences(
        [
            'I do',
            'I do not swear'
        ], 1) == [
               ('I', 'do'),
               ('do', 'I'),
               ('I', 'do'),
               ('do', 'I'),
               ('do', 'not'),
               ('not', 'do'),
               ('not', 'swear'),
               ('swear', 'not')
           ]


def test_samples_from_wordlists():
    assert samples_from_wordlists(
        [
            ['a', 'b', 'c'],
            ['d', 'e']
        ],
        2
    ) == [
               ('a', 'b'),
               ('a', 'c'),
               ('b', 'a'),
               ('b', 'c'),
               ('c', 'a'),
               ('c', 'b'),
               ('d', 'e'),
               ('e', 'd')
           ]


def test_vocabulary_from_wordlists():
    wordlist1: List[str] = ['This', 'is', 'the', 'first', 'word', 'list']
    wordlist2: List[str] = ['This', 'list', 'is', 'another', 'word', 'list']

    # If max_vocabulary_size >= (number of unique words), all unique words are in vocabulary
    assert set(vocabulary_from_wordlists([wordlist1, wordlist2], 7)) == \
        {'This', 'is', 'the', 'first', 'word', 'list', 'another'}
    assert set(vocabulary_from_wordlists([wordlist1, wordlist2], 1000)) == \
        {'This', 'is', 'the', 'first', 'word', 'list', 'another'}

    # If max_vocabulary_size < (number of unique words), the most frequent words are taken
    assert set(vocabulary_from_wordlists([wordlist1, wordlist2], 4)) == \
        {'This', 'is', 'word', 'list'}
    assert set(vocabulary_from_wordlists([wordlist1, wordlist2], 1)) == \
        {'list'}

    # If several words have the same frequency, they are taken in the sequence of their occurrence.
    # However, this depends on how nltk.FreqDist is implemented.
    assert set(vocabulary_from_wordlists([wordlist1, wordlist2], 2)) == \
        {'list', 'This'}


def test_filter_samples_by_vocabulary():
    vocabulary: List[str] = ['a', 'b', 'c']
    samples: List[Tuple[str, str]] = \
        [('a', 'a'), ('a', 'b'), ('b', 'c'), ('a', 'd'), ('d', 'a'), ('x', 'y'), ('b', 'c')]
    assert filter_samples_by_vocabulary(samples, vocabulary) == \
        [('a', 'a'), ('a', 'b'), ('b', 'c'), ('b', 'c')]
