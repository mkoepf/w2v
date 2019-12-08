from w2v.prepare_samples import samples_from_sentence, take_window, vocabulary_from_wordlists, samples_from_wordlists


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
    assert set(vocabulary_from_wordlists(
        [
            ['This', 'is', 'the', 'first', 'word', 'list'],
            ['This', 'is', 'another', 'word', 'list']
        ])) == {'This', 'is', 'the', 'first', 'word', 'list', 'another'}
