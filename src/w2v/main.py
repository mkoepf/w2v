# -*- coding: utf-8 -*-
"""
This file is the starting point of thw w2v Word to Vec implementation.
"""

import argparse
import sys
import logging
from typing import List, Tuple

from nltk import download
from nltk.corpus import gutenberg

from w2v import __version__, vectorize, prepare_samples, network

download('gutenberg')
download('punkt')

__author__ = "Michael Köpf"
__copyright__ = "Michael Köpf"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Simple implementation of word2vec.")
    parser.add_argument(
        "--version",
        action="version",
        version="w2v {ver}".format(ver=__version__))
    parser.add_argument(
        "--dim-embedding",
        dest="dim_embed",
        help="dimension of the embedding space",
        type=int,
        default=30,
        metavar="DIM_EMBED")
    parser.add_argument(
        "--dim-input",
        dest="max_vocabulary_size",
        help="DIM_INPUT",
        type=int,
        default=100,
        metavar="DIM_INPUT")
    parser.add_argument(
        "--window-width",
        dest="window_width",
        help="text window within which words are considered part of the same skip-gram",
        type=int,
        default=2,
        metavar="WIN_WIDTH")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def word2vec(dim_embed: int, max_vocabulary_size: int,  window_width: int):
    _logger.info("Max size of vocabulary: {}".format(max_vocabulary_size))
    _logger.info("Window width: {}".format(window_width))

    # Get vocabulary
    all_files: List[str] = gutenberg.fileids()
    wordlists_unprocessed: List[List[str]] = gutenberg.sents(all_files)

    wordlists: List[List[str]] = prepare_samples.preprocess_wordlists(wordlists_unprocessed)

    _logger.info("Number of sentences: {}".format(len(wordlists)))
    _logger.info("Number of words: {}".format(sum([len(cur_list) for cur_list in wordlists])))

    word_pairs: List[Tuple[str, str]] = prepare_samples.samples_from_wordlists(wordlists, window_width)
    _logger.info("Number of word pairs: {}".format(len(word_pairs)))

    vocabulary: List[str] = prepare_samples.vocabulary_from_wordlists(wordlists, max_vocabulary_size)
    _logger.info("Vocabulary size: {}".format(len(vocabulary)))

    word_pairs_filtered: List[Tuple[str, str]] = prepare_samples.filter_samples_by_vocabulary(word_pairs, vocabulary)
    _logger.info("Number of word pairs with both words in vocabulary: {}".format(len(word_pairs_filtered)))

    # Get input and output vectors as column matrices
    (X, Y) = vectorize.training_matrix(word_pairs_filtered, vocabulary)
    _logger.debug("dim(X)=%dx%d dim(Y)=%dx%d", X.shape[0], X.shape[1], Y.shape[0], Y.shape[1])

    model = network.net(len(vocabulary), dim_embed, len(vocabulary))

    network.train_model(model, X, Y)


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.info("Starting ...")

    word2vec(args.dim_embed, args.max_vocabulary_size, args.window_width)

    _logger.info("Done!")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
