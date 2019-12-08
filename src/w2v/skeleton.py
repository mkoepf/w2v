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

import argparse
import sys
import logging
from typing import List, Tuple

from books.alice_in_wonderland import alice_in_wonderland

from w2v import __version__, vectorize, prepare_samples, network

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
        dest="dim_embed",
        help="dimension of the embedding space",
        type=int,
        metavar="DIM_EMBED")
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


def word2vec(dim_embed: int):
    # Get vocabulary
    vocabulary: List[str] = prepare_samples.vocabulary_from_wordlists(alice_in_wonderland)
    _logger.debug("Vocabulary size: %d", len(vocabulary))

    # Get word pairs
    word_pairs: List[Tuple[str, str]] = prepare_samples.samples_from_wordlists(alice_in_wonderland, 1)
    _logger.debug("Number of word pairs: %d", len(word_pairs))

    # Get input and output vectors as column matrices
    (X, Y) = vectorize.training_matrix(word_pairs, vocabulary)
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

    word2vec(args.dim_embed)

    _logger.info("Done!")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
