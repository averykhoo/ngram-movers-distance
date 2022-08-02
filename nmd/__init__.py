"""
todo: write a useful docstring
"""

__version__ = '0.0.4'
__description__ = 'Yet another string matching algorithm'

from nmd.nmd import ngram_movers_distance
from nmd.nmd_bow import bow_ngram_movers_distance
from nmd.nmd_index import WordList

__all__ = [
    ngram_movers_distance,
    bow_ngram_movers_distance,
    WordList,
]
