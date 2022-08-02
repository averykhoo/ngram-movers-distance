"""
A string similarity measure based on the Earth Mover's Distance
"""

__version__ = '0.0.5'

from nmd.nmd import ngram_movers_distance
from nmd.nmd_index import WordList

__all__ = [
    ngram_movers_distance,
    WordList,
]
