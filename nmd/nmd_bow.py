from typing import Iterable
from typing import Union

import scipy.optimize

from nmd.nmd import ngram_movers_distance


def bow_ngram_movers_distance(bag_of_words_1: Union[str, Iterable[str]],
                              bag_of_words_2: Union[str, Iterable[str]],
                              n: int = 2,
                              invert: bool = False,
                              normalize: bool = False,
                              ) -> float:
    """
    calculates the n-gram mover's distance between two bags of words (for some specified n)
    case-sensitive by default, so lowercase/casefold the input words for case-insensitive results

    :param bag_of_words_1: a list of strings
    :param bag_of_words_2: another list of strings
    :param n: number of chars per n-gram (default 2)
    :param invert: return similarity instead of distance
    :param normalize: normalize to a score from 0 to 1 (inclusive of 0 and 1)
    :return: n-gram mover's distance, possibly inverted and/or normalized
    """

    # convert to list
    bag_of_words_1 = list(bag_of_words_1)  # rows
    bag_of_words_2 = list(bag_of_words_2)  # columns

    # optimize cost matrix using EMD
    costs = []
    for word_1 in bag_of_words_1:
        row = []
        for word_2 in bag_of_words_2:
            try:
                row.append(ngram_movers_distance(word_1, word_2, n=n, normalize=True))
            except ZeroDivisionError:
                row.append(int(word_1 != word_2))
        costs.append(row)
    if costs:
        row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(costs)  # 1D equivalent of EMD
    else:
        row_idxs, col_idxs = [], []

    # sum
    if invert:
        out = min(len(bag_of_words_1), len(bag_of_words_2))
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            out -= costs[row_idx][col_idx]
    else:
        out = abs(len(bag_of_words_1) - len(bag_of_words_2))
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            out += costs[row_idx][col_idx]

    if normalize:
        out /= max(len(bag_of_words_1), len(bag_of_words_2))

    return out
