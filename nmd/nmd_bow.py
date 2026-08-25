from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import scipy.optimize

from nmd.emd_1d import emd_1d_fast


def _n_gram_locations(word: str, n: int) -> Tuple[Dict[str, List[float]], int]:
    """
    n_gram -> [normalized position, ...] for one word, plus its total n-gram count

    this is the same decomposition `ngram_movers_distance` does internally; it is split out
    here so that a word's n-grams can be built once and reused across a whole matrix row
    """
    padded = f'\2{word}\3'
    num_grams = len(padded) - n + 1
    denominator = max(1, num_grams - 1)
    locations: Dict[str, List[float]] = dict()
    for idx in range(num_grams):
        locations.setdefault(padded[idx:idx + n], []).append(idx / denominator)
    return locations, num_grams


def _similarity(locations_1: Dict[str, List[float]],
                locations_2: Dict[str, List[float]],
                ) -> float:
    """
    summed n-gram overlap of two words, i.e. the inverted (un-normalized) nmd

    identical to the similarity loop in `ngram_movers_distance`, except that it takes
    pre-computed n-gram locations and walks whichever side has fewer distinct n-grams
    """
    if len(locations_1) > len(locations_2):
        locations_1, locations_2 = locations_2, locations_1
    get_2 = locations_2.get
    similarity = 0.0
    for n_gram, positions_1 in locations_1.items():
        positions_2 = get_2(n_gram)
        if positions_2 is not None:
            similarity += len(positions_1) + len(positions_2) - emd_1d_fast(positions_1, positions_2)
    return similarity


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
    :param invert: return similarity instead of difference
    :param normalize: normalize to a score from 0 to 1 (inclusive of 0 and 1)
    :return: n-gram mover's distance, possibly inverted and/or normalized
    """

    # convert to list
    bag_of_words_1 = list(bag_of_words_1)  # rows
    bag_of_words_2 = list(bag_of_words_2)  # columns

    # split each DISTINCT word into n-grams exactly once
    # calling ngram_movers_distance per cell would redo this for both words of every pair,
    # which is len(bag_1) * len(bag_2) splits where len(bag_1) + len(bag_2) will do
    # (repeated words share an entry too, so duplicate rows and columns cost nothing extra)
    n_grams_by_word: Dict[str, Tuple[Dict[str, List[float]], int]] = dict()
    for word in bag_of_words_1:
        if word not in n_grams_by_word:
            n_grams_by_word[word] = _n_gram_locations(word, n)
    for word in bag_of_words_2:
        if word not in n_grams_by_word:
            n_grams_by_word[word] = _n_gram_locations(word, n)

    prepared_2 = [(word, *n_grams_by_word[word]) for word in bag_of_words_2]

    # optimize cost matrix using EMD
    costs = []
    for word_1 in bag_of_words_1:
        locations_1, num_grams_1 = n_grams_by_word[word_1]
        row = []
        for word_2, locations_2, num_grams_2 in prepared_2:
            total_grams = num_grams_1 + num_grams_2
            if total_grams == 0:
                # both words are shorter than the n-grams asked for, so there is nothing to
                # compare; this is the ZeroDivisionError branch of the previous version
                row.append(int(word_1 != word_2))
            else:
                # normalized distance == 1 - normalized similarity
                row.append(1.0 - _similarity(locations_1, locations_2) / total_grams)
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
        # both bags empty means there is nothing to normalize against
        out /= max(1, len(bag_of_words_1), len(bag_of_words_2))

    return out
