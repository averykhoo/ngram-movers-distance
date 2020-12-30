import itertools
from functools import lru_cache
from typing import List
from typing import Tuple


def n_gram_emd(word_1: str, word_2: str, n: int = 2):
    """
    optimized for readability, not speed
    test cases: https://www.watercoolertrivia.com/blog/schwarzenegger
    """

    assert isinstance(word_1, str) and '\2' not in word_1 and '\3' not in word_1
    assert isinstance(word_2, str) and '\2' not in word_2 and '\3' not in word_2
    assert isinstance(n, int) and n >= 2

    n_grams_1 = [f'\2{word_1}\3'[i:i + n] for i in range(len(word_1) - n + 3)]
    n_grams_2 = [f'\2{word_2}\3'[i:i + n] for i in range(len(word_2) - n + 3)]

    n_gram_locations_1 = dict()
    for idx, n_gram in enumerate(n_grams_1):
        n_gram_locations_1.setdefault(n_gram, []).append(idx / (len(n_grams_1) - 1))

    n_gram_locations_2 = dict()
    for idx, n_gram in enumerate(n_grams_2):
        n_gram_locations_2.setdefault(n_gram, []).append(idx / (len(n_grams_2) - 1))

    distance = 0
    total = 0
    for n_gram, locations in n_gram_locations_1.items():
        total += len(locations)
        if n_gram not in n_gram_locations_2:
            distance += len(locations)
    for n_gram, locations in n_gram_locations_2.items():
        total += len(locations)
        if n_gram not in n_gram_locations_1:
            distance += len(locations)
        else:
            print(n_gram, locations, n_gram_locations_1[n_gram])
            distance += emd_1d(locations, n_gram_locations_1[n_gram])

    return distance, total


@lru_cache(maxsize=0xFFFF)
def _emd_1d(locations_1: Tuple[float], locations_2: Tuple[float]) -> float:
    if len(locations_1) == len(locations_2):
        return sum(abs(l1 - l2) for l1, l2 in zip(locations_1, locations_2))

    elif len(locations_2) == 1:
        return len(locations_1) - 1 + min(abs(l1 - locations_2[0]) for l1 in locations_1)

    else:
        # noinspection PyTypeChecker
        return 1 + min(_emd_1d(locations_1[:i] + locations_1[i + 1:], locations_2) for i in range(len(locations_1)))


def emd_1d_fast(locations_x: List[float], locations_y: List[float]) -> float:
    """
    distance needed to move
    todo: optimize worst case
    """

    # all inputs must be in the unit interval
    assert all(0 <= x <= 1 for x in locations_x)
    assert all(0 <= x <= 1 for x in locations_y)

    # in our use case, there should be no duplicates in each list
    assert len(locations_x) == len(set(locations_x))
    assert len(locations_y) == len(set(locations_y))

    # locations_1 will be the longer list
    if len(locations_x) < len(locations_y):
        locations_x, locations_y = locations_y, locations_x

    # empty list, so just count the l1 items and exit early
    if len(locations_y) == 0:
        return len(locations_x)

    # only one item, so take min distance and count the rest of the l1 items
    if len(locations_y) == 1:
        return min(abs(l1 - locations_y[0]) for l1 in locations_x) + len(locations_x) - 1

    # make a COPY of the list, sorted in reverse (descending order)
    # we'll be modifying in-place later, and we don't want to update the input
    locations_x = sorted(locations_x, reverse=True)
    locations_y = sorted(locations_y, reverse=True)

    # accumulated distance as we simplify the problem
    acc = 0

    # greedy-match constrained points with only one possible match (at the smaller end of locations_y)
    while locations_y and locations_x:
        if locations_y[-1] <= locations_x[-1]:
            acc += locations_x.pop(-1) - locations_y.pop(-1)
        elif len(locations_x) >= 2 and (locations_y[-1] - locations_x[-1]) <= (locations_x[-2] - locations_y[-1]):
            acc += locations_y.pop(-1) - locations_x.pop(-1)
        else:
            break

    # reverse both lists IN PLACE, so now they are sorted in ascending order
    locations_x.reverse()
    locations_y.reverse()

    # greedy-match constrained points with only one possible match (at the larger end of locations_y)
    while locations_y and locations_x:
        if locations_y[-1] >= locations_x[-1]:
            acc += locations_y.pop(-1) - locations_x.pop(-1)
        elif len(locations_x) >= 2 and (locations_x[-1] - locations_y[-1]) <= (locations_y[-1] - locations_x[-2]):
            acc += locations_x.pop(-1) - locations_y.pop(-1)
        else:
            break

    # remove any matching points in x and y
    new_x = []
    new_y = []
    locations_x.reverse()
    locations_y.reverse()
    while locations_x and locations_y:
        if locations_x[-1] < locations_y[-1]:
            new_x.append(locations_x.pop(-1))
        elif locations_x[-1] > locations_y[-1]:
            new_y.append(locations_y.pop(-1))
        else:
            # discard duplicate
            locations_x.pop(-1)
            locations_y.pop(-1)
    if locations_x:
        locations_x.reverse()
        new_x.extend(locations_x)
    if locations_y:
        locations_y.reverse()
        new_y.extend(locations_y)
    locations_x = new_x
    locations_y = new_y

    # another chance to early exit
    if len(locations_y) == 0:
        return acc + len(locations_x)
    if len(locations_y) == 1:
        return acc + min(abs(x - locations_y[0]) for x in locations_x) + len(locations_x) - 1

    # todo: build the bipartite graph
    # backward and forward pass

    # todo: split into connected components
    # this will remove all unmatchable points from the graph
    # [x1 y1 x2 x3 x4 y2 x3] ==> [x1 y1 x2], [x4 y2 x5] (x3 can never be matched)

    # todo: try to greedy-match unshared points for each component
    # [x1 y1 ... x2 ...]       ==> if x1y1 < y1x2, then y1 -> x1
    # [... x3 x4 y1 x5 x6 ...] ==> y1 can only match x4 or x5 (assuming there are no y-chains)
    # if it succeeds, then remove the component

    # enumerate the options instead of recursing
    acc += len(locations_x) - len(locations_y)
    min_cost = len(locations_y)
    for x_combination in itertools.combinations(locations_x, len(locations_y)):
        min_cost = min(min_cost, sum(abs(x - y) for x, y in zip(x_combination, locations_y)))
    return acc + min_cost


def emd_1d_slow(locations_x: List[float], locations_y: List[float]) -> float:
    if len(locations_x) < len(locations_y):
        return emd_1d_slow(locations_y, locations_x)

    if len(locations_x) == len(locations_y):
        return sum(abs(l1 - l2) for l1, l2 in zip(sorted(locations_x), sorted(locations_y)))

    return 1 + min(emd_1d_slow(locations_x[:i] + locations_x[i + 1:], locations_y) for i in range(len(locations_x)))


def emd_1d(locations_x: List[float], locations_y: List[float]) -> float:
    answer_1 = emd_1d_slow(locations_x, locations_y)
    answer_2 = emd_1d_fast(locations_x, locations_y)
    assert abs(answer_1 - answer_2) < 0.00001, (answer_2, answer_1)
    return answer_2
