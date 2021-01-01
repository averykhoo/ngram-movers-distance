import itertools
from typing import List


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
            # print(n_gram, locations, n_gram_locations_1[n_gram])
            distance += emd_1d(locations, n_gram_locations_1[n_gram])

    return distance, total


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
    # todo: do this before removing endpoints
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

    # there shouldn't be any duplicates across both lists now
    assert len(locations_x) + len(locations_y) == len(set(locations_x + locations_y))

    # enumerate the options instead of recursing
    # todo: actually build the bipartite graph to exclude impossible match options?
    acc += len(locations_x) - len(locations_y)
    min_cost = len(locations_y)
    for x_combination in itertools.combinations(locations_x, len(locations_y)):
        min_cost = min(min_cost, sum(abs(x - y) for x, y in zip(x_combination, locations_y)))
    return acc + min_cost


def emd_1d_fast_v2(locations_x: List[float], locations_y: List[float]) -> float:
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

    # enumerate the options instead of recursing
    min_cost = len(locations_y)
    for x_combination in itertools.combinations(locations_x, len(locations_y)):
        min_cost = min(min_cost, sum(abs(x - y) for x, y in zip(x_combination, locations_y)))
    return len(locations_x) - len(locations_y) + min_cost


def emd_1d_faster(locations_x: List[float], locations_y: List[float]) -> float:
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
    # todo: do this before removing endpoints
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

    # there shouldn't be any duplicates across both lists now
    assert len(locations_x) + len(locations_y) == len(set(locations_x + locations_y))

    # merge the lists for now to find sets of possibly paired points without actually building a bipartite graph
    locations = sorted([(loc, False) for loc in locations_x] + [(loc, True) for loc in locations_y])
    component_ranges = []
    n = 0
    current_left = None
    current_right = None

    # get ranges of forward alignments
    for idx, (loc, is_y) in enumerate(locations):
        if not is_y:
            n += 1
        if n:
            if current_left is None:
                current_left = idx
                current_right = idx
            else:
                current_right = idx
        if is_y:
            n -= 1
        if not n:
            component_ranges.append((current_left, current_right))
            current_left = None
            current_right = None
    if current_left is not None:
        component_ranges.append((current_left, current_right))
        current_left = None
        current_right = None

    # get ranges of backward alignments
    for idx in range(len(locations) - 1, -1, -1):
        loc, is_y = locations[idx]
        if not is_y:
            n += 1
        if n:
            if current_right is None:
                current_right = idx
                current_left = idx
            else:
                current_left = idx
        if is_y:
            n -= 1
        if not n:
            component_ranges.append((current_left, current_right))
            current_right = None
            current_left = None
    if current_right is not None:
        component_ranges.append((current_left, current_right))

    # merge ranges to get the sets of connected components
    # [x1 y1 x2 x3 x4 y2 x3] ==> [x1 y1 x2], [x4 y2 x5] (x3 can never be matched)
    component_ranges = sorted(component_ranges, reverse=True)
    last_seen = -1
    while component_ranges:
        left, right = component_ranges.pop(-1)
        while component_ranges and component_ranges[-1][0] < right:
            _, right = component_ranges.pop(-1)

        # count unmatched points since last seen
        if left > last_seen + 1:
            acc += left - last_seen - 1  # count unmatchable points

        # split into x and y lists again
        connected_x = [idx for idx, is_y in locations[left:right + 1] if not is_y]
        connected_y = [idx for idx, is_y in locations[left:right + 1] if is_y]

        # todo: greedy match endpoints (again)?

        # todo: try to greedy-match unshared points (must match all points in this component)?
        # [x1 y1 ... x2 ...]       ==> if x1y1 < y1x2, then y1 -> x1
        # [... x3 x4 y1 x5 x6 ...] ==> y1 can only match x4 or x5 (assuming there are no y-chains)
        # if it succeeds, then remove the component

        # enumerate the options instead of recursing
        # todo: actually build the bipartite graph to exclude impossible match options?
        acc += len(connected_x) - len(connected_y)
        min_cost = len(connected_y)
        for x_combination in itertools.combinations(connected_x, len(connected_y)):
            min_cost = min(min_cost, sum(abs(x - y) for x, y in zip(x_combination, connected_y)))
        acc += min_cost

        # update last seen
        last_seen = right

    # count unmatched points after last seen
    if len(locations) > last_seen + 1:
        acc += len(locations) - last_seen - 1

    return acc


def emd_1d_slow(locations_x: List[float], locations_y: List[float]) -> float:
    if len(locations_x) < len(locations_y):
        return emd_1d_slow(locations_y, locations_x)

    if len(locations_x) == len(locations_y):
        return sum(abs(l1 - l2) for l1, l2 in zip(sorted(locations_x), sorted(locations_y)))

    return 1 + min(emd_1d_slow(locations_x[:i] + locations_x[i + 1:], locations_y) for i in range(len(locations_x)))


def emd_1d_slow_v2(locations_x: List[float], locations_y: List[float]) -> float:
    if len(locations_x) < len(locations_y):
        locations_x, locations_y = locations_y, locations_x

    locations_x = sorted(locations_x)
    locations_y = sorted(locations_y)

    min_cost = len(locations_y)
    for x_combination in itertools.combinations(locations_x, len(locations_y)):
        min_cost = min(min_cost, sum(abs(x - y) for x, y in zip(x_combination, locations_y)))
    return len(locations_x) - len(locations_y) + min_cost


def emd_1d(locations_x: List[float], locations_y: List[float]) -> float:
    answer_1 = emd_1d_fast_v2(locations_x, locations_y)
    answer_2 = emd_1d_slow_v2(locations_x, locations_y)
    assert abs(answer_1 - answer_2) < 0.00001, (answer_2, answer_1)
    return answer_2


if __name__ == '__main__':

    # print(n_gram_emd('aaaabbbbbbbbaaaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(n_gram_emd('banana', 'bababanananananananana'))
    print(n_gram_emd('banana', 'bababanananananananananna'))
    print(n_gram_emd('banana', 'nanananananabababa'))
    print(n_gram_emd('banana', 'banana'))
    print(n_gram_emd('nanananananabababa', 'banana'))
    print(n_gram_emd('banana', 'bababananananananananannanananananananana'))
    print(n_gram_emd('banana', 'bababananananananananannananananananananananananananannanananananananana'))
    print(n_gram_emd('bananabababanana', 'bababananananananananannananananananananananananananannananabanananananana'))
