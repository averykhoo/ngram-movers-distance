import itertools
from typing import List
from typing import Sequence
from typing import Union


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


def emd_1d_faster(locations_x: Sequence[Union[int, float]],
                  locations_y: Sequence[Union[int, float]],
                  ) -> float:
    """
    kind of like earth mover's distance
    but positions are limited to within the unit interval
    and must be quantized
    """

    # all inputs must be in the unit interval
    assert all(0 <= x <= 1 for x in locations_x)
    assert all(0 <= x <= 1 for x in locations_y)

    # locations_1 will be the longer list
    if len(locations_x) < len(locations_y):
        locations_x, locations_y = locations_y, locations_x

    # empty list, so just count the l1 items and exit early
    if len(locations_y) == 0:
        return float(len(locations_x))

    # only one item, so take min distance and count the rest of the l1 items
    if len(locations_y) == 1:
        return float(min(abs(l1 - locations_y[0]) for l1 in locations_x) + len(locations_x) - 1)

    # make a COPY of the list, sorted in reverse (descending order)
    # we'll be modifying in-place later, and we don't want to update the input
    locations_x = sorted(locations_x)
    locations_y = sorted(locations_y)

    # last chance to early exit
    if len(locations_x) == len(locations_y):
        return float(sum(abs(x - y) for x, y in zip(locations_x, locations_y)))

    # accumulate distance as we simplify the problem
    acc = 0.0

    # remove any matching points in x and y
    # reverses the list (converts ascending -> descending)
    new_x = []
    new_y = []
    while locations_x and locations_y:
        if locations_x[-1] > locations_y[-1]:
            new_x.append(locations_x.pop(-1))
        elif locations_y[-1] > locations_x[-1]:
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

    # there shouldn't be any duplicates across both lists now
    # there can be duplicates within each list, but that's okay

    # greedy-match constrained points with only one possible match (at the smaller end of locations_y)
    while locations_y and locations_x:
        if locations_y[-1] <= locations_x[-1]:
            acc += locations_x.pop(-1) - locations_y.pop(-1)

        # this must be < and not <= if there are duplicates in each list, otherwise it removes the wrong pairs of points
        elif len(locations_x) >= 2 and abs(locations_y[-1] - locations_x[-1]) < abs(locations_x[-2] - locations_y[-1]):
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
        elif len(locations_x) >= 2 and abs(locations_x[-1] - locations_y[-1]) < abs(locations_y[-1] - locations_x[-2]):
            acc += locations_x.pop(-1) - locations_y.pop(-1)
        else:
            break

    # another chance to early exit
    if len(locations_y) == 0:
        return acc + len(locations_x)
    if len(locations_y) == 1:
        return acc + min(abs(x - locations_y[0]) for x in locations_x) + len(locations_x) - 1

    # merge the lists for now to find sets of possibly paired points without actually building a bipartite graph
    locations = sorted([(loc, False) for loc in locations_x] + [(loc, True) for loc in locations_y])
    component_ranges = []

    # get ranges of forward alignments
    n = 0
    current_left = None
    for idx, (loc, is_y) in enumerate(locations):
        if is_y:
            n += 1
            if current_left is None:
                current_left = idx
        elif n > 0:
            n -= 1
            if n == 0:
                component_ranges.append((current_left, idx))
                current_left = None
    if current_left is not None:  # current_left could be 0, so don't just test truthiness
        component_ranges.append((current_left, len(locations) - 1))

    # get ranges of backward alignments
    n = 0
    current_right = None
    for idx in range(len(locations) - 1, -1, -1):
        if locations[idx][1]:  # if is_y:
            n += 1
            if current_right is None:
                current_right = idx
        elif n > 0:
            n -= 1
            if n == 0:
                component_ranges.append((idx, current_right))
                current_right = None
    if current_right is not None:
        component_ranges.append((0, current_right))

    # merge ranges to get the sets of connected components
    # [x1 y1 x2 x3 x4 y2 x3] ==> [x1 y1 x2], [x4 y2 x5] (x3 can never be matched)
    component_ranges = sorted(component_ranges, reverse=True)
    last_seen = -1
    while component_ranges:
        left, right = component_ranges.pop(-1)
        while component_ranges and component_ranges[-1][0] <= right:
            right = max(right, component_ranges.pop(-1)[1])

        # count unmatched points since last seen
        if left > last_seen + 1:
            acc += left - last_seen - 1  # count unmatchable points

        # split into x and y lists again
        connected_x = [idx for idx, is_y in locations[left:right + 1] if not is_y]
        connected_y = [idx for idx, is_y in locations[left:right + 1] if is_y]

        # todo: greedy match endpoints again?

        # todo: try to greedy-match unshared points?
        # must match all points in this component
        # [x1 y1 ... x2 ...]       ==> if x1y1 < y1x2, then y1 -> x1
        # [... x3 x4 y1 x5 x6 ...] ==> y1 can only match x4 or x5 (assuming there are no y-chains)
        # if it succeeds, then remove the component

        # todo: actually build a bipartite graph to exclude impossible match options?

        # enumerate all possible matches for this connected component
        # this code block works even if connected_y is empty
        min_cost = len(connected_y)
        for x_combination in itertools.combinations(connected_x, len(connected_y)):
            min_cost = min(min_cost, sum(abs(x - y) for x, y in zip(x_combination, connected_y)))
        acc += min_cost + len(connected_x) - len(connected_y)

        # update last seen
        last_seen = right

    # count unmatched points after last seen
    if len(locations) > last_seen + 1:
        acc += len(locations) - last_seen - 1

    return acc


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
    answer_fast = emd_1d_faster(locations_x, locations_y)
    answer_slow = emd_1d_slow_v2(locations_x, locations_y)
    assert abs(answer_fast - answer_slow) < 0.00001, (answer_slow, answer_fast, locations_x, locations_y)
    return answer_slow


if __name__ == '__main__':
    emd_1d([0.0, 0.25], [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.42857142857142855])

    print(n_gram_emd('aabbbbbbbbaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(n_gram_emd('aaaabbbbbbbbaaaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(n_gram_emd('banana', 'bababanananananananana'))
    print(n_gram_emd('banana', 'bababanananananananananna'))
    print(n_gram_emd('banana', 'nanananananabababa'))
    print(n_gram_emd('banana', 'banana'))
    print(n_gram_emd('nanananananabababa', 'banana'))
    print(n_gram_emd('banana', 'bababananananananananannanananananananana'))
    print(n_gram_emd('banana', 'bababananananananananannananananananananananananananannanananananananana'))
    print(n_gram_emd('bananabababanana', 'bababananananananananannananananananananananananananannananabanananananana'))

    # num_x = 3
    # num_y = 7
    #
    # xs = [i / (num_x - 1) for i in range(num_x)]
    # ys = [i / (num_y - 1) for i in range(num_y)]
    # print(xs)
    # print(ys)
    # xs = xs + xs + xs + xs
    #
    # for x_len in range(len(xs) + 1):
    #     for y_len in range(len(ys) + 1):
    #         print(x_len, y_len)
    #         for x_combi in itertools.combinations(xs, x_len):
    #             for y_combi in itertools.combinations(ys, y_len):
    #                 assert abs(emd_1d(x_combi, y_combi) - emd_1d(y_combi, x_combi)) < 0.0001, (x_combi, y_combi)
