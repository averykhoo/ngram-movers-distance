import itertools
from typing import List
from typing import Sequence
from typing import Union


def speed_test(word_1: str, word_2: str):
    edit_distance(word_1, word_2)
    damerau_levenshtein_distance(word_1, word_2)

    return n_gram_emd(word_1, word_2)


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
    return answer_fast


def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def damerau_levenshtein_distance(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> damerau_levenshtein_distance('ba', 'abc')
    2
    >>> damerau_levenshtein_distance('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> damerau_levenshtein_distance('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


if __name__ == '__main__':

    num_x = 4
    num_y = 7

    xs = [i / (num_x - 1) for i in range(num_x)]
    ys = [i / (num_y - 1) for i in range(num_y)]
    print(xs)
    print(ys)
    xs = xs + xs + xs

    for x_len in range(len(xs) + 1):
        for y_len in range(len(ys) + 1):
            print(x_len, y_len)
            for x_combi in itertools.combinations(xs, x_len):
                for y_combi in itertools.combinations(ys, y_len):
                    assert abs(emd_1d(x_combi, y_combi) - emd_1d(y_combi, x_combi)) < 0.0001, (x_combi, y_combi)

    print(speed_test('aabbbbbbbbaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(speed_test('aaaabbbbbbbbaaaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(speed_test('banana', 'bababanananananananana'))
    print(speed_test('banana', 'bababanananananananananna'))
    print(speed_test('banana', 'nanananananabababa'))
    print(speed_test('banana', 'banana'))
    print(speed_test('nanananananabababa', 'banana'))
    print(speed_test('banana', 'bababananananananananannanananananananana'))
    print(speed_test('banana', 'bababananananananananannananananananananananananananannanananananananana'))
    print(speed_test('bananabababanana', 'bababananananananananannananananananananananananananannananabanananananana'))

    a = [
        'Schwartzenegger',
        'Schwarzeneger',
        'Schwarzenager',
        'Schwartzenager',
        'Schwartzeneger',
        'Schwarzeneggar',
        'Schwarzenneger',
        'Swartzenegger',
        'Swarzenegger',
        'Schwarzenagger',
        'Schwarznegger',
        'Swartzenager',
        'Schwarzanegger',
        'Shwarzenegger',
        'Schwartzenagger',
        'Swartzeneger',
        'Schwartznegger',
        'Schwarzenegar',
        'Shwartzenegger',
        'Schwarzennegger',
        'Schwarzennager',
        'Schwartzanegger',
        'Schwartzenneger',
        'Schwarzanager',
        'Schwarzengger',
        'Schwarzennegar',
        'Shwartzeneger',
        'Schwartzeneggar',
        'Schwarzneger',
        'Schwarzneggar',
        'Schwartzenegar',
        'Schwartzneger',
        'Schwazenegger',
        'Shwartzenager',
        'Swartzanegger',
        'Swarzeneger',
        'Swarzeneggar',
        'Schwarenegger',
        'Schwartzennager',
        'Schwartzneggar',
        'Shwarzeneger',
        'Swartzeneggar',
        'Swartznegger',
        'Swarzenager',
        'Swarzenagger',
        'Scharzenegger',
        'Schwarnegger',
        'Schwartnegger',
        'Schwartzanager',
        'Schwartzaneger',
        'Schwartzinager',
        'Schwarzzenager',
        'Shwarzenager',
        'Swartzenagger',
        'Swartzineger',
        'Scharzeneger',
        'Schwarnzenegger',
        'Schwartenager',
        'Schwartenegar',
        'Schwarteneger',
        'Schwartnegar',
        'Schwartzanegar',
        'Schwartzenger',
        'Schwartzenggar',
        'Schwartzineger',
        'Schwartznager',
        'Schwarzaneger',
        'Schwarzaneggar',
        'Schwarzanger',
        'Schwarzenaeger',
        'Schwarzeniger',
        'Schwarzinager',
        'Schwarznager',
        'Schwarztenegger',
        'Schwarzzeneger',
        'Schwarzzenegger',
        'Schwazenager',
        'Schwazeneger',
        'Scwartzenegger',
        'Scwarzenegger',
        'Shwartznegger',
        'Shwarzenegar',
        'Swarteneger',
        'Swartzanager',
        'Swartznager',
        'Swartzneger',
        'Swarzanegger',
        'Swarzennager',
        'Swarzenneger',
        'Swazeneger',
        'Schartzenager',
        'Schartzennager',
        'Schartznager',
        'Scharwzeneger',
        'Scharzenager',
        'Schawarzneneger',
        'Schawrknegger',
        'Schazenegger',
        'Schneckenger',
        'Schrarznegger',
        'Schrawzenneger',
        'Schrwazeneggar',
        'Schrwazenegger',
        'Schrwtzanagger',
        'Schsargdneger',
        'Schwaranagger',
        'Schwararzenegger',
        'Schwarezenegger',
        'Schwarganzer',
        'Schwarnznegar',
        'Schwarsanegger',
        'Schwarsenagger',
        'Schwarsnegger',
        'Schwartaneger',
        'Schwartenagger',
        'Schwartenegger',
        'Schwartenneger',
        'Schwarterneger',
        'Schwartineger',
        'Schwartnager',
        'Schwartnehar',
        'Schwartsaneger',
        'Schwartsinager',
        'Schwartzaneggar',
        'Schwartzanger',
        'Schwartzeiojaweofjaweneger',
        'Schwartzenagar',
        'Schwartzenegget',
        'Schwartzeneiger',
        'Schwartzengar',
        'Schwartzenkangaroo',
        'Schwartzennegar',
        'Schwartzinagger',
        'Schwartzinegar',
        'Schwartziniger',
        'Schwartznagger',
        'Schwartznegar',
        'Schwarz',
        'Schwarzamegger',
        'Schwarzanagger',
        'Schwarzatwizzler',
        'Schwarzeggar',
        'Schwarzegger',
        'Schwarzenaega',
        'Schwarzenagher',
        'Schwarzeneeger',
        'Schwarzenegor',
        'Schwarzenenergy',
        'Schwarzengeggar',
        'Schwarzgenar',
        'Schwarzinagger',
        'Schwarzineggar',
        'Schwarztenegar',
        'Schwarzzanager',
        'Schwatzeneggar',
        'Schwatzenneger',
        'Schwazenaeger',
        'Schwazenegrr',
        'Schwazerneger',
        'Schwazinager',
        'Schwaznagger',
        'Schwazneger',
        'Schwaznnager',
        'Schwazzeneger',
        'Schwazzenger',
        'Schwazzinager',
        'Schzwarnegger',
        'Scwarrzenegger',
        'Scwarzenager',
        'Scwarzeneggar',
        'Scwarzenneger',
        'Scwharzanegger',
        'Scwharzeneggar',
        'Shwarsneger',
        'Shwartaneger',
        'Shwarteneger',
        'Shwartinznegar',
        'Shwartnierger',
        'Shwartsnagger',
        'Shwartzanager',
        'Shwartzanegar',
        'Shwartzaneger',
        'Shwartzanegger',
        'Shwartzenagor',
        'Shwartzeneggar',
        'Shwartzengar',
        'Shwartzennegar',
        'Shwartzganeger',
        'Shwartznager',
        'Shwartzneger',
        'Shwarzanegger',
        'Shwarzenagger',
        'Shwarzenneger',
        'Shwarznager',
        'Shwaztsinager',
        'Swarchneger',
        'Swarchzinager',
        'Swarchznegger',
        'Swartenager',
        'Swartenegger',
        'Swartenzager',
        'Swartiznager',
        'Swartschenager',
        'Swartseneger',
        'Swartseneggar',
        'Swartsenenger',
        'Swartshanaiger',
        'Swarttenegger',
        'Swartz.',
        'Swartzanagger',
        'Swartzaneger',
        'Swartzanegga',
        'Swartzeigner',
        'Swartzenagar',
        'Swartzeneagar',
        'Swartzenegar',
        'Swartzenegher',
        'Swartzengger',
        'Swartzennager',
        'Swartzennegar',
        'Swartzenneger',
        'Swartzerniger',
        'Swartzinager',
        'Swartzineggar',
        'Swartznagger',
        'Swartznegar',
        'Swartzneggar',
        'Swarzenaeger',
        'Swarzenaggar',
        'Swarzenaider',
        'Swarzengger',
        'Swarzneger',
        'Swarznegger',
        'Swarzshnegger',
        'Swarzzeneggar',
        'Swarzzenegger',
        'Swatgnezzer',
        'Swatz..',
        'Swatzinagger',
        'Swazenegger',
        'Swazernager',
        'Swchwartzignegeridknga',
        'Swchwazaneger',
        'Swertizager',
        'Swertzeneggar',
        'Swhartznegar',
        'Switzenagger',
        'Swiztinager',
        'Swuartzenegar',
        'Schwartzanagger',
        'Schwartzennnnnnn',
        'Schwarzenger',
        'Swartasenegger',
        'Swazenegar',
    ]
    b = 'Schwarzenegger'
    for aa in a:
        speed_test(aa, b)
