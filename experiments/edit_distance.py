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


def extended_edit_distance(hyp: str,
                           ref: str,
                           deletion: float = 0.2,
                           insertion: float = 1.0,
                           substitution: float = 1.0,
                           jump: float = 2.0,
                           rho: float = 0.3,
                           debug: bool = False,
                           ) -> float:
    """
    Extended Edit Distance (https://www.aclweb.org/anthology/W19-5359)
    Calculates the (non-symmetric) EED from hypothesis to reference string
    https://github.com/averykhoo/extended-edit-distance/blob/master/EED.py

    Note: EED builds on CDER (https://www.aclweb.org/anthology/E06-1031)
    with default settings, the algorithm seems to:
    - skip the first char in each hyp word, even if it matches the ref
    - repeat the last hyp character until it sees a space in the ref
    - but it doesn't skip the first char if there's no jump
    - the c code stores the entire DP table for no reason
    other notes:
    - skipping the first char seems to be solved by setting jump cost to 0.9
    - setting jump cost higher than insertion cost tends to insert space as first word char
    - setting jump cost higher than substitution cost tends to insert random chars instead
    - rho changes nothing since it's computed at the end
    - attempts to integrate rho into the cost function failed miserably

    :param hyp: hypothesis sentence
    :param ref: reference sentence
    :param deletion: deletion cost
    :param insertion: insertion cost
    :param substitution: substitution cost
    :param jump: jump cost
    :param rho: coverage cost weight
    :param debug: print the full table of costs
    :return: EED score of hyp given ref (not symmetric)
    """

    # start and end with whitespace to facilitate jumps to front/end
    # works better when you use the most common whitespace (usually spaces)
    # this step is defined as part of the algorithm
    hyp = f' {hyp} '
    ref = f' {ref} '

    # only for debugging
    debug_table = []  # store full DP table
    debug_str = []  # matched string
    debug_cost = []  # costs
    debug_idx = []  # indices of matched string

    # coverage: count how many times each char is visited
    visit_coverage = [0.0] * (len(hyp) + 1)

    # the i-th row stores cost of the cheapest path from (0,0) to (i,l) in CDER alignment grid
    row = [0.0] + [1.0] * len(hyp)  # CDER initial row

    for ref_idx, ref_char in enumerate(ref):
        next_row = [float('inf')] * (len(hyp) + 1)

        # add 1 to the cost per row (same as edit distance)
        next_row[0] = row[0] + 1.0

        # do the normal edit distance calculation for the hyp sentence
        for hyp_idx, hyp_char in enumerate(hyp):
            next_row[hyp_idx + 1] = min([next_row[hyp_idx] + deletion,
                                         row[hyp_idx] + (0.0 if ref_char == hyp_char else substitution),
                                         row[hyp_idx + 1] + insertion])

        # this is the next char to be visited according to the EED algo
        min_cost, min_cost_idx = min((cost, idx) for idx, cost in enumerate(next_row))

        # increment the visit count
        visit_coverage[min_cost_idx] += 1.0

        # long jump allowed only if ref char is whitespace
        # the original algo only checks if ord(ref_char) == 32 (ascii space)
        if ref_char.isspace():
            long_jump_cost = jump + min_cost
            next_row = [min(x, long_jump_cost) for x in next_row]

        # for debug
        if debug:
            debug_table.append(row)
            debug_str.append(hyp[min_cost_idx - 1])
            debug_cost.append(min_cost)
            debug_idx.append(min_cost_idx)

        row = next_row

    # overall error == final cell of final row
    errors = row[-1]
    weighted_coverage = rho * sum(abs(num_visits - 1.0) for num_visits in visit_coverage)
    result = (errors + weighted_coverage) / (len(ref) + weighted_coverage)

    # debug
    if debug:
        debug_table.append(row)
        print(hyp)
        print(''.join(debug_str))
        print(ref)
        print(list(zip(debug_str, debug_idx)))
        for row in debug_table:
            print(row)

    return min(1.0, result)
