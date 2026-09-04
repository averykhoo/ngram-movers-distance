from nmd.emd_1d import emd_1d_dp


def ngram_movers_distance(word_1: str,
                          word_2: str,
                          n: int = 2,
                          invert: bool = False,
                          normalize: bool = False,
                          ) -> float:
    """
    calculates the n-gram mover's distance between two words (for some specified n)
    case-sensitive by default, so lowercase/casefold the input words for case-insensitive results

    :param word_1: a string
    :param word_2: another string, or possibly the same string
    :param n: number of chars per n-gram (default 2); n == 1 drops the START/END markers, making it a
              positional comparison of character multisets, which carries no adjacency information
    :param invert: return similarity instead of difference
    :param normalize: normalize to a score from 0 to 1 (inclusive of 0 and 1)
    :return: n-gram mover's distance, possibly inverted and/or normalized
    """
    # sanity checks
    if not isinstance(word_1, str):
        raise TypeError(word_1)
    if '\2' in word_1 or '\3' in word_1:
        raise ValueError(word_1)

    if not isinstance(word_2, str):
        raise TypeError(word_2)
    if '\2' in word_2 or '\3' in word_2:
        raise ValueError(word_2)

    if not isinstance(n, int):
        raise TypeError(n)
    if n < 1:
        raise ValueError(n)

    # add START_TEXT and END_TEXT markers to each word
    # https://en.wikipedia.org/wiki/Control_character#Transmission_control
    # the usage of these characters in any text is almost certainly a bug
    # it is possible to avoid using these characters by using a tuple of optional strings for each n-gram
    # but that's slightly slower and uses more memory
    # n == 1 is the exception: the markers would become two unigrams that match between any pair of words,
    # so they are dropped, and n == 1 is then a pure character multiset comparison with positions
    if n > 1:
        word_1 = f'\2{word_1}\3'
        word_2 = f'\2{word_2}\3'

    # number of n-grams per word
    num_grams_1 = len(word_1) - n + 1
    num_grams_2 = len(word_2) - n + 1

    # generate n_gram indices and index their locations
    n_gram_locations_1 = dict()
    for idx in range(num_grams_1):
        n_gram_locations_1.setdefault(word_1[idx:idx + n], []).append(idx / max(1, num_grams_1 - 1))
    n_gram_locations_2 = dict()
    for idx in range(num_grams_2):
        n_gram_locations_2.setdefault(word_2[idx:idx + n], []).append(idx / max(1, num_grams_2 - 1))

    # we want to calculate the earth mover distance for all n-grams in both words, which uses the following equation:
    # > distance = sum(emd_1d(n_gram_locations_1.get(n_gram, []), n_gram_locations_2.get(n_gram, []))
    # >                for n_gram in set(n_gram_locations_1).union(set(n_gram_locations_2)))
    # this could be optimized by only calculating emd for n-grams in common and just counting the symmetric difference
    # but calculating similarity (i.e. inverted distance) runs even faster than that
    # so instead we calculate the similarity and then find distance using the following identity:
    # > distance + similarity == num_grams_1 + num_grams_2
    similarity = 0
    for n_gram, locations_1 in n_gram_locations_1.items():
        if n_gram in n_gram_locations_2:
            similarity += len(locations_1) + len(n_gram_locations_2[n_gram])
            similarity -= emd_1d_dp(locations_1, n_gram_locations_2[n_gram])

    # return similarity or distance, optionally normalized
    output = similarity if invert else num_grams_1 + num_grams_2 - similarity
    if normalize:
        # two words can have no n-grams at all between them (both empty at n == 1, or both exactly
        # n - 2 characters long for larger n), which used to raise ZeroDivisionError. there is no
        # n-gram evidence either way, so fall back to comparing the strings -- which is exactly what
        # the callers of this function already did with the exception they caught, e.g.
        # `int(word_1 != word_2)` in nmd_bow.bow_ngram_movers_distance
        if num_grams_1 + num_grams_2 == 0:
            identical = word_1 == word_2
            return float(identical) if invert else float(not identical)
        output /= num_grams_1 + num_grams_2
    return output
