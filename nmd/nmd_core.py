from nmd.emd_1d import emd_1d_old


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
    :param n: number of chars per n-gram (default 2)
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
    if n < 2:
        raise ValueError(n)  # technically it would work for n==1, but we'd want to drop the START and END flags

    # add START_TEXT and END_TEXT markers to each word
    # https://en.wikipedia.org/wiki/Control_character#Transmission_control
    # the usage of these characters in any text is almost certainly a bug
    # it is possible to avoid using these characters by using a tuple of optional strings for each n-gram
    # but that's slightly slower and uses more memory
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
            similarity -= emd_1d_old(locations_1, n_gram_locations_2[n_gram])

    # return similarity or distance, optionally normalized
    output = similarity if invert else num_grams_1 + num_grams_2 - similarity
    if normalize:
        output /= num_grams_1 + num_grams_2
    return output
