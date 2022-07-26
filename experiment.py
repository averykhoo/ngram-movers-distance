import difflib
import itertools
import time
from typing import Sequence

from automata import Matcher
from automata import find_all_matches
from nmd.nmd import emd_1d as emd_1d_fast
from nmd.nmd import ngram_movers_distance
from nmd.nmd_index import ApproxWordList3
from nmd.nmd_index import ApproxWordList5


def emd_1d_slow(positions_x: Sequence[float],
                positions_y: Sequence[float],
                ) -> float:
    # positions_x must be longer
    if len(positions_x) < len(positions_y):
        positions_x, positions_y = positions_y, positions_x

    # sort both lists
    positions_x = sorted(positions_x)
    positions_y = sorted(positions_y)

    # find the minimum cost alignment
    costs = [len(positions_y)]
    for x_combination in itertools.combinations(positions_x, len(positions_y)):
        costs.append(sum(abs(x - y) for x, y in zip(x_combination, positions_y)))

    # the distance is the min cost alignment plus a count of unmatched points
    return len(positions_x) - len(positions_y) + min(costs)


def check_correct_emd_1d(positions_x: Sequence[float],
                         positions_y: Sequence[float],
                         ) -> float:
    """
    kind of like earth mover's distance
    but positions are limited to within the unit interval
    and must be quantized

    :param positions_x: list of positions (each a float from 0 to 1 inclusive)
    :param positions_y: list of positions (each a float from 0 to 1 inclusive)
    :return:
    """

    # sanity checks
    assert isinstance(positions_x, Sequence)
    assert isinstance(positions_y, Sequence)
    assert all(isinstance(x, (int, float)) for x in positions_x)
    assert all(isinstance(y, (int, float)) for y in positions_y)

    # all inputs must be in the unit interval
    assert all(0 <= x <= 1 for x in positions_x)
    assert all(0 <= y <= 1 for y in positions_y)

    # run both slow and fast and check them
    answer_fast = emd_1d_fast(positions_x, positions_y)
    answer_slow = emd_1d_slow(positions_x, positions_y)
    assert abs(answer_fast - answer_slow) < 0.00000001, (answer_slow, answer_fast, positions_x, positions_y)
    return answer_fast


if __name__ == '__main__':

    from levenshtein import damerau_levenshtein_distance
    from levenshtein import edit_distance


    def speed_test(word_1: str, word_2: str):
        edit_distance(word_1, word_2)
        damerau_levenshtein_distance(word_1, word_2)
        return ngram_movers_distance(word_1, word_2)


    num_x = 3
    num_y = 7

    xs = [i / (num_x - 1) for i in range(num_x)]
    ys = [i / (num_y - 1) for i in range(num_y)]
    # print(xs)
    # print(ys)
    xs = xs + xs + xs

    for x_len in range(len(xs) + 1):
        for y_len in range(len(ys) + 1):
            print(x_len, y_len)
            for x_combi in itertools.combinations(xs, x_len):
                for y_combi in itertools.combinations(ys, y_len):
                    assert abs(
                        check_correct_emd_1d(x_combi, y_combi) - check_correct_emd_1d(y_combi, x_combi)) < 0.0001, (
                    x_combi, y_combi)

    for _ in range(1000):
        speed_test('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                   'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        speed_test('aabbbbbbbbaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        speed_test('aaaabbbbbbbbaaaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        speed_test('banana', 'bababanananananananana')
        speed_test('banana', 'bababanananananananananna')
        speed_test('banana', 'nanananananabababa')
        speed_test('banana', 'banana')
        speed_test('nanananananabababa', 'banana')
        speed_test('banana', 'bababananananananananannanananananananana')
        speed_test('banana', 'bababananananananananannananananananananananananananannanananananananana')
        speed_test('bananabababanana', 'bababananananananananannananananananananananananananannananabanananananana')

    # test cases: https://www.watercoolertrivia.com/blog/schwarzenegger
    with open('schwarzenegger.txt') as f:
        for line in f:
            print('schwarzenegger', line.strip(), speed_test(line.strip(), 'schwarzenegger'))

    # real world test cases
    with open('words_en.txt') as f1:
        with open('words_ms.txt') as f2:
            for en, ms in zip(f1, f2):
                speed_test(en.strip(), ms.strip())
                speed_test(en.strip(), en.strip())
                speed_test(ms.strip(), ms.strip())

    with open('words_ms.txt', encoding='utf8') as f:
        words_ms = set(f.read().split())

    awl3_ms = ApproxWordList3((2,))
    for word in words_ms:
        awl3_ms.add_word(word)

    awl5_ms = ApproxWordList5((2,))
    for word in words_ms:
        awl5_ms.add_word(word)

    with open('words_en.txt', encoding='utf8') as f:
        # with open('british-english-insane.txt', encoding='utf8') as f:
        words = set(f.read().split())

    awl3_en = ApproxWordList3((2,))
    for word in words:
        awl3_en.add_word(word)

    awl5_en = ApproxWordList5((2,))
    for word in words:
        awl5_en.add_word(word)

    # bananana
    # supercallousedfragilemisticexepialidocus
    # asalamalaikum
    # beewilldermant
    # blackbary
    # kartweel
    # chomosrome
    # chrisanthumem
    # instalatiomn
    print(awl3_ms.lookup('bananananaanananananana'))
    print(awl5_ms.lookup('bananananaanananananana'))
    print(awl3_en.lookup('bananananaanananananana'))
    print(awl5_en.lookup('bananananaanananananana'))

    m = Matcher(sorted(words))

    while True:
        word = input('word:\n')
        word = word.strip()
        if not word:
            break

        t = time.time()
        print('awl3_ms', awl3_ms.lookup(word))
        print(time.time() - t)
        print()

        t = time.time()
        print('awl5_ms', awl5_ms.lookup(word))
        print(time.time() - t)
        print()

        t = time.time()
        print('difflib_ms', difflib.get_close_matches(word, words_ms, n=10))
        print(time.time() - t)
        print()

        t = time.time()
        print('difflib_ms', difflib.get_close_matches(word, words_ms, n=10, cutoff=0.3))
        print(time.time() - t)
        print()

        t = time.time()
        print('awl3_en', awl3_en.lookup(word))
        print(time.time() - t)
        print()

        t = time.time()
        print('awl5_en', awl5_en.lookup(word))
        print(time.time() - t)
        print()

        t = time.time()
        print('difflib_en', difflib.get_close_matches(word, words, n=10))
        print(time.time() - t)
        print()

        t = time.time()
        print('difflib_en', difflib.get_close_matches(word, words, n=10, cutoff=0.3))
        print(time.time() - t)
        print()

        t = time.time()
        print('automata dist 1 en', list(find_all_matches(word, 1, m)))
        print(time.time() - t)
        print()

        t = time.time()
        print('automata dist 2 en', list(find_all_matches(word, 2, m)))
        print(time.time() - t)
        print()

        t = time.time()
        print('automata dist 3 en', list(find_all_matches(word, 3, m)))
        print(time.time() - t)
        print()
