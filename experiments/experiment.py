import itertools
import pickle
import time
from typing import Sequence

from automata import Matcher
from nmd.nmd_core import emd_1d as emd_1d_fast
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_index import ApproxWordListV5
from nmd.nmd_index import ApproxWordListV6
from nmd.nmd_word_set import WordSet


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


import itertools
from typing import Sequence, Union, List

# Assume this exists for comparison
# def emd_1d_slow(...): ... as provided in the prompt

def emd_1d_dp(positions_x: Sequence[Union[int, float]],
              positions_y: Sequence[Union[int, float]],
             ) -> float:
    """
    Calculates the 1D Earth Mover's Distance using Dynamic Programming.

    This version handles unequal list sizes by assigning a penalty cost of 1
    for each unmatched point. It uses a space-optimized DP approach with
    O(min(N, M)) space complexity and O(N * M) time complexity, where N and M
    are the lengths of the input sequences.

    Args:
        positions_x: A sequence of numbers representing point positions.
        positions_y: Another sequence of numbers representing point positions.

    Returns:
        The calculated Earth Mover's Distance.
    """
    # --- Input Handling & Sorting ---
    # Ensure inputs are lists for potential sorting/modification
    # Sort lists first, as required by DP approach and helps with comparisons
    x = sorted(list(positions_x))
    y = sorted(list(positions_y))

    n = len(x)
    m = len(y)

    # Ensure x is the shorter list to optimize space complexity O(min(N,M))
    if n > m:
        x, y = y, x
        n, m = m, n

    # --- DP Initialization (Two Rows) ---
    # prev_dp_row represents the cost when considering 0 elements from x
    # Corresponds to dp[0][j] = j (cost of leaving j elements of y unmatched)
    prev_dp_row: List[float] = [float(j) for j in range(m + 1)]
    curr_dp_row: List[float] = [0.0] * (m + 1)

    # --- DP Calculation ---
    # Iterate through each element of the shorter list x
    for i in range(1, n + 1):
        # Base case for the current row: dp[i][0] = i
        # (cost of leaving i elements of x unmatched)
        curr_dp_row[0] = float(i)

        # Iterate through each element of the longer list y
        for j in range(1, m + 1):
            # Cost of matching x[i-1] with y[j-1]
            match_cost = abs(x[i-1] - y[j-1]) + prev_dp_row[j-1]

            # Cost of leaving x[i-1] unmatched (penalty 1)
            leave_x_cost = 1.0 + prev_dp_row[j]

            # Cost of leaving y[j-1] unmatched (penalty 1)
            leave_y_cost = 1.0 + curr_dp_row[j-1]

            # Choose the minimum cost path
            curr_dp_row[j] = min(match_cost, leave_x_cost, leave_y_cost)

        # Update prev_dp_row for the next iteration of i
        # Use list() constructor for a shallow copy, preventing aliasing issues
        prev_dp_row = list(curr_dp_row)
        # Or: prev_dp_row = curr_dp_row[:]
        # Avoid: prev_dp_row = curr_dp_row (this would just make them point to the same list)


    # --- Result ---
    # The final EMD is in the last cell calculated, corresponding to dp[n][m]
    return prev_dp_row[m]

# --- Testing Rig ---
def check_dp_correctness(pos_x, pos_y):
    """Compares emd_1d_dp against emd_1d_slow."""
    try:
        slow_result = emd_1d_slow(pos_x, pos_y)
        dp_result = emd_1d_dp(pos_x, pos_y)
        # Use assert with a tolerance for floating point comparisons
        assert abs(slow_result - dp_result) < 1e-9, \
            f"Mismatch! Slow={slow_result}, DP={dp_result}\nInputs: x={pos_x}, y={pos_y}"
        # print(f"Match: Slow={slow_result}, DP={dp_result}")
        return True
    except AssertionError as e:
        print(e)
        return False
    except Exception as e:
        print(f"Error during check: {e}\nInputs: x={pos_x}, y={pos_y}")
        return False

if __name__ == '__main__':
    print("Running EMD DP Correctness Checks...")
    test_cases = [
        ([], []),
        ([0.5], []),
        ([], [0.5]),
        ([0.1], [0.9]),
        ([0.9], [0.1]),
        ([0.1, 0.9], [0.1, 0.9]),
        ([0.1, 0.2], [0.8, 0.9]),
        ([0.1, 0.9], [0.5]),
        ([0.5], [0.1, 0.9]),
        ([0.1, 0.5, 0.9], [0.2, 0.8]),
        ([0.2, 0.8], [0.1, 0.5, 0.9]),
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.4]),
        ([0.1, 0.2, 0.3], [0.1, 0.3, 0.5]), # Mismatch potential
        ([0.1, 0.3], [0.2, 0.4]), # Simple interleaving
        ([0.1, 0.2, 0.8, 0.9], [0.15, 0.85]),
        ([0.15, 0.85], [0.1, 0.2, 0.8, 0.9]),
        ([0.1]*5, [0.9]*3),
        ([0.1]*3, [0.9]*5),
        ([0.1, 0.1, 0.9, 0.9], [0.1, 0.9]),
        ([0.1, 0.9], [0.1, 0.1, 0.9, 0.9]),
        (list(range(10)), list(range(5,15))), # Using integers too
    ]

    all_passed = True
    for i, (x, y) in enumerate(test_cases):
        print(f"--- Test Case {i+1} ---")
        if not check_dp_correctness(x, y):
             all_passed = False

    # Add some random tests
    print("\n--- Random Tests ---")
    import random
    random.seed(42)
    for i in range(20):
         len_x = random.randint(0, 15)
         len_y = random.randint(0, 15)
         x_rand = sorted([random.random() for _ in range(len_x)])
         y_rand = sorted([random.random() for _ in range(len_y)])
         if not check_dp_correctness(x_rand, y_rand):
             all_passed = False

    print("\n--- Summary ---")
    if all_passed:
        print("All EMD DP tests passed!")
    else:
        print("Some EMD DP tests failed!")

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
    answer_slow = emd_1d_dp(positions_x, positions_y)
    assert abs(answer_fast - answer_slow) < 0.00000001, (answer_slow, answer_fast, positions_x, positions_y)
    return answer_fast


if __name__ == '__main__':

    from experiments.edit_distance import damerau_levenshtein_distance
    from experiments.edit_distance import edit_distance


    def speed_test(word_1: str, word_2: str):
        edit_distance(word_1, word_2)
        damerau_levenshtein_distance(word_1, word_2)
        return ngram_movers_distance(word_1, word_2)


    num_x = 7
    num_y = 10

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
    #
    # for _ in range(1000):
    #     speed_test('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    #                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    #     speed_test('aabbbbbbbbaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    #     speed_test('aaaabbbbbbbbaaaa', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    #     speed_test('banana', 'bababanananananananana')
    #     speed_test('banana', 'bababanananananananananna')
    #     speed_test('banana', 'nanananananabababa')
    #     speed_test('banana', 'banana')
    #     speed_test('nanananananabababa', 'banana')
    #     speed_test('banana', 'bababananananananananannanananananananana')
    #     speed_test('banana', 'bababananananananananannananananananananananananananannanananananananana')
    #     speed_test('bananabababanana', 'bababananananananananannananananananananananananananannananabanananananana')
    #
    # # test cases: https://www.watercoolertrivia.com/blog/schwarzenegger
    # with open('schwarzenegger.txt') as f:
    #     for line in f:
    #         print('schwarzenegger', line.strip(), speed_test(line.strip(), 'schwarzenegger'))
    #
    # # real world test cases
    # with open('words_en.txt') as f1:
    #     with open('words_ms.txt') as f2:
    #         for en, ms in zip(f1, f2):
    #             speed_test(en.strip(), ms.strip())
    #             speed_test(en.strip(), en.strip())
    #             speed_test(ms.strip(), ms.strip())

    with open('words_ms.txt', encoding='utf8') as f:
        words_ms = set(f.read().split())
    print(f'{len(words_ms)=}')

    # t = time.perf_counter()
    # awl3_ms = ApproxWordListV3((1, 2, 3, 4))
    # for word in words_ms:
    #     awl3_ms.add_word(word)
    # print('build awl3_ms', time.perf_counter() - t)

    t = time.perf_counter()
    awl5_ms = ApproxWordListV5((1, 2, 3, 4))
    for word in words_ms:
        awl5_ms.add_word(word)
    print('build awl5_ms', time.perf_counter() - t)

    t = time.perf_counter()
    awl6_ms = ApproxWordListV6((1, 2, 3, 4))
    for word in words_ms:
        awl6_ms.add_word(word)
    print('build awl6_ms', time.perf_counter() - t)

    t = time.perf_counter()
    ws_ms = WordSet(ngram_sizes=(1, 2, 3, 4))
    for word in words_ms:
        ws_ms.add(word)
    print('build ws_ms', time.perf_counter() - t)

    # with open('british-english-insane.txt', encoding='utf8') as f:
    with open('words_en.txt', encoding='utf8') as f:
        words = set(f.read().split())
    print(f'{len(words)=}')

    # t = time.perf_counter()
    # awl3_en = ApproxWordListV3((1, 2, 3, 4))
    # for word in words:
    #     awl3_en.add_word(word)
    # print('build awl3_en', time.perf_counter() - t)

    t = time.perf_counter()
    awl5_en = ApproxWordListV5((1, 2, 3, 4))
    for word in words:
        awl5_en.add_word(word)
    print('build awl5_en', time.perf_counter() - t)

    t = time.perf_counter()
    awl6_en = ApproxWordListV6((1, 2, 3, 4))
    for word in words:
        awl6_en.add_word(word)
    print('build awl6_en', time.perf_counter() - t)

    t = time.perf_counter()
    ws_en = WordSet(ngram_sizes=(1, 2, 3, 4))
    for word in words:
        ws_en.add(word)
    print('build ws_en', time.perf_counter() - t)

    test_words = [
        'bananana',
        'supercallousedfragilemisticexepialidocus',
        'asalamalaikum',
        'beewilldermant',
        'blackbary',
        'kartweel',
        'chomosrome',
        'chrisanthumem',
        'instalatiomn',
    ]
    # print('awl3_ms', awl3_ms.lookup('bananananaanananananana'))
    print('awl5_ms', awl5_ms.lookup('bananananaanananananana', normalize=True))
    print('awl6_ms', awl6_ms.lookup('bananananaanananananana', normalize=True))
    print('ws_ms', ws_ms.find_similar('bananananaanananananana'))
    # print('awl3_en', awl3_en.lookup('bananananaanananananana'))
    print('awl5_en', awl5_en.lookup('bananananaanananananana', normalize=True))
    print('awl6_en', awl6_en.lookup('bananananaanananananana', normalize=True))
    print('ws_en', ws_en.find_similar('bananananaanananananana'))

    with open('pickles/awl5_ms.pkl', 'wb') as f:
        pickle.dump(awl5_ms, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/awl6_ms.pkl', 'wb') as f:
        pickle.dump(awl6_ms, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/ws_ms.pkl', 'wb') as f:
        pickle.dump(ws_ms, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/awl5_en.pkl', 'wb') as f:
        pickle.dump(awl5_en, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/awl6_en.pkl', 'wb') as f:
        pickle.dump(awl6_en, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/ws_en.pkl', 'wb') as f:
        pickle.dump(ws_en, f, pickle.HIGHEST_PROTOCOL)
    print('pickled')

    m = Matcher(sorted(words))

    # while True:
    #     word = input('word:\n')
    for word in test_words:
        word = word.strip()
        if not word:
            break

        # t = time.perf_counter()
        # print('awl3_ms', awl3_ms.lookup(word))
        # print(time.perf_counter() - t)
        # print()

        t = time.perf_counter()
        print('awl5_ms', awl5_ms.lookup(word, normalize=True))
        print(time.perf_counter() - t)
        print()

        # t = time.perf_counter()
        # print('awl5_ms_denorm', awl5_ms.lookup(word, normalize=False))
        # print(time.perf_counter() - t)
        # print()

        t = time.perf_counter()
        print('awl6_ms', awl6_ms.lookup(word, normalize=True))
        print(time.perf_counter() - t)
        print()

        # t = time.perf_counter()
        # print('awl6_ms_denorm', awl6_ms.lookup(word))
        # print(time.perf_counter() - t)
        # print()

        t = time.perf_counter()
        print('ws_ms', ws_ms.find_similar(word))
        print(time.perf_counter() - t)
        print()

        # t = time.perf_counter()
        # print('difflib_ms', difflib.get_close_matches(word, words_ms, n=10))
        # print(time.perf_counter() - t)
        # print()
        #
        # t = time.perf_counter()
        # print('difflib_ms', difflib.get_close_matches(word, words_ms, n=10, cutoff=0.3))
        # print(time.perf_counter() - t)
        # print()

        # t = time.perf_counter()
        # print('awl3_en', awl3_en.lookup(word))
        # print(time.perf_counter() - t)
        # print()

        t = time.perf_counter()
        print('awl5_en', awl5_en.lookup(word, normalize=True))
        print(time.perf_counter() - t)
        print()

        # t = time.perf_counter()
        # print('awl5_en_denorm', awl5_en.lookup(word, normalize=False))
        # print(time.perf_counter() - t)
        # print()

        t = time.perf_counter()
        print('awl6_en', awl6_en.lookup(word, normalize=True))
        print(time.perf_counter() - t)
        print()

        # t = time.perf_counter()
        # print('awl6_en_denorm', awl6_en.lookup(word, normalize=False))
        # print(time.perf_counter() - t)
        # print()

        t = time.perf_counter()
        print('ws_en', ws_en.find_similar(word))
        print(time.perf_counter() - t)
        print()

        # t = time.perf_counter()
        # print('difflib_en', difflib.get_close_matches(word, words, n=10))
        # print(time.perf_counter() - t)
        # print()
        #
        # t = time.perf_counter()
        # print('difflib_en', difflib.get_close_matches(word, words, n=10, cutoff=0.3))
        # print(time.perf_counter() - t)
        # print()
        #
        # t = time.perf_counter()
        # print('automata dist 1 en', list(find_all_matches(word, 1, m)))
        # print(time.perf_counter() - t)
        # print()
        #
        # t = time.perf_counter()
        # print('automata dist 2 en', list(find_all_matches(word, 2, m)))
        # print(time.perf_counter() - t)
        # print()
        #
        # t = time.perf_counter()
        # print('automata dist 3 en', list(find_all_matches(word, 3, m)))
        # print(time.perf_counter() - t)
        # print()

# if __name__ == '__main__':
#
#     with open('translate-reference.txt') as f:
#         ref_lines = f.readlines()
#     with open('translate-google-offline.txt') as f:
#         hyp_lines = f.readlines()
#
#     scores_bow = []
#     scores_nmd = []
#     scores_sim = []
#     for ref_line, hyp_line in zip(ref_lines, hyp_lines):
#         ref_tokens = list(unicode_tokenize(ref_line.casefold(), words_only=True, merge_apostrophe_word=True))
#         hyp_tokens = list(unicode_tokenize(hyp_line.casefold(), words_only=True, merge_apostrophe_word=True))
#         scores_bow.append(bow_ngram_movers_distance(ref_tokens, hyp_tokens, 4) / max(len(ref_tokens), len(hyp_tokens)))
#         scores_sim.append(
#             bow_ngram_movers_distance(ref_tokens, hyp_tokens, 4, invert=True) / max(len(ref_tokens), len(hyp_tokens)))
#         scores_nmd.append(ngram_movers_distance(' '.join(ref_tokens), ' '.join(hyp_tokens), 4, normalize=True))
#         print(' '.join(ref_tokens))
#         print(' '.join(hyp_tokens))
#         print(scores_bow[-1])
#         print(scores_sim[-1])
#         print(scores_nmd[-1])
#
#     from matplotlib import pyplot as plt
#
#     plt.scatter(scores_bow, scores_nmd, marker='.')
#     plt.show()
#     scores_diff = [a - b for a, b in zip(scores_bow, scores_nmd)]
#     tmp = sorted(zip(scores_diff, scores_bow, scores_sim, scores_nmd, ref_lines, hyp_lines))
#     print(tmp[0])
#     print(tmp[1])
#     print(tmp[2])
#     print(tmp[3])
#     print(tmp[-1])
#     print(tmp[-2])
#     print(tmp[-3])
#     print(tmp[-4])
