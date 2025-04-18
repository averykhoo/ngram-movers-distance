import pickle
import time

from automata import Matcher
from experiments.gemini_optimized_v5 import ApproxWordListV5b
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_index import ApproxWordListV5
from nmd.nmd_index import ApproxWordListV6
from nmd.nmd_word_set import WordSet

if __name__ == '__main__':

    from experiments.edit_distance import damerau_levenshtein_distance
    from experiments.edit_distance import edit_distance


    def speed_test(word_1: str, word_2: str):
        edit_distance(word_1, word_2)
        damerau_levenshtein_distance(word_1, word_2)
        return ngram_movers_distance(word_1, word_2)


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
    awl5b_ms = ApproxWordListV5b((1, 2, 3, 4))
    for word in words_ms:
        awl5b_ms.add_word(word)
    print('build awl5b_ms', time.perf_counter() - t)

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
    awl5b_en = ApproxWordListV5b((1, 2, 3, 4))
    for word in words:
        awl5b_en.add_word(word)
    print('build awl5b_en', time.perf_counter() - t)

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
    print('awl5b_ms', awl5b_ms.lookup('bananananaanananananana', normalize=True))
    print('awl6_ms', awl6_ms.lookup('bananananaanananananana', normalize=True))
    print('ws_ms', ws_ms.find_similar('bananananaanananananana'))
    # print('awl3_en', awl3_en.lookup('bananananaanananananana'))
    print('awl5_en', awl5_en.lookup('bananananaanananananana', normalize=True))
    print('awl5b_en', awl5b_en.lookup('bananananaanananananana', normalize=True))
    print('awl6_en', awl6_en.lookup('bananananaanananananana', normalize=True))
    print('ws_en', ws_en.find_similar('bananananaanananananana'))

    with open('pickles/awl5_ms.pkl', 'wb') as f:
        pickle.dump(awl5_ms, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/awl5b_ms.pkl', 'wb') as f:
        pickle.dump(awl5b_ms, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/awl6_ms.pkl', 'wb') as f:
        pickle.dump(awl6_ms, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/ws_ms.pkl', 'wb') as f:
        pickle.dump(ws_ms, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/awl5_en.pkl', 'wb') as f:
        pickle.dump(awl5_en, f, pickle.HIGHEST_PROTOCOL)
    with open('pickles/awl5b_en.pkl', 'wb') as f:
        pickle.dump(awl5b_en, f, pickle.HIGHEST_PROTOCOL)
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

        t = time.perf_counter()
        print('awl5b_ms', awl5b_ms.lookup(word, normalize=True))
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

        t = time.perf_counter()
        print('awl5b_en', awl5b_en.lookup(word, normalize=True))
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
