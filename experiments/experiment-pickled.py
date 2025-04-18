import pickle
import time

if __name__ == '__main__':

    with open('pickles/awl5_ms.pkl', 'rb') as f:
        awl5_ms = pickle.load(f)
    with open('pickles/awl6_ms.pkl', 'rb') as f:
        awl6_ms = pickle.load(f)
    with open('pickles/ws_ms.pkl', 'rb') as f:
        ws_ms = pickle.load(f)
    with open('pickles/awl5_en.pkl', 'rb') as f:
        awl5_en = pickle.load(f)
    with open('pickles/awl6_en.pkl', 'rb') as f:
        awl6_en = pickle.load(f)
    with open('pickles/ws_en.pkl', 'rb') as f:
        ws_en = pickle.load(f)

    test_words = [
        'bananananaanananananana',
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

    for word in test_words:
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
