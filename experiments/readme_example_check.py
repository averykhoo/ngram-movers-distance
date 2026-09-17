"""Regenerate the measured numbers in the README's "Why another string matching
algorithm?" section, so they are re-measurable rather than remembered.

    python experiments/readme_example_check.py

Claims checked, all over experiments/words_en.txt (41.5k words), first measured
2026-09-17:

    * Damerau-Levenshtein finds nothing within distance 2 of 'fotografer'
    * it finds 65 words within distance 5
    * WordList((2, 4), filter_n=0) ranks 'photographer' first

Runs in ~2s (2026-09-17). Not part of the pytest gate only because it reads
experiments/words_en.txt, and the gate covers the package rather than the
experiment data -- promote it if that stops being the right line.

Sabotage-checked 2026-09-17: changing the expected count to 64, and pointing
QUERY at a correctly spelled word, each turn it red.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edit_distance import damerau_levenshtein_distance
from nmd import WordList

QUERY = 'fotografer'
TARGET = 'photographer'
WORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'words_en.txt')


def main() -> int:
    with open(WORDS_PATH, encoding='utf8') as f:
        words = [w.strip() for w in f if w.strip()]

    # the length guard is only a speed-up: a word differing in length by more
    # than k is necessarily more than k edits away
    within_2 = [w for w in words if abs(len(w) - len(QUERY)) <= 2
                and damerau_levenshtein_distance(QUERY, w) <= 2]
    within_5 = [w for w in words if abs(len(w) - len(QUERY)) <= 5
                and damerau_levenshtein_distance(QUERY, w) <= 5]

    t = time.time()
    word_list = WordList((2, 4), filter_n=0)
    for word in words:
        word_list.add_word(word)
    build = time.time() - t
    ranked = [w for w, _ in word_list.lookup(QUERY, top_k=10)]

    print(f'{len(words)} words, indexed in {build:.1f}s')
    print(f'damerau-levenshtein <= 2 from {QUERY!r}: {len(within_2)} {within_2}')
    print(f'damerau-levenshtein <= 5 from {QUERY!r}: {len(within_5)}')
    print(f'nmd top 3: {ranked[:3]}')

    failures = []
    if within_2:
        failures.append(f'README says nothing is within distance 2; found {within_2}')
    if len(within_5) != 65:
        failures.append(f'README says 65 words within distance 5; found {len(within_5)}')
    if not ranked or ranked[0] != TARGET:
        failures.append(f'README says {TARGET!r} ranks first; got {ranked[:1]}')

    for line in failures:
        print(f'MISMATCH: {line}')
    print('README numbers reproduce.' if not failures else 'README numbers are STALE.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
