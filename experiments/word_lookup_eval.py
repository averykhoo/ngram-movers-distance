"""
the dictionary lookup task nmd was actually designed for -- does tf-idf cosine win here too?

every measurement in docs/bow-plan.md parts 5-6 is on product names, where the winning signal turned
out to be rare-substring (model number) matching. this is the other task: index a word list, corrupt a
word with a few edits, and see whether the original comes back first. n-grams here are within a single
short word, and idf is computed over the word list, so it is a genuinely different regime.

data: experiments/words_en.txt (41554 words, already in the repo)
run:  python experiments/word_lookup_eval.py [dict_size] [num_queries]
"""
import math
import random
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nmd.nmd_core import ngram_movers_distance
from v7_ranking import build_ranker

WORDS = Path(__file__).resolve().parent / 'words_en.txt'
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def corrupt(word, edits, rng):
    for _ in range(edits):
        if len(word) < 2:
            break
        op = rng.choice(['insert', 'delete', 'substitute', 'transpose'])
        i = rng.randrange(len(word))
        if op == 'insert':
            word = word[:i] + rng.choice(ALPHABET) + word[i:]
        elif op == 'delete':
            word = word[:i] + word[i + 1:]
        elif op == 'substitute':
            word = word[:i] + rng.choice(ALPHABET) + word[i + 1:]
        else:
            j = min(i + 1, len(word) - 1)
            chars = list(word)
            chars[i], chars[j] = chars[j], chars[i]
            word = ''.join(chars)
    return word


def grams(text, n):
    text = text if n == 1 else f'\2{text}\3'
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def main(dict_size=8000, num_queries=250, seed=0):
    rng = random.Random(seed)
    with open(WORDS, encoding='utf8') as f:
        vocab = sorted({w.strip().casefold() for w in f if w.strip().isalpha() and 3 <= len(w.strip()) <= 15})
    targets = rng.sample(vocab, num_queries)
    rest = [w for w in vocab if w not in set(targets)]
    dictionary = sorted(set(targets) | set(rng.sample(rest, max(0, dict_size - num_queries))))
    queries = [(corrupt(w, rng.choice([1, 1, 2]), rng), w) for w in targets]
    queries = [(q, w) for q, w in queries if q != w]
    print(f'dictionary {len(dictionary)} words, {len(queries)} corrupted queries '
          f'(1-2 edits, exact-match queries dropped)')
    print(f'examples: {[(q, w) for q, w in queries[:6]]}')

    # idf over the dictionary: df(g) = number of words containing g
    idf_tables = {}
    for n in (1, 2, 3, 4):
        df = Counter()
        for word in dictionary:
            df.update(set(grams(word, n)))
        idf_tables[n] = ({g: math.log(len(dictionary) / d) for g, d in df.items()}, math.log(len(dictionary)))

    def idf(g, n, p):
        table, ceiling = idf_tables[n]
        return table.get(g, ceiling) ** p

    def nmd_multi(a, b, n_list):
        return sum(ngram_movers_distance(a, b, n=n, invert=True, normalize=True) for n in n_list) / len(n_list)

    def cosine_tfidf(a, b, n, p=1.0, log_tf=True):
        ca, cb = Counter(grams(a, n)), Counter(grams(b, n))
        tf = (lambda c: 1 + math.log(c)) if log_tf else float
        va = {g: tf(c) * idf(g, n, p) for g, c in ca.items()}
        vb = {g: tf(c) * idf(g, n, p) for g, c in cb.items()}
        num = sum(v * vb.get(g, 0.0) for g, v in va.items())
        den = math.sqrt(sum(v * v for v in va.values()) * sum(v * v for v in vb.values()))
        return num / den if den else 0.0

    def jaccard(a, b, n):
        sa, sb = set(grams(a, n)), set(grams(b, n))
        return len(sa & sb) / len(sa | sb) if sa or sb else 1.0

    def weighted_nmd(a, b, n, p):
        """nmd with w(g) = idf(g)^p as a per-type scalar -- see experiments/idf_variants.py"""
        from nmd.emd_1d import emd_1d_dp
        def locate(text):
            text = text if n == 1 else f'\2{text}\3'
            count = len(text) - n + 1
            out = {}
            for i in range(count):
                out.setdefault(text[i:i + n], []).append(i / max(1, count - 1))
            return out
        la, lb = locate(a), locate(b)
        ta = sum(idf(g, n, p) * len(v) for g, v in la.items())
        tb = sum(idf(g, n, p) * len(v) for g, v in lb.items())
        sim = sum(idf(g, n, p) * (len(v) + len(lb[g]) - emd_1d_dp(v, lb[g])) for g, v in la.items() if g in lb)
        return sim / (ta + tb) if ta + tb else 1.0

    # every nmd row is served by an ApproxWordListV7 index rather than by scoring the whole
    # dictionary pairwise. this is an exact substitution, not an approximation: V7's pruning
    # bound is two-sided, and `lookup(dim=1, normalize=True)` computes the same mean-over-n of
    # per-n normalized similarity that `nmd_multi` computes. verified 2026-09-05 over 1200 words
    # x 57 queries at n = 1, 2, 3, (1,2), (2,4), (1,2,4): worst score difference 2.2e-16,
    # identical hit@1 everywhere, and MRR identical except one query at n=3 where the answer
    # ties with another word and the two disagree on tie order (0.9167 vs 0.9168)
    INDEX_METRICS = {
        'nmd n=1': dict(n=(1,)),
        'nmd n=2': dict(n=(2,)),
        'nmd n=3': dict(n=(3,)),
        'nmd n=4': dict(n=(4,)),
        'nmd n=(1,2)': dict(n=(1, 2)),
        'nmd n=(2,3)': dict(n=(2, 3)),
        'nmd n=(2,4)  [README default]': dict(n=(2, 4)),
        'nmd n=(1,2,4)': dict(n=(1, 2, 4)),
        # ⚠ these three are not quite the same formula the pairwise `weighted_nmd` below
        # computes: V7 smooths idf as log((V + 1) / df) where weighted_nmd uses the unsmoothed
        # log(V / df). measured 2026-09-05 the difference changed nothing -- all three rows
        # reproduced the pre-port numbers exactly (0.896/0.921/0.961, 0.870/0.898/0.948,
        # 0.532/0.622/0.818) because log(V/df) and log((V+1)/df) rank identically at this
        # vocabulary size. expect them to diverge on a small vocabulary, where +1 is a larger
        # relative shift, or at a high exponent that amplifies it
        'nmd n=2 + idf^1': dict(n=(2,), idf_exponent=1.0),
        'nmd n=2 + idf^2': dict(n=(2,), idf_exponent=2.0),
        'nmd n=2 + idf^6': dict(n=(2,), idf_exponent=6.0),
    }
    PAIRWISE_METRICS = {
        'tf-idf cosine n=2': lambda a, b: cosine_tfidf(a, b, 2),
        'tf-idf cosine n=3': lambda a, b: cosine_tfidf(a, b, 3),
        'tf-idf cosine n=2, idf^2': lambda a, b: cosine_tfidf(a, b, 2, p=2.0),
        'cosine n=2, no idf': lambda a, b: cosine_tfidf(a, b, 2, p=0.0),
        'jaccard n=2': lambda a, b: jaccard(a, b, 2),
        'jaccard n=3': lambda a, b: jaccard(a, b, 3),
        'difflib ratio': lambda a, b: SequenceMatcher(None, a, b).ratio(),
    }

    n_q = len(queries)
    print(f'\n{"metric":<32} {"hit@1":>8} {"MRR":>8} {"hit@10":>8} {"build":>7} {"sec":>7}')
    for name, kwargs in INDEX_METRICS.items():
        t0 = time.perf_counter()
        rank = build_ranker(dictionary, **kwargs)
        build_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        hits = rr = top10 = 0
        for query, answer in queries:
            ranked = [word for word, _score in rank(query)]
            # a word sharing no n-gram with the query never enters the candidate set; pairwise
            # it would have scored 0 and sorted last, so treat it as unranked
            position = ranked.index(answer) + 1 if answer in ranked else None
            hits += position == 1
            rr += 1 / position if position else 0.0
            top10 += bool(position and position <= 10)
        print(f'{name:<32} {hits / n_q:>8.3f} {rr / n_q:>8.3f} {top10 / n_q:>8.3f} '
              f'{build_sec:>7.1f} {time.perf_counter() - t0:>7.1f}')

    for name, metric in PAIRWISE_METRICS.items():
        t0 = time.perf_counter()
        hits = rr = top10 = 0
        for query, answer in queries:
            scored = sorted(((metric(query, word), word) for word in dictionary), key=lambda x: -x[0])
            rank_position = next(i for i, (_, word) in enumerate(scored, 1) if word == answer)
            hits += rank_position == 1
            rr += 1 / rank_position
            top10 += rank_position <= 10
        print(f'{name:<32} {hits / n_q:>8.3f} {rr / n_q:>8.3f} {top10 / n_q:>8.3f} '
              f'{"-":>7} {time.perf_counter() - t0:>7.1f}')


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:]))
