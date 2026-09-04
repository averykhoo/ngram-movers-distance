"""
hyperparameter search for char-nmd / bow-nmd / segment mover's distance on Abt-Buy

protocol: ER-Magellan Abt-Buy split, `name` attribute only. configs are ranked by average precision
on TRAIN (threshold-free, so the search never touches the test set); the winner of each family then
gets a threshold tuned on train and is reported as test F1. this is the same yardstick as
experiments/er_magellan_eval.py, so the numbers are directly comparable.

char-nmd and bow-nmd are cheap enough for an exhaustive grid. the segment ILP is not, so it gets a
coordinate search on a stratified train subsample (every positive + 3x as many negatives), which is
not exhaustive -- it can miss interactions between parameters.

note: n=1 is unavailable, `nmd_core.ngram_movers_distance` rejects n < 2 because the STX/ETX markers
would have to be dropped first (nmd/nmd_core.py:34).

run: python experiments/hyperparam_search.py
"""
import functools
import itertools
import random
import sys
import time

import nmd.nmd_segments
from experiments.er_magellan_eval import DATA
from experiments.er_magellan_eval import best_threshold
from experiments.er_magellan_eval import f1_at
from experiments.er_magellan_eval import read_split
from experiments.position_penalty import average_precision
from nmd.nmd_bow import bow_ngram_movers_distance
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_segments import segment_movers_distance

nmd.nmd_segments._pair_raw = functools.lru_cache(maxsize=None)(nmd.nmd_segments._pair_raw)


@functools.lru_cache(maxsize=None)
def _nmd(a, b, n):
    return ngram_movers_distance(a, b, n=n, invert=True, normalize=True)


def char_nmd(a, b, n_list):
    return sum(_nmd(a['joined'], b['joined'], n) for n in n_list) / len(n_list)


def bow_nmd(a, b, n):
    return bow_ngram_movers_distance(a['toks'], b['toks'], n=n, invert=True, normalize=True)


def segment(a, b, n_list, max_len, lam, mu, mass, min_sim):
    return segment_movers_distance(a['toks'], b['toks'], n=n_list, max_len=max_len, lam=lam, mu=mu,
                                   mass=mass, min_sim=min_sim, invert=True, normalize=True)


def evaluate(metric, pairs):
    return average_precision([metric(a, b) for a, b, _ in pairs], [y for _, _, y in pairs])


def report(name, metric, train, test):
    t0 = time.perf_counter()
    tr = [metric(a, b) for a, b, _ in train]
    te = [metric(a, b) for a, b, _ in test]
    thresh, train_f1 = best_threshold(tr, [y for _, _, y in train])
    f1, precision, recall = f1_at(te, [y for _, _, y in test], thresh)
    print(f'{name:<44} {100 * f1:>8.2f} {100 * precision:>7.2f} {100 * recall:>7.2f} '
          f'{100 * train_f1:>8.2f} {time.perf_counter() - t0:>7.1f}')
    return f1


def main():
    raw_train = read_split(DATA / 'train.txt', use_desc=False)
    raw_test = read_split(DATA / 'test.txt', use_desc=False)

    @functools.lru_cache(maxsize=None)
    def features(text):
        toks = [t for t in text.casefold().split() if any(c.isalnum() for c in t)]
        return {'toks': toks, 'joined': ' '.join(toks)}

    prep = lambda rows: [(features(a), features(b), y) for a, b, y in rows]
    train, test = prep(raw_train), prep(raw_test)

    # stratified subsample for the expensive family: every positive + 3x negatives
    rng = random.Random(0)
    pos = [r for r in train if r[2]]
    neg = [r for r in train if not r[2]]
    sub = pos + rng.sample(neg, min(len(neg), 3 * len(pos)))
    rng.shuffle(sub)
    print(f'train {len(train)} pairs ({len(pos)} positive), test {len(test)} pairs '
          f'({sum(r[2] for r in test)} positive), segment subsample {len(sub)} pairs')

    # ---------------------------------------------------------------- char-nmd: exhaustive
    print('\n' + '=' * 88)
    print('char-nmd: exhaustive grid over n (train AP)')
    print('=' * 88)
    n_options = [(2,), (3,), (4,), (5,), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5),
                 (2, 3, 4), (2, 4, 5), (3, 4, 5), (2, 3, 4, 5)]
    scored = []
    for n_list in n_options:
        ap = evaluate(lambda a, b, nl=n_list: char_nmd(a, b, nl), train)
        scored.append((ap, n_list))
        print(f'  n={str(n_list):<16} AP {ap:.4f}')
    scored.sort(reverse=True)
    best_char = scored[0]
    print(f'  -> best n={best_char[1]} AP {best_char[0]:.4f}')

    # ---------------------------------------------------------------- bow-nmd: exhaustive
    print('\n' + '=' * 88)
    print('bow-nmd: exhaustive grid over n (train AP)')
    print('=' * 88)
    scored = []
    for n in (2, 3, 4, 5):
        ap = evaluate(lambda a, b, n=n: bow_nmd(a, b, n), train)
        scored.append((ap, n))
        print(f'  n={n:<18} AP {ap:.4f}')
    scored.sort(reverse=True)
    best_bow = scored[0]
    print(f'  -> best n={best_bow[1]} AP {best_bow[0]:.4f}')

    # ---------------------------------------------------------------- segment: coordinate search
    print('\n' + '=' * 88)
    print('segment: coordinate search on the subsample (subsample AP)')
    print('=' * 88)
    cfg = {'n_list': (2, 4), 'max_len': 3, 'lam': 0.0, 'mu': 0.0, 'mass': 'ngrams', 'min_sim': 0.2}
    axes = [('n_list', [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]),
            ('max_len', [1, 2, 3, 4]),
            ('mass', ['ngrams', 'tokens']),
            ('lam', [0.0, 0.25, 0.5, 1.0]),
            ('mu', [0.0, 0.25, 0.5]),
            ('min_sim', [0.0, 0.1, 0.2, 0.3, 0.4])]
    base_ap = evaluate(lambda a, b: segment(a, b, **cfg), sub)
    print(f'  start {cfg} AP {base_ap:.4f}')
    for axis, values in axes:
        results = []
        for value in values:
            if value == cfg[axis]:
                results.append((base_ap, value))
                continue
            trial = dict(cfg, **{axis: value})
            t0 = time.perf_counter()
            ap = evaluate(lambda a, b: segment(a, b, **trial), sub)
            results.append((ap, value))
            print(f'  {axis}={str(value):<12} AP {ap:.4f}   ({time.perf_counter() - t0:.0f}s)')
        results.sort(key=lambda x: (-x[0], str(x[1])))
        base_ap, cfg[axis] = results[0]
        print(f'  -> {axis} = {cfg[axis]} (AP {base_ap:.4f})')
    print(f'  final config: {cfg}')

    # ---------------------------------------------------------------- winners on test
    print('\n' + '=' * 88)
    print('tuned configs: threshold from train, F1 on test')
    print('=' * 88)
    print(f'{"config":<44} {"test F1":>8} {"prec":>7} {"rec":>7} {"trainF1":>8} {"sec":>7}')
    report('char-nmd n=(2,)  [previous default]', lambda a, b: char_nmd(a, b, (2,)), train, test)
    report(f'char-nmd n={best_char[1]}  [tuned]', lambda a, b: char_nmd(a, b, best_char[1]), train, test)
    report('bow-nmd n=2  [previous default]', lambda a, b: bow_nmd(a, b, 2), train, test)
    report(f'bow-nmd n={best_bow[1]}  [tuned]', lambda a, b: bow_nmd(a, b, best_bow[1]), train, test)
    report('segment L3 lam0 ms.2 n=(2,4)  [previous]',
           lambda a, b: segment(a, b, (2, 4), 3, 0.0, 0.0, 'ngrams', 0.2), train, test)
    report(f'segment {cfg}  [tuned]', lambda a, b: segment(a, b, **cfg), train, test)

    print('\nfor reference on this split (published, supervised, all attributes):')
    print('  Magellan 43.6   DeepMatcher+ 62.8   Ditto 89.33')
    print('measured here previously (experiments/er_magellan_eval.py, name only, one threshold):')
    print('  tf-idf cos char-3gram 64.88   soft tf-idf 62.39   jaccard char-3gram 46.82')


if __name__ == '__main__':
    main(*sys.argv[1:])
