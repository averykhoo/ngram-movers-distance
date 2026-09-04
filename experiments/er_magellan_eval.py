"""
put nmd / segment mover's distance on the SAME yardstick as the published Abt-Buy numbers

the entity-matching literature (Magellan 43.6, DeepMatcher+ 62.8, Ditto 89.33 F1) evaluates on the
ER-Magellan Abt-Buy split: 9575 blocked candidate pairs, 5743 train / 1916 valid / 1916 test, labelled
match / non-match. those systems are supervised and read every attribute; here each string metric is
turned into the simplest possible classifier -- one global threshold, tuned on train, applied to test --
using the `name` attribute only. that is a much weaker learner, so the comparison is a lower bound.

data: files from https://github.com/megagonlabs/ditto/tree/master/data/er_magellan/Textual/Abt-Buy
      -> .scratch/data/er_magellan/{train,valid,test}.txt
run:  python experiments/er_magellan_eval.py [--desc]
"""
import functools
import math
import sys
import time
from collections import Counter
from pathlib import Path

import nmd.nmd_segments
from experiments.baseline_eval import char_ngrams
from experiments.baseline_eval import cosine
from experiments.baseline_eval import jaccard
from experiments.baseline_eval import monge_elkan
from experiments.baseline_eval import soft_tfidf
from experiments.baseline_eval import tfidf_vector
from experiments.baseline_eval import tokenize
from nmd.nmd_bow import bow_ngram_movers_distance
from nmd.nmd_core import ngram_movers_distance
from nmd.nmd_segments import segment_movers_distance

DATA = Path(__file__).resolve().parent.parent / '.scratch' / 'data' / 'er_magellan'

nmd.nmd_segments._pair_raw = functools.lru_cache(maxsize=None)(nmd.nmd_segments._pair_raw)


def parse_entity(blob, use_desc):
    fields = {}
    for chunk in blob.split(' COL '):
        chunk = chunk[4:] if chunk.startswith('COL ') else chunk
        key, _, value = chunk.partition(' VAL ')
        fields[key.strip()] = value.strip()
    text = fields.get('name', '')
    if use_desc:
        text = f'{text} {fields.get("description", "")}'.strip()
    return text


def read_split(path, use_desc):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            left, right, label = line.rstrip('\n').split('\t')
            rows.append((parse_entity(left, use_desc), parse_entity(right, use_desc), int(label)))
    return rows


def best_threshold(scores, labels):
    """threshold maximizing F1, scanning every midpoint between observed scores"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_pos = sum(labels)
    best_f1, best_thresh, tp = 0.0, 1.0, 0
    for rank, i in enumerate(order, 1):
        tp += labels[i]
        precision, recall = tp / rank, tp / total_pos
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        if f1 > best_f1:
            best_f1 = f1
            nxt = scores[order[rank]] if rank < len(order) else scores[i] - 1e-9
            best_thresh = (scores[i] + nxt) / 2
    return best_thresh, best_f1


def f1_at(scores, labels, thresh):
    tp = sum(1 for s, y in zip(scores, labels) if s >= thresh and y)
    fp = sum(1 for s, y in zip(scores, labels) if s >= thresh and not y)
    fn = sum(1 for s, y in zip(scores, labels) if s < thresh and y)
    if not tp:
        return 0.0, 0.0, 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall), precision, recall


def main(*args):
    use_desc = '--desc' in args
    train = read_split(DATA / 'train.txt', use_desc)
    test = read_split(DATA / 'test.txt', use_desc)
    print(f'field(s): {"name + description" if use_desc else "name only"}')
    print(f'train {len(train)} pairs ({sum(r[2] for r in train)} positive), '
          f'test {len(test)} pairs ({sum(r[2] for r in test)} positive)')

    # idf over every entity string in the split
    corpus = [r[i] for r in train + test for i in (0, 1)]
    tok_df, chr_df = Counter(), Counter()
    for text in corpus:
        tok_df.update(set(tokenize(text)))
        chr_df.update(set(char_ngrams(text.casefold())))
    n_docs = len(corpus)
    tok_idf = {t: math.log(n_docs / df) for t, df in tok_df.items()}
    chr_idf = {g: math.log(n_docs / df) for g, df in chr_df.items()}

    @functools.lru_cache(maxsize=None)
    def features(text):
        toks = tokenize(text)
        grams = char_ngrams(text.casefold())
        return {'toks': toks, 'joined': ' '.join(toks), 'tokset': set(toks), 'gramset': set(grams),
                'tokvec': tfidf_vector(toks, tok_idf), 'gramvec': tfidf_vector(grams, chr_idf)}

    METRICS = {
        'jaccard token': lambda a, b: jaccard(a['tokset'], b['tokset']),
        'jaccard char-3gram': lambda a, b: jaccard(a['gramset'], b['gramset']),
        'tf-idf cos token': lambda a, b: cosine(a['tokvec'], b['tokvec']),
        'tf-idf cos char-3gram': lambda a, b: cosine(a['gramvec'], b['gramvec']),
        'monge-elkan (JW)': lambda a, b: monge_elkan(a['toks'], b['toks']),
        'soft tf-idf (JW .9)': lambda a, b: soft_tfidf(a['tokvec'], b['tokvec']),
        'char-nmd(2)': lambda a, b: ngram_movers_distance(a['joined'], b['joined'], n=2, invert=True, normalize=True),
        'char-nmd(2,4)': lambda a, b: (ngram_movers_distance(a['joined'], b['joined'], n=2, invert=True, normalize=True)
                                       + ngram_movers_distance(a['joined'], b['joined'], n=4, invert=True,
                                                               normalize=True)) / 2,
        'bow-nmd(2)': lambda a, b: bow_ngram_movers_distance(a['toks'], b['toks'], n=2, invert=True, normalize=True),
        'segment L3 lam0': lambda a, b: segment_movers_distance(a['toks'], b['toks'], max_len=3, lam=0.0,
                                                                invert=True, normalize=True),
        'segment L3 lam0 ms.2': lambda a, b: segment_movers_distance(a['toks'], b['toks'], max_len=3, lam=0.0,
                                                                     min_sim=0.2, invert=True, normalize=True),
    }

    print()
    print(f'{"metric":<24} {"test F1":>8} {"prec":>7} {"rec":>7} {"thresh":>7} {"trainF1":>8} {"sec":>7}')
    for name, metric in METRICS.items():
        t0 = time.perf_counter()
        train_scores = [metric(features(a), features(b)) for a, b, _ in train]
        thresh, train_f1 = best_threshold(train_scores, [y for _, _, y in train])
        test_scores = [metric(features(a), features(b)) for a, b, _ in test]
        f1, precision, recall = f1_at(test_scores, [y for _, _, y in test], thresh)
        elapsed = time.perf_counter() - t0
        print(f'{name:<24} {100 * f1:>8.2f} {100 * precision:>7.2f} {100 * recall:>7.2f} '
              f'{thresh:>7.3f} {100 * train_f1:>8.2f} {elapsed:>7.1f}')

    print()
    print('published on this split (supervised, all attributes):')
    print('  Magellan 43.6   DeepMatcher+ 62.8   Ditto 89.33')


if __name__ == '__main__':
    main(*sys.argv[1:])
