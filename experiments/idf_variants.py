"""
how should idf be added to nmd?

tf-idf cosine over char 3-grams scores 64.88 F1 on Abt-Buy where every unweighted overlap measure
(jaccard / dice / nmd / bow / segment) sits at 38-47. that gap could come from three separable things:

  1. idf weighting of the n-grams
  2. cosine (geometric-mean) normalization instead of dice (arithmetic-mean) normalization
  3. sublinear (1 + log tf) term frequency instead of raw counts

this measures each one, then measures idf inside nmd itself. weighting nmd is structurally free:
nmd_core accumulates similarity per n-gram TYPE independently (nmd/nmd_core.py:64-68), and idf is
constant within a type, so w(g) is a scalar multiplier on that type's contribution -- the 1-D EMD and
every matching decision are untouched. non-negative weights also keep it a metric.

    similarity_w = sum_g  w(g) * [ c1(g) + c2(g) - emd(locs1(g), locs2(g)) ]
    total_w      = sum_g  w(g) * c1(g)  +  sum_g w(g) * c2(g)
    distance_w   = total_w - similarity_w

run: python experiments/idf_variants.py
"""
import math
import sys
from collections import Counter

from experiments.er_magellan_eval import DATA
from experiments.er_magellan_eval import best_threshold
from experiments.er_magellan_eval import f1_at
from experiments.er_magellan_eval import read_split
from experiments.position_penalty import average_precision
from nmd.emd_1d import emd_1d_dp


def grams(text, n):
    text = f'\2{text}\3'
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def gram_locations(text, n):
    text = f'\2{text}\3'
    count = len(text) - n + 1
    locations = {}
    for i in range(count):
        locations.setdefault(text[i:i + n], []).append(i / max(1, count - 1))
    return locations


# ------------------------------------------------------------------ set / vector measures

def dice_set(a, b, n, w):
    sa, sb = set(grams(a, n)), set(grams(b, n))
    num = 2 * sum(w(g) for g in sa & sb)
    den = sum(w(g) for g in sa) + sum(w(g) for g in sb)
    return num / den if den else 1.0


def dice_multiset(a, b, n, w):
    ca, cb = Counter(grams(a, n)), Counter(grams(b, n))
    num = 2 * sum(w(g) * min(ca[g], cb[g]) for g in ca.keys() & cb.keys())
    den = sum(w(g) * c for g, c in ca.items()) + sum(w(g) * c for g, c in cb.items())
    return num / den if den else 1.0


def cosine_set(a, b, n, w):
    sa, sb = set(grams(a, n)), set(grams(b, n))
    num = sum(w(g) ** 2 for g in sa & sb)
    den = math.sqrt(sum(w(g) ** 2 for g in sa) * sum(w(g) ** 2 for g in sb))
    return num / den if den else 1.0


def cosine_logtf(a, b, n, w):
    ca, cb = Counter(grams(a, n)), Counter(grams(b, n))
    va = {g: (1 + math.log(c)) * w(g) for g, c in ca.items()}
    vb = {g: (1 + math.log(c)) * w(g) for g, c in cb.items()}
    num = sum(v * vb.get(g, 0.0) for g, v in va.items())
    den = math.sqrt(sum(v * v for v in va.values()) * sum(v * v for v in vb.values()))
    return num / den if den else 1.0


# ------------------------------------------------------------------ weighted nmd

def weighted_nmd(a, b, n, w, norm='dice'):
    loc_a, loc_b = gram_locations(a, n), gram_locations(b, n)
    total_a = sum(w(g) * len(v) for g, v in loc_a.items())
    total_b = sum(w(g) * len(v) for g, v in loc_b.items())
    similarity = 0.0
    for g, la in loc_a.items():
        lb = loc_b.get(g)
        if lb is not None:
            similarity += w(g) * (len(la) + len(lb) - emd_1d_dp(la, lb))
    if norm == 'dice':
        den = total_a + total_b
    else:  # geometric mean, the cosine analogue: forgiving of length mismatch
        den = 2 * math.sqrt(total_a * total_b)
    return similarity / den if den else 1.0


def main():
    train = read_split(DATA / 'train.txt', use_desc=False)
    test = read_split(DATA / 'test.txt', use_desc=False)
    norm = lambda t: ' '.join(t.casefold().split())
    corpus = [norm(r[i]) for r in train + test for i in (0, 1)]

    print(f'train {len(train)} pairs ({sum(r[2] for r in train)} positive), '
          f'test {len(test)} pairs ({sum(r[2] for r in test)} positive)')

    for n in (2, 3):
        df = Counter()
        for text in corpus:
            df.update(set(grams(text, n)))
        n_docs = len(corpus)
        idf_table = {g: math.log(n_docs / d) for g, d in df.items()}
        max_idf = math.log(n_docs)
        idf = lambda g: idf_table.get(g, max_idf)
        # cosine over idf-weighted vectors puts w(g)^2 in the numerator for a shared gram, so "idf" in a
        # dice-shaped measure and "idf" in a cosine are not the same weighting. treat the exponent as a
        # free parameter and test it separately from the normalization.
        idf2 = lambda g: idf_table.get(g, max_idf) ** 2
        flat = lambda g: 1.0

        variants = {
            f'dice set                 ': lambda a, b: dice_set(a, b, n, flat),
            f'dice set        + idf    ': lambda a, b: dice_set(a, b, n, idf),
            f'dice multiset            ': lambda a, b: dice_multiset(a, b, n, flat),
            f'dice multiset   + idf    ': lambda a, b: dice_multiset(a, b, n, idf),
            f'cosine set               ': lambda a, b: cosine_set(a, b, n, flat),
            f'cosine set      + idf    ': lambda a, b: cosine_set(a, b, n, idf),
            f'cosine log-tf   + idf    ': lambda a, b: cosine_logtf(a, b, n, idf),
            f'nmd                      ': lambda a, b: weighted_nmd(a, b, n, flat, 'dice'),
            f'nmd             + idf    ': lambda a, b: weighted_nmd(a, b, n, idf, 'dice'),
            f'nmd + geometric norm     ': lambda a, b: weighted_nmd(a, b, n, flat, 'geo'),
            f'nmd + idf + geometric norm': lambda a, b: weighted_nmd(a, b, n, idf, 'geo'),
            f'dice multiset   + idf^2  ': lambda a, b: dice_multiset(a, b, n, idf2),
            f'nmd             + idf^2  ': lambda a, b: weighted_nmd(a, b, n, idf2, 'dice'),
            f'nmd + idf^2 + geometric norm': lambda a, b: weighted_nmd(a, b, n, idf2, 'geo'),
        }

        print()
        print('=' * 82)
        print(f'char {n}-grams')
        print('=' * 82)
        print(f'{"variant":<28} {"train AP":>9} {"test F1":>8} {"prec":>7} {"rec":>7}')
        for name, fn in variants.items():
            tr = [fn(norm(a), norm(b)) for a, b, _ in train]
            te = [fn(norm(a), norm(b)) for a, b, _ in test]
            ap = average_precision(tr, [y for _, _, y in train])
            thresh, _ = best_threshold(tr, [y for _, _, y in train])
            f1, precision, recall = f1_at(te, [y for _, _, y in test], thresh)
            print(f'{name:<28} {ap:>9.3f} {100 * f1:>8.2f} {100 * precision:>7.2f} {100 * recall:>7.2f}')

        # is the idf exponent peaked at 2, or is "square it" just a point on a monotone climb?
        print(f'\n  weighted nmd, w(g) = idf(g)^p, geometric normalization:')
        print(f'  {"p":>5} {"train AP":>9} {"test F1":>8}')
        for p in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
            w = (lambda g, p=p: idf_table.get(g, max_idf) ** p)
            tr = [weighted_nmd(norm(a), norm(b), n, w, 'geo') for a, b, _ in train]
            te = [weighted_nmd(norm(a), norm(b), n, w, 'geo') for a, b, _ in test]
            ap = average_precision(tr, [y for _, _, y in train])
            thresh, _ = best_threshold(tr, [y for _, _, y in train])
            f1, _, _ = f1_at(te, [y for _, _, y in test], thresh)
            print(f'  {p:>5.1f} {ap:>9.3f} {100 * f1:>8.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
