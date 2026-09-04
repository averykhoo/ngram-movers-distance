"""
why is nmd worse than jaccard over character n-grams?

claim: normalized nmd is NOT "jaccard plus reordering tolerance". a set has no order, so jaccard is
already fully order-invariant. nmd is that same n-gram overlap with a penalty *charged for* positional
displacement -- it is strictly less order-tolerant than jaccard, not more.

concretely, dropping the position term from nmd leaves
    sim = 2 * |A n B|_multiset,  normalized by |A| + |B|
which is the Dice coefficient over n-gram multisets, and Dice is a monotone transform of Jaccard
(J = D / (2 - D)), hence rank-equivalent to it. so:

    nmd(normalized)  ==  dice over n-gram multisets  -  (positional discount)

this script verifies that identity and measures what the discount costs on Abt-Buy.

run: python experiments/position_penalty.py
"""
import sys
from collections import Counter

from experiments.er_magellan_eval import DATA
from experiments.er_magellan_eval import best_threshold
from experiments.er_magellan_eval import f1_at
from experiments.er_magellan_eval import read_split
from nmd.nmd_core import ngram_movers_distance


def grams(text, n, markers=True):
    text = f'\2{text}\3' if markers else text
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def dice_multiset(a, b, n):
    """nmd with the position term removed: 2 * multiset intersection / total grams"""
    ca, cb = Counter(grams(a, n)), Counter(grams(b, n))
    total = sum(ca.values()) + sum(cb.values())
    if not total:
        return 1.0
    overlap = sum(min(ca[g], cb[g]) for g in ca.keys() & cb.keys())
    return 2 * overlap / total


def dice_set(a, b, n):
    sa, sb = set(grams(a, n)), set(grams(b, n))
    return 2 * len(sa & sb) / (len(sa) + len(sb)) if sa or sb else 1.0


def jaccard_set(a, b, n):
    sa, sb = set(grams(a, n)), set(grams(b, n))
    return len(sa & sb) / len(sa | sb) if sa or sb else 1.0


def average_precision(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_pos = sum(labels)
    tp, total = 0, 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
            total += tp / rank
    return total / total_pos if total_pos else 0.0


def main():
    train = read_split(DATA / 'train.txt', use_desc=False)
    test = read_split(DATA / 'test.txt', use_desc=False)
    norm = lambda t: ' '.join(t.casefold().split())

    # --- 1. the identity: nmd <= dice-multiset, equal only when everything lines up ---
    print('=' * 88)
    print('1. is normalized nmd == dice over n-gram multisets minus a positional discount?')
    print('=' * 88)
    checks = [('hello', 'hello'), ('hello', 'yellow'), ('abcdef', 'defabc'),
              ('sony camera dsc123', 'dsc123 sony camera'), ('pineapple', 'pine apple')]
    print(f'{"a":<20} {"b":<20} {"nmd":>7} {"dice-ms":>8} {"discount":>9}')
    violations = 0
    for a, b in checks:
        for n in (2, 3, 4):
            s_nmd = ngram_movers_distance(a, b, n=n, invert=True, normalize=True)
            s_dice = dice_multiset(a, b, n)
            violations += s_nmd > s_dice + 1e-9
            if n == 3:
                print(f'{a:<20} {b:<20} {s_nmd:>7.3f} {s_dice:>8.3f} {s_dice - s_nmd:>9.3f}')
    print(f'\nnmd > dice-multiset in {violations} of {len(checks) * 3} checks (expected 0: the '
          f'discount is never negative)')

    # --- 2. what the discount costs, matches vs non-matches ---
    print()
    print('=' * 88)
    print('2. the positional discount on Abt-Buy test pairs (n=3)')
    print('=' * 88)
    pos_disc, neg_disc, pos_dice, neg_dice = [], [], [], []
    for a, b, y in test:
        a, b = norm(a), norm(b)
        d = dice_multiset(a, b, 3)
        s = ngram_movers_distance(a, b, n=3, invert=True, normalize=True)
        (pos_disc if y else neg_disc).append(d - s)
        (pos_dice if y else neg_dice).append(d)
    mean = lambda xs: sum(xs) / len(xs)
    print(f'{"":<14} {"dice-ms":>9} {"discount":>9} {"nmd":>9} {"disc/dice":>10}')
    print(f'{"matches":<14} {mean(pos_dice):>9.3f} {mean(pos_disc):>9.3f} '
          f'{mean(pos_dice) - mean(pos_disc):>9.3f} {mean(pos_disc) / mean(pos_dice):>10.1%}')
    print(f'{"non-matches":<14} {mean(neg_dice):>9.3f} {mean(neg_disc):>9.3f} '
          f'{mean(neg_dice) - mean(neg_disc):>9.3f} {mean(neg_disc) / mean(neg_dice):>10.1%}')
    print(f'{"separation":<14} {mean(pos_dice) - mean(neg_dice):>9.3f} {"":>9} '
          f'{(mean(pos_dice) - mean(pos_disc)) - (mean(neg_dice) - mean(neg_disc)):>9.3f}')

    # --- 3. ranking quality of each variant ---
    print()
    print('=' * 88)
    print('3. train-AP / test-F1 of each variant (threshold tuned on train)')
    print('=' * 88)
    variants = {}
    for n in (2, 3, 4, 5):
        variants[f'jaccard set n={n}'] = lambda a, b, n=n: jaccard_set(a, b, n)
        variants[f'dice set n={n}'] = lambda a, b, n=n: dice_set(a, b, n)
        variants[f'dice multiset n={n}'] = lambda a, b, n=n: dice_multiset(a, b, n)
        variants[f'nmd n={n}'] = lambda a, b, n=n: ngram_movers_distance(a, b, n=n, invert=True, normalize=True)

    print(f'{"variant":<24} {"train AP":>9} {"test F1":>8} {"prec":>7} {"rec":>7}')
    for name, fn in variants.items():
        tr = [fn(norm(a), norm(b)) for a, b, _ in train]
        te = [fn(norm(a), norm(b)) for a, b, _ in test]
        ap = average_precision(tr, [y for _, _, y in train])
        thresh, _ = best_threshold(tr, [y for _, _, y in train])
        f1, precision, recall = f1_at(te, [y for _, _, y in test], thresh)
        print(f'{name:<24} {ap:>9.3f} {100 * f1:>8.2f} {100 * precision:>7.2f} {100 * recall:>7.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
