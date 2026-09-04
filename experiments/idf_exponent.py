"""
four questions about idf weighting, measured on Abt-Buy

1. does a high idf exponent cost anything at runtime?
2. does the exponent help tf-idf cosine too, or only nmd?
3. does the positional information help AT ALL, holding everything else fixed?
   (isolating it needs a controlled pair: identical weighting, identical normalization, differing only
   in whether a shared n-gram's overlap is min(c1, c2) or nmd's emd-discounted matched mass)
4. can the two be combined?

run: python experiments/idf_exponent.py
"""
import math
import sys
import time
from collections import Counter

from experiments.baseline_eval import read_csv
from experiments.baseline_eval import tokenize
from experiments.er_magellan_eval import DATA as ER_DATA
from experiments.er_magellan_eval import best_threshold
from experiments.er_magellan_eval import f1_at
from experiments.er_magellan_eval import read_split
from experiments.position_penalty import average_precision
from nmd.emd_1d import emd_1d_dp

ABT_DATA = ER_DATA.parent


def gram_info(text, n):
    """{gram: (count, [positions])} with the same markers and normalized positions nmd uses"""
    text = f'\2{text}\3' if n > 1 else text
    count = len(text) - n + 1
    out = {}
    for i in range(count):
        out.setdefault(text[i:i + n], []).append(i / max(1, count - 1))
    return out


def soft_cosine(a, b, n, w, positional, log_tf=False):
    """
    one family, two settings. for each shared n-gram type:
        positional=False -> overlap = min(c1, c2)                        (order-blind)
        positional=True  -> overlap = (c1 + c2 - emd(locs1, locs2)) / 2  (nmd's matched mass, same scale)
    numerator sum w(g)^2 * overlap, denominator the geometric mean of the two self-overlaps.
    everything except the overlap definition is held fixed, so the difference IS the ordering signal.
    """
    ga, gb = gram_info(a, n), gram_info(b, n)
    tf = (lambda c: 1 + math.log(c)) if log_tf else (lambda c: c)
    norm_a = sum(w(g) ** 2 * tf(len(v)) for g, v in ga.items())
    norm_b = sum(w(g) ** 2 * tf(len(v)) for g, v in gb.items())
    if not norm_a or not norm_b:
        return 1.0 if not ga and not gb else 0.0
    num = 0.0
    for g, la in ga.items():
        lb = gb.get(g)
        if lb is None:
            continue
        if positional:
            overlap = (len(la) + len(lb) - emd_1d_dp(la, lb)) / 2
        else:
            overlap = min(len(la), len(lb))
        num += w(g) ** 2 * (tf(overlap) if log_tf and overlap > 0 else overlap)
    return num / math.sqrt(norm_a * norm_b)


def main():
    train = read_split(ER_DATA / 'train.txt', use_desc=False)
    test = read_split(ER_DATA / 'test.txt', use_desc=False)
    norm = lambda t: ' '.join(t.casefold().split())
    corpus = [norm(r[i]) for r in train + test for i in (0, 1)]

    def make_idf(n):
        df = Counter()
        for text in corpus:
            df.update(set(gram_info(text, n)))
        n_docs = len(corpus)
        table = {g: math.log(n_docs / d) for g, d in df.items()}
        ceiling = math.log(n_docs)
        return lambda g, p: table.get(g, ceiling) ** p

    def score(fn):
        tr = [fn(norm(a), norm(b)) for a, b, _ in train]
        te = [fn(norm(a), norm(b)) for a, b, _ in test]
        ap = average_precision(tr, [y for _, _, y in train])
        thresh, _ = best_threshold(tr, [y for _, _, y in train])
        f1, _, _ = f1_at(te, [y for _, _, y in test], thresh)
        return ap, f1

    for n in (2, 3):
        idf = make_idf(n)
        print('=' * 84)
        print(f'char {n}-grams')
        print('=' * 84)

        # --- q1: does p cost runtime? ---
        pairs = [(norm(a), norm(b)) for a, b, _ in test[:400]]
        for p in (0, 25):
            w = lambda g, p=p: idf(g, p)
            t0 = time.perf_counter()
            for a, b in pairs:
                soft_cosine(a, b, n, w, positional=True)
            print(f'  timing: p={p:<3} {1e6 * (time.perf_counter() - t0) / len(pairs):>7.1f} us/pair')

        # --- q2 + q3: exponent sweep, order-blind vs positional, all else fixed ---
        print(f'\n  {"p":>5} | {"order-blind AP":>14} {"F1":>7} | {"positional AP":>14} {"F1":>7} | {"AP delta":>9}')
        for p in (0, 1, 2, 3, 4, 6, 10, 15, 25):
            w = lambda g, p=p: idf(g, p)
            ap0, f10 = score(lambda a, b, w=w: soft_cosine(a, b, n, w, positional=False))
            ap1, f11 = score(lambda a, b, w=w: soft_cosine(a, b, n, w, positional=True))
            print(f'  {p:>5} | {ap0:>14.3f} {100 * f10:>7.2f} | {ap1:>14.3f} {100 * f11:>7.2f} | {ap1 - ap0:>+9.3f}')

    # --- q4: does blending the two help? ---
    print()
    print('=' * 84)
    print('blending order-blind (n=2, p=2) with positional (n=2, p=2)')
    print('=' * 84)
    idf2 = make_idf(2)
    w2 = lambda g: idf2(g, 2)
    cache = {}
    for a, b, _ in train + test:
        key = (norm(a), norm(b))
        if key not in cache:
            cache[key] = (soft_cosine(*key, 2, w2, positional=False), soft_cosine(*key, 2, w2, positional=True))
    print(f'  {"alpha":>6} {"train AP":>9} {"test F1":>8}   (0 = fully positional, 1 = fully order-blind)')
    for alpha in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        blend = lambda a, b: (lambda v: alpha * v[0] + (1 - alpha) * v[1])(cache[(a, b)])
        ap, f1 = score(blend)
        print(f'  {alpha:>6.2f} {ap:>9.3f} {100 * f1:>8.2f}')

    # --- does the tuned exponent transfer to the retrieval task? ---
    print()
    print('=' * 84)
    print('retrieval check: does a high exponent still win on ranking? (100 Abt queries x 1092 Buy)')
    print('=' * 84)
    import random
    from collections import defaultdict
    abt = {r['id']: r['name'] for r in read_csv(ABT_DATA / 'Abt.csv')}
    buy = {r['id']: r['name'] for r in read_csv(ABT_DATA / 'Buy.csv')}
    truth = defaultdict(set)
    for r in read_csv(ABT_DATA / 'abt_buy_perfectMapping.csv'):
        truth[r['idAbt']].add(r['idBuy'])
    queries = random.Random(0).sample(sorted(truth), 100)
    buy_norm = {bid: ' '.join(tokenize(name)) for bid, name in buy.items()}
    idf2 = make_idf(2)
    print(f'  {"p":>5} | {"order-blind hit@1":>18} {"MRR":>7} | {"positional hit@1":>17} {"MRR":>7}')
    for p in (0, 2, 6, 15):
        w = lambda g, p=p: idf2(g, p)
        row = []
        for positional in (False, True):
            hits = rr = 0
            for aid in queries:
                qa = ' '.join(tokenize(abt[aid]))
                scored = sorted(((soft_cosine(qa, buy_norm[bid], 2, w, positional), bid) for bid in buy),
                                key=lambda x: -x[0])
                rank = next(i for i, (_, bid) in enumerate(scored, 1) if bid in truth[aid])
                hits += rank == 1
                rr += 1 / rank
            row.append((hits / len(queries), rr / len(queries)))
        print(f'  {p:>5} | {row[0][0]:>18.3f} {row[0][1]:>7.3f} | {row[1][0]:>17.3f} {row[1][1]:>7.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
