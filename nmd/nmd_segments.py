"""
segment mover's distance: token-sequence matching that tolerates word splits / merges
and prefers in-order matches. see docs/bow-plan.md part 2.

WARNING: requires `scipy.optimize`, so it's not available by default in the `nmd` namespace

this is the simple reference implementation: every candidate segment pair is scored,
then a tiny 0/1 ILP picks a matching. no indexing, no pruning, no caching.
"""
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import scipy.optimize
import scipy.sparse

from nmd.nmd_core import ngram_movers_distance


class Segment(NamedTuple):
    start: int  # index of first token
    length: int  # number of tokens
    text: str  # separator-free concatenation of the tokens
    pos: float  # normalized center of the segment in [0, 1]
    mass: float  # tokens (mass='tokens') or n-gram count of the tokens averaged over n (mass='ngrams')


class Match(NamedTuple):
    segment_a: Segment
    segment_b: Segment
    sim: float  # normalized nmd similarity of the concatenated strings (0 to 1)
    shift: float  # |pos_a - pos_b|
    gain: float  # contribution to the total similarity


class Alignment(NamedTuple):
    total: float  # total mass on both sides; similarity + distance == total
    similarity: float  # sum of gains
    distance: float  # total - similarity
    matches: List[Match]
    unmatched_a: List[int]  # token indices
    unmatched_b: List[int]


def _token_grams(token: str, n: int) -> int:
    # number of n-grams in f'\2{token}\3', same as nmd_core
    return max(0, len(token) + 3 - n)


def _segments(tokens: Sequence[str], max_len: int, n_list: Tuple[int, ...], mass: str) -> List[Segment]:
    k = len(tokens)
    out = []
    for start in range(k):
        for length in range(1, min(max_len, k - start) + 1):
            # tokens are unit intervals, so the center of a segment covering [start, start + length) is
            # start + length / 2, and a segment covering the whole sequence is at 0.5 regardless of k
            pos = (start + length / 2) / k
            if mass == 'tokens':
                seg_mass = float(length)
            else:
                seg_mass = sum(_token_grams(tok, n) for tok in tokens[start:start + length] for n in n_list) / len(n_list)
            out.append(Segment(start, length, ''.join(tokens[start:start + length]), pos, seg_mass))
    return out


def _pair_raw(text_a: str, text_b: str, n: int) -> Tuple[float, int]:
    """un-normalized nmd similarity of two strings, and the total number of n-grams in both"""
    grams = _token_grams(text_a, n) + _token_grams(text_b, n)
    return ngram_movers_distance(text_a, text_b, n=n, invert=True), grams


def _pair_gain(s: Segment, t: Segment, n_list: Tuple[int, ...], mass: str, lam: float, mu: float) -> Tuple[float, float]:
    """returns (normalized sim, gain) for matching segment s to segment t"""
    sims, gains = [], []
    for n in n_list:
        raw_sim, grams = _pair_raw(s.text, t.text, n)
        sims.append(raw_sim / grams if grams else float(s.text == t.text))
        if mass == 'tokens':
            gains.append((s.length + t.length) * sims[-1])
        else:
            # token mass minus the nmd distance of the concatenations. the concatenation has a different
            # n-gram count from its tokens ((p - 1) * (n - 3) per side), so a perfect split / merge is exactly
            # zero distance only if mass is counted on the tokens, not on the concatenation
            gains.append(s.mass + t.mass - (grams - raw_sim))
    sim = sum(sims) / len(sims)
    gain = sum(gains) / len(gains)
    gain -= lam * (s.mass + t.mass) / 2 * abs(s.pos - t.pos)
    gain -= mu * (s.length + t.length - 2)
    return sim, gain


def align_segments(tokens_a: Sequence[str],
                   tokens_b: Sequence[str],
                   n: Union[int, Iterable[int]] = (2, 4),
                   max_len: int = 3,
                   lam: float = 1.0,
                   mu: float = 0.0,
                   mass: str = 'ngrams',
                   min_sim: float = 0.0,
                   ) -> Alignment:
    """
    find the max-gain matching of token segments between two token sequences

    mass='tokens':
        gain(s, t) = (p + q) * sim(s, t)                 p + q tokens, sim = normalized nmd of concatenations
    mass='ngrams':
        gain(s, t) = mass(s) + mass(t) - nmd_distance(concat(s), concat(t))
                                                         mass = n-gram count of the tokens
    both:
                   - lam * (mass(s) + mass(t)) / 2 * |Δpos|     positional shift -> in-order preference
                   - mu * (p + q - 2)                            merge penalty per boundary crossed

    :param tokens_a: token sequence (already casefolded / normalized as desired)
    :param tokens_b: token sequence
    :param n: n-gram size(s) used for the per-segment nmd similarity; multiple sizes are averaged
    :param max_len: max number of consecutive tokens in a segment (1 = plain token assignment)
    :param lam: weight of the positional shift penalty (0 = order-invariant, like bow_ngram_movers_distance)
    :param mu: merge penalty per boundary crossed (0 = merging is free)
    :param mass: 'tokens' (every token weighs 1) or 'ngrams' (a token weighs its n-gram count)
    :param min_sim: ignore segment pairs whose normalized similarity is below this (cuts bigram noise-floor matches)
    """
    n_list = (n,) if isinstance(n, int) else tuple(n)
    if mass not in ('tokens', 'ngrams'):
        raise ValueError(mass)
    tokens_a = list(tokens_a)
    tokens_b = list(tokens_b)
    k, m = len(tokens_a), len(tokens_b)

    segs_a = _segments(tokens_a, max_len, n_list, mass)
    segs_b = _segments(tokens_b, max_len, n_list, mass)
    total = sum(s.mass for s in segs_a if s.length == 1) + sum(t.mass for t in segs_b if t.length == 1)
    if k == 0 or m == 0:
        return Alignment(total, 0.0, total, [], list(range(k)), list(range(m)))

    # score every candidate pair, keep the ones with positive gain
    candidates: List[Match] = []
    for s in segs_a:
        for t in segs_b:
            sim, gain = _pair_gain(s, t, n_list, mass, lam, mu)
            if gain > 0 and sim >= min_sim:
                candidates.append(Match(s, t, sim, abs(s.pos - t.pos), gain))

    if not candidates:
        return Alignment(total, 0.0, total, [], list(range(k)), list(range(m)))

    # 0/1 ILP: one variable per candidate pair, each token covered at most once
    rows, cols = [], []
    for var_idx, match in enumerate(candidates):
        for tok in range(match.segment_a.start, match.segment_a.start + match.segment_a.length):
            rows.append(tok)
            cols.append(var_idx)
        for tok in range(match.segment_b.start, match.segment_b.start + match.segment_b.length):
            rows.append(k + tok)
            cols.append(var_idx)
    coverage = scipy.sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(k + m, len(candidates)))
    result = scipy.optimize.milp(c=-np.array([c.gain for c in candidates]),
                                 constraints=scipy.optimize.LinearConstraint(coverage, ub=np.ones(k + m)),
                                 integrality=np.ones(len(candidates)),
                                 bounds=scipy.optimize.Bounds(0, 1))
    assert result.success, result.message

    chosen = [c for c, x in zip(candidates, result.x) if x > 0.5]
    chosen.sort(key=lambda c: (c.segment_a.start, c.segment_b.start))
    used_a = {tok for c in chosen for tok in range(c.segment_a.start, c.segment_a.start + c.segment_a.length)}
    used_b = {tok for c in chosen for tok in range(c.segment_b.start, c.segment_b.start + c.segment_b.length)}
    similarity = sum(c.gain for c in chosen)
    return Alignment(total,
                     similarity,
                     total - similarity,
                     chosen,
                     [i for i in range(k) if i not in used_a],
                     [j for j in range(m) if j not in used_b])


def segment_movers_distance(tokens_a: Sequence[str],
                            tokens_b: Sequence[str],
                            n: Union[int, Iterable[int]] = (2, 4),
                            max_len: int = 3,
                            lam: float = 1.0,
                            mu: float = 0.0,
                            mass: str = 'ngrams',
                            min_sim: float = 0.0,
                            invert: bool = False,
                            normalize: bool = False,
                            ) -> float:
    """
    distance between two token sequences, tolerant of word splits / merges, preferring in-order matches
    see `align_segments` for the parameters and `explain_alignment` for introspection
    """
    alignment = align_segments(tokens_a, tokens_b, n=n, max_len=max_len, lam=lam, mu=mu, mass=mass, min_sim=min_sim)
    if alignment.total == 0:
        return 1.0 if (invert and normalize) else 0.0
    out = alignment.similarity if invert else alignment.distance
    if normalize:
        out /= alignment.total
    return out


def explain_alignment(tokens_a: Sequence[str], tokens_b: Sequence[str], alignment: Alignment) -> str:
    tokens_a = list(tokens_a)
    tokens_b = list(tokens_b)
    norm = alignment.similarity / alignment.total if alignment.total else 1.0
    lines = [f'similarity {alignment.similarity:.3f} / {alignment.total:g}  (normalized {norm:.3f})']
    for c in alignment.matches:
        a = ' '.join(tokens_a[c.segment_a.start:c.segment_a.start + c.segment_a.length])
        b = ' '.join(tokens_b[c.segment_b.start:c.segment_b.start + c.segment_b.length])
        shape = f'{c.segment_a.length}:{c.segment_b.length}'
        lines.append(f'  [{a}] <-> [{b}]  {shape}  sim={c.sim:.2f} shift={c.shift:.2f} gain={c.gain:.2f}')
    if alignment.unmatched_a:
        lines.append(f'  unmatched a: {[tokens_a[i] for i in alignment.unmatched_a]}')
    if alignment.unmatched_b:
        lines.append(f'  unmatched b: {[tokens_b[j] for j in alignment.unmatched_b]}')
    return '\n'.join(lines)
