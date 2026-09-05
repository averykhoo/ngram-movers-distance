"""
ranked-retrieval metrics, factored out of the copy-pasted `rank = next(...)` loops

every eval script in this directory (baseline_eval, word_lookup_eval, segment_eval,
benchmark_all, idf_exponent) inlines its own ranking loop, and every one of them takes only
the rank of the *first* relevant item -- fine when there is exactly one right answer, but it
silently collapses multi-answer ground truth. these functions take the whole relevant set.

all of them take `ranked`, a list of ids best-first, and `relevant`, the set of ids that are
correct for that query. ids not in `ranked` are treated as ranked at infinity, i.e. a relevant
item the system never retrieved contributes 0 rather than being dropped from the denominator.
that makes every number here a *cutoff* metric: reporting them requires stating the cutoff, or
they are not comparable across runs with different `top_k`.

run `python experiments/ranking_metrics.py` to execute the worked examples below.
"""
import math
from typing import Collection
from typing import Hashable
from typing import List
from typing import Sequence

__all__ = ('reciprocal_rank', 'hit_at_k', 'recall_at_k', 'precision_at_k', 'r_precision',
           'average_precision', 'ndcg_at_k')


def reciprocal_rank(ranked: Sequence[Hashable], relevant: Collection[Hashable]) -> float:
    """
    1 / (rank of the first relevant item), or 0.0 if none was retrieved

    >>> reciprocal_rank(['a', 'b', 'c'], {'b'})
    0.5
    >>> reciprocal_rank(['a', 'b', 'c'], {'z'})
    0.0
    """
    relevant = set(relevant)
    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int = 1) -> float:
    """
    1.0 if any relevant item appears in the top k, else 0.0

    >>> hit_at_k(['a', 'b'], {'b'}, k=1)
    0.0
    >>> hit_at_k(['a', 'b'], {'b'}, k=2)
    1.0
    """
    relevant = set(relevant)
    return float(any(item in relevant for item in ranked[:k]))


def recall_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """
    fraction of all relevant items that appear in the top k

    >>> recall_at_k(['a', 'b', 'c'], {'a', 'c'}, k=2)
    0.5
    >>> recall_at_k(['a', 'b', 'c'], {'a', 'c'}, k=3)
    1.0
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / len(relevant)


def precision_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """
    fraction of the top k that is relevant

    the denominator is k even when fewer than k results came back, so a system that returns two
    perfect results and stops does not out-score one that returns five. that is the right
    convention here because `top_k` is fixed across a sweep: a short list means the index pruned
    everything else away, not that the task had nothing more to offer

    >>> precision_at_k(['a', 'x', 'b'], {'a', 'b'}, k=2)
    0.5
    >>> precision_at_k(['a', 'b'], {'a', 'b'}, k=4)
    0.5
    """
    if k <= 0:
        return 0.0
    relevant = set(relevant)
    return sum(1 for item in ranked[:k] if item in relevant) / k


def r_precision(ranked: Sequence[Hashable], relevant: Collection[Hashable]) -> float:
    """
    precision at k == len(relevant), the cutoff at which precision and recall coincide

    self-calibrating across queries with different numbers of right answers, which is why it is
    worth having alongside a fixed-cutoff precision on a task where that count varies

    >>> r_precision(['a', 'b', 'x'], {'a', 'b'})
    1.0
    >>> r_precision(['a', 'x', 'b'], {'a', 'b'})
    0.5
    >>> r_precision(['x', 'a'], {'a'})
    0.0
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return precision_at_k(ranked, relevant, k=len(relevant))


def average_precision(ranked: Sequence[Hashable], relevant: Collection[Hashable]) -> float:
    """
    mean of precision@i taken at each rank i holding a relevant item

    divided by the total number of relevant items, NOT by the number retrieved, so relevant
    items missing from `ranked` drag the score down instead of being ignored. averaging this
    over queries is MAP.

    perfect ranking, both relevant items first:
    >>> average_precision(['a', 'b', 'c'], {'a', 'b'})
    1.0

    one relevant item at rank 1, the other never retrieved -- precision@1 is 1.0 but it is
    averaged over 2 relevant items:
    >>> average_precision(['a', 'x', 'y'], {'a', 'b'})
    0.5

    relevant at ranks 2 and 3: (1/2 + 2/3) / 2
    >>> round(average_precision(['x', 'a', 'b'], {'a', 'b'}), 6)
    0.583333
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    num_hits = 0
    total = 0.0
    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            num_hits += 1
            total += num_hits / rank
    return total / len(relevant)


def ndcg_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """
    binary-gain normalized discounted cumulative gain at cutoff k

    gain is 1 for a relevant item and 0 otherwise, discounted by 1 / log2(rank + 1). the ideal
    ranking puts min(len(relevant), k) relevant items at the top, so a query with more relevant
    items than k can still reach 1.0.

    >>> ndcg_at_k(['a', 'b', 'c'], {'a', 'b'}, k=3)
    1.0
    >>> ndcg_at_k(['x', 'y', 'z'], {'a'}, k=3)
    0.0

    single relevant item at rank 2: (1/log2(3)) / (1/log2(2)) == 1/log2(3)
    >>> round(ndcg_at_k(['x', 'a'], {'a'}, k=2), 6)
    0.63093
    """
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, item in enumerate(ranked[:k], 1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def _self_check() -> None:
    """
    worked examples that a plain doctest run would not catch, plus the sabotage checks

    an assurance step that passes when the thing it guards is broken is worse than no check,
    so each of these confirms the metric actually *moves* in the direction it should.
    """
    # MAP must reward putting relevant items higher, not merely retrieving them
    assert average_precision(['a', 'b', 'x'], {'a', 'b'}) > average_precision(['x', 'a', 'b'], {'a', 'b'})
    # ... and must not be fooled by a longer list that finds the same item at the same rank
    assert average_precision(['a'], {'a'}) == average_precision(['a', 'x', 'y', 'z'], {'a'}) == 1.0
    # a relevant item beyond the returned list costs score rather than being dropped
    assert average_precision(['a'], {'a', 'b'}) == 0.5
    # NDCG must be discounted, i.e. rank 1 strictly better than rank 2
    assert ndcg_at_k(['a', 'x'], {'a'}, k=2) > ndcg_at_k(['x', 'a'], {'a'}, k=2)
    # NDCG at a cutoff shorter than the relevant set can still be perfect
    assert ndcg_at_k(['a', 'b'], {'a', 'b', 'c'}, k=2) == 1.0
    # cutoff is respected: the same list scores 0 when the hit falls outside k
    assert ndcg_at_k(['x', 'a'], {'a'}, k=1) == 0.0
    assert hit_at_k(['x', 'a'], {'a'}, k=1) == 0.0
    assert recall_at_k(['x', 'a'], {'a'}, k=1) == 0.0
    assert precision_at_k(['x', 'a'], {'a'}, k=1) == 0.0
    # precision@k divides by k, not by the length of a short result list
    assert precision_at_k(['a'], {'a'}, k=5) == 0.2
    assert precision_at_k(['a'], {'a'}, k=1) == 1.0
    # r-precision follows the size of the relevant set, not a fixed cutoff
    assert r_precision(['a', 'b', 'x'], {'a', 'b'}) == 1.0
    assert r_precision(['a', 'x', 'b'], {'a', 'b'}) == 0.5
    # ... so a ranking that is perfect for one query size is not automatically perfect for another
    assert r_precision(['a', 'x'], {'a'}) == 1.0
    assert r_precision(['a', 'x'], {'a', 'b'}) == 0.5
    # empty relevant set is 0 everywhere rather than a ZeroDivisionError
    for fn in (lambda: average_precision(['a'], set()),
               lambda: ndcg_at_k(['a'], set(), k=1),
               lambda: recall_at_k(['a'], set(), k=1),
               lambda: precision_at_k(['a'], set(), k=1),
               lambda: r_precision(['a'], set()),
               lambda: reciprocal_rank(['a'], set())):
        assert fn() == 0.0
    # an empty ranked list is 0 everywhere
    empty: List[Hashable] = []
    assert average_precision(empty, {'a'}) == 0.0
    assert ndcg_at_k(empty, {'a'}, k=5) == 0.0
    assert precision_at_k(empty, {'a'}, k=5) == 0.0
    assert r_precision(empty, {'a'}) == 0.0
    assert reciprocal_rank(empty, {'a'}) == 0.0
    # a non-positive cutoff is 0, not a ZeroDivisionError
    assert precision_at_k(['a'], {'a'}, k=0) == 0.0
    assert ndcg_at_k(['a'], {'a'}, k=0) == 0.0
    print('ranking_metrics self-check: ok')


if __name__ == '__main__':
    import doctest

    results = doctest.testmod(verbose=False)
    print(f'doctests: {results.attempted - results.failed}/{results.attempted} passed')
    assert results.failed == 0
    _self_check()
