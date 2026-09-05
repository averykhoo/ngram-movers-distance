"""
drive `ApproxWordListV7` as a ranker, so eval scripts stop scoring every candidate pairwise

every other eval script in this directory ranks by calling `metric(query, candidate)` once per
candidate. that measures the scoring formula but pays no attention to the index, and it is slow:
word_lookup_eval spends ~12 minutes and benchmark_all ~21 minutes largely on that loop.

the substitution is exact rather than approximate. V7's pruning bound is two-sided, so
`lookup` returns the same scores an exhaustive scan would (see the module docstring of
nmd/nmd_index_v7.py and tests/test_index_v7.py::test_pruning_never_drops_a_true_top_k), and
`lookup(dim=1, normalize=True, invert=True)` computes the mean over n of the per-n normalized
similarity -- which is exactly what `sum(ngram_movers_distance(a, b, n=n, invert=True,
normalize=True) for n in n_list) / len(n_list)` computes pairwise.

two differences are real and are NOT hidden by this module:

* an indexed word sharing no n-gram with the query never enters the candidate set, so it has no
  rank. `rank_of` reports those as missing, and callers should count them as a rank of infinity
  (contributing 0 to MRR), which is what the pairwise loop effectively did anyway -- a candidate
  with no shared n-gram scores 0 and sorts to the bottom.
* V7's idf is `log((V + 1) / df)` while the ad-hoc `weighted_nmd` in word_lookup_eval.py and
  idf_variants.py uses the unsmoothed `log(V / df)`. those two do not agree, so an idf-weighted
  metric ported to V7 is a *different* metric, not the same one computed faster. see
  nmd/nmd_index_v7.py:353-355 for why the smoothing is there.

verify_matches_pairwise() checks the first of those claims against the pairwise implementation
rather than asserting it, because "the index agrees with the formula" is exactly the kind of
thing that is worth measuring once per port instead of believing.
"""
import sys
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nmd.nmd_index_v7 import ApproxWordListV7

__all__ = ('build_ranker', 'verify_matches_pairwise')


def build_ranker(documents: Sequence[str],
                 n: Iterable[int],
                 idf_exponent: float = 0.0,
                 dim: float = 1.0,
                 normalize: bool = True,
                 position_weight: float = 1.0,
                 denominator: str = 'dice',
                 top_k: Optional[int] = None,
                 ) -> Callable[[str], List[Tuple[str, float]]]:
    """
    index `documents` once and return `rank(query) -> [(document, score), ...]`, best first

    `top_k` defaults to the whole corpus, which is what a full-list metric like MRR needs. that
    disables pruning, but most of V7's speed comes from scoring the vocabulary as numpy arrays
    rather than looping in python, so it is still far faster than the pairwise alternative --
    pruning is the smaller half of the win
    """
    index = ApproxWordListV7(n=n, idf_exponent=idf_exponent).add_words(documents)
    index._freeze()
    limit = top_k if top_k is not None else max(1, len(index))

    def rank(query: str) -> List[Tuple[str, float]]:
        return index.lookup(query, top_k=limit, dim=dim, normalize=normalize,
                            position_weight=position_weight, denominator=denominator)

    rank.index = index  # exposed so callers can report len(index) or reuse the build
    return rank


def rank_of(ranked: Sequence[Tuple[str, float]], answers) -> Optional[int]:
    """1-based rank of the first correct document, or None if none was retrieved at all"""
    answers = {answers} if isinstance(answers, str) else set(answers)
    for position, (document, _score) in enumerate(ranked, 1):
        if document in answers:
            return position
    return None


def verify_matches_pairwise(documents: Sequence[str],
                            queries: Sequence[str],
                            n: Sequence[int],
                            pairwise: Callable[[str, str], float],
                            tolerance: float = 1e-9,
                            ) -> Dict[str, float]:
    """
    check the indexed ranking against the pairwise one it replaces

    returns a dict of counts rather than asserting, so a port can print the evidence. an exact
    match is the expected outcome for unweighted nmd; anything else means the substitution
    changed the metric and the caller should say so instead of quietly reporting new numbers
    """
    rank = build_ranker(documents, n)
    worst_score_diff = 0.0
    ranking_differences = 0
    missing = 0
    for query in queries:
        indexed = rank(query)
        indexed_scores = dict(indexed)
        exhaustive = sorted(((pairwise(query, document), document) for document in documents),
                            key=lambda pair: (-pair[0], pair[1]))
        # only compare documents the index actually returned; the rest scored zero by definition
        for score, document in exhaustive:
            if document in indexed_scores:
                worst_score_diff = max(worst_score_diff, abs(score - indexed_scores[document]))
            elif score > tolerance:
                missing += 1
        indexed_order = [document for document, _ in indexed]
        expected_order = [document for _, document in exhaustive if document in indexed_scores]
        if indexed_order != expected_order:
            ranking_differences += 1
    return {'queries': len(queries),
            'worst_score_diff': worst_score_diff,
            'ranking_differences': ranking_differences,
            'nonzero_documents_missed': missing}
