# N-gram Mover's Distance

* A string similarity measure based on Earth Mover's Distance
* See [ngram-movers-distance](https://github.com/averykhoo/ngram-movers-distance) for code

## Why another string matching algorithm?

* Edit distance really wasn't cutting it when I needed to look up a dictionary for a misspelled word
    * With an edit distance of 1 or 2, the results are not useful since the target word isn't found
    * With a distance >=5, the results are meaningless since it contains half the dictionary
    * Same goes for Damerau-Levenshtein
* Also, edit distance is pretty slow when looking up long words in a large dictionary
    * Even after building a finite state automaton or using a trie to optimize lookup
    * NMD was designed with indexing in mind
        * A simpler index could be used for Jaccard or cosine similarity over ngrams
* EMD (and hence NMD) can be optimized to run really fast with some constraints
    * Values are 1-dimensional scalars
    * Values are always quantized

# Installation and dependencies

**`nmd` itself has no dependencies, and is meant to stay that way.** `import nmd` gives you
the two things that are pure python, and pulls in nothing third-party:

```python
from nmd import ngram_movers_distance  # the metric
from nmd import WordList               # the dictionary index
```

Everything else is a deliberate opt-in: the extra modules are *not* exported from the `nmd`
namespace, so installing the package never obliges you to install their dependencies. Import
them by their full path and install what they need yourself:

| import | needs | what it is for |
|---|---|---|
| `nmd.nmd_index_v7.ApproxWordListV7` | `numpy` | same index as `WordList`, ~20-29x faster lookups, ~11-14x less memory |
| `nmd.nmd_bow.bow_ngram_movers_distance` | `scipy` | compare two sequences of tokens |
| `nmd.nmd_segments.segment_movers_distance` | `scipy`, `numpy` | as above, but tolerant of split / merged words |
| `nmd.nmd_word_set.WordSet` | `pyroaring`, `regex` | a mutable, set-like container (`add` / `discard` / `in`) with unicode normalization |

This is why `pyproject.toml` declares no `dependencies` and no extras. If you want the fast
index, `pip install numpy` alongside `nmd`.

# Usage

## `ngram_movers_distance()`

* string distance metric, use this to compare two strings

```python
from nmd import ngram_movers_distance

# n-gram mover's distance
print(ngram_movers_distance(f'hello', f'yellow'))

# n=1 compares character multisets by position, with no adjacency information
# on typo correction it beats every larger n, see docs/bow-plan.md part 7
print(ngram_movers_distance(f'hello', f'yellow', n=1))

# similarity (inverted distance)
print(ngram_movers_distance(f'hello', f'yellow', invert=True))

# distance, normalized to the range 0 to 1 (inclusive of 0 and 1)
print(ngram_movers_distance(f'hello', f'yellow', normalize=True))

# similarity, normalized to the range 0 to 1 (inclusive of 0 and 1)
print(ngram_movers_distance(f'hello', f'yellow', invert=True, normalize=True))
```

## `WordList`

* use this for dictionary lookups of words

```python
from nmd import WordList

# get words from a text file
with open(f'dictionary.txt', encoding=f'utf8') as f:
    words = set(f.read().split())

# index words
# combined 2- and 4-grams seem to work best as index keys; note that for *scoring* alone,
# n=(1,2) beats (2,4) by 11 points of hit@1 on typo correction (docs/bow-plan.md part 7),
# but unigrams have almost no selectivity as a filter, so they are not useful for the index
word_list = WordList((2, 4), filter_n=0)
for word in words:
    word_list.add_word(word)

# lookup a word
print(word_list.lookup(f'asalamalaikum'))  # -> 'assalamualaikum'
print(word_list.lookup(f'walaikumalasam'))  # -> 'waalaikumsalam'
```

## `ApproxWordListV7`

* WARNING: requires `numpy`, so it's not available by default in the `nmd` namespace
* same idea as `WordList`, but the lookup is vectorised instead of looping in python
* the index score is the *exact* nmd similarity, not an approximation of it -- the n-gram
  count bound used to avoid scoring the whole vocabulary is two-sided, so it can only
  discard words that genuinely cannot reach the top k

measured against `WordList` (`ApproxWordListV6`, `filter_n=3`) with `n=(2, 4)`:

| vocabulary | build | index size | lookup |
|------------|-------|------------|--------|
| 41.5k, V6  | 2.5s  | 168.7 MB   | 102.5 ms |
| 41.5k, V7  | 3.2s  | 14.8 MB    | 5.2 ms |
| 250k, V6   | 31.5s | 1137.1 MB  | 798.4 ms |
| 250k, V7   | 21.1s | 82.4 MB    | 27.2 ms |

the gap widens with vocabulary size (20x at 41.5k, 29x at 250k) because V6 loops in python
over every word touched by any shared n-gram, and that set grows with the vocabulary

```python
from nmd.nmd_index_v7 import ApproxWordListV7

word_list = ApproxWordListV7((2, 4))  # combined 2- and 4-grams seem to work best
word_list.add_words(words)  # or add_word() one at a time

# optionally weight rare n-grams more heavily: idf(gram) ** idf_exponent
# 0.0 (the default) is unweighted and takes the original code path bit-for-bit.
# helps on product names, HURTS on typo correction -- see docs/bow-plan.md parts 6-7
weighted = ApproxWordListV7((2, 4), idf_exponent=2.0)

# lookup returns [(word, score), ...], most similar first
print(word_list.lookup(f'asalamalaikum'))
print(word_list.lookup(f'walaikumalasam', top_k=3))

# two more scoring knobs, both defaulting to the plain nmd behaviour:
#   position_weight  how much of the positional displacement to charge for, in [0, 1].
#                    1.0 (default) is nmd exactly; 0.0 discards position and leaves dice
#                    over n-gram multisets, which is BETTER on product names and worse on
#                    typos (docs/bow-plan.md part 9)
#   denominator      'dice' (default) normalizes by total_query + total_word;
#                    'geo' by 2 * sqrt(total_query * total_word), which softens a length
#                    mismatch instead of charging the full difference
print(word_list.lookup(f'sony camera dsc123', position_weight=0.0, denominator='geo'))
```

### which parameters to use

`docs/bow-plan.md` part 9 searched a 432-point grid over six retrieval benchmarks. The tasks
split into two clusters that disagree on **every** knob, so there is no single best setting:

| your data looks like | `n` | `idf_exponent` | `dim` | `normalize` | `position_weight` |
|---|---|---|---|---|---|
| short strings, typos | `(1, 2)` | 0.0-0.5 | 2 | `True` | 1.0 |
| product names, records | `(2, 3)` | 3.0 | 1 | `False` | 0.0 |
| unknown / mixed | `(1, 2)` | 0.5 | 2 | `True` | 1.0, with `denominator='geo'` |

⚠ `normalize` and `idf_exponent` substitute for each other -- both counteract length bias, so
turning both up over-corrects. On product names `normalize=True` is worth **+0.145** MAP at
`idf_exponent=0` and **-0.079** at `idf_exponent=3`.

⚠ `n=(1, 2)` scores best on every typo benchmark but is a poor *index key*: unigrams have no
selectivity, and on 138-character documents `n=(1,)` scores MAP 0.008. Use it for short strings
only.

* differences from `WordList` (`ApproxWordListV6`):
    * `lookup()` returns a plain float score per word instead of a tuple of
      (index score, recomputed nmd) -- the index score already is the exact nmd similarity,
      so V6's second value only ever repeated the first (to within float noise)
    * words with identical scores come back in alphabetical order rather than in
      posting-list insertion order, so results are reproducible run to run
    * there is no `filter_n` prefilter: it existed to keep the python candidate loop short,
      and once that loop is vectorised it costs more than it saves
    * queries whose length is exactly `n - 2` raise `ZeroDivisionError` in `WordList`, and
      work here
    * `invert=False` returns an actual distance (`ApproxWordListV5` returned
      `normalize - score`, i.e. a negative number)
    * ⚠ **`normalize` defaults to `True` here**, where `WordList` and
      `ngram_movers_distance()` both default to `False`. An un-normalized score is a raw
      similarity sum, so it grows with candidate length and biases ranking toward long
      entries; turning it on gained MAP on all six benchmarks in `docs/bow-plan.md` part 9,
      from +0.050 on Malay typo correction to +0.339 on long documents

## `bow_ngram_movers_distance()`

* WARNING: requires `scipy.optimize`, so it's not available by default in the `nmd` namespace
* use this to compare sequences of tokens (not necessarily unique)
* note that this does not merge or split words, so if you're matching `["pineapple"]` and `["pine", "apple"]` the
  similarity will be low. consider just using nmd in this case.

```python
from nmd.nmd_bow import bow_ngram_movers_distance

text_1 = f'Clementi Sports Hub'
text_2 = f'sport hubs clemmeti'
print(bow_ngram_movers_distance(bag_of_words_1=text_1.casefold().split(),
                                bag_of_words_2=text_2.casefold().split(),
                                invert=True,  # invert: return similarity instead of distance
                                normalize=True,  # return a score between 0 and 1
                                ))
```

## `segment_movers_distance()`

* WARNING: requires `scipy.optimize` and `numpy`, so it's not available by default in the `nmd` namespace
* like `bow_ngram_movers_distance` but tolerant of split / merged words (`["pineapple"]` matches `["pine", "apple"]`
  perfectly) and optionally preferring in-order matches. reference implementation, not optimized.
* see [docs/bow-plan.md](docs/bow-plan.md) for the design and an evaluation on product names

```python
from nmd.nmd_segments import align_segments, explain_alignment, segment_movers_distance

a = 'sony black earbud style headphones mdrex55bk'.split()
b = 'ex series earbuds black mdr ex55/blk'.split()
print(segment_movers_distance(a, b, lam=0.0, min_sim=0.2, invert=True, normalize=True))

# introspect the matching: which segments were paired, with what similarity and positional shift
print(explain_alignment(a, b, align_segments(a, b, lam=0.0, min_sim=0.2)))
```

# Testing

The project includes a test suite using pytest. To run the tests:

```bash
pytest
```

For more details about the tests, see the [tests/README.md](tests/README.md) file.

# known bugs in the older index classes

`ApproxWordListV3` and `ApproxWordListV5` are **frozen**: they are kept unchanged for
comparison against the versions that replaced them, and their bugs are documented rather than
fixed. `ApproxWordListV6` (the shipped `WordList`) and `ApproxWordListV7` are not frozen, and
have none of these.

still present, in the frozen classes only:

* `ApproxWordListV3.vocabulary` is `return sorted(self.vocabulary)` -- infinite recursion,
  `RecursionError` on any access
* `ApproxWordListV5.lookup(invert=False)` returns `normalize - match_score`, i.e. bool
  arithmetic, so distances come out negative and sorted worst-first. it also returns
  `top_k * 2` results instead of `top_k`
* `ApproxWordListV5.lookup()` raises `ZeroDivisionError` for a query of length `n - 2`, as V6
  used to (below)
* `num_grams()` disagrees with `get_n_grams()` in two places: it over-counts by 2 at `n=1`
  (adding START/END flags that `get_n_grams` omits for 1-grams), and it returns a negative
  count once the word is shorter than `n - 3`. it is still called by V3 and V5;
  `num_n_grams()` is the corrected version, and is what V6 and V7 use

fixed 2026-09-05 in `ApproxWordListV6`, i.e. in `WordList` (see `tests/test_index_v6.py`):

* `lookup()` raised `ZeroDivisionError` for any query of length `n - 2` (e.g. a 2-character
  query when 4-grams are indexed), because that produces exactly one n-gram and the location
  denominator `len(n_grams) - 1` is zero. `add_word()` had always handled this case; only the
  lookup path had not
* normalization used the `num_grams()` above, so at `n=1` every denominator was 2 too large
  (a word matched against itself scored 0.714 rather than 1.0)

neither fix changes any score on the documented `n=(2, 4)` default, where `num_grams()` and
`num_n_grams()` agree for every non-empty word.

# todo

* `from nmd import nmd` should return a function, not a module -> refactor this
* try out cython 3? maybe in pure python mode
* todo: try [this paper's algo](https://www.aclweb.org/anthology/C10-1096.pdf)
    * which referenced [this paper](https://www.cse.iitb.ac.in/~sunita/papers/sigmod04.pdf)
* use less bizarre test strings
* note where the algorithm breaks down
    * matching long strings with many n-grams
    * matching strings with significantly different lengths
* rename nmd_bow because it isn't really a bag-of-words, it's a token sequence
* index for nmd_bow, and split/merge-tolerant token matching: see [docs/bow-plan.md](docs/bow-plan.md)
* consider a `real_quick_ratio`-like optimization, or maybe calculate length bounds?
    * needs a cutoff to actually speed up though, makes a huge difference for difflib
    * a sufficiently low cutoff is not unreasonable, although the default of 0.6 might be a little high for nmd
    * that said the builtin diff performs pretty badly at low similarities, so 0.6 is reasonable for them

```python
def real_quick_ratio(self):
    """Return an upper bound on ratio() very quickly.

    This isn't defined beyond that it is an upper bound on .ratio(), and
    is faster to compute than either .ratio() or .quick_ratio().
    """

    la, lb = len(self.a), len(self.b)
    # can't have more matches than the number of elements in the shorter sequence
    matches, length = min(la, lb), la + lb
    if length:
        return 2.0 * matches / length
    return 1.0
```

* create a better string container for the index, more like a `set`
    * `add(word: str)`
    * `remove(word: str)`
    * `clear()`
    * `__contains__(word: str)`
    * `__iter__()`
* better lookup
    * add a min_similarity filter (float, based on normalized distance)
        * `lookup(word: str, min_similarity: float = 0, filter: bool = True)`
    * try `__contains__` first
        * try levenshtein automaton (distance=1) second?
            * sort by nmd, since most likely there will only be a few results
        * but how to get multiple results?
            * still need to run full search?
            * or maybe just return top 1 result?
* prefix lookup
    * look for all strings that are approximately prefixed
    * like existing index but not normalized and ignoring unmatched ngrams from target

## Publishing (notes for myself)

* init
    * `pip install flit`
    * `flit init`
    * make sure `nmd/__init__.py` contains a docstring and version
* publish / update
    * increment `__version__` in `nmd/__init__.py`
    * `flit publish`
