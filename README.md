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

# Usage

## `ngram_movers_distance()`

* string distance metric, use this to compare two strings

```python
from nmd import ngram_movers_distance

# n-gram mover's distance
print(ngram_movers_distance(f'hello', f'yellow'))

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
word_list = WordList((2, 4), filter_n=0)  # combined 2- and 4-grams seem to work best
for word in words:
    word_list.add_word(word)

# lookup a word
print(word_list.lookup(f'asalamalaikum'))  # -> 'assalamualaikum'
print(word_list.lookup(f'walaikumalasam'))  # -> 'waalaikumsalam'
```

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