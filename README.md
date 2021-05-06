#   N-gram Mover's Distance

a string similarity measure based on Earth Mover's Distance

#   Usage

##  `nmd.py`
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

##  `nmd_index.py`
```python
from nmd_index import ApproxWordList5

# get words from a text file
with open(f'words_ms.txt', encoding=f'utf8') as f:
    words = set(f.read().split())

# index words
word_list = ApproxWordList5((2, 4), filter_n=0)  # combined 2- and 4-grams seem to work best
for word in words:
    word_list.add_word(word)

# lookup a word
print(word_list.lookup(f'asalamalaikum'))  # -> 'assalamualaikum'
print(word_list.lookup(f'walaikumalasam'))  # -> 'waalaikumsalam'
```

#   todo
*   real_quick_ratio, or maybe calculate length bounds?
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
*   create a better string container for the index, more like a `set`
    *   `add(word: str)`
    *   `remove(word: str)`
    *   `clear()`
    *   `__contains__(word: str)`
    *   `__iter__()`
*   better lookup
    *   add a min_similarity filter (float, based on normalized distance)
        *   `lookup(word: str, min_similarity: float = 0, filter: bool = True)`
    *   try `__contains__` first
        *   try levenshtein automaton (distance=1) second?
            *   sort by nmd, since most likely there will only be a few results
        *   but how to get multiple results?
            *   still need to run full search?
            *   or maybe just return top 1 result?
    *   make the 3-gram filter optional
*   prefix lookup
    *   look for all strings that are approximately prefixed
    *   like existing index but not normalized and ignoring unmatched ngrams from target
*   bag of words
    *   use WMD with NMD word distances
    *   may require proper EMD implementation?
    
    
    