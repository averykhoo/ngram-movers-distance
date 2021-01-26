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
*   create a better string container for the index, more like a `set`
    *   `add(word: str)`
    *   `remove(word: str)`
    *   `clear()`
    *   `__contains__(word: str)`
    *   `__iter__()`
*   better lookup
    *   `lookup(word: str, min_similarity: float = 0, filter: bool = True)`
    *   try `__contains__` first
    *   try levenshtein automaton (distance=1) first?
    *   make the 3-gram filter optional
    *   add a min_similarity filter (float, based on normalized distance)
*   prefix lookup
    *   look for all strings that are approximately prefixed
    *   like existing index but not normalized and ignoring unmatched ngrams from target
    
    
    