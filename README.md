# N-gram Mover's Distance

a string similarity measure based on Earth Mover's Distance

# Usage
```python
from nmd import n_gram_emd

# n-gram mover's distance
print(n_gram_emd(f'hello', f'yellow'))

# similarity (inverted distance)
print(n_gram_emd(f'hello', f'yellow', invert=True))

# distance, normalized to the range 0 to 1 (inclusive of 0 and 1)
print(n_gram_emd(f'hello', f'yellow', normalize=True))

# similarity, normalized to the range 0 to 1 (inclusive of 0 and 1)
print(n_gram_emd(f'hello', f'yellow', invert=True, normalize=True))
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
    
    
    