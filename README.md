#   N-gram Mover's Distance
a string similarity measure based on Earth Mover's Distance

##  Why another string matching algorithm?
*   Edit distance really wasn't cutting it when I needed to lookup a dictionary for a misspelled word
    *   When distance >1, the results are not very useful
    *   When distance >5, the results are meaningless
    *   Same goes for Damerau-Levenshtein
*   Also, edit distance was a bit slow when looking up a large dictionary
    *   Even with a decent automaton or trie implementation
    *   NMD was designed with indexing in mind
    *   A simpler index could be used for Jaccard or cosine similarity over ngrams
        *   todo: try [this paper's algo](https://www.aclweb.org/anthology/C10-1096.pdf)
        *   which referenced [this paper](https://www.cse.iitb.ac.in/~sunita/papers/sigmod04.pdf)
    *   also todo: try matching automaton against a trie

#   Usage

##  `nmd.py`
string distance metric, use this to compare two strings
```python
from nmd.nmd import ngram_movers_distance

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
use this for dictionary lookups of words
```python
from nmd.nmd_index import ApproxWordList5

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

##  `nmd_bow.py`
use this to compare sequences of tokens (not necessarily unique)
```python
from nmd.nmd_bow import bow_ngram_movers_distance
from tokenizer import unicode_tokenize

text_1 = f'Clementi Sports Hub'
text_2 = f'sport hubs clemmeti'
print(bow_ngram_movers_distance(bag_of_words_1=unicode_tokenize(text_1.casefold(), words_only=True),
                                bag_of_words_2=unicode_tokenize(text_2.casefold(), words_only=True),
                                invert=True,  # invert: return similarity instead of distance
                                normalize=True,  # return a score between 0 and 1
                                ))
```

#   todo
*   rename nmd_bow because it isn't really a bow, it's a token sequence
*   real_quick_ratio, or maybe calculate length bounds?
    *   needs a cutoff to actually speed up though, makes a huge difference for difflib
    *   a sufficiently low cutoff is not unreasonable, although the default of 0.6 might be a little high for nmd
    *   that said the builtin diff performs pretty badly at low similarities, so 0.6 is reasonable for them
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
    
    
    