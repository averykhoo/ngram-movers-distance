"""
split into a separate file because this needs `pyroaring` and `regex`
"""
import time
from collections import defaultdict
from collections.abc import MutableSet
from functools import partial
from typing import Callable
from typing import Dict
from typing import Final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
from typing import cast

import unicodedata
from pyroaring import BitMap

from nmd.emd_1d import emd_1d_old
from nmd.nmd_index import ApproxWordListV5

# BitMap=set  # this works for testing without pyroaring

# --- Default Normalizers ---

#: NFC normalization function (most common default)
normalize_nfc: Final[Callable[[str], str]] = partial(unicodedata.normalize, 'NFC')
#: NFKC normalization function (more aggressive compatibility normalization)
normalize_nfkc: Final[Callable[[str], str]] = partial(unicodedata.normalize, 'NFKC')

# Optional: Define normalize_nfd_strip_marks using regex if needed
# Requires 'pip install regex'
import regex

_UNICODE_MARK_PATTERN = regex.compile(r'\p{M}')


def normalize_nfd_strip_marks(text: str) -> str:
    """Normalizes via NFD and removes combining marks (ignores accents)."""
    # Decompose, then filter out combining marks
    return "".join(
        c for c in unicodedata.normalize('NFD', text)
        if not _UNICODE_MARK_PATTERN.match(c)
    )


_DEFAULT_START_MARKER = '\2'
_DEFAULT_END_MARKER = '\3'


class WordSet(MutableSet[str]):
    """
    A set-like collection optimized for fast, approximate string lookup.

    Compares words based on shared character sequences (n-grams) and the
    'distance' their positions need to shift to align, using a variation
    of Earth Mover's Distance (N-gram Mover's Distance).

    Configuration (n-gram sizes, case sensitivity, Unicode handling) is
    fixed upon initialization. Implements the collections.abc.MutableSet interface.
    """

    # --- Type Hint Alias ---
    UnicodeNormalizer = Optional[Callable[[str], str]]

    # --- Initialization ---
    def __init__(self,
                 *,  # Force keyword-only arguments
                 case_sensitive: bool = False,
                 unicode_normalizer: UnicodeNormalizer = normalize_nfc,
                 ngram_sizes: Union[int, Iterable[int]] = (2, 3, 4)
                 ) -> None:
        """
        Initializes the WordSet and its matching configuration.

        Args:
            case_sensitive: If False (default), matching ignores case ('Apple' == 'apple').
            unicode_normalizer: Function to standardize Unicode characters before
               comparison. Defaults to NFC normalization (most common).
               Provide `None` to disable normalization. Other options include
               `nmd.normalize_nfkc` (more aggressive) or
               `nmd.normalize_nfd_strip_marks` (ignores accents), or a custom function.
            ngram_sizes: Size(s) of character sequences (n-grams) to compare.
               Smaller values (e.g., 2) focus on local similarity/typos.
               Larger values (e.g., 4) focus on word structure.
               Using multiple sizes like `(2, 4)` (default) combines these.
        """
        # --- Validate and Store Configuration ---
        if not isinstance(case_sensitive, bool):
            raise TypeError("case_sensitive must be a boolean")
        self._case_sensitive: Final[bool] = case_sensitive

        if unicode_normalizer is not None and not callable(unicode_normalizer):
            raise TypeError("unicode_normalizer must be a callable or None")
        self._unicode_normalizer: Final[WordSet.UnicodeNormalizer] = unicode_normalizer

        if isinstance(ngram_sizes, int):
            if ngram_sizes <= 0: raise ValueError("ngram_sizes must be positive")
            self._n_list: Final[Tuple[int, ...]] = (ngram_sizes,)
        elif isinstance(ngram_sizes, Iterable):
            try:
                unique_sizes = sorted(set(int(val) for val in ngram_sizes))
                if not unique_sizes or any(val <= 0 for val in unique_sizes):
                    raise ValueError("All values in ngram_sizes must be positive integers")
                self._n_list = tuple(unique_sizes)
            except (TypeError, ValueError) as e:
                raise TypeError("ngram_sizes must be an int or iterable of positive ints") from e
        else:
            raise TypeError("ngram_sizes must be an int or iterable of positive ints")

        # --- Determine Effective Filter N-gram Size (Internal Optimization) ---
        # Use a reasonable default, e.g., 3 if present, otherwise min size if <= 4?
        self._effective_filter_n: Final[Optional[int]]
        if 3 in self._n_list:
            self._effective_filter_n = 3
        else:
            min_n = self._n_list[0]  # Smallest n-gram size specified
            self._effective_filter_n = min_n if min_n <= 4 else None  # Only use filter if small n exists

        # --- Internal State Initialization ---
        self._vocab_normalized_to_id: Dict[str, int] = {}  # Normalized -> ID
        self._vocab_id_to_original: Dict[int, str] = {}  # ID -> Original word
        self._word_id_counter: int = 0
        # Store precomputed info needed for lookup per word ID
        self._word_info: Dict[
            int, Dict] = {}  # ID -> {'norm_len': int, 'num_grams': Tuple[int,...], 'ngrams_pos': Dict[str, Tuple[float,...]], 'filter_grams': Set[str]}
        # Filter index using RoaringBitmaps
        self._filter_index: Dict[str, BitMap] = {}  # filter-n-gram -> RoaringBitmap{word_id}

    # --- Internal Helper Methods ---

    def _normalize_word(self, word: str) -> str:
        """Applies case folding and Unicode normalization based on config."""
        if not self._case_sensitive:
            word = word.casefold()
        if self._unicode_normalizer:
            try:
                word = self._unicode_normalizer(word)
            except Exception as e:
                # Wrap potential errors from user-provided function
                raise ValueError(f"Unicode normalizer failed for word '{word[:50]}': {e}") from e
        # Basic check after potential normalization
        if '\0' in word:
            raise ValueError("Word contains null byte after normalization")
        return word

    def _get_n_grams_list(self, text: str, n: int) -> List[str]:
        """Generates n-grams for a given text and n."""
        # Note: Consider Cythonizing this if profiling shows it's a bottleneck
        if n <= 0: return []
        padded_text = f'{_DEFAULT_START_MARKER}{text}{_DEFAULT_END_MARKER}'
        len_padded = len(padded_text)
        num_grams = len_padded - n + 1
        if num_grams <= 0: return []
        return [padded_text[i:i + n] for i in range(num_grams)]

    def _get_num_grams(self, len_text: int, n: int) -> int:
        """Calculates number of n-grams including start/end markers."""
        if n <= 0: return 0
        effective_len = len_text + 2  # Account for markers
        num = effective_len - n + 1
        return max(0, num)

    def _compute_word_info(self, normalized_word: str) -> Dict:
        """Computes the necessary info for a normalized word."""
        info: Dict[str, Union[int, Tuple[int, ...], Dict[str, Tuple[float, ...]], Set[str]]] = {}
        info['norm_len'] = len(normalized_word)
        info['num_grams'] = tuple(self._get_num_grams(cast(int, info['norm_len']), n) for n in self._n_list)

        # Generate n-grams and positions for all specified Ns for the main index
        ngrams_pos_dict: Dict[str, List[float]] = defaultdict(list)
        for n_idx, n in enumerate(self._n_list):
            n_grams_list = self._get_n_grams_list(normalized_word, n)
            num_grams_for_n = len(n_grams_list)
            if num_grams_for_n > 0:
                divisor = float(max(1, num_grams_for_n - 1))
                for idx, gram in enumerate(n_grams_list):
                    ngrams_pos_dict[gram].append(idx / divisor)
        # Convert lists to tuples for storage
        info['ngrams_pos'] = {gram: tuple(pos_list) for gram, pos_list in ngrams_pos_dict.items()}

        # Generate filter n-grams (if filter enabled)
        info['filter_grams'] = set()
        if self._effective_filter_n:
            info['filter_grams'] = set(self._get_n_grams_list(normalized_word, self._effective_filter_n))

        return info

    # --- Set Methods (MutableSet Implementation) ---

    def add(self, word: str) -> None:
        """
        Adds a word to the set. Normalizes based on config. Idempotent.
        Raises TypeError or ValueError on invalid input.
        """
        if not isinstance(word, str):
            raise TypeError(f"Can only add strings, got {type(word).__name__}")
        if not word:
            raise ValueError("Cannot add empty string")

        original_word = word
        normalized_word = self._normalize_word(word)

        if not normalized_word:
            raise ValueError(f"Word '{original_word[:50]}' became empty after normalization")

        if normalized_word in self._vocab_normalized_to_id:
            return  # Already present

        # Assign new ID
        word_id = self._word_id_counter
        self._word_id_counter += 1

        # Store mappings
        self._vocab_normalized_to_id[normalized_word] = word_id
        self._vocab_id_to_original[word_id] = original_word

        # Precompute and store info needed for lookup
        info = self._compute_word_info(normalized_word)
        self._word_info[word_id] = info

        # Add to filter index (using BitMap)
        if self._effective_filter_n:
            filter_grams = cast(Set[str], info['filter_grams'])
            for gram in filter_grams:
                if gram not in self._filter_index:
                    self._filter_index[gram] = BitMap()  # Changed RoaringBitmap -> BitMap
                self._filter_index[gram].add(word_id)

    def discard(self, word: str) -> None:
        """
        Removes a word if present (based on normalized form), otherwise does nothing.
        Raises TypeError if input is not a string.
        """
        if not isinstance(word, str):
            raise TypeError(f"Can only discard strings, got {type(word).__name__}")

        normalized_word = self._normalize_word(word)

        if normalized_word not in self._vocab_normalized_to_id:
            return  # Not found, do nothing

        word_id = self._vocab_normalized_to_id.pop(normalized_word)
        del self._vocab_id_to_original[word_id]
        info = self._word_info.pop(word_id)

        # Remove from filter index
        if self._effective_filter_n:
            filter_grams = cast(Set[str], info['filter_grams'])
            for gram in filter_grams:
                if gram in self._filter_index:
                    bitmap = self._filter_index[gram]
                    bitmap.discard(word_id)  # Use discard for safety
                    if not bitmap:  # If bitmap becomes empty, remove key
                        del self._filter_index[gram]

    def __contains__(self, word: object) -> bool:
        """Checks for exact (normalized) membership using 'in'."""
        if not isinstance(word, str):
            return False  # Per MutableSet requirements, handle non-str gracefully
        normalized_word = self._normalize_word(word)
        return normalized_word in self._vocab_normalized_to_id

    def __len__(self) -> int:
        """Returns the number of unique words."""
        return len(self._vocab_normalized_to_id)

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the *original* words added."""
        return iter(self._vocab_id_to_original.values())

    # --- Concrete Methods from MutableSet (Optimized where possible) ---

    def remove(self, word: str) -> None:
        """
        Removes a word (based on normalized form). Raises KeyError if not found.
        """
        if not isinstance(word, str):  # Check type before normalization
            raise TypeError(f"Can only remove strings, got {type(word).__name__}")
        normalized_word = self._normalize_word(word)
        if normalized_word not in self._vocab_normalized_to_id:
            raise KeyError(f"Word (normalized: '{normalized_word[:50]}') not found in WordSet")
        # discard handles the actual removal logic
        self.discard(word)  # Pass original word, discard will normalize again

    def clear(self) -> None:
        """Removes all words from the set."""
        self._vocab_normalized_to_id.clear()
        self._vocab_id_to_original.clear()
        self._word_info.clear()
        self._filter_index.clear()
        self._word_id_counter = 0

    # --- Fuzzy Lookup Methods ---

    def find_similar(self,
                     query: str,
                     k: int = 5,
                     min_similarity: Optional[float] = None
                     ) -> List[Tuple[str, float]]:
        """
        Finds words in the set similar to the query, sorted by similarity score.

        Args:
            query: The string to search for.
            k: Max number of results.
            min_similarity: Minimum normalized similarity score [0.0, 1.0].

        Returns:
            List of (original_word, similarity_score) tuples, best matches first.
        """
        # --- Input Validation ---
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if min_similarity is not None:
            if not isinstance(min_similarity, (float, int)) or not (0.0 <= min_similarity <= 1.0):
                raise ValueError(f"min_similarity must be a float between 0.0 and 1.0, got {min_similarity}")

        # --- Handle Edge Cases ---
        if not query or not self._vocab_normalized_to_id:
            return []

        original_query = query
        normalized_query = self._normalize_word(query)
        if not normalized_query:
            return []  # Query became empty after normalization

        # --- 1. Candidate Filtering (using RoaringBitmaps) ---
        target_ids: Iterable[int]
        if self._effective_filter_n:
            query_filter_grams = set(self._get_n_grams_list(normalized_query, self._effective_filter_n))
            # print(f'{query_filter_grams=}')
            if not query_filter_grams:
                # Query too short for filter N, fallback depends on desired behavior
                # Option 1: Check all words (might be slow)
                # target_ids = self._word_info.keys()
                # Option 2: Return empty (safer default if filtering is expected)
                return []
            else:
                candidate_bitmap = BitMap()
                # Optimization: Process grams likely to be rarer first? Hard to know easily.
                for gram in query_filter_grams:
                    gram_bitmap = self._filter_index.get(gram, [])
                    # print(gram, gram_bitmap[:10])

                    if gram_bitmap:  # Only process if gram exists in index
                        candidate_bitmap.update(gram_bitmap)
                if not candidate_bitmap:
                    return []  # No candidates share required filter n-grams
                target_ids = candidate_bitmap  # Iterate directly over the bitmap
        else:
            # No filtering enabled, check all words
            target_ids = self._word_info.keys()

        # --- 2. Approximate Scoring ---
        query_info = self._compute_word_info(normalized_query)
        query_ngrams_pos = cast(Dict[str, Tuple[float, ...]], query_info['ngrams_pos'])
        query_num_grams = cast(Tuple[int, ...], query_info['num_grams'])
        candidate_approx_scores: List[Tuple[float, int]] = []  # (negative_approx_score, word_id)

        for word_id in target_ids:
            word_info = self._word_info[word_id]
            word_ngrams_pos = cast(Dict[str, Tuple[float, ...]], word_info['ngrams_pos'])
            word_num_grams = cast(Tuple[int, ...], word_info['num_grams'])

            total_similarity_contribution = 0.0

            # Find common n-grams efficiently (faster than set conversion for each word?)
            # Iterate through smaller set of n-grams and check presence in the other.
            query_keys = query_ngrams_pos.keys()
            word_keys = word_ngrams_pos.keys()
            iter_keys = query_keys if len(query_keys) <= len(word_keys) else word_keys
            check_dict = word_ngrams_pos if len(query_keys) <= len(word_keys) else query_ngrams_pos

            for n_idx, n in enumerate(self._n_list):
                norm_factor_for_n = query_num_grams[n_idx] + word_num_grams[n_idx]
                if norm_factor_for_n == 0: continue

                # Calculate similarity only for n-grams of size n
                # This requires knowing which n-grams belong to which size n
                # Current structure `ngrams_pos` mixes them. Refactor needed?
                # Let's stick to the simpler combined approach for now: iterate all common grams
                # and normalize at the end. This implicitly weights by n-gram frequency.

            # Calculate total similarity contribution across all Ns
            for gram in iter_keys:
                if gram in check_dict:  # Found a common gram
                    q_pos = query_ngrams_pos[gram]
                    w_pos = word_ngrams_pos[gram]
                    # Similarity = total possible matches - EMD cost
                    # Need emd_1d here. Assumes it's imported.
                    total_similarity_contribution += len(q_pos) + len(w_pos) - emd_1d_old(q_pos, w_pos)

            # Calculate total normalization factor across all Ns
            total_normalization_factor = sum(query_num_grams) + sum(word_num_grams)

            if total_normalization_factor > 0:
                # Ensure score is between 0 and 1 (due to EMD properties)
                approx_norm_sim = max(0.0, min(1.0, total_similarity_contribution / total_normalization_factor))
                candidate_approx_scores.append((-approx_norm_sim, word_id))  # Store negative

        # --- 3. Sort & Select Top Approx ---
        candidate_approx_scores.sort()  # Sorts by score (most negative first = highest sim)
        # num_to_rescore = min(len(candidate_approx_scores), k * 2)  # Rescore slightly more
        # # --- 4. Exact Re-scoring (using original strings) ---
        # final_candidates: List[Tuple[float, int, float]] = []  # (negative_final_score, word_id)
        #
        # for i in range(num_to_rescore):
        #     approx_score_neg, word_id = candidate_approx_scores[i]
        #     # original_word = self._vocab_id_to_original[word_id]
        #
        #     # # Calculate actual normalized similarity using the public function
        #     # try:
        #     #     # Assume ngram_movers_distance accepts iterable n
        #     #     final_score = ngram_movers_distance(
        #     #         original_query,
        #     #         original_word,
        #     #         n=3,
        #     #         invert=True,
        #     #         normalize=True
        #     #     )
        #     # except ZeroDivisionError:
        #     #     final_score = 1.0 if original_query == original_word else 0.0
        #
        #     # Apply min_similarity threshold based on the *final* score
        #     if min_similarity is None or final_score >= min_similarity:
        #         final_candidates.append((-final_score, word_id, approx_score_neg))
        #
        # # --- 5. Final Sort & Format Output ---
        # final_candidates.sort()  # Sort by final score (descending similarity)

        # Prepare output
        results: List[Tuple[str, float]] = []
        # for i in range(min(k, len(final_candidates))):
        #     final_score_neg, word_id, approx_score_neg = final_candidates[i]
        for i in range(min(k, len(candidate_approx_scores))):
            approx_score_neg, word_id = candidate_approx_scores[i]
            original_word = self._vocab_id_to_original[word_id]
            results.append((original_word, -approx_score_neg))  # Return positive score

        return results

    def suggest(self, query: str, k: int = 5) -> List[str]:
        """
        Suggests the top k most similar words (names only), best matches first.
        """
        # Basic validation is done within find_similar
        similar_results = self.find_similar(query=query, k=k, min_similarity=None)
        # Extract just the words
        return [word for word, score in similar_results]

    # --- Configuration Properties (Read-Only) ---

    @property
    def ngram_sizes(self) -> Tuple[int, ...]:
        """The n-gram sizes used for similarity calculation (read-only)."""
        return self._n_list

    @property
    def case_sensitive(self) -> bool:
        """Whether matching is case-sensitive (read-only)."""
        return self._case_sensitive

    @property
    def unicode_normalizer(self) -> UnicodeNormalizer:
        """The Unicode normalization function used (read-only)."""
        return self._unicode_normalizer

    # --- Representation ---
    def __repr__(self) -> str:
        """Returns a developer-friendly representation."""
        norm_name = 'None'
        if self._unicode_normalizer:
            # Try to get a standard name, fallback to function name
            if self._unicode_normalizer is normalize_nfc:
                norm_name = 'NFC'
            elif self._unicode_normalizer is normalize_nfkc:
                norm_name = 'NFKC'
            elif self._unicode_normalizer is normalize_nfd_strip_marks:
                norm_name = 'NFD_STRIP_MARKS'
            else:
                norm_name = getattr(self._unicode_normalizer, '__name__', repr(self._unicode_normalizer))

        return (
            f"{self.__class__.__name__}("
            f"ngram_sizes={self.ngram_sizes!r}, "
            f"case_sensitive={self.case_sensitive!r}, "
            f"unicode_normalizer={norm_name}, "
            f"size={len(self)})"
        )


MutableSet.register(WordSet)

if __name__ == '__main__':
    ws = WordSet()
    ws.add('asdf')
    print(ws)

    with open('../experiments/words_en.txt', encoding='utf8') as f:
        # with open('british-english-insane.txt', encoding='utf8') as f:
        words = set(f.read().split())

    t = time.perf_counter()
    for word in words:
        ws.add(word)
    print('WordSet', time.perf_counter() - t)
    print(ws)

    wl = ApproxWordListV5()
    t = time.perf_counter()
    for word in words:
        wl.add_word(word)
    print('ApproxWordListV5', time.perf_counter() - t)

    t = time.perf_counter()
    for _ in range(100):
        res = ws.find_similar('bananananaanananananana')
    print('WordSet', time.perf_counter() - t)
    print(res)

    t = time.perf_counter()
    for _ in range(100):
        res = wl.lookup('bananananaanananananana', normalize=True)
    print('ApproxWordListV5', time.perf_counter() - t)
    print(res)
