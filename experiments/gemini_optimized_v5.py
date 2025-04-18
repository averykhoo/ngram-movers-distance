import heapq
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union, Iterable, Final

# Assume these helper functions exist and are reasonably optimized:
# Make sure these are imported correctly from your project structure
from .nmd_core import emd_1d
from .nmd_index_commons import get_n_grams, num_grams, mean

# Requires: pip install pyroaring
try:
    from pyroaring import BitMap
except ImportError:
    print("Warning: pyroaring not installed. Using standard sets for filtering (less efficient).")
    BitMap = set # Fallback for testing without pyroaring


class ApproxWordListV5b:
    """
    Provides approximate string matching based on N-gram Mover's Distance (NMD).

    This version uses an inverted index structure (n-gram -> word info) and
    employs several optimizations over earlier versions, including:
    - Using pyroaring.BitMap for the initial filter index for efficiency.
    - Pre-calculating query n-gram information.
    - Implementing a two-stage filtering process (bitmap filter -> bounds filter).
    - Optimizing list initializations.
    - Improved variable naming and commenting.

    Configuration (n-gram sizes, case sensitivity, filter size) is fixed
    upon initialization.
    """

    # --- Initialization ---
    def __init__(self,
                 n: Union[int, Iterable[int]] = (2, 4),
                 case_sensitive: bool = False,
                 filter_n: Optional[int] = 3,
                 ):
        """
        Initializes the approximate word list index.

        Args:
            n: The size(s) of n-grams to use for the main similarity calculation.
               Can be a single integer or an iterable of integers (e.g., (2, 4)).
            case_sensitive: If False (default), matching ignores case.
            filter_n: The size of n-grams to use for the initial fast filter index.
                      Set to None to disable the filter (not recommended for large lists).
        """

        # --- Validate and Store N-gram Sizes ---
        if isinstance(n, int):
            if n <= 0: raise ValueError("n must be positive")
            self.__n_list: Final[Tuple[int, ...]] = (n,)
        elif isinstance(n, Iterable):
            try:
                unique_sizes = sorted(set(int(val) for val in n))
                if not unique_sizes or any(val <= 0 for val in unique_sizes):
                    raise ValueError("All values in n must be positive integers")
                self.__n_list = tuple(unique_sizes)
            except (TypeError, ValueError) as e:
                raise TypeError("n must be an int or iterable of positive ints") from e
        else:
            raise TypeError("n must be an int or iterable of positive ints")
        # Precompute for efficiency inside loops
        self.__n_indices: Final[Dict[int, int]] = {n_val: i for i, n_val in enumerate(self.__n_list)}

        # --- Validate and Store Case Sensitivity ---
        if not isinstance(case_sensitive, bool):
            raise TypeError("case_sensitive must be a boolean")
        self.__case_insensitive: Final[bool] = not case_sensitive

        # --- Validate and Store Filter N Size ---
        self.__filter_n: Final[Optional[int]] = filter_n
        if self.__filter_n is not None:
            if not isinstance(self.__filter_n, int) or self.__filter_n <= 0:
                raise ValueError("filter_n must be a positive integer or None")

        # --- Vocabulary Storage ---
        # Maps normalized word to its internal integer index
        self.__word_to_index: Dict[str, int] = {}
        # Maps internal integer index back to the normalized word
        self.__index_to_word: List[str] = []
        # Stores length of normalized word for each index
        self.__index_to_word_length: List[int] = []
        # Stores tuple of ngram counts (#grams for n1, #grams for n2, ...) for each index
        self.__index_to_num_grams: List[Tuple[int,...]] = []

        # --- Filter Index ---
        # Maps a filter n-gram string to a Roaring Bitmap of word indices containing it
        self.__ngram_filter_index: Dict[str, BitMap] = {}

        # --- N-gram Count Index (Inverted) ---
        # Maps an n-gram string to a list of (word_index, count_in_word) tuples
        self.__ngram_counts_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        # --- N-gram Position Index (Inverted) ---
        # Maps an n-gram string to a list of (word_index, tuple_of_normalized_positions) tuples
        self.__ngram_positions_index: Dict[str, List[Tuple[int, Tuple[float, ...]]]] = defaultdict(list)

        # --- Preallocated Zero List for Score Initialization ---
        # Used with .copy() for potentially faster list creation in loops
        self.__zeros_list_for_scores: Final[List[float]] = [0.0] * len(self.__n_list)

    # --- Public Properties ---
    @property
    def vocabulary(self) -> List[str]:
        """Returns a sorted list of unique normalized words in the index."""
        return sorted(self.__index_to_word)

    @property
    def ngram_sizes(self) -> Tuple[int, ...]:
        """Returns the n-gram sizes used for similarity calculation."""
        return self.__n_list

    @property
    def case_sensitive(self) -> bool:
        """Returns whether matching is case-sensitive."""
        return not self.__case_insensitive

    @property
    def filter_ngram_size(self) -> Optional[int]:
        """Returns the n-gram size used for the filter index (or None if disabled)."""
        return self.__filter_n

    # --- Internal Helper Methods ---
    def _normalize_word(self, word: str) -> str:
        """Applies case folding based on configuration."""
        # TODO: Consider adding optional Unicode normalization like WordSet
        if self.__case_insensitive:
            word = word.casefold()
        return word

    def _resolve_word_index(self, normalized_word: str, auto_add: bool = True) -> Optional[int]:
        """
        Finds the index for a normalized word, optionally adding it.

        Args:
            normalized_word: The word after normalization.
            auto_add: If True, adds the word to the vocabulary if not found.

        Returns:
            The integer index of the word, or None if not found and auto_add is False.
        """
        if not isinstance(normalized_word, str) or not normalized_word:
            # Should generally be caught earlier, but provides safety
            return None

        if normalized_word in self.__word_to_index:
            return self.__word_to_index[normalized_word]

        if not auto_add:
            return None

        # Add unknown word
        word_index = len(self.__index_to_word)
        self.__word_to_index[normalized_word] = word_index
        self.__index_to_word.append(normalized_word)
        word_length = len(normalized_word)
        self.__index_to_word_length.append(word_length)
        self.__index_to_num_grams.append(tuple(num_grams(word_length, n) for n in self.__n_list))

        # Invariant check (optional, good for debugging)
        # assert len(self.__word_to_index) == len(self.__index_to_word) == word_index + 1

        return word_index

    # --- Index Building ---
    def add_word(self, word: str) -> 'ApproxWordListV5b':
        """
        Adds a word to the index after normalization.

        Args:
            word: The word string to add.

        Returns:
            The instance itself, allowing chaining.

        Raises:
            ValueError: If the word is empty, becomes empty after normalization,
                        or contains reserved control characters ('\2', '\3').
        """
        if not isinstance(word, str) or not word:
            raise ValueError("Word must be a non-empty string")

        normalized_word = self._normalize_word(word)
        if not normalized_word:
             raise ValueError(f"Word '{word[:50]}' became empty after normalization")
        # Ensure no reserved padding characters are present
        if '\2' in normalized_word or '\3' in normalized_word:
             raise ValueError("Word contains reserved control characters ('\\2', '\\3') after normalization")

        # Check if already present before resolving index
        if normalized_word in self.__word_to_index:
            return self # Idempotent

        word_index = self._resolve_word_index(normalized_word, auto_add=True)
        # _resolve_word_index handles adding to core lists

        # --- Populate Filter Index ---
        if self.__filter_n:
            # Get unique filter n-grams first
            filter_grams = set(get_n_grams(normalized_word, self.__filter_n))
            for n_gram in filter_grams:
                # setdefault ensures BitMap exists, then add index
                self.__ngram_filter_index.setdefault(n_gram, BitMap()).add(word_index)

        # --- Populate Counts and Positions Indices ---
        for n_size in self.__n_list:
            n_grams_list = get_n_grams(normalized_word, n_size)
            num_n_grams = len(n_grams_list)
            if num_n_grams == 0: continue # Skip if word is too short for this n

            # Calculate locations for unique n-grams
            n_gram_locations = defaultdict(list)
            divisor = float(max(1, num_n_grams - 1))
            for index, n_gram in enumerate(n_grams_list):
                n_gram_locations[n_gram].append(index / divisor)

            # Add counts and position tuples to inverted indices
            for n_gram, locations_list in n_gram_locations.items():
                locations_tuple = tuple(locations_list) # Store immutable tuple
                count = len(locations_tuple)
                self.__ngram_counts_index[n_gram].append((word_index, count))
                self.__ngram_positions_index[n_gram].append((word_index, locations_tuple))
        return self

    # --- Lookup Implementation ---
    def __lookup_similarity(self,
                            normalized_query_word: str,
                            dim: Union[int, float],
                            top_k: int,
                            normalize: bool,
                            ) -> Counter:
        """
        Internal method to calculate approximate similarity scores for candidates.

        This implements a multi-stage filtering and scoring process:
        1. Pre-calculates query n-gram information.
        2. Filters candidates using the fast bitmap filter index (if enabled).
        3. Calculates lower-bound scores based on shared n-gram counts.
        4. Determines a threshold based on the k-th best lower-bound score.
        5. Filters remaining candidates based on an upper-bound score proxy.
        6. Calculates the final approximate NMD score using emd_1d for the finalists.
        7. Aggregates scores across different n-gram sizes.

        Args:
            normalized_query_word: The query word after normalization.
            dim: The dimension parameter for the `mean` function used for score aggregation.
            top_k: The target number of results.
            normalize: Whether to normalize similarity scores (0-1 range).

        Returns:
            A collections.Counter mapping word_index to final approximate score.
        """

        # === Section 1: Pre-calculate Query Info ===
        query_length = len(normalized_query_word)
        if query_length == 0: return Counter()

        # Calculate filter n-grams for the query
        query_filter_grams: Optional[Set[str]] = None
        if self.__filter_n:
            query_filter_grams = set(get_n_grams(normalized_query_word, self.__filter_n))
            # If query is too short for filter_n, no matches are possible via filter
            if not query_filter_grams: return Counter()

        # Calculate query n-grams, counts, locations, and total counts ONCE
        # query_ngrams_by_n: Maps n_size -> list of n_grams for query
        query_ngrams_by_n: Dict[int, List[str]] = {}
        # query_counters: Maps n_size -> Counter of n_grams for query
        query_counters: Dict[int, Counter] = {}
        # query_locations: Maps n_size -> {n_gram -> tuple_of_positions} for query
        query_locations: Dict[int, Dict[str, Tuple[float, ...]]] = {}
        # query_num_grams_tuple: Tuple of total #n-grams for query (len = len(n_list))
        query_num_grams_tuple: Tuple[int, ...] = tuple(num_grams(query_length, n) for n in self.__n_list)

        for n_index, n_size in enumerate(self.__n_list):
            # Skip if query word is too short for this n-gram size
            if query_num_grams_tuple[n_index] == 0: continue

            n_grams_list = get_n_grams(normalized_query_word, n_size)
            query_ngrams_by_n[n_size] = n_grams_list
            query_counters[n_size] = Counter(n_grams_list)

            num_n_grams = len(n_grams_list)
            if num_n_grams > 0:
                query_loc_dict = defaultdict(list)
                divisor = float(max(1, num_n_grams - 1))
                for index, n_gram in enumerate(n_grams_list):
                    query_loc_dict[n_gram].append(index / divisor)
                # Store locations as {gram: (pos_tuple)}
                query_locations[n_size] = {gram: tuple(pos) for gram, pos in query_loc_dict.items()}
            else:
                 query_locations[n_size] = {} # Should not happen if num_grams > 0 check passed

        # === Section 2: Initial Bitmap Filtering ===
        candidate_bitmap = BitMap() # Stores indices passing the filter
        if self.__filter_n and query_filter_grams: # Ensure filter enabled and query usable
            for n_gram in query_filter_grams:
                gram_bitmap = self.__ngram_filter_index.get(n_gram) # Look up n-gram in filter index
                if gram_bitmap:
                    candidate_bitmap.update(gram_bitmap) # Efficient union
            # If no words share any filter n-grams, return early
            if not candidate_bitmap: return Counter()
        elif self.__filter_n and not query_filter_grams:
             # Should have been caught earlier, but safety check
             return Counter()
        else: # No filter index enabled - consider all words
            if self.__index_to_word: # Check if vocabulary exists
                 # Create bitmap containing all valid word indices
                 candidate_bitmap = BitMap(range(len(self.__index_to_word)))
            else:
                 return Counter() # No words in index

        # === Section 3: Bounds Calculation (Lower Bound Proxy) ===
        # Calculates a score based on min(query_count, candidate_count) for shared n-grams.
        # This serves as a proxy for the lower bound of the final similarity score.
        # min_scores_per_candidate: Maps word_index -> list_of_min_scores (one per n_size)
        min_scores_per_candidate: Dict[int, List[float]] = defaultdict(
            lambda: self.__zeros_list_for_scores.copy() # Use preallocated list + copy
        )
        # Keep track of candidates actually encountered during count lookup
        candidate_indices_in_min_scores: Set[int] = set()

        for n_index, n_size in enumerate(self.__n_list):
            if n_size not in query_counters: continue # Skip if query had no grams for this n

            query_num_grams_for_n = query_num_grams_tuple[n_index]
            query_counter_for_n = query_counters[n_size]

            for n_gram, query_count in query_counter_for_n.items():
                # Iterate through stored (word_index, count) for this n_gram
                for other_word_index, other_count in self.__ngram_counts_index.get(n_gram, []):
                    # Fast check: Is this candidate still viable after initial bitmap filter?
                    if other_word_index not in candidate_bitmap:
                        continue

                    # Candidate is relevant, calculate min contribution
                    other_num_grams_for_n = self.__index_to_num_grams[other_word_index][n_index]
                    denominator = query_num_grams_for_n + other_num_grams_for_n

                    if denominator > 0:
                        min_shared_count = min(query_count, other_count)
                        # Calculate contribution (normalized or raw)
                        score_contribution = min_shared_count / denominator if normalize else float(min_shared_count)
                        # Add contribution to the correct index in the score list
                        min_scores_per_candidate[other_word_index][n_index] += score_contribution
                        candidate_indices_in_min_scores.add(other_word_index)

        # If no candidates share any n-grams (across all sizes), return early
        if not min_scores_per_candidate: return Counter()

        # === Section 4: Thresholding (Filter by Max Score Bound) ===
        # Combine the lower-bound scores across different n-sizes using 'mean'
        combined_min_scores = {
            word_index: mean(scores, dim)
            for word_index, scores in min_scores_per_candidate.items()
        }

        # Determine the k-th best lower-bound score to set a threshold
        num_candidates_after_min_calc = len(combined_min_scores)
        actual_k = min(top_k, num_candidates_after_min_calc)
        if actual_k == 0: return Counter()

        # Find the k candidates with the highest minimum scores efficiently
        top_k_min_scores = heapq.nlargest(actual_k, combined_min_scores.values())
        # The threshold is the score of the k-th candidate found
        min_acceptable_score_threshold = top_k_min_scores[-1]

        # Filter candidates further: keep only those whose *maximum* possible score
        # (estimated as 2 * min_score contribution) could possibly meet the threshold.
        finalist_bitmap = BitMap() # Stores indices passing the bounds check
        for word_index, lower_bound_scores in min_scores_per_candidate.items():
             # The maximum contribution for an n-gram is 2 * min contribution
             # This holds true even with normalization applied earlier
             upper_bound_scores = [score * 2.0 for score in lower_bound_scores]
             max_possible_score = mean(upper_bound_scores, dim=dim)

             if max_possible_score >= min_acceptable_score_threshold:
                 finalist_bitmap.add(word_index)

        # If no candidates remain after bounds filtering, return early
        if not finalist_bitmap: return Counter()

        # === Section 5: Actual EMD Scoring ===
        # Now calculate the actual approximate NMD score (using emd_1d) only for finalists.
        # actual_scores_per_candidate: Maps word_index -> list_of_actual_scores (one per n_size)
        actual_scores_per_candidate: Dict[int, List[float]] = defaultdict(
            lambda: self.__zeros_list_for_scores.copy()
        )

        for n_index, n_size in enumerate(self.__n_list):
            if n_size not in query_locations: continue # Skip if query had no locations for this n

            query_num_grams_for_n = query_num_grams_tuple[n_index]
            query_locations_for_n = query_locations[n_size]

            for n_gram, query_pos_tuple in query_locations_for_n.items():
                # Iterate through stored (word_index, positions) for this n_gram
                for other_word_index, other_pos_tuple in self.__ngram_positions_index.get(n_gram, []):
                    # Fast check: Is this candidate a finalist?
                    if other_word_index not in finalist_bitmap:
                        continue

                    # This candidate needs full scoring
                    # Calculate similarity contribution: num_matches - emd_cost
                    similarity_contribution = float(len(query_pos_tuple) + len(other_pos_tuple)) \
                                              - emd_1d(query_pos_tuple, other_pos_tuple)

                    # Normalize if required
                    if normalize:
                         other_num_grams_for_n = self.__index_to_num_grams[other_word_index][n_index]
                         denominator = query_num_grams_for_n + other_num_grams_for_n
                         if denominator > 0:
                             actual_scores_per_candidate[other_word_index][n_index] += similarity_contribution / denominator
                         # else: score remains 0.0 for this n if denominator is 0
                    else:
                         actual_scores_per_candidate[other_word_index][n_index] += similarity_contribution

        # === Section 6: Final Score Aggregation ===
        # Combine scores across different n-sizes using the specified 'mean' function
        final_scores = Counter({
            word_index: mean(scores, dim)
            for word_index, scores in actual_scores_per_candidate.items()
        })

        return final_scores # Return Counter {word_index: final_approx_score}

    # --- Public Lookup Method ---
    def lookup(self,
               word: str,
               top_k: int = 5,
               dim: Union[int, float] = 1,
               normalize: bool = False,
               ) -> List[Tuple[str, float]]:
        """
        Finds the top_k words in the index most similar to the input word.

        Args:
            word: The query word string.
            top_k: The maximum number of similar words to return.
            dim: The dimension parameter for the `mean` function used for score
                 aggregation across different n-gram sizes (e.g., 1 for arithmetic mean).
            normalize: If True, similarity scores are normalized to the range [0.0, 1.0].
                       If False, scores are raw similarity contributions.

        Returns:
            A list of (word, score) tuples, sorted by score descending.

        Raises:
            ValueError: If word is empty or top_k is not positive.
        """
        # --- Input Validation ---
        if not isinstance(word, str) or not word:
            raise ValueError("Query word must be a non-empty string")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        # TODO: Validate 'dim' if 'mean' function requires it

        # --- Normalization ---
        normalized_word = self._normalize_word(word)
        if not normalized_word:
            # Word became empty after normalization
            return []
        # Check for reserved chars again just in case normalization introduced them
        if '\2' in normalized_word or '\3' in normalized_word:
             # Or raise ValueError("Query word contains reserved control characters after normalization")
             return []

        # --- Perform Similarity Lookup ---
        # Calls the internal method which does the heavy lifting
        word_scores_counter = self.__lookup_similarity(normalized_word, dim, top_k, normalize)

        # --- Get Top Results ---
        # Retrieve the top_k results directly from the Counter
        top_results = word_scores_counter.most_common(top_k)

        # --- Format Output ---
        # Convert word indices back to original normalized words
        # Results are already sorted by score descending by most_common
        return [(self.__index_to_word[word_index], score) for word_index, score in top_results]

    def __len__(self) -> int:
        """Returns the number of unique words in the index."""
        return len(self.__index_to_word)

    def __contains__(self, word: str) -> bool:
        """Checks if a word exists in the index (after normalization)."""
        if not isinstance(word, str):
             return False
        normalized_word = self._normalize_word(word)
        return normalized_word in self.__word_to_index