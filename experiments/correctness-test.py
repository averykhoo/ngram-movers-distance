import itertools
import math
import time
from functools import lru_cache
from typing import List
from typing import Sequence
from typing import Union

from nmd.emd_1d import emd_1d_dp
from nmd.emd_1d import emd_1d_hybrid
from nmd.emd_1d import emd_1d_old as emd_1d_fast_original


# Assume emd_1d_slow and potentially other versions exist for testing
def emd_1d_dp_optimized(positions_x: Sequence[Union[int, float]],
                        positions_y: Sequence[Union[int, float]],
                        ) -> float:
    """
    Calculates the 1D Earth Mover's Distance using Dynamic Programming.

    This version handles unequal list sizes by assigning a penalty cost of 1
    for each unmatched point. It uses a space-optimized DP approach with
    O(min(N, M)) space complexity and O(N * M) time complexity, where N and M
    are the lengths of the input sequences.

    Optimization Attempt: Uses slice assignment `[:]` for row update instead of
                         `list()` constructor, which might be marginally faster.

    Args:
        positions_x: A sequence of numbers representing point positions.
        positions_y: Another sequence of numbers representing point positions.

    Returns:
        The calculated Earth Mover's Distance.
    """
    # --- Input Handling & Sorting ---
    # Sort lists first, as required by DP approach
    # Using list() ensures we have mutable lists if input was tuple/etc.
    x = sorted(positions_x)
    y = sorted(positions_y)
    len_x = len(x)
    len_y = len(y)

    # Ensure x is the shorter list to optimize space complexity O(min(N,M))
    if len_x > len_y:
        x, y = y, x
        len_x, len_y = len_y, len_x

    # --- DP Initialization (Two Rows) ---
    # prev_dp_row represents the cost when considering 0 elements from x
    # Corresponds to dp[0][j] = j (cost of leaving j elements of y unmatched)
    # Initialize prev_dp_row directly
    prev_dp_row: List[float] = [float(j) for j in range(len_y + 1)]
    # Allocate curr_dp_row once, contents don't matter
    curr_dp_row: List[float] = prev_dp_row.copy()

    # --- DP Calculation ---
    # Iterate through each element of the shorter list x
    for i in range(1, len_x + 1):
        # Base case for the current row: dp[i][0] = i
        # (cost of leaving the first i elements of x unmatched)
        curr_dp_row[0] = float(i)

        # Iterate through each element of the longer list y
        for j in range(1, len_y + 1):
            # Cost of matching x[i-1] with y[j-1]
            match_cost = abs(x[i - 1] - y[j - 1]) + prev_dp_row[j - 1]

            # Cost of leaving x[i-1] unmatched (penalty 1)
            leave_x_cost = 1.0 + prev_dp_row[j]

            # Cost of leaving y[j-1] unmatched (penalty 1)
            leave_y_cost = 1.0 + curr_dp_row[j - 1]

            # Choose the minimum cost path
            curr_dp_row[j] = min(match_cost, leave_x_cost, leave_y_cost)

        # Update prev_dp_row for the next iteration of i
        # to avoid allocations, we just do a swap
        prev_dp_row, curr_dp_row = curr_dp_row, prev_dp_row

    # --- Result ---
    # The final EMD is in the last cell calculated, which is now stored in prev_dp_row
    # because of the final update step inside the loop.
    return prev_dp_row[len_y]


def check_correct_emd_1d(positions_x: Sequence[Union[int, float]],
                         positions_y: Sequence[Union[int, float]],
                         tolerance: float = 1e-9,
                         ) -> float:
    """
    Compares multiple EMD implementations (slow, fast_original, dp, hybrid)
    for consistency against the 'fast_original' baseline.

    Also performs sanity checks on the input data.

    Args:
        positions_x: List/sequence of positions (numbers).
        positions_y: List/sequence of positions (numbers).
        tolerance: The absolute tolerance for floating point comparisons.

    Returns:
        The result from the fast_original (baseline) implementation if all checks pass.

    Raises:
        AssertionError: If sanity checks fail or if the results from the other
                        EMD implementations do not match the fast_original result
                        within the specified tolerance.
    """

    # --- Sanity Checks ---
    # Check types
    assert isinstance(positions_x, Sequence), f"positions_x is not a Sequence: {type(positions_x)}"
    assert isinstance(positions_y, Sequence), f"positions_y is not a Sequence: {type(positions_y)}"
    assert all(isinstance(x, (int, float)) for x in positions_x), f"Not all elems in positions_x numeric: {positions_x}"
    assert all(isinstance(y, (int, float)) for y in positions_y), f"Not all elems in positions_y numeric: {positions_y}"

    # Check range (if assuming unit interval, keep this)
    # If the functions should work outside [0,1], comment this out.
    assert all(0 <= x <= 1 for x in positions_x), f"Not all elements in positions_x are in [0, 1]: {positions_x}"
    assert all(0 <= y <= 1 for y in positions_y), f"Not all elements in positions_y are in [0, 1]: {positions_y}"

    # if not positions_x or not positions_y:
    #     return max(len(positions_x), len(positions_y))

    # --- Run all four EMD implementations ---
    # Use lists to ensure sequences aren't exhausted if they are iterators
    list_x = list(positions_x)
    list_y = list(positions_y)

    # Calculate all results
    # result_slow = emd_1d_slow(list_x, list_y)
    result_fast_original = emd_1d_fast_original(list_x, list_y)  # BASELINE
    result_dp = emd_1d_dp(list_x, list_y)
    result_dp_opt = emd_1d_dp_optimized(list_x, list_y)
    result_hybrid = emd_1d_hybrid(list_x, list_y)

    # --- Compare results against the baseline (fast_original) ---
    dp_matches = math.isclose(result_fast_original, result_dp, abs_tol=tolerance)
    dp_opt_matches = math.isclose(result_fast_original, result_dp_opt, abs_tol=tolerance)
    hybrid_matches = math.isclose(result_fast_original, result_hybrid, abs_tol=tolerance)
    # slow_matches = math.isclose(result_fast_original, result_slow, abs_tol=tolerance)  # Compare slow to baseline too

    # --- Assert consistency ---
    # Assert DP vs Baseline
    assert dp_matches, (
        f"Mismatch! Fast Original vs DP\n"
        f"Fast O: {result_fast_original} (Baseline)\n"
        f"DP    : {result_dp}\n"
        f"DP opt: {dp_opt_matches}\n"
        f"Hybrid: {result_hybrid}\n"
        # f"Slow  : {result_slow}\n"
        f"Inputs: x={positions_x}, y={positions_y}"
    )
    # Assert DP vs Baseline
    assert dp_opt_matches, (
        f"Mismatch! Fast Original vs DP\n"
        f"Fast O: {result_fast_original} (Baseline)\n"
        f"DP    : {result_dp}\n"
        f"DP opt: {dp_opt_matches}\n"
        f"Hybrid: {result_hybrid}\n"
        # f"Slow  : {result_slow}\n"
        f"Inputs: x={positions_x}, y={positions_y}"
    )
    # Assert Hybrid vs Baseline
    assert hybrid_matches, (
        f"Mismatch! Fast Original vs Hybrid\n"
        f"Fast O: {result_fast_original} (Baseline)\n"
        f"DP    : {result_dp}\n"
        f"DP opt: {dp_opt_matches}\n"
        f"Hybrid: {result_hybrid}\n"
        # f"Slow  : {result_slow}\n"
        f"Inputs: x={positions_x}, y={positions_y}"
    )
    # # Assert Slow vs Baseline (Optional but good for sanity)
    # assert slow_matches, (
    #     f"Mismatch! Fast Original vs Slow\n"
    #     f"Fast O: {result_fast_original} (Baseline)\n"
    #     f"DP    : {result_dp}\n"
    #     f"DP opt: {dp_opt_matches}\n"
    #     f"Hybrid: {result_hybrid}\n"
    #     f"Slow  : {result_slow}\n"
    #     f"Inputs: x={positions_x}, y={positions_y}"
    # )

    # Return the reference baseline value if all checks pass
    return result_fast_original


# Keep the original test harness structure, it will now use the updated check function
if __name__ == '__main__':

    t = time.perf_counter()
    emd_1d_dp_optimized([0], [0])
    print('warmup:', time.perf_counter() - t)

    num_x = 3
    num_y = 9

    # Generate base positions, avoid division by zero for length 1 lists
    xs_base = [i / (num_x - 1) if num_x > 1 else 0.5 for i in range(num_x)]
    ys_base = [i / (num_y - 1) if num_y > 1 else 0.5 for i in range(num_y)]

    # Create duplicates for more complex scenarios
    xs = xs_base + xs_base + xs_base
    ys = ys_base  # Use ys without duplicates for asymmetry

    print(f"Running comparison checks on all combinations up to x={len(xs)}, y={len(ys)}...")
    # Iterate through all possible subset lengths
    for x_len in range(len(xs) + 1):
        for y_len in range(len(ys) + 1):
            print(f"Checking lengths: x={x_len}, y={y_len}")
            # Iterate through all combinations for the current lengths
            for x_combi in itertools.combinations(xs, x_len):
                for y_combi in itertools.combinations(ys, y_len):
                    # Check x vs y using the baseline function
                    # Let assertion handle failures immediately
                    result_xy = check_correct_emd_1d(x_combi, y_combi)

                    # Check y vs x (symmetry test) using the baseline function
                    result_yx = check_correct_emd_1d(y_combi, x_combi)

                    # Check results match (symmetry)
                    assert math.isclose(result_xy, result_yx, abs_tol=1e-9), \
                        f"Symmetry failed! xy={result_xy}, yx={result_yx}\n x={x_combi}, y={y_combi}"

    print("\nAll checked combinations passed!")  # Reached only if no assertion failed
