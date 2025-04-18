import itertools
import math
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from experiment import emd_1d_dp
# from experiment import emd_1d_slow
from nmd.nmd_core import emd_1d as emd_1d_fast_original


def emd_1d_hybrid(positions_x: Sequence[Union[int, float]],
                  positions_y: Sequence[Union[int, float]],
                  ) -> float:
    """
    Calculates the 1D Earth Mover's Distance using a hybrid approach.

    It combines the efficient pre-processing and greedy matching from the
    original fast `emd_1d` with a Dynamic Programming fallback to replace
    the potentially slow `itertools.combinations` step, ensuring polynomial
    worst-case time complexity.

    Handles unequal list sizes by assigning a penalty cost of 1 for each
    unmatched point.

    Args:
        positions_x: A sequence of numbers representing point positions.
        positions_y: Another sequence of numbers representing point positions.

    Returns:
        The calculated Earth Mover's Distance.
    """
    # x will be the longer list initially (as in original)
    if len(positions_x) < len(positions_y):
        positions_x, positions_y = positions_y, positions_x

    # --- Initial Edge Case Handling (Identical to original) ---
    if len(positions_y) == 0:
        return float(len(positions_x))

    if len(positions_y) == 1:
        # Optimization: Calculate min distance directly if y has only one point
        # Use abs() for correct difference calculation
        min_dist = min(abs(x - positions_y[0]) for x in positions_x)
        # Add penalty for unmatched points in x
        return float(min_dist + len(positions_x) - 1)

    # --- Sorting and Copying (Identical to original) ---
    # Make copies, sort in reverse for initial duplicate removal processing
    positions_x = sorted(positions_x, reverse=True)
    positions_y = sorted(positions_y, reverse=True)

    # --- Handling Equal Length Case (Identical to original) ---
    if len(positions_x) == len(positions_y):
        # If lists are equal length after sorting, pair them directly
        # This requires sorting (already done, reversed is fine for zip)
        return float(sum(abs(x - y) for x, y in zip(positions_x, positions_y)))

    # --- Remove Matching Points (Identical to original) ---
    # Reduces problem size by removing points present in both lists
    # Also reverses lists to ascending order
    new_x = []
    new_y = []
    while positions_x and positions_y:
        if positions_x[-1] < positions_y[-1]:
            new_x.append(positions_x.pop(-1))
        elif positions_y[-1] < positions_x[-1]:
            new_y.append(positions_y.pop(-1))
        else:  # discard matching points
            positions_x.pop(-1)
            positions_y.pop(-1)
    # Extend with remaining elements (already sorted ascending)
    if positions_x:
        positions_x.reverse()
        new_x.extend(positions_x)
    if positions_y:
        positions_y.reverse()
        new_y.extend(positions_y)
    positions_x = new_x
    positions_y = new_y

    # --- Post-Removal Edge Case Handling (Identical to original) ---
    # Re-check lengths after removing common elements
    if len(positions_y) == 0:
        return float(len(positions_x))  # Only unmatched x points remain
    if len(positions_y) == 1:
        # Find min distance to the single y point + penalty for other x points
        min_dist = min(abs(x - positions_y[0]) for x in positions_x)
        return float(min_dist + len(positions_x) - 1)
    # We know len(positions_x) > len(positions_y) >= 2 at this point

    # --- Connected Component Analysis (Identical to original) ---
    # Merge lists to identify potentially connected matching regions
    locations = sorted([(loc, False) for loc in positions_x] + [(loc, True) for loc in positions_y])
    component_ranges: List[Tuple[int, int]] = []

    # Find forward connected components
    n = 0
    current_left = None
    for idx, (loc, is_y) in enumerate(locations):
        if is_y:
            n += 1
            if current_left is None:
                current_left = idx
        elif n > 0:
            n -= 1
            if n == 0:
                component_ranges.append((current_left, idx))
                current_left = None
    if current_left is not None:
        component_ranges.append((current_left, len(locations) - 1))

    # Find backward connected components
    n = 0
    current_right = None
    for idx in range(len(locations) - 1, -1, -1):
        if locations[idx][1]:  # if is_y
            n += 1
            if current_right is None:
                current_right = idx
        elif n > 0:
            n -= 1
            if n == 0:
                component_ranges.append((idx, current_right))
                current_right = None
    if current_right is not None:
        component_ranges.append((0, current_right))

    # --- Process Components (Main Loop - Modified Fallback) ---
    distance = 0.0
    component_ranges.sort(reverse=True)  # Sort for efficient merging
    last_seen = -1
    while component_ranges:
        # Merge overlapping ranges (identical to original)
        left, right = component_ranges.pop(-1)
        while component_ranges and component_ranges[-1][0] <= right:
            right = max(right, component_ranges.pop(-1)[1])

        # Count unmatched points between components (identical to original)
        if left > last_seen + 1:
            # Count points in 'locations' between last_seen+1 and left-1
            # These points cannot be matched
            unmatched_count = 0
            for k in range(last_seen + 1, left):
                if not locations[k][1]:  # Count only x points (y points were handled)
                    unmatched_count += 1
            distance += float(unmatched_count)

        # Split the current merged component back into x and y lists (ascending order)
        # Note: Original sliced reversed, then reversed back. Simpler: slice directly.
        component_x = [loc for idx, (loc, is_y) in enumerate(locations) if left <= idx <= right and not is_y]
        component_y = [loc for idx, (loc, is_y) in enumerate(locations) if left <= idx <= right and is_y]

        # --- Greedy Endpoint Matching (Modified to add abs() distance) ---
        # Match points at the ends if they are the unambiguous best match
        # Match at the SMALLER end
        while component_y and component_x:  # Ensure both non-empty
            x_val = component_x[0]
            y_val = component_y[0]
            if y_val <= x_val:
                distance += abs(x_val - y_val)  # Add absolute distance
                component_x.pop(0)
                component_y.pop(0)
            elif len(component_x) >= 2 \
                    and y_val < component_x[1] \
                    and (y_val - x_val) <= (component_x[1] - y_val):
                distance += abs(y_val - x_val)  # Add absolute distance
                component_x.pop(0)
                component_y.pop(0)
            else:
                break  # Cannot greedily match at the start

        # Match at the LARGER end
        while component_y and component_x:  # Ensure both non-empty
            x_val = component_x[-1]
            y_val = component_y[-1]
            if y_val >= x_val:
                distance += abs(y_val - x_val)  # Add absolute distance
                component_x.pop(-1)
                component_y.pop(-1)
            elif len(component_x) >= 2 \
                    and y_val > component_x[-2] \
                    and (x_val - y_val) <= (y_val - component_x[-2]):
                distance += abs(y_val - x_val)  # Add absolute distance
                component_x.pop(-1)
                component_y.pop(-1)
            else:
                break  # Cannot greedily match at the end

        # --- Core Matching Fallback (Replaced `combinations` with DP) ---
        if len(component_y) == 0:
            # If all y points were matched greedily, remaining x points are unmatched
            distance += float(len(component_x))
        elif len(component_x) == 0:
            # Should not happen if len(y)>0 initially and len(x)>=len(y)
            # If it did, it implies an error earlier or len(x)<len(y) initially
            pass  # Or raise error? distance is correct if all y matched, 0 otherwise
        else:
            # --- START: Dynamic Programming Fallback ---
            # Use DP to solve the EMD for the remaining points in the component.
            # component_x and component_y are already sorted.

            # Ensure x_dp is the shorter list for DP space optimization
            x_dp = component_x
            y_dp = component_y
            n_dp = len(x_dp)
            m_dp = len(y_dp)

            if n_dp > m_dp:
                x_dp, y_dp = y_dp, x_dp
                n_dp, m_dp = m_dp, n_dp

            # Initialize DP rows (allocate inside loop for simplicity)
            prev_dp_row: List[float] = [float(j) for j in range(m_dp + 1)]
            curr_dp_row: List[float] = [0.0] * (m_dp + 1)

            # Fill DP table
            for i in range(1, n_dp + 1):
                curr_dp_row[0] = float(i)  # Cost of leaving i elements of x_dp unmatched
                for j in range(1, m_dp + 1):
                    match_cost = abs(x_dp[i - 1] - y_dp[j - 1]) + prev_dp_row[j - 1]
                    leave_x_cost = 1.0 + prev_dp_row[j]
                    leave_y_cost = 1.0 + curr_dp_row[j - 1]
                    curr_dp_row[j] = min(match_cost, leave_x_cost, leave_y_cost)
                # Update previous row for next iteration
                prev_dp_row = list(curr_dp_row)  # Use list() for shallow copy

            # Add the calculated EMD for this component to the total distance
            distance += prev_dp_row[m_dp]
            # --- END: Dynamic Programming Fallback ---

        # Update last seen index (identical to original)
        last_seen = right

    # --- Count Unmatched Points After Last Component (Identical to original) ---
    if len(locations) > last_seen + 1:
        unmatched_count = 0
        for k in range(last_seen + 1, len(locations)):
            if not locations[k][1]:  # Count only x points
                unmatched_count += 1
        distance += float(unmatched_count)

    return distance


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
    assert all(
        isinstance(x, (int, float)) for x in positions_x), f"Not all elements in positions_x are numbers: {positions_x}"
    assert all(
        isinstance(y, (int, float)) for y in positions_y), f"Not all elements in positions_y are numbers: {positions_y}"

    # Check range (if assuming unit interval, keep this)
    # If the functions should work outside [0,1], comment this out.
    assert all(0 <= x <= 1 for x in positions_x), f"Not all elements in positions_x are in [0, 1]: {positions_x}"
    assert all(0 <= y <= 1 for y in positions_y), f"Not all elements in positions_y are in [0, 1]: {positions_y}"

    # --- Run all four EMD implementations ---
    # Use lists to ensure sequences aren't exhausted if they are iterators
    list_x = list(positions_x)
    list_y = list(positions_y)

    # Calculate all results
    # result_slow = emd_1d_slow(list_x, list_y)
    result_fast_original = emd_1d_fast_original(list_x, list_y)  # BASELINE
    result_dp = emd_1d_dp(list_x, list_y)
    result_hybrid = emd_1d_hybrid(list_x, list_y)

    # --- Compare results against the baseline (fast_original) ---
    dp_matches = math.isclose(result_fast_original, result_dp, abs_tol=tolerance)
    hybrid_matches = math.isclose(result_fast_original, result_hybrid, abs_tol=tolerance)
    # slow_matches = math.isclose(result_fast_original, result_slow, abs_tol=tolerance)  # Compare slow to baseline too

    # --- Assert consistency ---
    # Assert DP vs Baseline
    assert dp_matches, (
        f"Mismatch! Fast Original vs DP\n"
        f"Fast O: {result_fast_original} (Baseline)\n"
        f"DP    : {result_dp}\n"
        f"Hybrid: {result_hybrid}\n"
        # f"Slow  : {result_slow}\n"
        f"Inputs: x={positions_x}, y={positions_y}"
    )
    # Assert Hybrid vs Baseline
    assert hybrid_matches, (
        f"Mismatch! Fast Original vs Hybrid\n"
        f"Fast O: {result_fast_original} (Baseline)\n"
        f"DP    : {result_dp}\n"
        f"Hybrid: {result_hybrid}\n"
        # f"Slow  : {result_slow}\n"
        f"Inputs: x={positions_x}, y={positions_y}"
    )
    # # Assert Slow vs Baseline (Optional but good for sanity)
    # assert slow_matches, (
    #     f"Mismatch! Fast Original vs Slow\n"
    #     f"Fast O: {result_fast_original} (Baseline)\n"
    #     f"DP    : {result_dp}\n"
    #     f"Hybrid: {result_hybrid}\n"
    #     f"Slow  : {result_slow}\n"
    #     f"Inputs: x={positions_x}, y={positions_y}"
    # )

    # Return the reference baseline value if all checks pass
    return result_fast_original


# Keep the original test harness structure, it will now use the updated check function
if __name__ == '__main__':

    num_x = 4
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
