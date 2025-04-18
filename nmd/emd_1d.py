import itertools
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union


def emd_1d_slow(positions_x: Sequence[float],
                positions_y: Sequence[float],
                ) -> float:
    # positions_x must be longer
    if len(positions_x) < len(positions_y):
        positions_x, positions_y = positions_y, positions_x

    # sort both lists
    positions_x = sorted(positions_x)
    positions_y = sorted(positions_y)

    # find the minimum cost alignment
    costs = [len(positions_y)]
    for x_combination in itertools.combinations(positions_x, len(positions_y)):
        costs.append(sum(abs(x - y) for x, y in zip(x_combination, positions_y)))

    # the distance is the min cost alignment plus a count of unmatched points
    return len(positions_x) - len(positions_y) + min(costs)


def emd_1d_dp(positions_x: Sequence[Union[int, float]],
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


def emd_1d_old(positions_x: Sequence[Union[int, float]],
               positions_y: Sequence[Union[int, float]],
               ) -> float:
    """
    calculates the Earth Mover's Distance between two sets of floats
    the sets do not need to be of the same size, and either set may be empty (or both)

    difference from a real emd function:
    * restricted to one dimension only
    * input distributions must be quantized (to points on a line), and cannot be fractional
    * unmatched points are assigned a "distance" of 1, since for this use case the domain is from 0.0 to 1.0
      (integers and floats greater than 1.0 are allowed, but you'll need a bit more math to make sense of the output)
      (this tweak means the algo is a metric that obeys the triangle inequality)

    thanks to these differences, the algorithm can be optimized significantly, and runs in about linear time

    just found this pdf: http://infolab.stanford.edu/pub/cstr/reports/cs/tr/99/1620/CS-TR-99-1620.ch4.pdf
    * §4.3.1 says to use a transportation simplex (TODO: look into this)
    * §4.3.2 says zipping the lists together is correct for equal weight (equivalent to taking the area between CDFs)
    * it doesn't really seem to talk about unequal distributions in 1d though

    :param positions_x:
    :param positions_y:
    :return:
    """
    # x will be the longer list
    if len(positions_x) < len(positions_y):
        positions_x, positions_y = positions_y, positions_x

    # y is empty, so just count the x items and exit early
    if len(positions_y) == 0:
        return float(len(positions_x))

    # y has only one item, so take min distance and count the rest of the x items
    if len(positions_y) == 1:
        return float(min(abs(x - positions_y[0]) for x in positions_x) + len(positions_x) - 1)

    # make a COPY of the list, sorted in reverse (descending order)
    # we'll be modifying x and y in-place later, and we don't want to accidentally edit the input
    # also the input might be immutable (e.g. a tuple), you never know
    positions_x = sorted(positions_x, reverse=True)
    positions_y = sorted(positions_y, reverse=True)

    # if there are exactly the same number of objects in both lists
    # then there must be a 1-to-1 correspondence, so we can just zip the lists together
    # note that this step requires both lists to be sorted (both being in reverse is fine)
    if len(positions_x) == len(positions_y):
        return float(sum(abs(x - y) for x, y in zip(positions_x, positions_y)))

    # remove any matching points in x and y
    # this implementation also reverses the list (i.e. descending -> ascending)
    # matching points contribute 0 distance, so we don't need to account for them
    new_x = []
    new_y = []
    while positions_x and positions_y:
        if positions_x[-1] < positions_y[-1]:
            new_x.append(positions_x.pop(-1))
        elif positions_y[-1] < positions_x[-1]:
            new_y.append(positions_y.pop(-1))
        else:  # discard matching points in x and y
            positions_x.pop(-1)
            positions_y.pop(-1)
    if positions_x:
        positions_x.reverse()
        new_x.extend(positions_x)
    if positions_y:
        positions_y.reverse()
        new_y.extend(positions_y)
    positions_x = new_x
    positions_y = new_y

    # there are no more duplicates across both lists
    # there can still be duplicates within each list, but that's okay
    # both lists are now sorted normally (in ascending order)
    # we also know that the lists do not have the same number of items
    # after having removed duplicate items, this is the last chance to early exit
    if len(positions_y) == 0:
        return float(len(positions_x))
    if len(positions_y) == 1:
        return float(min(abs(x - positions_y[0]) for x in positions_x) + len(positions_x) - 1)

    # now is the hard part of the algorithm, matching possible points from both lists
    # [x1 y1 x2 x3 x4 y2 x3] ==> [x1 y1 x2], [x4 y2 x5] (x3 can never be matched)
    # we'll break the x-y matching problem into sub-problems which can be solved separately
    # the obvious thing to do is to build a bipartite graph and look for connected components
    # but implementing the full graph building and separation algorithm would eat too many cpu cycles
    # so instead we'll use a counting method to find the ranges of x and y that map the each other
    # we'll start by merging the lists in order to find which x can be mapped to from each y
    # thanks to timsort, this merge happens in more or less linear time
    locations = sorted([(loc, False) for loc in positions_x] + [(loc, True) for loc in positions_y])
    component_ranges = []

    # get ranges of FORWARD possible alignments
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
    if current_left is not None:  # current_left could be 0, so don't just test truthiness
        component_ranges.append((current_left, len(locations) - 1))

    # get ranges of BACKWARD possible alignments
    n = 0
    current_right = None
    for idx in range(len(locations) - 1, -1, -1):
        if locations[idx][1]:  # if is_y:
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

    # we'll accumulate distance as we simplify the problem
    distance = 0.0

    # merge ranges to get the sets of connected components
    component_ranges.sort(reverse=True)  # should only contain 2 runs -> about linear time to sort
    last_seen = -1
    while component_ranges:
        # take the first range, then keep taking overlapping ranges
        left, right = component_ranges.pop(-1)
        while component_ranges and component_ranges[-1][0] <= right:
            right = max(right, component_ranges.pop(-1)[1])  # range can be a proper subset

        # count unmatched points since last seen
        if left > last_seen + 1:
            distance += left - last_seen - 1  # count unmatchable points

        # split the range into x and y lists again, in reverse (descending order)
        connected_x = [idx for idx, is_y in locations[right:left - 1 if left else None:-1] if not is_y]
        connected_y = [idx for idx, is_y in locations[right:left - 1 if left else None:-1] if is_y]

        # greedy-match constrained points with only one possible match at the SMALLER end of connected_y
        while connected_y:  # don't need to check connected_x since it cannot be shorter than y
            # if y_min <= x_min, then they must be paired
            if connected_y[-1] <= connected_x[-1]:
                distance += connected_x.pop(-1) - connected_y.pop(-1)

            # x_min < y_min < x_next and abs(y_min - x_min) <= abs(y_min - x_next)
            # meaning that y_min's best option is x_min, for which there are no competing points
            elif len(connected_x) >= 2 \
                    and connected_y[-1] < connected_x[-2] \
                    and (connected_y[-1] - connected_x[-1]) <= (connected_x[-2] - connected_y[-1]):
                distance += connected_y.pop(-1) - connected_x.pop(-1)

            # endpoints do not match, break loop
            else:
                break

        # reverse both lists IN PLACE, so now they are sorted in ascending order
        connected_x.reverse()
        connected_y.reverse()

        # greedy-match constrained points with only one possible match at the LARGER end of connected_y
        while connected_y:
            # if y_max >= x_max, then they must be paired
            if connected_y[-1] >= connected_x[-1]:
                distance += connected_y.pop(-1) - connected_x.pop(-1)

            # x_prev < y_max < x_max and abs(y_max - x_max) <= abs(y_max - x_prev)
            # meaning that y_max's best option is x_max, for which there are no competing points
            elif len(connected_x) >= 2 \
                    and connected_y[-1] > connected_x[-2] \
                    and (connected_x[-1] - connected_y[-1]) <= (connected_y[-1] - connected_x[-1]):
                distance += connected_y.pop(-1) - connected_x.pop(-1)

            # endpoints don't match
            else:
                break

        # try for early exit, because itertools.combinations is slow
        if len(connected_y) == 0:
            distance += len(connected_x)
        elif len(connected_y) == 1:
            distance += float(min(abs(x - connected_y[0]) for x in connected_x)) + len(connected_x) - 1

        # enumerate all possible matches for this connected component
        # this code block works even if connected_y is empty
        # possible: try to greedy-match unshared points (greedy match must succeed for all y)
        # also possible: actually build the bipartite graph to exclude impossible match options
        else:
            costs = [len(connected_y)]
            for x_combination in itertools.combinations(connected_x, len(connected_y)):
                costs.append(sum(abs(x - y) for x, y in zip(x_combination, connected_y)))
            distance += min(costs) + len(connected_x) - len(connected_y)

        # update last seen
        last_seen = right

    # count unmatched points after last seen
    if len(locations) > last_seen + 1:
        distance += len(locations) - last_seen - 1

    return distance
