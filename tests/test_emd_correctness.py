"""
Tests for Earth Mover's Distance (EMD) implementations.

This file contains tests converted from the original correctness-test.py
in the experiments directory.
"""
import itertools
import math
import pytest
from typing import List, Sequence, Union

from nmd.emd_1d import emd_1d_dp
from nmd.emd_1d import emd_1d_hybrid
from nmd.emd_1d import emd_1d_old as emd_1d_fast_original


# Keep the optimized implementation for testing
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


def check_emd_implementations(positions_x, positions_y, tolerance=1e-9):
    """
    Helper function to check that all EMD implementations produce consistent results.
    
    Args:
        positions_x: List/sequence of positions (numbers).
        positions_y: List/sequence of positions (numbers).
        tolerance: The absolute tolerance for floating point comparisons.
        
    Returns:
        The result from the fast_original implementation.
        
    Raises:
        AssertionError: If the results from different implementations don't match.
    """
    # Use lists to ensure sequences aren't exhausted if they are iterators
    list_x = list(positions_x)
    list_y = list(positions_y)

    # Calculate results from all implementations
    result_fast_original = emd_1d_fast_original(list_x, list_y)  # BASELINE
    result_dp = emd_1d_dp(list_x, list_y)
    result_dp_opt = emd_1d_dp_optimized(list_x, list_y)
    result_hybrid = emd_1d_hybrid(list_x, list_y)

    # Check that all implementations produce the same result
    assert math.isclose(result_fast_original, result_dp, abs_tol=tolerance)
    assert math.isclose(result_fast_original, result_dp_opt, abs_tol=tolerance)
    assert math.isclose(result_fast_original, result_hybrid, abs_tol=tolerance)
    
    return result_fast_original


class TestEMDCorrectness:
    """Tests for Earth Mover's Distance implementations."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        num_x = 3
        num_y = 9
        
        # Generate base positions, avoid division by zero for length 1 lists
        xs_base = [i / (num_x - 1) if num_x > 1 else 0.5 for i in range(num_x)]
        ys_base = [i / (num_y - 1) if num_y > 1 else 0.5 for i in range(num_y)]
        
        # Create duplicates for more complex scenarios
        xs = xs_base + xs_base + xs_base
        ys = ys_base  # Use ys without duplicates for asymmetry
        
        return xs, ys
    
    def test_input_validation(self):
        """Test that inputs are properly validated."""
        # Test numeric inputs
        positions_x = [0.1, 0.5, 0.9]
        positions_y = [0.2, 0.4, 0.6, 0.8]
        
        # These should be valid inputs in [0, 1]
        check_emd_implementations(positions_x, positions_y)
        
        # # TODO: test with invalid inputs (outside [0, 1])
        # with pytest.raises(AssertionError):
        #     check_emd_implementations([-0.1, 0.5, 0.9], positions_y)
        #
        # with pytest.raises(AssertionError):
        #     check_emd_implementations(positions_x, [0.2, 1.1, 0.6])
    
    def test_symmetry(self, sample_data):
        """Test that EMD is symmetric: EMD(x, y) = EMD(y, x)."""
        xs, ys = sample_data
        
        # Test a few combinations
        for x_len in range(1, 4):  # Test with smaller subset for speed
            for y_len in range(1, 4):
                for x_combi in itertools.islice(itertools.combinations(xs, x_len), 3):
                    for y_combi in itertools.islice(itertools.combinations(ys, y_len), 3):
                        # Check x vs y
                        result_xy = check_emd_implementations(x_combi, y_combi)
                        
                        # Check y vs x (symmetry test)
                        result_yx = check_emd_implementations(y_combi, x_combi)
                        
                        # Check results match (symmetry)
                        assert math.isclose(result_xy, result_yx, abs_tol=1e-9)
    
    def test_edge_cases(self):
        """Test edge cases like empty lists and single elements."""
        # Empty lists
        check_emd_implementations([], [0.5])
        
        check_emd_implementations([0.5], [])

        # Single elements
        result = check_emd_implementations([0.5], [0.5])
        assert math.isclose(result, 0.0, abs_tol=1e-9)
        
        result = check_emd_implementations([0.2], [0.8])
        assert math.isclose(result, 0.6, abs_tol=1e-9)
    
    def test_comprehensive(self, sample_data):
        """More comprehensive test with various combinations."""
        xs, ys = sample_data
        
        # Test with smaller subsets for reasonable test time
        for x_len in range(1, 4):
            for y_len in range(1, 4):
                print(f"Testing with x_len={x_len}, y_len={y_len}")
                
                # Take just a few combinations to keep test time reasonable
                x_combinations = list(itertools.combinations(xs, x_len))[:3]
                y_combinations = list(itertools.combinations(ys, y_len))[:3]
                
                for x_combi in x_combinations:
                    for y_combi in y_combinations:
                        # Verify all implementations agree
                        check_emd_implementations(x_combi, y_combi)