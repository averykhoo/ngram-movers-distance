# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: infer_types=True

# Import C standard library functions for potentially faster operations
from libc.stdlib cimport malloc, free
from libc.math cimport fabs  # C absolute value for floats
from libc.float cimport DBL_MAX  # Max double value for min comparison start

# Import Python types for function signature
from typing import Sequence, Union, List

# Define C types for clarity and performance
ctypedef double double_t
ctypedef Py_ssize_t index_t  # Standard C type for sizes/indices

def emd_1d_dp_cython(positions_x: Sequence[Union[int, float]],
                     positions_y: Sequence[Union[int, float]],
                     ) -> double_t:
    """
    Calculates the 1D Earth Mover's Distance using Dynamic Programming (Cython).

    This version handles unequal list sizes by assigning a penalty cost of 1
    for each unmatched point. It uses a space-optimized DP approach with
    O(min(N, M)) space complexity and O(N * M) time complexity.

    Args:
        positions_x: A sequence of numbers representing point positions.
        positions_y: Another sequence of numbers representing point positions.

    Returns:
        The calculated Earth Mover's Distance as a float (double).
    """
    # --- Type Declarations for Local Variables ---
    cdef index_t n, m, i, j
    cdef double_t match_cost, leave_x_cost, leave_y_cost
    cdef double_t * x_ptr = NULL  # Pointer for C array holding sorted x values
    cdef double_t * y_ptr = NULL  # Pointer for C array holding sorted y values
    cdef double_t * prev_dp_row = NULL  # Pointer for previous DP row (C array)
    cdef double_t * curr_dp_row = NULL  # Pointer for current DP row (C array)
    cdef double_t final_emd
    cdef bint swapped = False  # Flag to track if x and y were swapped

    # --- Input Handling & Conversion to C Arrays ---
    # Convert Python sequences to sorted C arrays for direct, fast access
    # Note: This involves copying data initially.

    # Convert x
    n = len(positions_x)
    x_ptr = <double_t *> malloc(n * sizeof(double_t))
    if x_ptr is NULL:
        raise MemoryError("Failed to allocate memory for x array")
    try:
        for i in range(n):
            x_ptr[i] = positions_x[i]  # Direct conversion
        # Sort the C array (requires a C sort function - using Python's sort on a temp list is easier)
        x_list = [x_ptr[i] for i in range(n)]
        x_list.sort()
        for i in range(n):
            x_ptr[i] = x_list[i]
    except:  # Catch potential errors during conversion/allocation
        if x_ptr != NULL: free(x_ptr)
        raise  # Re-raise the exception

    # Convert y
    m = len(positions_y)
    y_ptr = <double_t *> malloc(m * sizeof(double_t))
    if y_ptr is NULL:
        if x_ptr != NULL: free(x_ptr)
        raise MemoryError("Failed to allocate memory for y array")
    try:
        for i in range(m):
            y_ptr[i] = positions_y[i]
        # Sort the C array via temp Python list
        y_list = [y_ptr[i] for i in range(m)]
        y_list.sort()
        for i in range(m):
            y_ptr[i] = y_list[i]
    except:
        if x_ptr != NULL: free(x_ptr)
        if y_ptr != NULL: free(y_ptr)
        raise

    # Ensure x is the shorter list for DP space optimization
    if n > m:
        # Swap pointers and lengths
        x_ptr, y_ptr = y_ptr, x_ptr
        n, m = m, n
        swapped = True  # Remember we swapped if we need original pointers for freeing

    # --- DP Initialization (C Arrays) ---
    # Allocate memory for the two DP rows
    prev_dp_row = <double_t *> malloc((m + 1) * sizeof(double_t))
    curr_dp_row = <double_t *> malloc((m + 1) * sizeof(double_t))
    if prev_dp_row is NULL or curr_dp_row is NULL:
        # Clean up allocated memory before raising error
        if prev_dp_row != NULL: free(prev_dp_row)
        if curr_dp_row != NULL: free(curr_dp_row)
        # Free original x/y pointers correctly considering potential swap
        if swapped:
            if y_ptr != NULL: free(y_ptr)  # Original x
            if x_ptr != NULL: free(x_ptr)  # Original y
        else:
            if x_ptr != NULL: free(x_ptr)  # Original x
            if y_ptr != NULL: free(y_ptr)  # Original y
        raise MemoryError("Failed to allocate memory for DP rows")

    # Initialize the 'previous' row (cost of matching empty x with y)
    for j in range(m + 1):
        prev_dp_row[j] = <double_t> j

    # --- DP Calculation (using C arrays and pointers) ---
    for i in range(1, n + 1):
        # Base case: cost of matching i elements of x with 0 elements of y
        curr_dp_row[0] = <double_t> i

        # Inner loop using pointer arithmetic for potential minor optimization
        # (standard indexing curr_dp_row[j] is usually optimized well by C compiler too)
        for j in range(1, m + 1):
            # Calculate costs using C-level operations
            # Use fabs from libc.math for C double absolute value
            match_cost = fabs(x_ptr[i - 1] - y_ptr[j - 1]) + prev_dp_row[j - 1]
            leave_x_cost = 1.0 + prev_dp_row[j]
            leave_y_cost = 1.0 + curr_dp_row[j - 1]

            # C-style min comparison (or use Python's min)
            # Python's min is often fine, but manual can ensure no Python overhead
            if match_cost <= leave_x_cost and match_cost <= leave_y_cost:
                curr_dp_row[j] = match_cost
            elif leave_x_cost <= match_cost and leave_x_cost <= leave_y_cost:
                curr_dp_row[j] = leave_x_cost
            else:
                curr_dp_row[j] = leave_y_cost
            # Alternatively: curr_dp_row[j] = min(match_cost, leave_x_cost, leave_y_cost)

        # Swap pointers for the next iteration (no data copying)
        prev_dp_row, curr_dp_row = curr_dp_row, prev_dp_row

    # --- Result & Cleanup ---
    final_emd = prev_dp_row[m]  # Result is in the row pointed to by prev_dp_row

    # Free all allocated C memory
    if prev_dp_row != NULL: free(prev_dp_row)
    if curr_dp_row != NULL: free(curr_dp_row)
    # Free original x/y pointers correctly considering potential swap
    if swapped:
        if y_ptr != NULL: free(y_ptr)  # Original x
        if x_ptr != NULL: free(x_ptr)  # Original y
    else:
        if x_ptr != NULL: free(x_ptr)  # Original x
        if y_ptr != NULL: free(y_ptr)  # Original y

    return final_emd
