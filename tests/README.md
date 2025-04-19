# Tests for Ngram Movers Distance

This directory contains tests for the Ngram Movers Distance project.

## Test Structure

- `test_emd_correctness.py`: Tests for Earth Mover's Distance (EMD) implementations
- `test_find_replace_trie.py`: Tests for the Trie implementation

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_emd_correctness.py
```

To run a specific test:

```bash
pytest tests/test_emd_correctness.py::TestEMDCorrectness::test_symmetry
```

## Test Coverage

The tests cover:

1. **EMD Implementations**:
   - Correctness of different EMD implementations
   - Symmetry property (EMD(x,y) = EMD(y,x))
   - Edge cases (empty lists, single elements)
   - Input validation

2. **Trie Implementation**:
   - Unicode space handling
   - Updating with tuples, dictionaries
   - Regex generation
   - Basic operations (add, delete, contains)
   - Finding patterns with and without overlapping