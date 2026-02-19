#!/usr/bin/env python3

import json
import os
import pytest


def test_solutions_file_exists():
    """Test that the solutions.json file exists."""
    assert os.path.exists('/workspace/solutions.json'), "solutions.json file does not exist"


def test_solutions_json_valid():
    """Test that solutions.json contains valid JSON."""
    with open('/workspace/solutions.json', 'r') as f:
        data = json.load(f)
    assert isinstance(data, dict), "JSON should contain a dictionary"


def test_solutions_has_correct_keys():
    """Test that solutions.json has exactly the required keys."""
    with open('/workspace/solutions.json', 'r') as f:
        data = json.load(f)
    
    expected_keys = {"4", "5", "6", "8"}
    actual_keys = set(data.keys())
    
    assert actual_keys == expected_keys, f"Expected keys {expected_keys}, but got {actual_keys}"


def test_solutions_values_are_integers():
    """Test that all solution counts are integers."""
    with open('/workspace/solutions.json', 'r') as f:
        data = json.load(f)
    
    for key, value in data.items():
        assert isinstance(value, int), f"Value for key '{key}' should be an integer, got {type(value)}"


def test_solutions_correct_counts():
    """Test that the N-Queens solution counts are correct."""
    with open('/workspace/solutions.json', 'r') as f:
        data = json.load(f)
    
    expected_counts = {
        "4": 2,
        "5": 10,
        "6": 4,
        "8": 92
    }
    
    for key, expected_value in expected_counts.items():
        actual_value = data.get(key)
        assert actual_value == expected_value, \
            f"For N={key}, expected {expected_value} solutions but got {actual_value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

```python
#!/usr/bin/env python3

import json


def is_safe(board, row, col, n):
    """Check if placing a queen at (row, col) is safe."""
    # Check column
    for i in range(row):
        if board[i] == col:
            return False
    
    # Check diagonal (upper left)
    for i in range(row):
        if abs(board[i] - col) == abs(i - row):
            return False
    
    return True


def solve_n_queens(n):
    """Find all solutions to the N-Queens puzzle for board size n."""
    solutions = []
    board = [-1] * n
    
    def backtrack(row):
        if row == n:
            solutions.append(board[:])
            return
        
        for col in range(n):
            if is_safe(board, row, col, n):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1
    
    backtrack(0)
    return solutions


def main():
    """Solve N-Queens for board sizes 4, 5, 6, and 8 and save results."""
    board_sizes = [4, 5, 6, 8]
    results = {}
    
    for size in board_sizes:
        solutions = solve_n_queens(size)
        results[str(size)] = len(solutions)
        print(f"N={size}: Found {len(solutions)} solutions")
    
    # Save results to JSON file
    with open('/workspace/solutions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to /workspace/solutions.json")


if __name__ == "__main__":
    main()