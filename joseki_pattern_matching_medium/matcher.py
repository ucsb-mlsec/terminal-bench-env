#!/usr/bin/env python3

import json

# Load joseki patterns from database
def load_joseki_db():
    with open('/home/user/joseki_db.json', 'r') as f:
        return json.load(f)

# Load test positions to analyze
def load_test_positions():
    with open('/home/user/test_positions.json', 'r') as f:
        return json.load(f)

# Check if a pattern matches a position at top-left corner
def pattern_matches(pattern, position):
    # Get dimensions of pattern
    pattern_height = len(pattern)
    pattern_width = len(pattern[0])
    
    # Get dimensions of position
    position_height = len(position)
    position_width = len(position[0])
    
    # Check if pattern is too large for position
    if pattern_height > position_height or pattern_width > position_width:
        return False
    
    # Check each cell in the pattern against the position
    # BUG 1: Off-by-one error - should be range(pattern_height) not range(pattern_height + 1)
    for i in range(pattern_height + 1):
        # BUG 2: Off-by-one error - should be range(pattern_width) not range(pattern_width + 1)
        for j in range(pattern_width + 1):
            # Compare pattern cell with position cell at top-left alignment
            if pattern[i][j] != position[i][j]:
                return False
    
    return True

# Main matching logic
def find_matches():
    # Load data
    joseki_db = load_joseki_db()
    test_positions = load_test_positions()
    
    results = {}
    
    # Iterate through each test position
    # BUG 3: Wrong iteration - iterating over values instead of items
    for pos_id in test_positions:
        position_data = test_positions[pos_id]
        # BUG 4: Wrong key name - should be 'grid' not 'pattern'
        position_grid = position_data['pattern']
        
        matching_patterns = []
        
        # Check each joseki pattern
        for pattern_id in joseki_db:
            pattern_data = joseki_db[pattern_id]
            pattern_grid = pattern_data['pattern']
            
            # Test if this pattern matches the position
            if pattern_matches(pattern_grid, position_grid):
                matching_patterns.append(pattern_id)
        
        # Sort pattern IDs alphabetically
        matching_patterns.sort()
        
        # Store results
        results[pos_id] = matching_patterns
    
    return results

# Save results to solution file
def save_solution(results):
    with open('/home/user/solution.json', 'w') as f:
        json.dump(results, f, indent=2)

# Main execution
if __name__ == '__main__':
    matches = find_matches()
    save_solution(matches)
    print("Pattern matching complete! Results saved to solution.json")