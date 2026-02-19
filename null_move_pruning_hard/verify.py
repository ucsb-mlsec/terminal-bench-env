#!/usr/bin/env python3

import json
import time
import sys
from game_ai import GameAI, GameState

def load_test_positions(filename='data/test_positions.txt'):
    """Load test positions from file."""
    positions = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    positions.append(line)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        sys.exit(1)
    return positions

def run_performance_tests(positions, depth=8):
    """Run performance tests on all positions at specified depth."""
    print(f"\n{'='*60}")
    print(f"Running performance tests at depth {depth}")
    print(f"{'='*60}")
    
    total_nodes = 0
    times = []
    
    for i, position in enumerate(positions):
        print(f"\nTest position {i+1}/{len(positions)}...")
        
        # Create game state from position string
        game_state = GameState.from_string(position)
        ai = GameAI()
        
        # Measure search time and node count
        start_time = time.time()
        best_move, nodes_evaluated = ai.search(game_state, depth)
        elapsed_time = time.time() - start_time
        
        total_nodes += nodes_evaluated
        times.append(elapsed_time)
        
        print(f"  Nodes evaluated: {nodes_evaluated:,}")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Best move: {best_move}")
    
    avg_time = sum(times) / len(times) if times else 0
    
    print(f"\n{'='*60}")
    print(f"Performance Test Results:")
    print(f"  Total nodes evaluated: {total_nodes:,}")
    print(f"  Average time per position: {avg_time:.2f}s")
    print(f"{'='*60}")
    
    return total_nodes, avg_time

def validate_tactical_positions(positions):
    """Validate that AI finds optimal moves in tactical positions."""
    print(f"\n{'='*60}")
    print(f"Running tactical validation tests")
    print(f"{'='*60}")
    
    # Use last 3 positions as tactical tests
    tactical_positions = positions[-3:]
    
    # Hardcoded expected optimal moves for tactical positions
    # These are the moves that lead to winning sequences
    expected_moves = {
        0: ['move_a5', 'move_b4', 'move_c3'],  # Multiple acceptable moves for position 0
        1: ['move_d7', 'move_e6'],  # Multiple acceptable moves for position 1
        2: ['move_f1']  # Only one winning move for position 2
    }
    
    all_passed = True
    
    for i, position in enumerate(tactical_positions):
        print(f"\nTactical test {i+1}/{len(tactical_positions)}...")
        
        game_state = GameState.from_string(position)
        ai = GameAI()
        
        # Search at depth 8 for tactical precision
        best_move, nodes_evaluated = ai.search(game_state, depth=8)
        
        # Check if move is one of the expected optimal moves
        expected = expected_moves.get(i, [])
        move_found = best_move in expected if expected else True
        
        print(f"  Best move found: {best_move}")
        print(f"  Expected moves: {', '.join(expected) if expected else 'any'}")
        print(f"  Result: {'PASS' if move_found else 'FAIL'}")
        
        if not move_found and expected:
            all_passed = False
    
    print(f"\n{'='*60}")
    print(f"Tactical Validation: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'='*60}")
    
    return all_passed

def main():
    print("="*60)
    print("Game AI Performance Verification")
    print("="*60)
    
    # Load test positions
    positions = load_test_positions('data/test_positions.txt')
    print(f"\nLoaded {len(positions)} test positions")
    
    if len(positions) == 0:
        print("Error: No test positions found")
        sys.exit(1)
    
    # Run performance tests at depth 8
    total_nodes, avg_time = run_performance_tests(positions, depth=8)
    
    # Validate tactical positions
    tactical_pass = validate_tactical_positions(positions)
    
    # Check requirements
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    nodes_requirement = total_nodes < 5_000_000
    time_requirement = avg_time < 3.0
    
    print(f"\nPerformance Requirements:")
    print(f"  Total nodes < 5,000,000: {total_nodes:,} - {'PASS' if nodes_requirement else 'FAIL'}")
    print(f"  Avg time < 3.0s: {avg_time:.2f}s - {'PASS' if time_requirement else 'FAIL'}")
    print(f"  Tactical tests: {'PASS' if tactical_pass else 'FAIL'}")
    
    all_passed = nodes_requirement and time_requirement and tactical_pass
    
    print(f"\n{'='*60}")
    if all_passed:
        print("OVERALL RESULT: PASS ✓")
        print(f"{'='*60}")
        print("\nThe optimized AI meets all requirements!")
        print(f"Performance improvement: {((19234561 - total_nodes) / 19234561 * 100):.1f}% reduction in nodes")
    else:
        print("OVERALL RESULT: FAIL ✗")
        print(f"{'='*60}")
        print("\nThe AI does not yet meet all requirements.")
        if not nodes_requirement:
            print(f"  - Need to reduce nodes by {total_nodes - 5_000_000:,}")
        if not time_requirement:
            print(f"  - Need to reduce time by {avg_time - 3.0:.2f}s")
        if not tactical_pass:
            print(f"  - Need to fix tactical move selection")
    
    print()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()