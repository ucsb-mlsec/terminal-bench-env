#!/usr/bin/env python3

import sys
import os
import json

def main():
    print("Chess Engine Position Tracking Test")
    print("=" * 50)
    
    # Check if solution.py exists
    solution_path = '/workspace/solution.py'
    if not os.path.exists(solution_path):
        print(f"ERROR: {solution_path} not found!")
        sys.exit(1)
    
    # Import the solution
    sys.path.insert(0, '/workspace')
    try:
        import solution
    except ImportError as e:
        print(f"ERROR: Failed to import solution.py: {e}")
        sys.exit(1)
    
    # Load positions
    positions_file = '/workspace/chess_engine/positions.txt'
    queries_file = '/workspace/chess_engine/queries.txt'
    
    if not os.path.exists(positions_file):
        print(f"ERROR: {positions_file} not found!")
        sys.exit(1)
    
    if not os.path.exists(queries_file):
        print(f"ERROR: {queries_file} not found!")
        sys.exit(1)
    
    print(f"\nLoading positions from {positions_file}...")
    with open(positions_file, 'r') as f:
        positions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(positions)} positions")
    
    print(f"\nLoading queries from {queries_file}...")
    with open(queries_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(queries)} queries")
    
    # Create position set for validation
    positions_set = set(positions)
    
    # Build the lookup structure using the solution
    print("\nBuilding position lookup structure...")
    try:
        if hasattr(solution, 'PositionTracker'):
            tracker = solution.PositionTracker()
            for pos in positions:
                tracker.add(pos)
        else:
            print("ERROR: solution.py must define a PositionTracker class with add() and query() methods")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to build lookup structure: {e}")
        sys.exit(1)
    
    # Get memory usage
    try:
        memory_bytes = tracker.get_memory_usage()
        print(f"Memory usage: {memory_bytes:,} bytes ({memory_bytes / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"ERROR: Failed to get memory usage: {e}")
        sys.exit(1)
    
    # Test queries
    print("\nTesting queries...")
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for query in queries:
        result = tracker.query(query)
        actual_seen = query in positions_set
        
        if result and actual_seen:
            true_positives += 1
        elif result and not actual_seen:
            false_positives += 1
        elif not result and actual_seen:
            false_negatives += 1
        else:  # not result and not actual_seen
            true_negatives += 1
    
    # Calculate metrics
    total_seen = sum(1 for q in queries if q in positions_set)
    total_unseen = len(queries) - total_seen
    false_positive_rate = false_positives / total_unseen if total_unseen > 0 else 0.0
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Memory Used: {memory_bytes:,} bytes ({memory_bytes / 1024 / 1024:.2f} MB)")
    print(f"True Positives: {true_positives} / {total_seen}")
    print(f"False Negatives: {false_negatives}")
    print(f"False Positives: {false_positives} / {total_unseen}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate * 100:.2f}%)")
    
    # Check pass criteria
    memory_pass = memory_bytes <= 1048576
    fp_rate_pass = false_positive_rate < 0.05
    no_false_negatives = false_negatives == 0
    
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)
    print(f"Memory ≤ 1MB: {'PASS' if memory_pass else 'FAIL'}")
    print(f"False Positive Rate < 5%: {'PASS' if fp_rate_pass else 'FAIL'}")
    print(f"No False Negatives: {'PASS' if no_false_negatives else 'FAIL'}")
    
    overall_pass = memory_pass and fp_rate_pass and no_false_negatives
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    
    # Save results
    results = {
        "memory_bytes": memory_bytes,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate
    }
    
    results_file = '/workspace/results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    sys.exit(0 if overall_pass else 1)

if __name__ == '__main__':
    main()