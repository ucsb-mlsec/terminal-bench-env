#!/usr/bin/env python3

import time
import json
from position_cache import PositionCache
from evaluator import evaluate_position

def load_test_positions(filename):
    """Load test positions from file."""
    positions = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    positions.append(line)
        print(f"Loaded {len(positions)} test positions")
        return positions
    except Exception as e:
        print(f"Error loading positions: {e}")
        return []

def run_benchmark():
    """Run the position cache benchmark with temporal locality patterns."""
    
    # Load test positions
    positions_file = '/opt/chess_engine/test_positions.txt'
    positions = load_test_positions(positions_file)
    
    if not positions:
        print("Error: No positions loaded")
        return
    
    # Create cache with capacity 1000
    cache = PositionCache(1000)
    
    # Track lookup times
    lookup_times = []
    
    # Start timing
    start_time = time.time()
    timeout = 60  # 60 second timeout
    
    print("Starting benchmark with temporal locality pattern...")
    
    # Create access pattern with temporal locality
    # Simulates chess engine tree search patterns
    access_sequence = []
    
    # Pattern: access blocks, then revisit recent blocks
    block_size = 100
    revisit_size = 25
    
    num_blocks = min(len(positions) // block_size, 50)  # Limit to reasonable size
    
    for i in range(num_blocks):
        # Access new block
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, len(positions))
        access_sequence.extend(range(start_idx, end_idx))
        
        # Revisit part of previous block (temporal locality)
        if i > 0:
            prev_start = (i - 1) * block_size + block_size // 2
            prev_end = min(prev_start + revisit_size, len(positions))
            access_sequence.extend(range(prev_start, prev_end))
        
        # Revisit part of current block (simulating deeper search)
        revisit_start = start_idx + block_size // 3
        revisit_end = min(revisit_start + revisit_size, end_idx)
        access_sequence.extend(range(revisit_start, revisit_end))
    
    print(f"Access sequence length: {len(access_sequence)}")
    
    # Process access sequence
    verification_samples = []
    
    for idx, pos_idx in enumerate(access_sequence):
        # Check timeout
        if time.time() - start_time > timeout:
            print("ERROR: Benchmark timeout exceeded!")
            break
        
        position = positions[pos_idx]
        
        # Time the cache lookup
        lookup_start = time.time()
        cached_eval = cache.get(position)
        lookup_end = time.time()
        
        lookup_time_ms = (lookup_end - lookup_start) * 1000
        lookup_times.append(lookup_time_ms)
        
        if cached_eval is None:
            # Cache miss - evaluate and store
            evaluation = evaluate_position(position)
            cache.put(position, evaluation)
            
            # Save some samples for verification
            if len(verification_samples) < 5:
                verification_samples.append((position, evaluation))
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1} accesses in {elapsed:.2f}s")
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nBenchmark completed in {total_time:.2f} seconds")
    
    # Get cache statistics
    stats = cache.get_stats()
    hits = stats['hits']
    misses = stats['misses']
    cache_size = stats['size']
    
    total_accesses = hits + misses
    cache_hit_rate = hits / total_accesses if total_accesses > 0 else 0.0
    
    # Calculate average lookup time
    avg_lookup_ms = sum(lookup_times) / len(lookup_times) if lookup_times else 0.0
    
    print(f"\nCache Statistics:")
    print(f"  Hits: {hits}")
    print(f"  Misses: {misses}")
    print(f"  Total Accesses: {total_accesses}")
    print(f"  Hit Rate: {cache_hit_rate:.4f}")
    print(f"  Cache Size: {cache_size}")
    print(f"  Average Lookup Time: {avg_lookup_ms:.6f} ms")
    
    # Verify correctness
    print("\nVerifying cached evaluations...")
    verification_passed = True
    for position, expected_eval in verification_samples:
        cached = cache.get(position)
        if cached is None or abs(cached - expected_eval) > 0.0001:
            print(f"  VERIFICATION FAILED for position")
            verification_passed = False
            break
    
    if verification_passed:
        print("  Verification PASSED")
    
    # Check performance targets
    target_hit_rate = 0.60
    target_lookup_ms = 0.001
    target_time = 30.0
    
    passed = (
        cache_hit_rate >= target_hit_rate and
        avg_lookup_ms < target_lookup_ms and
        total_time < target_time and
        verification_passed
    )
    
    print("\nPerformance Targets:")
    print(f"  Hit Rate >= {target_hit_rate}: {'PASS' if cache_hit_rate >= target_hit_rate else 'FAIL'} ({cache_hit_rate:.4f})")
    print(f"  Avg Lookup < {target_lookup_ms}ms: {'PASS' if avg_lookup_ms < target_lookup_ms else 'FAIL'} ({avg_lookup_ms:.6f}ms)")
    print(f"  Total Time < {target_time}s: {'PASS' if total_time < target_time else 'FAIL'} ({total_time:.2f}s)")
    print(f"\nOverall: {'PASSED' if passed else 'FAILED'}")
    
    # Save results
    results = {
        "cache_hit_rate": round(cache_hit_rate, 4),
        "avg_lookup_ms": round(avg_lookup_ms, 6),
        "total_time_seconds": round(total_time, 2),
        "passed": passed
    }
    
    results_file = '/opt/chess_engine/results.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    run_benchmark()