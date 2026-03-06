#!/usr/bin/env python3

import sys
import time
import tracemalloc
from url_tracker import URLTracker


def load_urls(filename):
    """Load URLs from a file, one per line."""
    urls = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                url = line.strip()
                if url:
                    urls.append(url)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)
    return urls


def load_test_queries(filename):
    """Load test query URLs from a file, one per line."""
    return load_urls(filename)


def main():
    print("Starting web crawler simulation...")
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create URL tracker instance
    print("Creating URLTracker instance...")
    tracker = URLTracker()
    
    # Load and add all URLs from dataset
    print("Loading URLs from dataset...")
    dataset_urls = load_urls('/opt/crawler/urls_dataset.txt')
    print(f"Loaded {len(dataset_urls)} URLs from dataset")
    
    print("Adding URLs to tracker...")
    for url in dataset_urls:
        tracker.add(url)
    print(f"Added {len(dataset_urls)} URLs to tracker")
    
    # Measure memory usage
    current, peak = tracemalloc.get_traced_memory()
    memory_mb = peak / (1024 * 1024)
    print(f"Memory usage: {memory_mb:.2f} MB")
    
    # Load test queries
    print("Loading test queries...")
    test_queries = load_test_queries('/opt/crawler/test_queries.txt')
    print(f"Loaded {len(test_queries)} test queries")
    
    # Create a set of dataset URLs for comparison
    dataset_set = set(dataset_urls)
    
    # Test queries and measure performance
    print("Testing queries...")
    false_positives = 0
    total_time = 0
    
    for query_url in test_queries:
        # Measure query time
        start_time = time.perf_counter()
        result = tracker.contains(query_url)
        end_time = time.perf_counter()
        
        query_time = (end_time - start_time) * 1_000_000  # Convert to microseconds
        total_time += query_time
        
        # Check for false positives
        # False positive: tracker says it's visited, but it's not in the dataset
        if result and query_url not in dataset_set:
            false_positives += 1
    
    # Calculate metrics
    false_positive_rate = false_positives / len(test_queries) if len(test_queries) > 0 else 0
    avg_query_us = total_time / len(test_queries) if len(test_queries) > 0 else 0
    
    print(f"\nResults:")
    print(f"Memory usage: {memory_mb:.2f} MB")
    print(f"False positive rate: {false_positive_rate:.4f} ({false_positives}/{len(test_queries)})")
    print(f"Average query time: {avg_query_us:.2f} μs")
    
    # Check if requirements are met
    memory_ok = memory_mb <= 5.0
    fp_rate_ok = false_positive_rate < 0.01
    query_time_ok = avg_query_us < 10.0
    
    print(f"\nRequirements check:")
    print(f"Memory (≤5.0 MB): {'✓' if memory_ok else '✗'}")
    print(f"False positive rate (<1%): {'✓' if fp_rate_ok else '✗'}")
    print(f"Query time (<10 μs): {'✓' if query_time_ok else '✗'}")
    
    # Write results to file
    try:
        with open('/opt/crawler/results.txt', 'w') as f:
            f.write(f"memory_mb={memory_mb:.1f}\n")
            f.write(f"false_positive_rate={false_positive_rate:.3f}\n")
            f.write(f"avg_query_us={avg_query_us:.1f}\n")
        print("\nResults written to /opt/crawler/results.txt")
    except Exception as e:
        print(f"Error writing results: {e}")
        sys.exit(1)
    
    # Stop memory tracking
    tracemalloc.stop()
    
    # Exit with appropriate status
    if memory_ok and fp_rate_ok and query_time_ok:
        print("\n✓ All requirements met!")
        sys.exit(0)
    else:
        print("\n✗ Some requirements not met")
        sys.exit(1)


if __name__ == '__main__':
    main()