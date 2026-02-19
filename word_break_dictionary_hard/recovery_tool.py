#!/usr/bin/env python3

import json
import time

def load_dictionary(filepath):
    """Load dictionary from file - converts to set for fast lookup"""
    with open(filepath, 'r') as f:
        # Bug: strip() might not handle all whitespace properly
        words = [line.strip() for line in f]
    return set(words)  # Using set for O(1) lookup

def segment_text(text, dictionary):
    """
    Segment text into valid words from dictionary.
    Returns all possible segmentations.
    """
    # Bug: No memoization - will be extremely slow for long strings
    # Bug: Missing proper handling of finding ALL solutions
    
    if not text:
        return [[]]  # Base case - empty string
    
    results = []
    
    # Bug: range is wrong - should be range(1, len(text) + 1)
    for i in range(len(text)):
        prefix = text[:i]  # Bug: off-by-one, missing last character
        
        # Bug: not checking if prefix is in dictionary before recursing
        suffix_segments = segment_text(text[i:], dictionary)
        
        # Bug: appending wrong structure
        if suffix_segments:
            for segment in suffix_segments:
                results.append([prefix] + segment)
    
    # Bug: returning incomplete results
    return results[0] if results else []  # Only returns first result!

def save_results(results, filepath):
    """Save segmentation results to JSON file"""
    output = {
        "results": results,
        "total_processed": len(results)
        # Bug: Missing total_time_seconds field
    }
    
    with open(filepath, 'w') as f:
        # Bug: might not format correctly
        json.dump(output, f)

def main():
    print("Loading dictionary...")
    dictionary = load_dictionary('/data/dictionary.txt')
    
    print("Loading corrupted samples...")
    with open('/data/corrupted_samples.txt', 'r') as f:
        samples = [line.strip() for line in f]
    
    print(f"Processing {len(samples)} samples...")
    start_time = time.time()
    
    results = []
    
    for idx, sample in enumerate(samples):
        print(f"Processing {idx + 1}/{len(samples)}: {sample}")
        
        # Bug: segment_text doesn't return all segmentations properly
        segmentations = segment_text(sample, dictionary)
        
        # Bug: wrong format for segmentations - should be strings not lists
        result = {
            "input": sample,
            "segmentations": segmentations,  # Bug: not converting to space-separated strings
            "count": len(segmentations) if segmentations else 0
        }
        results.append(result)
    
    end_time = time.time()
    
    # Bug: not including timing in output
    print(f"Processing complete in {end_time - start_time:.2f} seconds")
    
    print("Saving results...")
    save_results(results, '/workspace/recovery_results.json')
    print("Done!")

if __name__ == "__main__":
    main()