#!/usr/bin/env python3

import os
import sys

def burrows_wheeler_transform(text):
    """
    Apply Burrows-Wheeler Transform to the input text.
    Creates rotations, sorts them, and extracts characters.
    """
    # Add special end marker
    text = text + "_"
    
    # Generate all rotations of the text
    rotations = []
    for i in range(len(text)):
        # BUG: This creates incorrect rotations - should be text[i:] + text[:i]
        rotation = text[i+1:] + text[:i+1]
        rotations.append(rotation)
    
    # Sort rotations lexicographically
    sorted_rotations = sorted(rotations)
    
    # Extract the last character from each sorted rotation
    # BUG: Taking first character instead of last
    transformed = ''.join([rotation[0] for rotation in sorted_rotations])
    
    return transformed

def process_file(filepath):
    """
    Read a file and apply the transformation.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        
        # Apply transformation
        transformed = burrows_wheeler_transform(content)
        
        return transformed
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    """
    Main function to process all test files and save results.
    """
    input_dir = '/workspace/test_inputs'
    output_file = '/workspace/results.txt'
    
    test_files = ['test1.txt', 'test2.txt', 'test3.txt']
    
    results = []
    
    for filename in test_files:
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue
        
        transformed = process_file(filepath)
        
        if transformed is not None:
            results.append(f"{filename}={transformed}")
            print(f"Processed {filename}: {transformed}")
    
    # Write results to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()