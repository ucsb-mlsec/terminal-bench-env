#!/usr/bin/env python3

import numpy as np
import sys
import os

def main():
    """Calculator application that computes mean and standard deviation."""
    
    input_file = '/workspace/test_input.txt'
    output_file = '/workspace/expected_output.txt'
    
    try:
        # Read input numbers from file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            raise ValueError(f"Expected at least 2 numbers in {input_file}")
        
        # Parse numbers
        numbers = []
        for i, line in enumerate(lines[:2]):
            try:
                num = float(line.strip())
                numbers.append(num)
            except ValueError as e:
                raise ValueError(f"Invalid number on line {i+1}: {line.strip()}")
        
        # Convert to numpy array for calculations
        data = np.array(numbers)
        
        # Compute mean and standard deviation using numpy
        mean_value = np.mean(data)
        std_value = np.std(data)
        
        # Write results to output file
        with open(output_file, 'w') as f:
            f.write(f"Mean: {mean_value}\n")
            f.write(f"Standard Deviation: {std_value}\n")
        
        print(f"Calculation complete. Results written to {output_file}")
        print(f"Mean: {mean_value}")
        print(f"Standard Deviation: {std_value}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())