#!/usr/bin/env python3

import numpy as np
import array_ops

def main():
    # Read input data from file
    try:
        with open('input_data.txt', 'r') as f:
            numbers = [float(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: input_data.txt not found")
        return
    except ValueError:
        print("Error: Invalid number format in input file")
        return
    
    # Convert to numpy array
    data = np.array(numbers, dtype=np.float64)
    
    # Call Fortran subroutine
    result = array_ops.compute_sum(data, len(data))
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()