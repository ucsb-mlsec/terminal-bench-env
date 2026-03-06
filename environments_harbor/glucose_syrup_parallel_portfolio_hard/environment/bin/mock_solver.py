#!/usr/bin/env python3

import argparse
import random
import time
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Mock SAT Solver')
    parser.add_argument('-i', '--input', required=True, help='Path to CNF problem file')
    parser.add_argument('-s', '--seed', required=True, type=int, help='Random seed')
    parser.add_argument('-t', '--timeout', required=True, type=float, help='Maximum execution time in seconds')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize random number generator with seed
    random.seed(args.seed)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Read the input CNF file (basic validation)
    try:
        with open(args.input, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Simulate solving time: random duration between 0.5 and timeout
    solve_time = random.uniform(0.5, args.timeout)
    
    # Sleep to simulate solving
    time.sleep(solve_time)
    
    # Randomly decide result with specific probabilities
    # 40% SAT, 40% UNSAT, 20% TIMEOUT
    rand_val = random.random()
    
    if rand_val < 0.4:
        result = "SAT"
        exit_code = 0
    elif rand_val < 0.8:
        result = "UNSAT"
        exit_code = 0
    else:
        result = "TIMEOUT"
        exit_code = 1
    
    # Write result to output file
    try:
        with open(args.output, 'w') as f:
            f.write(result)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()