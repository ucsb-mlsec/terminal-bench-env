#!/usr/bin/env python3

import sys

def main():
    if len(sys.argv) != 3:
        print("Error: Expected 2 arguments")
        sys.exit(1)
    
    expression = sys.argv[1]
    precision = int(sys.argv[2])
    
    result = eval(expression)
    rounded_result = round(result, precision)
    
    print(f"{rounded_result:.{precision}f}")

if __name__ == "__main__":
    main()