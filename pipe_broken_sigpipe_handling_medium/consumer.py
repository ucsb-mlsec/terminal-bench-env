#!/usr/bin/env python3

import sys

def main():
    line_count = 0
    for line in sys.stdin:
        line_count += 1
        line = line.strip()
        
        if 'CRITICAL' in line:
            print(f"[CONSUMER] Found CRITICAL error at line {line_count}: {line}", file=sys.stderr)
            sys.exit(1)
        
        if any(level in line for level in ['INFO', 'WARNING', 'ERROR', 'DEBUG']):
            print(f"[CONSUMER] Processed: {line}")

if __name__ == "__main__":
    main()