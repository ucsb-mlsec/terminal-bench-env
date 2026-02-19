#!/usr/bin/env python3

import argparse
import json
import random
from datetime import datetime, timedelta

def generate_stream(count):
    """Generate a stream of log records."""
    base_timestamp = datetime(2024, 1, 1, 0, 0, 0)
    
    for i in range(1, count + 1):
        record = {
            "id": i,
            "timestamp": (base_timestamp + timedelta(seconds=i-1)).isoformat(),
            "value": random.randint(1, 100)
        }
        print(json.dumps(record))

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic log data stream')
    parser.add_argument('--count', type=int, required=True, 
                        help='Number of log records to generate')
    
    args = parser.parse_args()
    
    if args.count < 1:
        parser.error("Count must be at least 1")
    
    generate_stream(args.count)

if __name__ == "__main__":
    main()