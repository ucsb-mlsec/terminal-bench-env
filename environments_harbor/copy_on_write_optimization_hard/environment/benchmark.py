#!/usr/bin/env python3

import time
import tracemalloc
import random
from config_system import ConfigVersionControl

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Start tracking memory
    tracemalloc.start()
    
    # Start timing
    start_time = time.time()
    
    # Create ConfigVersionControl instance
    config = ConfigVersionControl()
    
    # Initialize with 5000 configuration settings
    for i in range(5000):
        config.set(f'setting_{i}', f'value_{i}')
    
    # Commit initial version (version 0)
    config.commit()
    
    # Create 1000 additional versions
    for version in range(1, 1001):
        # Randomly modify 10 settings
        indices = random.sample(range(5000), 10)
        for idx in indices:
            config.set(f'setting_{idx}', f'value_{idx}_v{version}')
        
        # Commit the version
        config.commit()
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert bytes to MB
    peak_mb = peak / (1024 * 1024)
    
    # Print results to console
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Peak memory usage: {peak_mb:.2f} MB")
    print(f"Number of versions created: 1001")
    
    # Write results to file
    with open('/workspace/results.txt', 'w') as f:
        f.write(f"TIME_SECONDS={elapsed_time:.1f}\n")
        f.write(f"MEMORY_MB={peak_mb:.1f}\n")
        f.write(f"VERSIONS=1001\n")

if __name__ == '__main__':
    main()