#!/usr/bin/env python3

import dask.array as da
import numpy as np

# Array 1: Misaligned chunks (2000, 2500)
# This represents data that was chunked for balanced row-column operations
array1 = da.random.random((10000, 5000), chunks=(2000, 2500))

# Array 2: Misaligned chunks (1000, 5000)
# This represents data that was chunked for row-wise processing
array2 = da.random.random((10000, 5000), chunks=(1000, 5000))

# Array 3: Misaligned chunks (5000, 1000)
# This represents data that was chunked for column-wise processing
array3 = da.random.random((10000, 5000), chunks=(5000, 1000))

# These arrays have different chunk sizes which will cause rechunking overhead
# during element-wise operations, leading to performance degradation
```

```python
#!/usr/bin/env python3

import dask.array as da
import json
from arrays import array1, array2, array3

def analyze_chunks():
    """Analyze current chunk configurations and determine optimal chunks."""
    
    # Get current chunk sizes
    print("Current chunk configurations:")
    print(f"Array 1 chunks: {array1.chunks}")
    print(f"Array 2 chunks: {array2.chunks}")
    print(f"Array 3 chunks: {array3.chunks}")
    
    # Array shape
    shape = (10000, 5000)
    
    # float64 = 8 bytes
    bytes_per_element = 8
    max_chunk_bytes = 100 * 1024 * 1024  # 100 MB
    
    # Calculate maximum elements per chunk
    max_elements_per_chunk = max_chunk_bytes / bytes_per_element
    
    # Find optimal chunk size that:
    # 1. Divides evenly into both dimensions (for alignment)
    # 2. Keeps chunks under 100MB
    # 3. Maximizes chunk size for fewer total chunks
    
    best_chunks = None
    best_total_chunks = float('inf')
    
    # Find divisors of 10000 and 5000
    def get_divisors(n):
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors, reverse=True)
    
    row_divisors = get_divisors(10000)
    col_divisors = get_divisors(5000)
    
    # Test combinations
    for row_chunk in row_divisors:
        for col_chunk in col_divisors:
            elements = row_chunk * col_chunk
            chunk_bytes = elements * bytes_per_element
            
            if chunk_bytes <= max_chunk_bytes:
                total_chunks = (shape[0] // row_chunk) * (shape[1] // col_chunk)
                
                # Prefer larger chunks (fewer total chunks) for better performance
                if total_chunks < best_total_chunks:
                    best_total_chunks = total_chunks
                    best_chunks = (row_chunk, col_chunk)
    
    # Calculate final statistics
    optimal_chunks = list(best_chunks)
    total_chunks = best_total_chunks
    chunk_size_mb = (best_chunks[0] * best_chunks[1] * bytes_per_element) / (1024 * 1024)
    
    print(f"\nOptimal configuration:")
    print(f"Chunks: {optimal_chunks}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunk size: {chunk_size_mb:.2f} MB")
    
    # Save solution
    solution = {
        "optimal_chunks": optimal_chunks,
        "total_chunks": total_chunks,
        "chunk_size_mb": round(chunk_size_mb, 2)
    }
    
    with open('/workspace/solution.json', 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"\nSolution saved to /workspace/solution.json")
    return solution

if __name__ == "__main__":
    analyze_chunks()