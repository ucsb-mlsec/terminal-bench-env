#!/usr/bin/env python3

import dask.array as da
import json
import numpy as np

# Import the arrays from arrays.py
from arrays import array1, array2, array3

# Step 1: Identify current chunk sizes
print("Analyzing chunk sizes...")
print(f"Array 1 shape: {array1.shape}, chunks: {array1.chunks}")
print(f"Array 2 shape: {array2.shape}, chunks: {array2.chunks}")
print(f"Array 3 shape: {array3.shape}, chunks: {array3.chunks}")

# Array dimensions
rows, cols = 10000, 5000
dtype_size = 8  # float64 = 8 bytes

# Step 2: Determine optimal chunk size
# Constraint: chunk size < 100MB
max_chunk_bytes = 100 * 1024 * 1024  # 100 MB in bytes
max_elements_per_chunk = max_chunk_bytes // dtype_size

# We need to find divisors of rows and cols that maximize chunk size while staying under 100MB
def find_divisors(n):
    """Find all divisors of n"""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)

row_divisors = find_divisors(rows)
col_divisors = find_divisors(cols)

# Find optimal chunk configuration
best_chunks = None
best_total_chunks = float('inf')

for row_chunk in row_divisors:
    for col_chunk in col_divisors:
        elements = row_chunk * col_chunk
        if elements <= max_elements_per_chunk:
            chunk_size_bytes = elements * dtype_size
            if chunk_size_bytes <= max_chunk_bytes:
                total_chunks = (rows // row_chunk) * (cols // col_chunk)
                # Prefer larger chunks (fewer total chunks) for better performance
                if total_chunks < best_total_chunks:
                    best_total_chunks = total_chunks
                    best_chunks = (row_chunk, col_chunk)

# If no valid configuration found, use a reasonable default
if best_chunks is None:
    # Calculate chunk size that uses ~80MB per chunk for safety margin
    target_elements = int((80 * 1024 * 1024) // dtype_size)
    # Try to make chunks roughly square
    chunk_rows = min(2000, rows)
    chunk_cols = min(target_elements // chunk_rows, cols)
    best_chunks = (chunk_rows, chunk_cols)
    best_total_chunks = (rows // chunk_rows) * (cols // chunk_cols)

optimal_rows, optimal_cols = best_chunks

# Step 3: Calculate metrics
elements_per_chunk = optimal_rows * optimal_cols
chunk_size_mb = (elements_per_chunk * dtype_size) / (1024 * 1024)
total_chunks = (rows // optimal_rows) * (cols // optimal_cols)

# Create solution dictionary
solution = {
    "optimal_chunks": [optimal_rows, optimal_cols],
    "total_chunks": total_chunks,
    "chunk_size_mb": round(chunk_size_mb, 2)
}

# Print solution
print("\n" + "="*60)
print("OPTIMAL CHUNK CONFIGURATION")
print("="*60)
print(f"Optimal chunks: {solution['optimal_chunks']}")
print(f"Total chunks: {solution['total_chunks']}")
print(f"Chunk size: {solution['chunk_size_mb']} MB")
print("="*60)

# Save solution to file
with open('/workspace/solution.json', 'w') as f:
    json.dump(solution, f, indent=2)

print(f"\nSolution saved to /workspace/solution.json")

# Verify the solution meets constraints
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
print(f"Chunk size under 100MB: {chunk_size_mb < 100} ({chunk_size_mb:.2f} MB)")
print(f"Arrays can be aligned: {rows % optimal_rows == 0 and cols % optimal_cols == 0}")
print(f"Dimensions evenly divisible: Rows {rows}/{optimal_rows}={rows//optimal_rows}, Cols {cols}/{optimal_cols}={cols//optimal_cols}")
print("="*60)