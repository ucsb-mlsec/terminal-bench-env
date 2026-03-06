#!/usr/bin/env python3
"""
CUDA Thrust Parallel Reduction Simulator
Simulates parallel reduction operations for CPU testing
"""

import math
from typing import List, Union, Callable


def _chunk_data(data: List[Union[int, float]], chunk_size: int) -> List[List[Union[int, float]]]:
    """Divide data into chunks to simulate parallel thread blocks"""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def _get_operation(op_name: str) -> Callable:
    """Return the appropriate reduction operation function"""
    operations = {
        'sum': lambda a, b: a + b,
        'max': lambda a, b: max(a, b),
        'min': lambda a, b: min(a, b),
        'product': lambda a, b: a * b
    }
    return operations.get(op_name, operations['sum'])


def _reduce_chunk(chunk: List[Union[int, float]], operation: Callable) -> Union[int, float]:
    """Reduce a single chunk using the specified operation"""
    if not chunk:
        return 0
    result = chunk[0]
    for i in range(1, len(chunk)):
        result = operation(result, chunk[i])
    return result


def _merge_partials(partials: List[Union[int, float]], operation: Callable) -> Union[int, float]:
    """
    Merge partial results in stages, simulating parallel tree reduction
    This simulates how CUDA Thrust combines results from different thread blocks
    """
    if not partials:
        return 0
    if len(partials) == 1:
        return partials[0]
    
    # Simulate tree-based parallel reduction in stages
    current_level = partials[:]
    
    while len(current_level) > 1:
        next_level = []
        
        # Combine pairs of elements in parallel
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                # BUG: Using addition instead of the specified operation
                combined = current_level[i] + current_level[i + 1]
                next_level.append(combined)
            else:
                # Odd element out, carry forward
                next_level.append(current_level[i])
        
        current_level = next_level
    
    return current_level[0]


def parallel_reduce(data: List[Union[int, float]], operation: str = 'sum') -> Union[int, float]:
    """
    Simulate CUDA Thrust parallel reduction on CPU
    
    Args:
        data: Input data to reduce
        operation: Reduction operation ('sum', 'max', 'min', 'product')
    
    Returns:
        Reduced result
    """
    if not data:
        return 0
    
    if len(data) == 1:
        return data[0]
    
    # Get the operation function
    op_func = _get_operation(operation)
    
    # Chunk size simulates CUDA thread block size
    CHUNK_SIZE = 16
    
    # Stage 1: Divide into chunks (simulating thread blocks)
    chunks = _chunk_data(data, CHUNK_SIZE)
    
    # Stage 2: Reduce each chunk in parallel (simulating per-block reduction)
    partial_results = []
    for chunk in chunks:
        partial = _reduce_chunk(chunk, op_func)
        partial_results.append(partial)
    
    # Stage 3: Merge partial results (simulating inter-block reduction)
    final_result = _merge_partials(partial_results, op_func)
    
    return final_result


if __name__ == "__main__":
    # Simple test
    test_data = list(range(1, 101))
    result = parallel_reduce(test_data, 'sum')
    print(f"Sum of 1-100: {result}")