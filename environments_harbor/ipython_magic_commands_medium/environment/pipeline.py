#!/usr/bin/env python3

def process_small_batch():
    '''Processes a small batch of data - minimal memory usage'''
    data = [i for i in range(50000)]
    result = sum(data)
    return result

def process_medium_dataset():
    '''Processes a medium-sized dataset with dictionary operations'''
    data = {i: str(i) * 10 for i in range(200000)}
    result = len(data)
    return result

def process_large_dataset():
    '''Processes a large dataset with string operations - HIGHEST MEMORY'''
    data = [str(i) * 200 for i in range(1000000)]
    processed = ''.join(data[:100])
    result = len(data)
    return result

def process_standard_batch():
    '''Processes a standard batch with nested structures'''
    data = [[i, i*2, i*3] for i in range(150000)]
    flattened = [item for sublist in data for item in sublist]
    result = sum(flattened)
    return result