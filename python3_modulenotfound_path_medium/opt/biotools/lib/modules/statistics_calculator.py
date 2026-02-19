#!/usr/bin/env python3
"""
Statistics Calculator Module

This module provides functions for calculating statistics on DNA sequences.
"""


def calculate_stats(sequences):
    """
    Calculate statistics for a list of sequences.
    
    Args:
        sequences: A list of dictionaries, each containing sequence information.
                  Each dictionary should have at least an 'id' and 'sequence' key.
    
    Returns:
        A dictionary containing:
            - count: Total number of sequences
            - avg_length: Average length of sequences
            - min_length: Minimum sequence length
            - max_length: Maximum sequence length
    """
    if not sequences:
        return {
            'count': 0,
            'avg_length': 0.0,
            'min_length': 0,
            'max_length': 0
        }
    
    lengths = get_sequence_lengths(sequences)
    
    return {
        'count': len(sequences),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }


def get_sequence_lengths(sequences):
    """
    Extract the lengths of all sequences.
    
    Args:
        sequences: A list of dictionaries, each containing sequence information.
                  Each dictionary should have at least a 'sequence' key.
    
    Returns:
        A list of integers representing the length of each sequence.
    """
    return [len(seq['sequence']) for seq in sequences]