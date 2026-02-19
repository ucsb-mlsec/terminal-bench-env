#!/usr/bin/env python3
"""
quality_checker.py

A module for checking quality metrics of DNA sequences.
"""


def calculate_gc_content(sequence):
    """
    Calculate the GC content of a DNA sequence.
    
    Args:
        sequence (str): A DNA sequence string containing A, T, G, C bases
        
    Returns:
        float: GC content as a fraction between 0.0 and 1.0
        
    Example:
        >>> calculate_gc_content("ATGC")
        0.5
    """
    if not sequence:
        return 0.0
    
    # Convert to uppercase for consistent handling
    sequence = sequence.upper()
    
    # Count G and C bases
    gc_count = sequence.count('G') + sequence.count('C')
    total_count = len(sequence)
    
    if total_count == 0:
        return 0.0
    
    return gc_count / total_count


def check_quality(sequences):
    """
    Check quality metrics for a list of DNA sequences.
    
    Evaluates sequences based on:
    - Minimum length requirement (50 bp)
    - GC content (sequences with extreme GC content may be flagged)
    
    Args:
        sequences (list): A list of sequence dictionaries, where each dictionary
                         should contain at least a 'sequence' key with the DNA string
                         
    Returns:
        dict: A dictionary containing quality metrics with keys:
              - 'total': total number of sequences
              - 'passed': number of sequences passing quality checks
              - 'failed': number of sequences failing quality checks
              
    Example:
        >>> seqs = [{'id': 'seq1', 'sequence': 'ATGC' * 20}]
        >>> check_quality(seqs)
        {'total': 1, 'passed': 1, 'failed': 0}
    """
    total = len(sequences)
    passed = 0
    failed = 0
    
    MIN_LENGTH = 50
    MIN_GC = 0.2
    MAX_GC = 0.8
    
    for seq_dict in sequences:
        # Extract sequence string from dictionary
        if 'sequence' not in seq_dict:
            failed += 1
            continue
            
        sequence = seq_dict['sequence']
        
        # Check minimum length
        if len(sequence) < MIN_LENGTH:
            failed += 1
            continue
        
        # Check GC content
        gc_content = calculate_gc_content(sequence)
        if gc_content < MIN_GC or gc_content > MAX_GC:
            failed += 1
            continue
        
        # Sequence passed all checks
        passed += 1
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed
    }