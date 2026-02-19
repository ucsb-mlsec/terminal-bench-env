#!/usr/bin/env python3
"""
Genomic Sequence Processor Module

This module provides functions for processing and analyzing genomic sequences,
including DNA sequence manipulation, composition analysis, and validation.
"""


def reverse_complement(sequence):
    """
    Generate the reverse complement of a DNA sequence.
    
    Args:
        sequence (str): DNA sequence string containing A, T, G, C bases
        
    Returns:
        str: Reverse complement of the input sequence
    """
    complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    complement = ''.join(complement_map.get(base.upper(), 'N') for base in sequence)
    return complement[::-1]


def validate_sequence(sequence):
    """
    Validate that a sequence contains only valid DNA bases.
    
    Args:
        sequence (str): DNA sequence to validate
        
    Returns:
        bool: True if sequence is valid, False otherwise
    """
    valid_bases = set('ATGCN')
    return all(base.upper() in valid_bases for base in sequence)


def gc_content(sequence):
    """
    Calculate the GC content percentage of a DNA sequence.
    
    Args:
        sequence (str): DNA sequence string
        
    Returns:
        float: GC content as percentage (0-100)
    """
    if not sequence:
        return 0.0
    sequence_upper = sequence.upper()
    gc_count = sequence_upper.count('G') + sequence_upper.count('C')
    return (gc_count / len(sequence)) * 100


def count_bases(sequence):
    """
    Count the occurrence of each base in a DNA sequence.
    
    Args:
        sequence (str): DNA sequence string
        
    Returns:
        dict: Dictionary with base counts {'A': n, 'T': n, 'G': n, 'C': n}
    """
    sequence_upper = sequence.upper()
    return {
        'A': sequence_upper.count('A'),
        'T': sequence_upper.count('T'),
        'G': sequence_upper.count('G'),
        'C': sequence_upper.count('C')
    }


def transcribe(sequence):
    """
    Transcribe DNA sequence to RNA (replace T with U).
    
    Args:
        sequence (str): DNA sequence string
        
    Returns:
        str: RNA sequence with U instead of T
    """
    return sequence.upper().replace('T', 'U')


def translate_codon(codon):
    """
    Translate a single DNA codon to amino acid (simplified).
    
    Args:
        codon (str): Three-letter DNA codon
        
    Returns:
        str: Single letter amino acid code or 'X' for unknown
    """
    codon_table = {
        'ATG': 'M', 'TAA': '*', 'TAG': '*', 'TGA': '*',
        'GCT': 'A', 'TGT': 'C', 'GAT': 'D', 'GAA': 'E'
    }
    return codon_table.get(codon.upper(), 'X')