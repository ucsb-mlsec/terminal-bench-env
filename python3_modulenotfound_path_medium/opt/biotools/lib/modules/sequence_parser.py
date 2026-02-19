#!/usr/bin/env python3
"""
sequence_parser.py - Module for parsing and validating DNA sequences in FASTA format
"""

def parse_fasta(filepath):
    """
    Parse a FASTA format file and extract sequences.
    
    Args:
        filepath (str): Path to the FASTA file to parse
        
    Returns:
        list: List of dictionaries with 'id' and 'sequence' keys
              Example: [{'id': 'seq1', 'sequence': 'ATCG...'}, ...]
    """
    sequences = []
    current_id = None
    current_sequence = []
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences.append({
                        'id': current_id,
                        'sequence': ''.join(current_sequence)
                    })
                # Start new sequence
                current_id = line[1:].split()[0]  # Remove '>' and take first word
                current_sequence = []
            else:
                # Add to current sequence
                current_sequence.append(line)
        
        # Save last sequence
        if current_id is not None:
            sequences.append({
                'id': current_id,
                'sequence': ''.join(current_sequence)
            })
    
    return sequences


def validate_sequence(sequence):
    """
    Validate that a DNA sequence contains only valid characters.
    
    Args:
        sequence (str): DNA sequence to validate
        
    Returns:
        bool: True if sequence contains only valid DNA characters (A, T, C, G, N),
              False otherwise
    """
    if not sequence:
        return False
    
    valid_chars = set('ATCGNatcgn')
    return all(char in valid_chars for char in sequence)