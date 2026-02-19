#!/usr/bin/env python3
"""
Alignment Tools Module

This module provides sequence alignment operations for genomic analysis,
including pairwise alignment and alignment scoring functions.
"""


def pairwise_align(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2):
    """
    Perform pairwise sequence alignment between two sequences.
    
    Args:
        seq1 (str): First sequence to align
        seq2 (str): Second sequence to align
        match_score (int): Score for matching bases (default: 2)
        mismatch_penalty (int): Penalty for mismatches (default: -1)
        gap_penalty (int): Penalty for gaps (default: -2)
    
    Returns:
        tuple: Aligned sequences as (aligned_seq1, aligned_seq2)
    """
    # Simple implementation - direct comparison with gap insertion
    aligned1, aligned2 = [], []
    i, j = 0, 0
    
    while i < len(seq1) or j < len(seq2):
        if i < len(seq1) and j < len(seq2):
            aligned1.append(seq1[i])
            aligned2.append(seq2[j])
            i += 1
            j += 1
        elif i < len(seq1):
            aligned1.append(seq1[i])
            aligned2.append('-')
            i += 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j])
            j += 1
    
    return (''.join(aligned1), ''.join(aligned2))


def alignment_score(aligned_seq1, aligned_seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2):
    """
    Calculate the alignment score for two aligned sequences.
    
    Args:
        aligned_seq1 (str): First aligned sequence
        aligned_seq2 (str): Second aligned sequence
        match_score (int): Score for matching bases (default: 2)
        mismatch_penalty (int): Penalty for mismatches (default: -1)
        gap_penalty (int): Penalty for gaps (default: -2)
    
    Returns:
        int: Total alignment score
    """
    score = 0
    for base1, base2 in zip(aligned_seq1, aligned_seq2):
        if base1 == '-' or base2 == '-':
            score += gap_penalty
        elif base1 == base2:
            score += match_score
        else:
            score += mismatch_penalty
    
    return score


def calculate_score(seq1, seq2):
    """
    Calculate alignment score directly from two sequences.
    
    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence
    
    Returns:
        int: Alignment score
    """
    aligned1, aligned2 = pairwise_align(seq1, seq2)
    return alignment_score(aligned1, aligned2)


def align_sequences(sequences):
    """
    Align multiple sequences and return alignment matrix.
    
    Args:
        sequences (list): List of sequences to align
    
    Returns:
        list: List of aligned sequences
    """
    if not sequences:
        return []
    
    # Simple multiple alignment - align all to first sequence
    reference = sequences[0]
    aligned = [reference]
    
    for seq in sequences[1:]:
        aligned_seq, _ = pairwise_align(seq, reference)
        aligned.append(aligned_seq)
    
    return aligned