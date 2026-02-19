"""
Quality control module for genomic sequence analysis.

This module provides functions to perform quality control operations on
genomic sequences, including quality score validation, read filtering,
and QC metric generation.
"""

def check_quality_scores(sequence, quality_scores, min_score=20):
    """
    Check if quality scores meet the minimum threshold.
    
    Args:
        sequence (str): The genomic sequence string
        quality_scores (list): List of quality scores (Phred scores)
        min_score (int): Minimum acceptable quality score (default: 20)
    
    Returns:
        bool: True if all quality scores meet threshold, False otherwise
    """
    if not quality_scores:
        return False
    
    if len(sequence) != len(quality_scores):
        return False
    
    return all(score >= min_score for score in quality_scores)


def filter_reads(reads, quality_threshold=25, min_length=50):
    """
    Filter low-quality reads from a collection.
    
    Args:
        reads (list): List of tuples (sequence, quality_scores)
        quality_threshold (int): Minimum average quality score
        min_length (int): Minimum sequence length
    
    Returns:
        list: Filtered list of high-quality reads
    """
    filtered = []
    
    for sequence, quality_scores in reads:
        if len(sequence) < min_length:
            continue
        
        if not quality_scores or len(quality_scores) != len(sequence):
            continue
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality >= quality_threshold:
            filtered.append((sequence, quality_scores))
    
    return filtered


def qc_metrics(reads):
    """
    Generate quality control metrics for a set of reads.
    
    Args:
        reads (list): List of tuples (sequence, quality_scores)
    
    Returns:
        dict: Dictionary containing QC metrics including:
            - total_reads: Total number of reads
            - avg_length: Average sequence length
            - avg_quality: Average quality score across all reads
            - pass_rate: Percentage of reads passing basic QC
    """
    if not reads:
        return {
            'total_reads': 0,
            'avg_length': 0,
            'avg_quality': 0,
            'pass_rate': 0
        }
    
    total_reads = len(reads)
    total_length = sum(len(seq) for seq, _ in reads)
    avg_length = total_length / total_reads if total_reads > 0 else 0
    
    all_scores = []
    for _, quality_scores in reads:
        if quality_scores:
            all_scores.extend(quality_scores)
    
    avg_quality = sum(all_scores) / len(all_scores) if all_scores else 0
    
    passing_reads = sum(1 for seq, scores in reads 
                       if scores and sum(scores) / len(scores) >= 20)
    pass_rate = (passing_reads / total_reads * 100) if total_reads > 0 else 0
    
    return {
        'total_reads': total_reads,
        'avg_length': round(avg_length, 2),
        'avg_quality': round(avg_quality, 2),
        'pass_rate': round(pass_rate, 2)
    }