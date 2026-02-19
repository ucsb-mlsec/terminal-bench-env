#!/usr/bin/env python3
"""
Variant Caller Module

This module handles variant calling from genomic sequence data.
It provides functions for identifying single nucleotide polymorphisms (SNPs),
insertions/deletions (indels), and variant annotation.
"""


def call_snps(reference_seq, sample_seq, min_quality=20):
    """
    Identify single nucleotide polymorphisms (SNPs) between reference and sample sequences.
    
    Args:
        reference_seq (str): Reference genome sequence
        sample_seq (str): Sample sequence to compare
        min_quality (int): Minimum quality score for SNP calling
        
    Returns:
        list: List of tuples containing (position, ref_base, alt_base)
    """
    snps = []
    min_len = min(len(reference_seq), len(sample_seq))
    
    for i in range(min_len):
        if reference_seq[i] != sample_seq[i]:
            snps.append((i, reference_seq[i], sample_seq[i]))
    
    return snps


def call_indels(reference_seq, sample_seq, min_length=1):
    """
    Detect insertions and deletions (indels) in sample sequence compared to reference.
    
    Args:
        reference_seq (str): Reference genome sequence
        sample_seq (str): Sample sequence to analyze
        min_length (int): Minimum indel length to report
        
    Returns:
        list: List of dictionaries containing indel information
    """
    indels = []
    
    # Simple length-based indel detection
    len_diff = len(sample_seq) - len(reference_seq)
    
    if abs(len_diff) >= min_length:
        indel_type = "insertion" if len_diff > 0 else "deletion"
        indels.append({
            'type': indel_type,
            'length': abs(len_diff),
            'position': min(len(reference_seq), len(sample_seq))
        })
    
    return indels


def annotate_variant(position, ref_base, alt_base, variant_type="SNP"):
    """
    Annotate a genetic variant with classification and functional information.
    
    Args:
        position (int): Genomic position of the variant
        ref_base (str): Reference base(s)
        alt_base (str): Alternate base(s)
        variant_type (str): Type of variant (SNP, insertion, deletion)
        
    Returns:
        dict: Annotated variant information
    """
    annotation = {
        'position': position,
        'reference': ref_base,
        'alternate': alt_base,
        'type': variant_type,
        'classification': classify_variant(ref_base, alt_base),
        'functional_impact': 'unknown'
    }
    
    return annotation


def classify_variant(ref_base, alt_base):
    """
    Classify a variant based on the reference and alternate bases.
    
    Args:
        ref_base (str): Reference base(s)
        alt_base (str): Alternate base(s)
        
    Returns:
        str: Variant classification
    """
    if len(ref_base) == len(alt_base) == 1:
        return "substitution"
    elif len(alt_base) > len(ref_base):
        return "insertion"
    elif len(alt_base) < len(ref_base):
        return "deletion"
    else:
        return "complex"