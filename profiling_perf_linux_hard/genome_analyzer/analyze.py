#!/usr/bin/env python3

import os
import sys
from collections import Counter

def read_sequences(filename):
    """Read DNA sequences from file."""
    sequences = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    sequences.append(line.upper())
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    return sequences

def compute_gc_content(sequence):
    """Calculate GC content percentage efficiently."""
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

def find_repeating_patterns(sequences):
    """
    Find repeating k-mer patterns in sequences.
    THIS IS THE BOTTLENECK - extremely inefficient implementation.
    Uses nested loops with repeated string operations.
    """
    all_patterns = {}
    k = 6  # k-mer size
    
    # Inefficient: checking every possible pattern against every position
    for seq in sequences:
        seq_len = len(seq)
        # First pass: collect all k-mers
        for i in range(seq_len - k + 1):
            pattern = seq[i:i+k]
            
            # BOTTLENECK: Inefficient nested loop checking
            # Count occurrences by scanning entire sequence multiple times
            count = 0
            for j in range(seq_len - k + 1):
                # Character by character comparison (very slow)
                match = True
                for pos in range(k):
                    if i + pos < seq_len and j + pos < seq_len:
                        if seq[i + pos] != seq[j + pos]:
                            match = False
                            break
                    else:
                        match = False
                        break
                if match:
                    count += 1
            
            # More inefficient operations
            if count > 1:
                # Rebuild pattern string character by character (wasteful)
                rebuilt_pattern = ""
                for char_idx in range(k):
                    if i + char_idx < seq_len:
                        rebuilt_pattern += seq[i + char_idx]
                
                # Inefficient pattern validation
                is_valid = True
                for c in rebuilt_pattern:
                    if c not in ['A', 'T', 'G', 'C']:
                        is_valid = False
                        break
                
                if is_valid:
                    if rebuilt_pattern in all_patterns:
                        all_patterns[rebuilt_pattern] += count
                    else:
                        all_patterns[rebuilt_pattern] = count
    
    return all_patterns

def calculate_statistics(sequences):
    """Calculate basic statistics about sequences."""
    if not sequences:
        return {}
    
    lengths = [len(seq) for seq in sequences]
    total_bases = sum(lengths)
    
    base_counts = Counter()
    for seq in sequences:
        base_counts.update(seq)
    
    stats = {
        'total_sequences': len(sequences),
        'total_bases': total_bases,
        'avg_length': total_bases / len(sequences),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'base_distribution': dict(base_counts)
    }
    
    return stats

def find_motifs(sequences, motif='ATG'):
    """Find start codon positions in sequences."""
    motif_positions = {}
    
    for idx, seq in enumerate(sequences):
        positions = []
        start = 0
        while True:
            pos = seq.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if positions:
            motif_positions[f'seq_{idx}'] = positions
    
    return motif_positions

def analyze_complexity(sequences):
    """Analyze sequence complexity using simple entropy measure."""
    complexities = []
    
    for seq in sequences:
        if not seq:
            complexities.append(0.0)
            continue
        
        # Calculate base frequencies
        base_counts = Counter(seq)
        total = len(seq)
        
        # Simple diversity score
        unique_bases = len(base_counts)
        evenness = unique_bases / 4.0  # DNA has 4 bases
        
        complexities.append(evenness)
    
    return complexities

def validate_sequences(sequences):
    """Validate that sequences contain only valid DNA bases."""
    valid_bases = set('ATGC')
    valid_sequences = []
    
    for seq in sequences:
        if set(seq).issubset(valid_bases):
            valid_sequences.append(seq)
    
    return valid_sequences

def main():
    """Main execution function."""
    filename = '/workspace/genome_analyzer/sequences.txt'
    
    print("Starting genome sequence analysis...")
    print(f"Reading sequences from {filename}")
    
    # Read sequences
    sequences = read_sequences(filename)
    print(f"Loaded {len(sequences)} sequences")
    
    # Validate sequences
    sequences = validate_sequences(sequences)
    print(f"Validated {len(sequences)} sequences")
    
    # Calculate basic statistics
    stats = calculate_statistics(sequences)
    print("\nSequence Statistics:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Total bases: {stats['total_bases']}")
    print(f"  Average length: {stats['avg_length']:.2f}")
    print(f"  Length range: {stats['min_length']} - {stats['max_length']}")
    
    # Compute GC content for each sequence
    gc_contents = [compute_gc_content(seq) for seq in sequences]
    avg_gc = sum(gc_contents) / len(gc_contents) if gc_contents else 0
    print(f"\nAverage GC content: {avg_gc:.2f}%")
    
    # Find start codon positions
    motif_positions = find_motifs(sequences, 'ATG')
    print(f"\nFound ATG motifs in {len(motif_positions)} sequences")
    
    # Analyze sequence complexity
    complexities = analyze_complexity(sequences)
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    print(f"Average sequence complexity: {avg_complexity:.3f}")
    
    # Find repeating patterns (BOTTLENECK FUNCTION)
    print("\nAnalyzing repeating patterns...")
    patterns = find_repeating_patterns(sequences)
    print(f"Found {len(patterns)} repeating patterns")
    
    # Show top patterns
    if patterns:
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop repeating 6-mers:")
        for pattern, count in top_patterns:
            print(f"  {pattern}: {count} occurrences")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()