#!/usr/bin/env python3

import sys
import os

# Import custom bioinformatics modules
from sequence_parser import parse_fasta
from quality_checker import check_quality
from statistics_calculator import calculate_stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_analysis.py <input_fasta_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"Starting bioinformatics pipeline analysis...")
    print(f"Input file: {input_file}")
    
    try:
        # Parse FASTA file
        print("Step 1: Parsing FASTA sequences...")
        sequences = parse_fasta(input_file)
        print(f"  Parsed {len(sequences)} sequences")
        
        # Check quality of sequences
        print("Step 2: Checking sequence quality...")
        quality_results = check_quality(sequences)
        print(f"  Quality check complete: {quality_results['passed']} sequences passed quality control")
        
        # Calculate statistics
        print("Step 3: Calculating sequence statistics...")
        stats = calculate_stats(sequences)
        print(f"  Average sequence length: {stats['avg_length']:.2f} bp")
        print(f"  GC content: {stats['gc_content']:.2f}%")
        print(f"  Total nucleotides: {stats['total_nucleotides']}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"Analysis complete. Processed {len(sequences)} sequences.")
        print(f"Quality-filtered sequences: {quality_results['passed']}/{len(sequences)}")
        print(f"Mean length: {stats['avg_length']:.2f} bp")
        print(f"GC content: {stats['gc_content']:.2f}%")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())