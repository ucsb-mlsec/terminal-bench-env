#!/bin/bash

echo "Starting Stage 3: Running alignment..."

# Check if stage2 output exists
if [ ! -f "data/stage2_output.txt" ]; then
    echo "Error: data/stage2_output.txt not found. Stage 2 must complete first."
    exit 1
fi

# Simulate processing time
sleep 3

# Read from stage2 output and create stage3 output
{
    echo "=== Alignment Results ==="
    echo "Input from Stage 2:"
    cat data/stage2_output.txt
    echo ""
    echo "Aligned Sequences:"
    echo "SEQ001: ATCGATCGATCGATCG aligned to reference position 1000-1016"
    echo "SEQ002: GCTAGCTAGCTAGCTA aligned to reference position 2050-2066"
    echo "SEQ003: TTAATTAATTAATTAA aligned to reference position 3100-3116"
    echo "SEQ004: CGCGCGCGCGCGCGCG aligned to reference position 4200-4216"
    echo "SEQ005: ATATATATATATAT aligned to reference position 5300-5314"
    echo "SEQ006: GCGCGCGCGCGCGC aligned to reference position 6400-6414"
    echo "SEQ007: TACGTACGTACGTACG aligned to reference position 7500-7516"
    echo "Total aligned reads: 7"
    echo "Alignment quality score: 98.5%"
} > data/stage3_output.txt

echo "Stage 3 completed successfully"
exit 0