#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="processed_images"
INPUT_DIR="test_images"
NUM_RUNS=10
CORRUPTION_COUNT=0
SUCCESS_COUNT=0

echo "Image Processor Test Suite"
echo "=========================="
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Compile the program
echo "Compiling image processor..."
make clean && make
if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi
echo -e "${GREEN}Compilation successful${NC}"
echo ""

# Run tests multiple times
for i in $(seq 1 $NUM_RUNS); do
    echo "Run $i/$NUM_RUNS..."
    
    # Clean output directory
    rm -rf "$OUTPUT_DIR"/*
    
    # Run the image processor
    ./image_processor "$INPUT_DIR" "$OUTPUT_DIR" 2>/dev/null
    
    # Check for corruption
    CORRUPTED=0
    
    # Verify output files exist and have valid format
    for input_file in "$INPUT_DIR"/*.pgm; do
        if [ -f "$input_file" ]; then
            filename=$(basename "$input_file")
            output_file="$OUTPUT_DIR/$filename"
            
            if [ ! -f "$output_file" ]; then
                echo -e "  ${RED}✗ Missing output: $filename${NC}"
                CORRUPTED=1
                continue
            fi
            
            # Check if file is readable and has content
            if [ ! -s "$output_file" ]; then
                echo -e "  ${RED}✗ Empty output: $filename${NC}"
                CORRUPTED=1
                continue
            fi
            
            # Check PGM header
            header=$(head -n 1 "$output_file")
            if [ "$header" != "P2" ]; then
                echo -e "  ${RED}✗ Invalid header in: $filename${NC}"
                CORRUPTED=1
                continue
            fi
            
            # Basic size check - output should be similar size to input
            input_size=$(wc -c < "$input_file")
            output_size=$(wc -c < "$output_file")
            size_diff=$((output_size - input_size))
            size_diff=${size_diff#-}  # absolute value
            
            if [ $size_diff -gt $((input_size / 2)) ]; then
                echo -e "  ${YELLOW}⚠ Size mismatch in: $filename${NC}"
                CORRUPTED=1
            fi
        fi
    done
    
    if [ $CORRUPTED -eq 1 ]; then
        echo -e "${RED}Run $i: CORRUPTED${NC}"
        CORRUPTION_COUNT=$((CORRUPTION_COUNT + 1))
    else
        echo -e "${GREEN}Run $i: SUCCESS${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
    echo ""
done

# Print summary statistics
echo "=========================="
echo "Test Summary"
echo "=========================="
echo "Total runs: $NUM_RUNS"
echo -e "${GREEN}Successful runs: $SUCCESS_COUNT${NC}"
echo -e "${RED}Corrupted runs: $CORRUPTION_COUNT${NC}"
echo "Corruption rate: $((CORRUPTION_COUNT * 100 / NUM_RUNS))%"
echo ""

if [ $CORRUPTION_COUNT -gt 0 ]; then
    echo -e "${RED}Data corruption detected! Thread synchronization issues present.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed successfully!${NC}"
    exit 0
fi