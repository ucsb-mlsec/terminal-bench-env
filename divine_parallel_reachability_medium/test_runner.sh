#!/bin/bash

echo "Testing state space exploration..."
echo "=================================="

OUTPUT_FILE="/workspace/solution/output.txt"
SCRIPT="/workspace/solution/explore.py"

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "FAIL: Script not found at $SCRIPT"
    exit 1
fi

# Run the exploration multiple times and store outputs
echo "Running exploration 5 times to verify determinism..."
for i in {1..5}; do
    echo "Run $i..."
    python3 "$SCRIPT"
    
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "FAIL: Output file not created at $OUTPUT_FILE"
        exit 1
    fi
    
    output=$(cat "$OUTPUT_FILE")
    
    # Verify format: single line with three space-separated integers
    if ! echo "$output" | grep -E '^[0-9]+ [0-9]+ [0-9]+$' > /dev/null; then
        echo "FAIL: Invalid output format. Expected: '<int> <int> <int>', Got: '$output'"
        exit 1
    fi
    
    # Store output for comparison
    eval "output$i='$output'"
done

# Compare all outputs
echo ""
echo "Comparing outputs for consistency..."
if [ "$output1" = "$output2" ] && [ "$output2" = "$output3" ] && [ "$output3" = "$output4" ] && [ "$output4" = "$output5" ]; then
    echo "PASS: All runs produced identical results"
    echo "Result: $output1"
    exit 0
else
    echo "FAIL: Outputs differ between runs (non-deterministic)"
    echo "Run 1: $output1"
    echo "Run 2: $output2"
    echo "Run 3: $output3"
    echo "Run 4: $output4"
    echo "Run 5: $output5"
    exit 1
fi