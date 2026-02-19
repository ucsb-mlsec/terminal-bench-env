#!/bin/bash

set -e

WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "=== Matrix Transpose Optimization Verification ==="
echo ""

# Cleanup old files
rm -f transpose_baseline transpose_optimized output_baseline.bin output_transposed.bin

# Compile baseline
echo "[1/6] Compiling baseline version..."
if ! gcc -O2 -march=native -std=c11 transpose_baseline.c -o transpose_baseline 2>compile_baseline.err; then
    echo "ERROR: Baseline compilation failed"
    cat compile_baseline.err
    exit 1
fi
echo "✓ Baseline compiled successfully"

# Compile optimized
echo "[2/6] Compiling optimized version..."
if ! gcc -O2 -march=native -std=c11 transpose_optimized.c -o transpose_optimized 2>compile_optimized.err; then
    echo "ERROR: Optimized compilation failed"
    cat compile_optimized.err
    exit 1
fi
echo "✓ Optimized compiled successfully"
echo ""

# Run baseline
echo "[3/6] Running baseline version..."
BASELINE_START=$(date +%s.%N)
if ! ./transpose_baseline > baseline_output.log 2>&1; then
    echo "ERROR: Baseline execution failed"
    cat baseline_output.log
    exit 1
fi
BASELINE_END=$(date +%s.%N)
BASELINE_TIME=$(echo "$BASELINE_END - $BASELINE_START" | bc)
echo "✓ Baseline completed in ${BASELINE_TIME}s"

# Save baseline output if it exists
if [ -f output_transposed.bin ]; then
    mv output_transposed.bin output_baseline.bin
fi

# Run optimized
echo "[4/6] Running optimized version..."
OPTIMIZED_START=$(date +%s.%N)
if ! ./transpose_optimized > optimized_output.log 2>&1; then
    echo "ERROR: Optimized execution failed"
    cat optimized_output.log
    exit 1
fi
OPTIMIZED_END=$(date +%s.%N)
OPTIMIZED_TIME=$(echo "$OPTIMIZED_END - $OPTIMIZED_START" | bc)
echo "✓ Optimized completed in ${OPTIMIZED_TIME}s"
echo ""

# Check if output file exists
if [ ! -f output_transposed.bin ]; then
    echo "ERROR: output_transposed.bin not found"
    exit 1
fi

# Verify file size
EXPECTED_SIZE=$((8192 * 8192 * 8))
ACTUAL_SIZE=$(stat -f%z output_transposed.bin 2>/dev/null || stat -c%s output_transposed.bin 2>/dev/null)
if [ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]; then
    echo "ERROR: Output file size incorrect (expected $EXPECTED_SIZE, got $ACTUAL_SIZE)"
    exit 1
fi

# Verify correctness
echo "[5/6] Verifying mathematical correctness..."
cat > verify_correctness.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 8192

int main() {
    FILE *fin = fopen("test_matrix.bin", "rb");
    FILE *fout = fopen("output_transposed.bin", "rb");
    
    if (!fin || !fout) {
        fprintf(stderr, "Failed to open files\n");
        return 1;
    }
    
    // Spot check 100 random positions
    srand(42);
    int errors = 0;
    
    for (int check = 0; check < 100; check++) {
        int i = rand() % N;
        int j = rand() % N;
        
        double input_val, output_val;
        
        // Read input[j][i]
        fseek(fin, (j * N + i) * sizeof(double), SEEK_SET);
        fread(&input_val, sizeof(double), 1, fin);
        
        // Read output[i][j]
        fseek(fout, (i * N + j) * sizeof(double), SEEK_SET);
        fread(&output_val, sizeof(double), 1, fout);
        
        if (fabs(input_val - output_val) > 1e-10) {
            fprintf(stderr, "Mismatch at (%d,%d): input[%d][%d]=%f != output[%d][%d]=%f\n",
                    i, j, j, i, input_val, i, j, output_val);
            errors++;
        }
    }
    
    fclose(fin);
    fclose(fout);
    
    if (errors > 0) {
        fprintf(stderr, "Found %d errors in spot check\n", errors);
        return 1;
    }
    
    printf("All spot checks passed\n");
    return 0;
}
EOF

gcc -O2 verify_correctness.c -o verify_correctness -lm
if ! ./verify_correctness; then
    echo "ERROR: Transpose verification failed"
    exit 1
fi
echo "✓ Transpose is mathematically correct"
echo ""

# Calculate speedup
SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $OPTIMIZED_TIME" | bc)

# Format times to 1 decimal place
BASELINE_TIME_FORMATTED=$(printf "%.1f" $BASELINE_TIME)
OPTIMIZED_TIME_FORMATTED=$(printf "%.1f" $OPTIMIZED_TIME)

# Check performance target
echo "[6/6] Checking performance target..."
TARGET_TIME=3.5
MEETS_TARGET=$(echo "$OPTIMIZED_TIME < $TARGET_TIME" | bc)

if [ "$MEETS_TARGET" -eq 1 ]; then
    echo "✓ Performance target MET (${OPTIMIZED_TIME_FORMATTED}s < ${TARGET_TIME}s)"
else
    echo "✗ Performance target MISSED (${OPTIMIZED_TIME_FORMATTED}s >= ${TARGET_TIME}s)"
fi
echo ""

# Write results
echo "[RESULTS] Writing to result.txt..."
cat > result.txt << EOF
BASELINE_TIME=${BASELINE_TIME_FORMATTED}
OPTIMIZED_TIME=${OPTIMIZED_TIME_FORMATTED}
SPEEDUP=${SPEEDUP}
EOF

echo "=== SUMMARY ==="
echo "Baseline time:  ${BASELINE_TIME_FORMATTED}s"
echo "Optimized time: ${OPTIMIZED_TIME_FORMATTED}s"
echo "Speedup:        ${SPEEDUP}x"
echo ""

if [ "$MEETS_TARGET" -eq 1 ]; then
    echo "✓✓✓ ALL TESTS PASSED ✓✓✓"
    exit 0
else
    echo "✗✗✗ PERFORMANCE TARGET NOT MET ✗✗✗"
    exit 1
fi