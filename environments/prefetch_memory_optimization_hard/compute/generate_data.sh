#!/bin/bash

# Test data generator for matrix operations
# Creates test matrices for performance benchmarking

# Configuration variables
DATA_DIR="./test_data"
DEFAULT_SIZE=1024
MATRIX_SIZE=${1:-$DEFAULT_SIZE}

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating test data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# File paths
MATRIX_A="$DATA_DIR/test_matrix_A.dat"
MATRIX_B="$DATA_DIR/test_matrix_B.dat"
VECTOR_C="$DATA_DIR/test_vector_C.dat"

echo "Generating test data for matrix operations"
echo "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "Element type: double-precision floating-point"

# Generate Matrix A (binary format)
echo "Generating Matrix A..."
TOTAL_ELEMENTS=$((MATRIX_SIZE * MATRIX_SIZE))
dd if=/dev/urandom of="$MATRIX_A" bs=8 count=$TOTAL_ELEMENTS status=none 2>/dev/null

# Generate Matrix B (binary format)
echo "Generating Matrix B..."
dd if=/dev/urandom of="$MATRIX_B" bs=8 count=$TOTAL_ELEMENTS status=none 2>/dev/null

# Generate Vector C (binary format)
echo "Generating Vector C..."
dd if=/dev/urandom of="$VECTOR_C" bs=8 count=$MATRIX_SIZE status=none 2>/dev/null

# Set appropriate permissions
chmod 644 "$MATRIX_A" "$MATRIX_B" "$VECTOR_C"

# Display file sizes
echo ""
echo "Test data generated successfully:"
echo "  Matrix A: $(ls -lh $MATRIX_A | awk '{print $5}')"
echo "  Matrix B: $(ls -lh $MATRIX_B | awk '{print $5}')"
echo "  Vector C: $(ls -lh $VECTOR_C | awk '{print $5}')"
echo ""
echo "Total elements per matrix: $TOTAL_ELEMENTS"
echo "Total memory per matrix: $((TOTAL_ELEMENTS * 8)) bytes"
echo ""
echo "Data directory: $DATA_DIR"
echo "Ready for testing!"

exit 0