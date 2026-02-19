#!/bin/bash

# parse_test.sh - Test Protocol Buffers text format parsing

set -e

# Check if protoc is available
if ! command -v protoc &> /dev/null; then
    echo "ERROR: protoc (Protocol Buffers compiler) is not installed or not in PATH"
    exit 1
fi

echo "Testing Protocol Buffers schema against export files..."
echo "=========================================================="

# Check if schema file exists
if [ ! -f "user_profile.proto" ]; then
    echo "ERROR: user_profile.proto not found"
    exit 1
fi

# Check if exports directory exists
if [ ! -d "exports" ]; then
    echo "ERROR: exports/ directory not found"
    exit 1
fi

# Find all .txtpb files in exports directory
EXPORT_FILES=(exports/*.txtpb)

if [ ${#EXPORT_FILES[@]} -eq 0 ] || [ ! -f "${EXPORT_FILES[0]}" ]; then
    echo "ERROR: No .txtpb files found in exports/ directory"
    exit 1
fi

# Counter for successes and failures
SUCCESS_COUNT=0
FAILURE_COUNT=0
TOTAL_FILES=${#EXPORT_FILES[@]}

# Get the message type name from the proto file
MESSAGE_TYPE=$(grep -E "^message [A-Za-z_]+" user_profile.proto | head -1 | awk '{print $2}')

if [ -z "$MESSAGE_TYPE" ]; then
    echo "ERROR: Could not determine message type from user_profile.proto"
    exit 1
fi

echo "Message type: $MESSAGE_TYPE"
echo ""

# Test each export file
for export_file in "${EXPORT_FILES[@]}"; do
    filename=$(basename "$export_file")
    echo -n "Testing $filename... "
    
    # Try to parse the text format file
    if protoc --decode="$MESSAGE_TYPE" user_profile.proto < "$export_file" > /dev/null 2>&1; then
        echo "✓ PASS"
        ((SUCCESS_COUNT++))
    else
        echo "✗ FAIL"
        ((FAILURE_COUNT++))
        # Show error details
        echo "  Error details:"
        protoc --decode="$MESSAGE_TYPE" user_profile.proto < "$export_file" 2>&1 | sed 's/^/    /'
    fi
done

echo ""
echo "=========================================================="
echo "Results: $SUCCESS_COUNT/$TOTAL_FILES files parsed successfully"

if [ $FAILURE_COUNT -eq 0 ]; then
    echo "Status: ALL TESTS PASSED ✓"
    exit 0
else
    echo "Status: $FAILURE_COUNT TEST(S) FAILED ✗"
    exit 1
fi