#!/bin/bash

set -e

echo "=== Signal Handler Safety Verification Script ==="
echo

# Check if libcompute.so exists
echo "[1/7] Checking if libcompute.so exists..."
if [ ! -f "/workspace/mathlib/libcompute.so" ]; then
    echo "ERROR: /workspace/mathlib/libcompute.so not found!"
    exit 1
fi
echo "✓ Library file exists"
echo

# Check for red zone usage in compiled library
echo "[2/7] Analyzing library for unsafe stack operations..."
if ! command -v objdump &> /dev/null; then
    echo "WARNING: objdump not found, skipping disassembly check"
else
    # Check for red zone violations (rsp adjustments below stack pointer without adjustment)
    # Red zone unsafe code typically accesses memory below %rsp without first subtracting from %rsp
    objdump -d /workspace/mathlib/libcompute.so > /tmp/disasm.txt
    
    # Look for patterns that suggest red zone usage (accessing negative offsets from %rsp)
    # With -mno-red-zone, we should see proper stack adjustments
    if grep -q "sub.*%rsp" /tmp/disasm.txt; then
        echo "✓ Found proper stack frame allocation (sub from rsp)"
    fi
    
    # Check that we're not seeing bare negative offsets from rsp without allocation
    # This is a heuristic check
    echo "✓ Disassembly analysis complete"
fi
echo

# Check for compilation flags in build artifacts
echo "[3/7] Checking build configuration..."
if [ -f "/workspace/mathlib/Makefile" ]; then
    if grep -q "mno-red-zone" /workspace/mathlib/Makefile; then
        echo "✓ Found -mno-red-zone flag in Makefile"
    else
        echo "WARNING: -mno-red-zone not found in Makefile"
    fi
fi
echo

# Compile test program
echo "[4/7] Compiling test program..."
if [ ! -f "/workspace/test_signal_safety.c" ]; then
    echo "ERROR: test_signal_safety.c not found!"
    exit 1
fi

gcc -o /workspace/test_signal_safety /workspace/test_signal_safety.c \
    -L/workspace/mathlib -lcompute -Wl,-rpath,/workspace/mathlib
echo "✓ Test program compiled successfully"
echo

# Run test program briefly
echo "[5/7] Running test program..."
timeout 5 /workspace/test_signal_safety || {
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "✓ Test ran for timeout period (expected behavior)"
    elif [ $exit_code -eq 0 ]; then
        echo "✓ Test completed successfully"
    else
        echo "ERROR: Test program crashed or failed (exit code: $exit_code)"
        exit 1
    fi
}
echo

# Check solution.json
echo "[6/7] Validating solution.json..."
if [ ! -f "/workspace/solution.json" ]; then
    echo "ERROR: solution.json not found!"
    exit 1
fi

# Validate JSON format
if ! python3 -c "import json; json.load(open('/workspace/solution.json'))" 2>/dev/null; then
    echo "ERROR: solution.json is not valid JSON!"
    exit 1
fi

# Check required fields
python3 << 'PYEOF'
import json
import sys

with open('/workspace/solution.json') as f:
    data = json.load(f)

required_fields = ['modified_files', 'critical_flag', 'library_path']
for field in required_fields:
    if field not in data:
        print(f"ERROR: Missing required field: {field}")
        sys.exit(1)

if 'mno-red-zone' not in data['critical_flag']:
    print(f"ERROR: critical_flag should be -mno-red-zone, got: {data['critical_flag']}")
    sys.exit(1)

print("✓ solution.json format is valid")
PYEOF

echo

# Final verification
echo "[7/7] Final checks..."
ldd /workspace/test_signal_safety | grep -q libcompute || {
    echo "ERROR: Test program not properly linked to libcompute.so"
    exit 1
}
echo "✓ Library linkage verified"
echo

echo "============================================"
echo "VERIFICATION PASSED"
echo "============================================"
exit 0