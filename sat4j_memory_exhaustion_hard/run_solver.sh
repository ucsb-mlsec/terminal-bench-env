#!/bin/bash

# Set working directory
cd /opt/sat_service || exit 1

# Create temporary file for time output
TEMP_OUTPUT=$(mktemp)
SOLVER_OUTPUT=$(mktemp)

# Cleanup function
cleanup() {
    rm -f "$TEMP_OUTPUT" "$SOLVER_OUTPUT"
}
trap cleanup EXIT

# Run solver with time monitoring
echo "Running SAT solver with memory monitoring..."
/usr/bin/time -v python3 solver.py test_problem.cnf > "$SOLVER_OUTPUT" 2> "$TEMP_OUTPUT"
SOLVER_EXIT=$?

# Check if solver completed successfully
if [ $SOLVER_EXIT -ne 0 ]; then
    echo "ERROR: Solver failed to execute"
    cat "$SOLVER_OUTPUT"
    exit 1
fi

# Extract memory usage (in KB from time output, convert to MB)
MEMORY_KB=$(grep "Maximum resident set size" "$TEMP_OUTPUT" | awk '{print $6}')
MEMORY_MB=$(echo "scale=2; $MEMORY_KB / 1024" | bc)

# Extract elapsed time
ELAPSED_TIME=$(grep "Elapsed (wall clock)" "$TEMP_OUTPUT" | awk '{print $8}' | tr -d '()')

# Extract solver result
RESULT=$(grep -E "(SATISFIABLE|UNSATISFIABLE)" "$SOLVER_OUTPUT" | head -1 | awk '{print $1}')

# Display results
echo "----------------------------------------"
echo "Peak Memory: ${MEMORY_MB}MB"
echo "Runtime: ${ELAPSED_TIME} seconds"
echo "Result: ${RESULT}"
echo "----------------------------------------"

# Output detailed timing if needed
if [ "$1" == "-v" ]; then
    echo ""
    echo "Detailed timing information:"
    cat "$TEMP_OUTPUT"
fi

exit 0