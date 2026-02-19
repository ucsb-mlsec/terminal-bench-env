#!/bin/bash
# run_parallel_test.sh - Test harness for parallel log processing

set -u

# Configuration
INCOMING_DIR="/var/logs/incoming"
PROCESSED_DIR="/var/logs/processed"
TEST_DATE="2024/01/15"
NUM_WORKERS=20
PROCESS_SCRIPT="/opt/logprocessor/process_logs.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Parallel Log Processing Test Harness"
echo "=========================================="
echo "Workers: ${NUM_WORKERS}"
echo "Test Date: ${TEST_DATE}"
echo ""

# Cleanup and setup
echo "Setting up test environment..."
rm -rf "${INCOMING_DIR}" "${PROCESSED_DIR}"
mkdir -p "${INCOMING_DIR}"
mkdir -p "${PROCESSED_DIR}"

# Create test log files
echo "Creating ${NUM_WORKERS} test log files..."
for i in $(seq -f "%02g" 1 ${NUM_WORKERS}); do
    LOG_FILE="${INCOMING_DIR}/test_log_${i}.txt"
    cat > "${LOG_FILE}" << EOF
2024-01-15 10:00:${i} INFO Worker ${i} started processing
2024-01-15 10:01:${i} DEBUG Worker ${i} reading configuration
2024-01-15 10:02:${i} INFO Worker ${i} processing batch 1
2024-01-15 10:03:${i} INFO Worker ${i} processing batch 2
2024-01-15 10:04:${i} WARN Worker ${i} encountered minor issue
2024-01-15 10:05:${i} INFO Worker ${i} completed successfully
EOF
done

echo "Test files created successfully."
echo ""

# Array to store PIDs and exit codes
declare -a WORKER_PIDS
declare -a WORKER_EXIT_CODES

# Launch parallel workers
echo "Launching ${NUM_WORKERS} parallel workers..."
for i in $(seq -f "%02g" 1 ${NUM_WORKERS}); do
    LOG_FILE="${INCOMING_DIR}/test_log_${i}.txt"
    "${PROCESS_SCRIPT}" "${LOG_FILE}" "${TEST_DATE}" &
    WORKER_PIDS+=($!)
done

echo "All workers launched. Waiting for completion..."
echo ""

# Wait for all workers and collect exit codes
SUCCESS_COUNT=0
FAILURE_COUNT=0

for i in "${!WORKER_PIDS[@]}"; do
    PID="${WORKER_PIDS[$i]}"
    wait "${PID}"
    EXIT_CODE=$?
    WORKER_EXIT_CODES+=("${EXIT_CODE}")
    
    WORKER_NUM=$((i + 1))
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Worker ${WORKER_NUM} (PID ${PID}) completed successfully"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗${NC} Worker ${WORKER_NUM} (PID ${PID}) failed with exit code ${EXIT_CODE}"
        ((FAILURE_COUNT++))
    fi
done

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo "Total Workers: ${NUM_WORKERS}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAILURE_COUNT}${NC}"
echo ""

# Check if output directory was created
if [ -d "${PROCESSED_DIR}/${TEST_DATE}" ]; then
    OUTPUT_FILES=$(find "${PROCESSED_DIR}/${TEST_DATE}" -type f | wc -l)
    echo "Output directory exists: ${PROCESSED_DIR}/${TEST_DATE}"
    echo "Output files created: ${OUTPUT_FILES}"
else
    echo -e "${RED}ERROR: Output directory was not created!${NC}"
fi

echo ""

# Final verdict
if [ ${FAILURE_COUNT} -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "✓ ALL TESTS PASSED!"
    echo -e "==========================================${NC}"
    exit 0
else
    echo -e "${RED}=========================================="
    echo "✗ TESTS FAILED!"
    echo "Some workers encountered errors."
    echo -e "==========================================${NC}"
    exit 1
fi