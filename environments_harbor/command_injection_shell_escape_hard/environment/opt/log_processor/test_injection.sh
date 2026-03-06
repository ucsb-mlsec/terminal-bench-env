#!/bin/bash

# Security Test Suite for Log Processor
# Tests for command injection vulnerabilities

TEST_DIR="/tmp/log_test_$$"
MARKER_FILE="/tmp/injection_marker_$$"
VULNERABLE_COUNT=0
TEST_COUNT=0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

setup_test_env() {
    mkdir -p "$TEST_DIR"
    mkdir -p "$TEST_DIR/logs"
    mkdir -p "$TEST_DIR/reports"
    rm -f "$MARKER_FILE"
}

cleanup_test_env() {
    rm -rf "$TEST_DIR"
    rm -f "$MARKER_FILE"
}

check_injection_success() {
    if [ -f "$MARKER_FILE" ]; then
        return 0  # Injection succeeded (bad)
    fi
    return 1  # Injection blocked (good)
}

# Test 1: Filename with semicolon command injection
test_semicolon_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 1] Testing semicolon injection in filename..."
    
    local malicious_file="test;touch ${MARKER_FILE};.log"
    rm -f "$MARKER_FILE"
    
    echo "Sample log data" > "$TEST_DIR/logs/legitimate.log"
    
    # Test against process_logs.sh
    cd /opt/log_processor/
    ./process_logs.sh "$TEST_DIR/logs/$malicious_file" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Semicolon injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Semicolon injection blocked"
    fi
}

# Test 2: Filename with pipe injection
test_pipe_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 2] Testing pipe injection in filename..."
    
    local malicious_file="test.log|touch ${MARKER_FILE}"
    rm -f "$MARKER_FILE"
    
    cd /opt/log_processor/
    ./process_logs.sh "$TEST_DIR/logs/$malicious_file" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Pipe injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Pipe injection blocked"
    fi
}

# Test 3: Filename with command substitution using $()
test_dollar_substitution() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 3] Testing \$() command substitution in filename..."
    
    local malicious_file="test\$(touch ${MARKER_FILE}).log"
    rm -f "$MARKER_FILE"
    
    cd /opt/log_processor/
    ./process_logs.sh "$TEST_DIR/logs/$malicious_file" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} \$() substitution injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} \$() substitution injection blocked"
    fi
}

# Test 4: Log entry with backtick injection
test_backtick_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 4] Testing backtick injection in log entry..."
    
    rm -f "$MARKER_FILE"
    echo "Error: \`touch ${MARKER_FILE}\` failed" > "$TEST_DIR/logs/test_backtick.log"
    
    cd /opt/log_processor/
    ./generate_report.sh "$TEST_DIR/logs/test_backtick.log" "$TEST_DIR/reports/output.txt" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Backtick injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Backtick injection blocked"
    fi
}

# Test 5: Log entry with $() command injection
test_log_dollar_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 5] Testing \$() injection in log entry..."
    
    rm -f "$MARKER_FILE"
    echo "Status: \$(touch ${MARKER_FILE}) complete" > "$TEST_DIR/logs/test_dollar.log"
    
    cd /opt/log_processor/
    ./generate_report.sh "$TEST_DIR/logs/test_dollar.log" "$TEST_DIR/reports/output2.txt" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} \$() log injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} \$() log injection blocked"
    fi
}

# Test 6: Path traversal with command injection
test_path_traversal_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 6] Testing path traversal with command injection..."
    
    local malicious_path="../../tmp/test;touch ${MARKER_FILE};.log"
    rm -f "$MARKER_FILE"
    
    cd /opt/log_processor/
    ./cleanup.sh "$malicious_path" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Path traversal injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Path traversal injection blocked"
    fi
}

# Test 7: Ampersand background process injection
test_ampersand_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 7] Testing ampersand injection..."
    
    local malicious_file="test.log&touch ${MARKER_FILE}&"
    rm -f "$MARKER_FILE"
    
    cd /opt/log_processor/
    ./cleanup.sh "$malicious_file" 2>/dev/null || true
    sleep 1  # Give background process time
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Ampersand injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Ampersand injection blocked"
    fi
}

# Test 8: Eval injection through log processing
test_eval_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 8] Testing eval-based injection..."
    
    rm -f "$MARKER_FILE"
    echo "Action: rm -rf /tmp; touch ${MARKER_FILE}" > "$TEST_DIR/logs/test_eval.log"
    
    cd /opt/log_processor/
    ./generate_report.sh "$TEST_DIR/logs/test_eval.log" "$TEST_DIR/reports/output3.txt" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Eval injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Eval injection blocked"
    fi
}

# Test 9: Unquoted variable in cleanup
test_unquoted_variable() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 9] Testing unquoted variable exploitation..."
    
    local malicious_file="test.log touch ${MARKER_FILE}"
    rm -f "$MARKER_FILE"
    
    cd /opt/log_processor/
    ./cleanup.sh "$malicious_file" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Unquoted variable injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Unquoted variable injection blocked"
    fi
}

# Test 10: Newline injection
test_newline_injection() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo "[TEST 10] Testing newline injection..."
    
    rm -f "$MARKER_FILE"
    printf "test.log\ntouch ${MARKER_FILE}\n" > "$TEST_DIR/malicious_input.txt"
    
    cd /opt/log_processor/
    local input=$(cat "$TEST_DIR/malicious_input.txt")
    ./cleanup.sh "$input" 2>/dev/null || true
    
    if check_injection_success; then
        echo -e "${RED}[FAIL]${NC} Newline injection succeeded"
        VULNERABLE_COUNT=$((VULNERABLE_COUNT + 1))
    else
        echo -e "${GREEN}[PASS]${NC} Newline injection blocked"
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "Log Processor Security Test Suite"
    echo "=========================================="
    echo ""
    
    setup_test_env
    
    # Run all tests
    test_semicolon_injection
    test_pipe_injection
    test_dollar_substitution
    test_backtick_injection
    test_log_dollar_injection
    test_path_traversal_injection
    test_ampersand_injection
    test_eval_injection
    test_unquoted_variable
    test_newline_injection
    
    echo ""
    echo "=========================================="
    echo "Test Results Summary"
    echo "=========================================="
    echo "Total tests: $TEST_COUNT"
    echo "Vulnerabilities found: $VULNERABLE_COUNT"
    echo "Tests passed: $((TEST_COUNT - VULNERABLE_COUNT))"
    echo ""
    
    cleanup_test_env
    
    if [ $VULNERABLE_COUNT -eq 0 ]; then
        echo -e "${GREEN}Status: SECURE${NC}"
        echo "SECURE"
    else
        echo -e "${RED}Status: VULNERABLE${NC}"
        echo "VULNERABLE"
    fi
}

main