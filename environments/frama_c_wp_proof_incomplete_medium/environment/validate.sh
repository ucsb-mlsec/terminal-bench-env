#!/bin/bash

# Buffer Safety Report Validation Script
# This script validates that all critical buffer safety issues have been identified
# in the packet_handler.c file and properly documented in safety_report.txt

REPORT_FILE="/workspace/safety_report.txt"

# Critical line numbers that MUST be identified (these correspond to severe buffer safety issues)
# These are the most dangerous buffer overflows and out-of-bounds accesses
CRITICAL_LINES=(42 58 73 89 105 127)

# Minimum total number of issues that should be reported
MIN_ISSUES=8

echo "=== Buffer Safety Report Validation ==="
echo

# Check 1: Verify report file exists
echo "[1] Checking if report file exists..."
if [ ! -f "$REPORT_FILE" ]; then
    echo "ERROR: Report file not found at $REPORT_FILE"
    exit 1
fi
echo "✓ Report file found"
echo

# Check 2: Validate format of each line
echo "[2] Validating report format..."
LINE_NUM=0
while IFS= read -r line; do
    LINE_NUM=$((LINE_NUM + 1))
    
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Check format: line=N,severity=S,reason=R
    if ! echo "$line" | grep -qE '^line=[0-9]+,severity=(critical|warning),reason=.+$'; then
        echo "ERROR: Invalid format on line $LINE_NUM of report: '$line'"
        echo "Expected format: line=N,severity=(critical|warning),reason=description"
        exit 1
    fi
done < "$REPORT_FILE"
echo "✓ All lines follow correct format"
echo

# Check 3: Verify severity values are valid
echo "[3] Checking severity values..."
INVALID_SEVERITY=$(grep -vE 'severity=(critical|warning)' "$REPORT_FILE" | grep -v '^$')
if [ -n "$INVALID_SEVERITY" ]; then
    echo "ERROR: Invalid severity value found. Must be 'critical' or 'warning'"
    echo "$INVALID_SEVERITY"
    exit 1
fi
echo "✓ All severity values are valid"
echo

# Check 4: Verify minimum number of issues reported
echo "[4] Checking minimum issue count..."
TOTAL_ISSUES=$(grep -c '^line=' "$REPORT_FILE")
if [ "$TOTAL_ISSUES" -lt "$MIN_ISSUES" ]; then
    echo "ERROR: Insufficient issues reported. Found $TOTAL_ISSUES, expected at least $MIN_ISSUES"
    echo "Please review the code more thoroughly for buffer safety issues"
    exit 1
fi
echo "✓ Sufficient issues reported: $TOTAL_ISSUES total"
echo

# Check 5: Verify all critical line numbers are present
echo "[5] Checking for critical line numbers..."
MISSING_CRITICAL=()
for CRIT_LINE in "${CRITICAL_LINES[@]}"; do
    if ! grep -q "^line=$CRIT_LINE," "$REPORT_FILE"; then
        MISSING_CRITICAL+=("$CRIT_LINE")
    fi
done

if [ ${#MISSING_CRITICAL[@]} -gt 0 ]; then
    echo "ERROR: Missing critical buffer safety issues at the following lines:"
    for MISSING in "${MISSING_CRITICAL[@]}"; do
        echo "  - Line $MISSING (CRITICAL buffer safety issue not identified)"
    done
    echo
    echo "These lines contain severe buffer overflow or out-of-bounds access issues"
    echo "that must be documented in the safety report."
    exit 1
fi
echo "✓ All critical line numbers identified"
echo

# Check 6: Verify that critical issues are marked appropriately
echo "[6] Verifying critical issues have appropriate severity..."
CRITICAL_COUNT=0
for CRIT_LINE in "${CRITICAL_LINES[@]}"; do
    if grep "^line=$CRIT_LINE," "$REPORT_FILE" | grep -q "severity=critical"; then
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
    fi
done

if [ "$CRITICAL_COUNT" -lt 4 ]; then
    echo "WARNING: Only $CRITICAL_COUNT out of ${#CRITICAL_LINES[@]} critical lines marked as 'critical' severity"
    echo "Consider reviewing if some warnings should be upgraded to critical"
fi
echo "✓ Critical severity assignments reviewed"
echo

# All checks passed
echo "========================================="
echo "✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓"
echo "========================================="
echo
echo "Summary:"
echo "  - Report file exists and is properly formatted"
echo "  - All $TOTAL_ISSUES issues follow the required format"
echo "  - All ${#CRITICAL_LINES[@]} critical buffer safety issues identified"
echo "  - Severity values are valid"
echo
echo "The safety analysis is complete and comprehensive."

exit 0