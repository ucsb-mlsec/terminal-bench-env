#!/bin/bash

# Ansible Playbook Idempotency Verification Script
# Usage: ./verify_idempotency.sh <playbook_path>

set -e

# Check if playbook path is provided
if [ -z "$1" ]; then
    echo "ERROR: No playbook path provided"
    echo "Usage: $0 <playbook_path>"
    exit 1
fi

PLAYBOOK=$1
INVENTORY="/opt/playbooks/inventory"

# Check if playbook exists
if [ ! -f "$PLAYBOOK" ]; then
    echo "ERROR: Playbook not found at $PLAYBOOK"
    exit 1
fi

# Check if inventory exists
if [ ! -f "$INVENTORY" ]; then
    echo "ERROR: Inventory not found at $INVENTORY"
    exit 1
fi

echo "=========================================="
echo "Idempotency Test for: $PLAYBOOK"
echo "=========================================="
echo ""

# First run
echo ">>> FIRST RUN - Initial Configuration <<<"
echo "------------------------------------------"
FIRST_RUN=$(ansible-playbook -i "$INVENTORY" "$PLAYBOOK" 2>&1)
FIRST_EXIT_CODE=$?

if [ $FIRST_EXIT_CODE -ne 0 ]; then
    echo "$FIRST_RUN"
    echo ""
    echo "ERROR: First playbook run failed with exit code $FIRST_EXIT_CODE"
    exit 1
fi

echo "$FIRST_RUN"
echo ""

# Extract statistics from first run
FIRST_CHANGED=$(echo "$FIRST_RUN" | grep -oP 'changed=\K[0-9]+' | tail -1)
FIRST_FAILED=$(echo "$FIRST_RUN" | grep -oP 'failed=\K[0-9]+' | tail -1)

echo "First Run Summary: changed=$FIRST_CHANGED, failed=$FIRST_FAILED"
echo ""
sleep 1

# Second run
echo ">>> SECOND RUN - Idempotency Check <<<"
echo "------------------------------------------"
SECOND_RUN=$(ansible-playbook -i "$INVENTORY" "$PLAYBOOK" 2>&1)
SECOND_EXIT_CODE=$?

if [ $SECOND_EXIT_CODE -ne 0 ]; then
    echo "$SECOND_RUN"
    echo ""
    echo "ERROR: Second playbook run failed with exit code $SECOND_EXIT_CODE"
    exit 1
fi

echo "$SECOND_RUN"
echo ""

# Extract statistics from second run
SECOND_CHANGED=$(echo "$SECOND_RUN" | grep -oP 'changed=\K[0-9]+' | tail -1)
SECOND_FAILED=$(echo "$SECOND_RUN" | grep -oP 'failed=\K[0-9]+' | tail -1)

echo "Second Run Summary: changed=$SECOND_CHANGED, failed=$SECOND_FAILED"
echo ""

# Check for idempotency
echo "=========================================="
echo "IDEMPOTENCY VALIDATION RESULT"
echo "=========================================="

if [ "$SECOND_CHANGED" = "0" ] && [ "$SECOND_FAILED" = "0" ]; then
    echo "✓ PASS: Playbook is idempotent"
    echo "  - No changes detected on second run"
    echo "  - All tasks reported 'ok' status"
    exit 0
else
    echo "✗ FAIL: Playbook is not idempotent"
    echo "  - Changes detected: $SECOND_CHANGED"
    echo "  - Tasks failed: $SECOND_FAILED"
    echo ""
    echo "The playbook must not report any changes when run twice consecutively."
    echo "Review the playbook for non-idempotent operations."
    exit 1
fi