#!/bin/bash

set -e

echo "=== Daemon Graceful Shutdown Test Harness ==="
echo

# Cleanup any existing results file
echo "Cleaning up previous test results..."
rm -f /workspace/worker/results.txt

# Start the daemon in the background
echo "Starting daemon..."
python /workspace/worker/daemon.py &
DAEMON_PID=$!

# Verify the daemon process started
sleep 1
if ! kill -0 $DAEMON_PID 2>/dev/null; then
    echo "ERROR: Daemon failed to start"
    exit 1
fi

echo "Daemon started with PID: $DAEMON_PID"

# Wait for daemon to begin processing a task
echo "Waiting for daemon to begin processing a task..."
sleep 3

# Verify daemon is still running
if ! kill -0 $DAEMON_PID 2>/dev/null; then
    echo "ERROR: Daemon died unexpectedly before signal was sent"
    exit 1
fi

# Send SIGTERM to the daemon
echo "Sending SIGTERM to daemon (PID: $DAEMON_PID)..."
kill -TERM $DAEMON_PID

# Wait for daemon to exit with timeout
echo "Waiting for daemon to complete current task and exit..."
TIMEOUT=20
ELAPSED=0
while kill -0 $DAEMON_PID 2>/dev/null; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Daemon did not exit within ${TIMEOUT} seconds"
        kill -9 $DAEMON_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# Get the exit code
wait $DAEMON_PID
DAEMON_EXIT_CODE=$?

echo "Daemon exited with code: $DAEMON_EXIT_CODE"

# Check that exit code is 0
if [ $DAEMON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Daemon exited with non-zero status code: $DAEMON_EXIT_CODE"
    exit 1
fi

# Verify results.txt exists
if [ ! -f /workspace/worker/results.txt ]; then
    echo "ERROR: results.txt was not created"
    exit 1
fi

# Verify at least one task completion was written
TASK_COUNT=$(wc -l < /workspace/worker/results.txt)
if [ $TASK_COUNT -lt 1 ]; then
    echo "ERROR: No task completions found in results.txt"
    exit 1
fi

echo "Results file contains $TASK_COUNT completed task(s)"
echo
echo "=== All checks passed! ==="
echo "✓ Daemon started successfully"
echo "✓ Daemon handled SIGTERM gracefully"
echo "✓ Daemon exited with status code 0"
echo "✓ Current task completed before exit"
echo
exit 0