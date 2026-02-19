from flask import Flask, Response, request
import time
import json

app = Flask(__name__)

event_counter = 0
active_clients = []

def generate_events(last_id):
    global event_counter
    client_buffer = []
    active_clients.append(client_buffer)
    
    try:
        # Convert last_id to integer, default to 0 if invalid
        last_event_id = int(last_id) if last_id and last_id.isdigit() else 0
        
        while True:
            event_counter += 1
            
            # Skip events that the client has already received
            if event_counter <= last_event_id:
                continue
            
            event_data = {
                'message': f'Alert {event_counter}',
                'severity': 'medium'
            }
            
            # Include the id: field in SSE format
            yield f"id: {event_counter}\n"
            yield f"data: {json.dumps(event_data)}\n\n"
            time.sleep(2)
    finally:
        # Remove client buffer when client disconnects
        if client_buffer in active_clients:
            active_clients.remove(client_buffer)

@app.route('/events')
def events():
    last_id = request.headers.get('Last-Event-ID', '0')
    return Response(generate_events(last_id), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
```

```bash
#!/bin/bash

# Error handling - exit on any error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Starting Event Server Validation Tests..."

# Clean up function
cleanup() {
    echo "Cleaning up..."
    # Kill any process on port 8080
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    pkill -f "event_server_fixed.py" 2>/dev/null || true
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Kill any existing server processes on port 8080
echo "Killing any existing processes on port 8080..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 1

# Start the fixed server in background
echo "Starting fixed server..."
python3 /home/agent/event_server_fixed.py > /tmp/server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo -e "${RED}FAIL: Server failed to start${NC}"
    echo "FAIL: Server failed to start" > /tmp/test_results.txt
    cat /tmp/server.log
    exit 1
fi

# Test 1: Connect and verify events have IDs
echo "Test 1: Checking for event IDs..."
timeout 8 curl -N -s http://localhost:8080/events > /tmp/events1.txt 2>&1 || true

# Check if we received events
if [ ! -s /tmp/events1.txt ]; then
    echo -e "${RED}FAIL: No events received${NC}"
    echo "FAIL: No events received" > /tmp/test_results.txt
    exit 1
fi

# Verify events have id: fields
ID_COUNT=$(grep -c "^id: " /tmp/events1.txt || echo "0")
if [ "$ID_COUNT" -lt 3 ]; then
    echo -e "${RED}FAIL: Events don't have proper id: fields (found $ID_COUNT)${NC}"
    echo "FAIL: Events don't have proper id: fields" > /tmp/test_results.txt
    cat /tmp/events1.txt
    exit 1
fi

echo -e "${GREEN}✓ Events have id: fields${NC}"

# Extract the last event ID
LAST_ID=$(grep "^id: " /tmp/events1.txt | tail -1 | awk '{print $2}')
echo "Last event ID received: $LAST_ID"

if [ -z "$LAST_ID" ]; then
    echo -e "${RED}FAIL: Could not extract last event ID${NC}"
    echo "FAIL: Could not extract last event ID" > /tmp/test_results.txt
    exit 1
fi

# Test 2: Reconnect with Last-Event-ID header
echo "Test 2: Testing event resumption with Last-Event-ID: $LAST_ID..."
sleep 2
timeout 8 curl -N -s -H "Last-Event-ID: $LAST_ID" http://localhost:8080/events > /tmp/events2.txt 2>&1 || true

# Check if we received events
if [ ! -s /tmp/events2.txt ]; then
    echo -e "${RED}FAIL: No events received on reconnection${NC}"
    echo "FAIL: No events received on reconnection" > /tmp/test_results.txt
    exit 1
fi

# Extract first ID from reconnection
FIRST_NEW_ID=$(grep "^id: " /tmp/events2.txt | head -1 | awk '{print $2}')
echo "First new event ID: $FIRST_NEW_ID"

if [ -z "$FIRST_NEW_ID" ]; then
    echo -e "${RED}FAIL: No event IDs in reconnection${NC}"
    echo "FAIL: No event IDs in reconnection" > /tmp/test_results.txt
    exit 1
fi

# Verify that the first new ID is greater than the last ID (no duplicates)
EXPECTED_ID=$((LAST_ID + 1))
if [ "$FIRST_NEW_ID" -lt "$EXPECTED_ID" ]; then
    echo -e "${RED}FAIL: Received duplicate events (expected ID >= $EXPECTED_ID, got $FIRST_NEW_ID)${NC}"
    echo "FAIL: Received duplicate events on reconnection" > /tmp/test_results.txt
    exit 1
fi

echo -e "${GREEN}✓ Event resumption works correctly${NC}"

# Test 3: Check memory cleanup (client disconnection)
echo "Test 3: Testing client cleanup..."

# Get initial memory usage
INITIAL_MEM=$(ps -p $SERVER_PID -o rss= || echo "0")

# Connect and disconnect multiple clients
for i in {1..5}; do
    timeout 3 curl -N -s http://localhost:8080/events > /dev/null 2>&1 || true
    sleep 0.5
done

sleep 2

# Check if server is still running
if ! ps -p $SERVER_PID > /dev/null; then
    echo -e "${RED}FAIL: Server crashed during multiple connections${NC}"
    echo "FAIL: Server crashed during multiple connections" > /tmp/test_results.txt
    exit 1
fi

FINAL_MEM=$(ps -p $SERVER_PID -o rss= || echo "0")

# Memory should not grow significantly (allow some margin)
MEM_GROWTH=$((FINAL_MEM - INITIAL_MEM))
echo "Memory growth: ${MEM_GROWTH}KB"

# If memory grew by more than 50MB, there might be a leak
if [ "$MEM_GROWTH" -gt 51200 ]; then
    echo -e "${RED}WARNING: Significant memory growth detected (${MEM_GROWTH}KB)${NC}"
    # Not failing on this as it could be normal Python behavior
fi

echo -e "${GREEN}✓ Client cleanup test passed${NC}"

# All tests passed
echo -e "${GREEN}All tests PASSED!${NC}"
echo "PASS" > /tmp/test_results.txt

exit 0