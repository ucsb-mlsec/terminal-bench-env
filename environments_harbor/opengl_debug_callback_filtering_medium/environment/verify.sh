#!/bin/bash

echo "Running message generation test..."

cd /workspace/graphics_debug/

if [ ! -f "generate_messages.py" ]; then
    echo "Error: generate_messages.py not found"
    exit 1
fi

python3 generate_messages.py

if [ $? -eq 0 ]; then
    echo "Test completed successfully"
    exit 0
else
    echo "Test failed"
    exit 1
fi