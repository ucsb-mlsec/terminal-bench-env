#!/bin/bash

set -e

INPUT_DIR='inputs'
TRACER='./tracer'

# Check if tracer executable exists, if not build it
if [ ! -f "$TRACER" ]; then
    echo "Tracer executable not found, building..."
    make
fi

echo "Starting batch processing..."
echo "================================"

# Loop through input files 1-5
for i in {1..5}; do
    echo ""
    echo "Processing input file $i..."
    
    $TRACER $INPUT_DIR/tracer_input$i.txt
    
    # Print memory usage
    echo -n "Memory usage: "
    ps aux | grep "[t]racer" | awk '{print $6 " KB (RSS: " $4 "%)"}' || echo "Process completed"
    
    # Sleep between iterations
    if [ $i -lt 5 ]; then
        sleep 1
    fi
done

echo ""
echo "================================"
echo "Batch processing complete."

exit 0