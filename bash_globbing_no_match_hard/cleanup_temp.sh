#!/bin/bash

# Temporary file cleanup script for batch processing system
# Cleans up temporary and scratch directories after processing

TEMP_DIR="/home/agent/temp"
SCRATCH_DIR="/home/agent/scratch"

echo "Starting cleanup of temporary directories..."

# Clean up temporary files
echo "Removing temporary files..."
rm /home/agent/temp/*.tmp
rm /home/agent/temp/*.log

# Clean up scratch cache files
echo "Checking cache file sizes..."
du -sh /home/agent/scratch/*.cache

# Remove old scratch files
for f in /home/agent/scratch/*.dat; do
    echo "Removing scratch file: $f"
    rm -f "$f"
done

# Clean up process ID files
if [ -f /home/agent/temp/*.pid ]; then
    echo "Removing PID files..."
    rm /home/agent/temp/*.pid
fi

# List and remove log files using pipe
echo "Cleaning up log files..."
ls /home/agent/temp/*.log | xargs rm

# Remove any remaining temporary files
for f in /home/agent/temp/*; do
    echo "Processing: $f"
    rm -f "$f"
done

# Verify cleanup
echo "Verifying cleanup completion..."
echo "Remaining temp files:"
ls -la /home/agent/temp/*.tmp

echo "Cleanup completed."