#!/bin/bash

# Sequence Aligner Service Startup Script
# This script starts the sequence-aligner Java service with appropriate JVM settings

SERVICE_NAME="sequence-aligner"

# Load configuration file
CONFIG_FILE="config/sequence-aligner.conf"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Heap size configuration - NEEDS OPTIMIZATION
# Current setting may need adjustment based on memory usage logs
# TODO: Analyze memory logs to determine optimal heap size
HEAP_SIZE="-Xmx4096m -Xms1024m"

# JVM Options
JVM_OPTS="$HEAP_SIZE -XX:+UseG1GC -XX:+PrintGCDetails -XX:+PrintGCDateStamps"
JVM_OPTS="$JVM_OPTS -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=logs/"
JVM_OPTS="$JVM_OPTS -Djava.awt.headless=true"

# Classpath
CLASSPATH="/app/sequence-aligner/lib/*"

# Main class
MAIN_CLASS="com.bioinformatics.SequenceAligner"

# Log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Start the service
echo "Starting $SERVICE_NAME service..."
echo "Heap configuration: $HEAP_SIZE"
echo "Log output: $LOG_DIR/${SERVICE_NAME}.log"

java $JVM_OPTS -cp "$CLASSPATH" $MAIN_CLASS >> "$LOG_DIR/${SERVICE_NAME}.log" 2>&1 &

# Save PID
PID=$!
echo $PID > "$LOG_DIR/${SERVICE_NAME}.pid"

echo "$SERVICE_NAME started with PID: $PID"