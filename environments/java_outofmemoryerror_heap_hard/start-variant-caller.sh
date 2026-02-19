#!/bin/bash

# Variant Caller Service Startup Script
# This service processes aligned genomic sequences to identify genetic variants

SERVICE_NAME="variant-caller"

# Load service-specific configuration
if [ -f "config/variant-caller.conf" ]; then
    source config/variant-caller.conf
fi

# JVM Heap Settings
# NOTE: These heap settings need optimization based on memory usage log analysis
# Current setting is placeholder and may need adjustment to prevent OutOfMemoryError
HEAP_SIZE="-Xmx3072m -Xms768m"

# Additional JVM Options
JVM_OPTS="-XX:+UseG1GC -XX:+PrintGCDetails"

# Classpath
CLASSPATH="/app/variant-caller/lib/*"

# Main Class
MAIN_CLASS="com.bioinformatics.VariantCaller"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the service
echo "Starting ${SERVICE_NAME} service..."
java ${HEAP_SIZE} ${JVM_OPTS} -cp ${CLASSPATH} ${MAIN_CLASS} >> logs/${SERVICE_NAME}.log 2>&1 &

# Save PID
echo $! > ${SERVICE_NAME}.pid

echo "${SERVICE_NAME} started with PID $(cat ${SERVICE_NAME}.pid)"