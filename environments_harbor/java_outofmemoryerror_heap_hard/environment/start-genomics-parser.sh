#!/bin/bash

# Genomics Parser Service Startup Script
# Description: Starts the genomics-parser Java service for processing genomic sequence data

# Service configuration
SERVICE_NAME="genomics-parser"
CONFIG_FILE="/app/config/genomics-parser.conf"
LOG_DIR="/app/logs"
APP_HOME="/app/genomics-parser"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Load configuration file
if [ -f "${CONFIG_FILE}" ]; then
    source ${CONFIG_FILE}
    echo "Configuration loaded from ${CONFIG_FILE}"
else
    echo "Warning: Configuration file ${CONFIG_FILE} not found, using defaults"
fi

# JVM Heap Settings
# TODO: Heap size needs optimization based on memory usage analysis
# Current settings may cause OutOfMemoryError during peak loads
HEAP_SIZE="-Xmx2048m -Xms512m"

# Additional JVM Options
JVM_OPTS="${HEAP_SIZE}"
JVM_OPTS="${JVM_OPTS} -XX:+UseG1GC"
JVM_OPTS="${JVM_OPTS} -XX:+PrintGCDetails"
JVM_OPTS="${JVM_OPTS} -XX:+PrintGCDateStamps"
JVM_OPTS="${JVM_OPTS} -Xloggc:${LOG_DIR}/${SERVICE_NAME}-gc.log"
JVM_OPTS="${JVM_OPTS} -XX:+HeapDumpOnOutOfMemoryError"
JVM_OPTS="${JVM_OPTS} -XX:HeapDumpPath=${LOG_DIR}/${SERVICE_NAME}-heap-dump.hprof"

# Classpath configuration
CLASSPATH="${APP_HOME}/lib/*"

# Main class
MAIN_CLASS="com.bioinformatics.GenomicsParser"

# Service startup
echo "Starting ${SERVICE_NAME} service..."
echo "JVM Options: ${JVM_OPTS}"
echo "Main Class: ${MAIN_CLASS}"
echo "Log output: ${LOG_DIR}/${SERVICE_NAME}.log"

# Start the Java application
java ${JVM_OPTS} \
    -cp ${CLASSPATH} \
    ${MAIN_CLASS} \
    >> ${LOG_DIR}/${SERVICE_NAME}.log 2>&1 &

# Capture process ID
PID=$!
echo ${PID} > ${LOG_DIR}/${SERVICE_NAME}.pid
echo "${SERVICE_NAME} started with PID: ${PID}"