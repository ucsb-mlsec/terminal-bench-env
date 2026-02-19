#!/bin/bash

# Baseline build script for legacy application
# Produces a working but unoptimized binary

CC=gcc
SRC_DIR=src
INC_DIR=include
OUTPUT=legacy_app

# Basic compiler flags - debug symbols, no optimization
CFLAGS="-g -O0 -I${INC_DIR}"

echo "Starting baseline build..."
echo "Compiler: ${CC}"
echo "Flags: ${CFLAGS}"
echo ""

# Compile all source files
echo "Compiling source files..."
${CC} ${CFLAGS} \
    ${SRC_DIR}/main.c \
    ${SRC_DIR}/utils.c \
    ${SRC_DIR}/processing.c \
    ${SRC_DIR}/config.c \
    -o ${OUTPUT}

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo "Output binary: ${OUTPUT}"
    echo ""
    echo "Binary size:"
    ls -lh ${OUTPUT} | awk '{print $5, $9}'
    du -h ${OUTPUT}
else
    echo "Build failed!"
    exit 1
fi