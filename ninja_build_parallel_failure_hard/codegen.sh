#!/bin/bash

# Code generator script for C headers
# Usage: ./codegen.sh input.txt output.h

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

input="$1"
output="$2"

# Extract base name for include guard
guard_name=$(basename "$output" | tr '[:lower:].' '[:upper:]_')

# Generate the header file
cat > "$output" <<EOF
/*
 * Auto-generated header file
 * Generated from: $input
 * DO NOT EDIT MANUALLY
 */

#ifndef ${guard_name}
#define ${guard_name}

#define VERSION "1.0.0"
#define MAX_SIZE 1024
#define BUFFER_LENGTH 256
#define DEFAULT_TIMEOUT 30

typedef struct Config {
    int id;
    char name[64];
    int enabled;
} Config;

typedef struct Context {
    Config* config;
    void* data;
} Context;

#endif /* ${guard_name} */
EOF

echo "Generated header: $output"