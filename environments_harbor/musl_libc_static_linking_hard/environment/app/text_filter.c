#!/bin/bash

# Solution script to build a fully static binary for minimal Alpine containers

set -e

# Navigate to the application directory
cd /workspace/app

# Create the C source file
cat > text_filter.c << 'EOF'
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 1024

/**
 * Text Filter Program
 * Reads text from stdin, converts to uppercase, and outputs to stdout
 * Also counts and reports statistics about the processed text
 */

int main(int argc, char *argv[]) {
    char line[MAX_LINE_LENGTH];
    int line_count = 0;
    int char_count = 0;
    int word_count = 0;
    
    // Process each line from stdin
    while (fgets(line, sizeof(line), stdin) != NULL) {
        line_count++;
        int in_word = 0;
        
        // Process each character in the line
        for (int i = 0; line[i] != '\0'; i++) {
            char c = line[i];
            
            // Count characters (excluding newlines)
            if (c != '\n') {
                char_count++;
            }
            
            // Count words
            if (isspace(c)) {
                in_word = 0;
            } else {
                if (!in_word) {
                    word_count++;
                    in_word = 1;
                }
            }
            
            // Convert to uppercase and output
            if (c != '\n') {
                putchar(toupper(c));
            }
        }
        
        // Add newline after each processed line
        putchar('\n');
    }
    
    // Print statistics to stderr to separate from processed output
    if (line_count > 0) {
        fprintf(stderr, "\n--- Statistics ---\n");
        fprintf(stderr, "Lines processed: %d\n", line_count);
        fprintf(stderr, "Characters: %d\n", char_count);
        fprintf(stderr, "Words: %d\n", word_count);
    }
    
    return 0;
}
EOF

# Create a Makefile that builds a fully static binary using musl-gcc
cat > Makefile << 'EOF'
CC = musl-gcc
CFLAGS = -Wall -O2 -static
TARGET = text_filter
SRC = text_filter.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
EOF

# Clean any previous builds
make clean 2>/dev/null || true

# Build the static binary
make

# Verify the binary was created
if [ ! -f text_filter ]; then
    echo "Error: Binary was not created"
    exit 1
fi

# Make sure the binary is executable
chmod +x text_filter

# Verify it's statically linked
if ldd text_filter 2>&1 | grep -q "not a dynamic executable"; then
    echo "Success: Static binary created at /workspace/app/text_filter"
else
    echo "Warning: Binary may have dynamic dependencies"
    ldd text_filter || true
fi

echo "Build complete!"