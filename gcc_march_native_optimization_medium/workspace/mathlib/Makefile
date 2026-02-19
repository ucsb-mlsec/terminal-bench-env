# Makefile for mathlib computational library

# Optimization flags - can be overridden to enable CPU-specific optimizations
OPTIMIZE_FLAGS =

# Compiler configuration
CC = gcc
CFLAGS = -O2 -fPIC -Wall $(OPTIMIZE_FLAGS)

# Library name
TARGET = libmathlib.so
SOURCE = vector_ops.c

# Default target
all: $(TARGET)

# Build shared library
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -shared -o $(TARGET) $(SOURCE)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean