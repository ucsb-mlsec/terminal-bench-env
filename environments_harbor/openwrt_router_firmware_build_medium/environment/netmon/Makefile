# Makefile for netmon - Network Monitor utility
# Cross-compilation configuration for MIPS architecture

# Compiler and flags
CC=mips-gcc
CFLAGS=-O2 -Wall -static
TARGET=netmon

# Source directory
SRCDIR=src

# Default target
all: $(TARGET)

# Build the netmon binary
$(TARGET): $(SRCDIR)/netmon.c
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCDIR)/netmon.c

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

.PHONY: clean all