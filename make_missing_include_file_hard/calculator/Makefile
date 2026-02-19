# Calculator Project Makefile

# Include configuration files
include config/compiler.mk
include config/flags.mk

# Project settings
TARGET = calculator
SRCDIR = src
OBJDIR = obj

# Source files
SOURCES = $(SRCDIR)/calculator.c
OBJECTS = $(OBJDIR)/calculator.o

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(TARGET)

# Compile source files
$(OBJDIR)/calculator.o: $(SRCDIR)/calculator.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/calculator.c -o $(OBJDIR)/calculator.o

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean