# Makefile for vecmath library
# TODO: This needs to be fixed to use libtool properly!

CC=gcc
CFLAGS=-fPIC -Wall -O2
LDFLAGS=-shared

# Source files
SRC=src/vector.c
OBJ=vector.o
HEADER=src/vector.h

# Library name
LIBNAME=libvecmath.so

# Default target
all: $(LIBNAME)

# Compile source to object - TODO: should use libtool
vector.o: $(SRC) $(HEADER)
	$(CC) $(CFLAGS) -c src/vector.c -o vector.o

# Create shared library - BROKEN: not using libtool link mode
$(LIBNAME): $(OBJ)
	$(CC) $(LDFLAGS) -o $(LIBNAME) $(OBJ)
	# Missing version info and proper library naming

# Install target - INCOMPLETE
install:
	# TODO: need to set up prefix and install directories
	# TODO: need to install headers too
	cp $(LIBNAME) /usr/local/lib/

# Test compilation - probably won't work
test: tests/test_vector.c
	$(CC) -o test_vector tests/test_vector.c -I. -L. -lvecmath

clean:
	rm -f $(OBJ) $(LIBNAME) test_vector

.PHONY: all install test clean