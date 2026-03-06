CC=gcc
CFLAGS=-Wall -fPIC
LIBTOOL=libtool
LDFLAGS=-lm

# Library version - TODO: is this the right format?
VERSION=1:0:0

# Target library name
LIBNAME=libmathops.la

all: $(LIBNAME) test_mathlib

# Compile mathops.c - not sure if I need libtool here?
mathops.o: mathops.c mathops.h
	$(CC) $(CFLAGS) -c mathops.c -o mathops.o

# Link the library - trying to use libtool but getting errors
$(LIBNAME): mathops.o
	$(CC) -shared mathops.o -o libmathops.so $(LDFLAGS)

# Build test program - link errors happening here
test_mathlib: test_mathlib.c $(LIBNAME)
	$(CC) $(CFLAGS) test_mathlib.c -o test_mathlib -L. -lmathops $(LDFLAGS)

# Clean up build files
clean:
	rm -f *.o *.lo *.la test_mathlib libmathops.so
	rm -rf .libs

# Notes from previous dev:
# - libtool should create .lo files instead of .o?
# - need to use libtool --mode=compile and --mode=link?
# - version-info flag goes somewhere but where exactly?
# - test program can't find library at runtime even when it compiles

.PHONY: all clean