CC = gcc
CFLAGS = -fPIC -Wall
LIBDIR = ../lib
LIBNAME = libmathops

all: libmathops.so test_program

mathops.o: mathops.c mathops.h
	$(CC) $(CFLAGS) -c mathops.c -o mathops.o

libmathops.so: mathops.o
	$(CC) -shared -o libmathops.so mathops.o

test_program: test_program.c libmathops.so
	$(CC) -o test_program test_program.c -L. -lmathops -Wl,-rpath,/workspace/lib

install: libmathops.so
	mkdir -p /workspace/lib
	cp libmathops.so /workspace/lib/

clean:
	rm -f *.o *.so test_program
	rm -f /workspace/lib/libmathops.so*