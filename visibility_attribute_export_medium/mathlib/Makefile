CC=gcc
CFLAGS=-Wall -fPIC
LDFLAGS=-shared

all: libmath.so test_mathlib

libmath.so: mathlib.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o libmath.so mathlib.c

test_mathlib: test_mathlib.c libmath.so
	$(CC) $(CFLAGS) -o test_mathlib test_mathlib.c -L. -lmath -Wl,-rpath,.

clean:
	rm -f libmath.so test_mathlib