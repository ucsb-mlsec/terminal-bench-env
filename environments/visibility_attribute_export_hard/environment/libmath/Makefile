.PHONY: all clean

CC = gcc
CFLAGS = -Wall -fPIC
LDFLAGS = -shared

all: libmath.so

libmath.so: math_operations.o internal_helpers.o
	$(CC) $(LDFLAGS) -o libmath.so math_operations.o internal_helpers.o

math_operations.o: math_operations.c
	$(CC) $(CFLAGS) -c math_operations.c

internal_helpers.o: internal_helpers.c
	$(CC) $(CFLAGS) -c internal_helpers.c

clean:
	rm -f *.o *.so