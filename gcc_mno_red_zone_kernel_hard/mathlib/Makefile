CC = gcc
CFLAGS = -fPIC -O2 -Wall

all: libcompute.so

libcompute.o: libcompute.c
	$(CC) $(CFLAGS) -c libcompute.c -o libcompute.o

libcompute.so: libcompute.o
	$(CC) -shared libcompute.o -o libcompute.so

clean:
	rm -f *.o *.so

.PHONY: all clean