CC=gcc
CFLAGS=-fopenmp -O2 -Wall -lm

all: process

process: process.c
	$(CC) $(CFLAGS) -o process process.c

run: process
	./process

clean:
	rm -f process *.o

.PHONY: all run clean