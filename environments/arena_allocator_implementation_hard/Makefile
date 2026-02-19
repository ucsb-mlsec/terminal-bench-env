CC=gcc
CFLAGS=-O3 -Wall -Wextra -std=c99
LDFLAGS=-lm

all: process

process: process.c
	$(CC) $(CFLAGS) -o process process.c $(LDFLAGS)

clean:
	rm -f process *.o

.PHONY: all clean