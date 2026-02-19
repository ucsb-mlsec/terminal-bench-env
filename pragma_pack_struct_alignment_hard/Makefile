CC=gcc
CFLAGS=-Wall -Wextra -std=c99

all: verify_sizes

verify_sizes: verify_sizes.c
	$(CC) $(CFLAGS) -o verify_sizes verify_sizes.c

test: verify_sizes
	./verify_sizes

clean:
	rm -f verify_sizes *.o

.PHONY: all test clean