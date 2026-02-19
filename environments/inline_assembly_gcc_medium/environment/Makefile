.PHONY: all clean run

CC = gcc
CFLAGS = -O2 -Wall -march=native
LDFLAGS = -lm

all: test_vecsum

test_vecsum: test_vecsum.o vecsum.o
	$(CC) -o test_vecsum test_vecsum.o vecsum.o $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o test_vecsum

run: test_vecsum
	./test_vecsum