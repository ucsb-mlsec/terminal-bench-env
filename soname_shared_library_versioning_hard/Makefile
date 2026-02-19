CC = gcc
CFLAGS = -Wall -I./libmath -I./libstring
LDFLAGS = -L./libmath -L./libstring -lmath -lstring -lm

.PHONY: all clean run

all: test_program

test_program: test_program.c
	$(CC) $(CFLAGS) -o test_program test_program.c $(LDFLAGS)

run: test_program
	LD_LIBRARY_PATH=./libmath:./libstring:$$LD_LIBRARY_PATH ./test_program

clean:
	rm -f test_program