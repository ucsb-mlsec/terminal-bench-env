CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -pthread -O2

all: test_stack

lockfree_stack.o: lockfree_stack.c lockfree_stack.h
	$(CC) $(CFLAGS) -c lockfree_stack.c -o lockfree_stack.o

test_stack: lockfree_stack.o
	$(CC) $(CFLAGS) lockfree_stack.o -o test_stack

clean:
	rm -f *.o test_stack