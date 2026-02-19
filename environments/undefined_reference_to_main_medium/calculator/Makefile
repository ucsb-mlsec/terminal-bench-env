CC=gcc
CFLAGS=-Wall -g

all: calculator

calculator: src/calculator.c src/operations.c
	$(CC) $(CFLAGS) -o calculator src/calculator.c src/operations.c

clean:
	rm -f *.o calculator

.PHONY: all clean