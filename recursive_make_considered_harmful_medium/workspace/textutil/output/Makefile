# Legacy recursive makefile for output module
CC = gcc
CFLAGS = -I../include -Wall

all: output.o

output.o: output.c
	$(CC) $(CFLAGS) -c output.c -o output.o

clean:
	rm -f *.o