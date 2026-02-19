# Legacy recursive makefile for parser module
CC = gcc
CFLAGS = -I../include -Wall

all: parser.o

parser.o: parser.c
	$(CC) $(CFLAGS) -c parser.c -o parser.o

clean:
	rm -f *.o