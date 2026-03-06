# Legacy recursive makefile for formatter module
CC = gcc
CFLAGS = -I../include -Wall

all: formatter.o

formatter.o: formatter.c
	$(CC) $(CFLAGS) -c formatter.c -o formatter.o

clean:
	rm -f *.o