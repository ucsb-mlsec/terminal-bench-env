CC=gcc
CFLAGS=-Wall -g

all: logparser

logparser: logparser.c
	$(CC) $(CFLAGS) -o logparser logparser.c

clean:
	rm -f logparser *.o

.PHONY: all clean