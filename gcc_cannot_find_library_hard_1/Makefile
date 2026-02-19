CC=gcc
CFLAGS=-Wall -O2
LDFLAGS=

all: analyzer

analyzer: analyzer.o
	$(CC) $(LDFLAGS) -o analyzer analyzer.o

analyzer.o: analyzer.c
	$(CC) $(CFLAGS) -c analyzer.c

clean:
	rm -f *.o analyzer

.PHONY: all clean