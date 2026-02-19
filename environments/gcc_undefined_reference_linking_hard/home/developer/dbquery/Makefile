CC=gcc
CFLAGS=-Wall -Wextra -O2
LDFLAGS=-lsqlite3

all: dbquery

dbquery.o: dbquery.c
	$(CC) $(CFLAGS) -c dbquery.c -o dbquery.o

dbquery: dbquery.o
	$(CC) $(LDFLAGS) dbquery.o -o dbquery

clean:
	rm -f *.o dbquery