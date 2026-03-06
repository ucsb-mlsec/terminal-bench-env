CC=g++
CFLAGS=-Wall -g -I../common

all: libnetwork.a

socket.o: socket.cpp socket.h
	$(CC) $(CFLAGS) -c socket.cpp -o socket.o

libnetwork.a: socket.o
	ar rcs libnetwork.a socket.o

clean:
	rm -f *.o *.a