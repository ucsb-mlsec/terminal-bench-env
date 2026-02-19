CC=g++
CFLAGS=-Wall -g

all: libutils.a

utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) -c utils.cpp -o utils.o

libutils.a: utils.o
	ar rcs libutils.a utils.o

clean:
	rm -f *.o *.a