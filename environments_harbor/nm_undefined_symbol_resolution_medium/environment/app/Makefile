CC=gcc
CFLAGS=-Wall

all: myapp

myapp: main.o
	$(CC) -o myapp main.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f *.o myapp