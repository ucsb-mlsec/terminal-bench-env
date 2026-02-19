CC=gcc
CFLAGS=-g -O2

objects=main.o operations.o utils.o

all: calculator

calculator: $(objects)
	$(CC) $(CFLAGS) -o calculator $(objects)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

operations.o: operations.c
	$(CC) $(CFLAGS) -c operations.c

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f *.o calculator