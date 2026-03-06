CC=g++
CFLAGS=-Wall -g -I../common -I../network
LDFLAGS=-L../common -L../network -lnetwork -lutils

all: app

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp -o main.o

app: main.o
	$(CC) main.o $(LDFLAGS) -o app

clean:
	rm -f *.o app