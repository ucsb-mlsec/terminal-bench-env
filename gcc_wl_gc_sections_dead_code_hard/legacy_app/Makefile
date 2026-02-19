CC=gcc
CFLAGS=-Wall -O2
LDFLAGS=

all: app

app: main.o utils.o config.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o app main.o utils.o config.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c

config.o: config.c
	$(CC) $(CFLAGS) -c config.c

test: app
	./app || exit 1

clean:
	rm -f *.o app