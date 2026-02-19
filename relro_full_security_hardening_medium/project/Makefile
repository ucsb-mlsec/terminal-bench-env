CC=gcc
CFLAGS=-Wall -O2
LDFLAGS=

.PHONY: all clean

all: server client libutils.so

libutils.so: utils.c
	$(CC) -shared -o libutils.so utils.c

server: server.c libutils.so
	$(CC) $(CFLAGS) -o server server.c -L. -lutils

client: client.c libutils.so
	$(CC) $(CFLAGS) -o client client.c -L. -lutils

clean:
	rm -f *.o *.so server client