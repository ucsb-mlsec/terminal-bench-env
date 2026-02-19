CC = gcc
CFLAGS = -fPIC -Wall
LDFLAGS = -shared

all: libgraphics.so

libgraphics.so: graphics.o
	$(CC) $(LDFLAGS) -o libgraphics.so graphics.o

graphics.o: graphics.c
	$(CC) $(CFLAGS) -c graphics.c

clean:
	rm -f *.o *.so