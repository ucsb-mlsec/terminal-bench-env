CC=gcc
CFLAGS=-Wall -g
OBJS = network.o

all: libnetwork.a

libnetwork.a: $(OBJS)
	ar rcs libnetwork.a $(OBJS)

network.o: network.c network.h
        $(CC) $(CFLAGS) -c network.c

clean:
	rm -f *.o *.a