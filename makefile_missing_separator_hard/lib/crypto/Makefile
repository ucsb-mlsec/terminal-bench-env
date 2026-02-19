CC=gcc
CFLAGS=-Wall -g
OBJS = crypto.o

all: libcrypto.a

libcrypto.a: $(OBJS)
	ar rcs libcrypto.a $(OBJS)

crypto.o: crypto.c crypto.h
    $(CC) $(CFLAGS) -c crypto.c

clean:
	rm -f *.o *.a