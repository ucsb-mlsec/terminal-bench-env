CC = gcc
CFLAGS = -fPIC -Wall
LDFLAGS = -shared

libmath.so.1: math_lib.c
	$(CC) $(CFLAGS) math_lib.c
	$(LDFLAGS) -Wl,--version-script=libmath.map -o libmath.so.1

clean:
	rm -f *.so.* *.o

all: libmath.so.1