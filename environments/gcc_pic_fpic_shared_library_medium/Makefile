CC=gcc
CFLAGS=-Wall
LDFLAGS=-L. -ltools -Wl,-rpath,.

all: app

tools.o: tools.c
	$(CC) -c tools.c -o tools.o

libtools.so: tools.o libcalc.a
	$(CC) -shared -o libtools.so tools.o libcalc.a

app: app.c libtools.so
	$(CC) app.c -o app $(LDFLAGS)

clean:
	rm -f app *.o *.so

.PHONY: all clean