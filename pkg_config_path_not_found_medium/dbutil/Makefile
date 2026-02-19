CC=gcc
CFLAGS=$(shell pkg-config --cflags sqlite3)
LIBS=$(shell pkg-config --libs sqlite3)

all: dbutil

dbutil: dbutil.c
	$(CC) $(CFLAGS) -o dbutil dbutil.c $(LIBS)

clean:
	rm -f dbutil *.o