CC=gcc
CFLAGS=-Wall -g
LIBDIRS = lib/crypto lib/network
LIBS = lib/crypto/libcrypto.a lib/network/libnetwork.a

all: netutil

netutil: src/main.o $(LIBS)
	$(CC) -o netutil src/main.o $(LIBS)

src/main.o: src/main.c
        $(CC) $(CFLAGS) -c src/main.c -o src/main.o

libs:
	@for dir in $(LIBDIRS); do \
		$(MAKE) -C $$dir; \
	done

$(LIBS): libs

clean:
	rm -f netutil src/main.o
	@for dir in $(LIBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

.PHONY: all clean libs