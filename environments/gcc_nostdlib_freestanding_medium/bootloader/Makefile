CC = gcc
BINARY = boot.elf
CFLAGS = -m64 -c
LDFLAGS = -T linker.ld

all: $(BINARY)

$(BINARY): boot.o
	$(CC) $(LDFLAGS) -o $(BINARY) boot.o

boot.o: boot.c
	$(CC) $(CFLAGS) boot.c -o boot.o

clean:
	rm -f *.o $(BINARY)

.PHONY: all clean