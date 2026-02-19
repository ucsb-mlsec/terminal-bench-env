CC=gcc
CFLAGS=-I./include -Wall
ARCH=x86
SOURCES=memory.c processor.c utils.c main.c
OBJECTS=$(SOURCES:.c=.o)

all: program

program: $(OBJECTS)
	$(CC) -o program $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o program