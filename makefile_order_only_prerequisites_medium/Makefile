CC = gcc
CFLAGS = -Iinclude -Wall

all: bin/program

bin/program: build/main.o build/utils.o | bin/
	$(CC) build/main.o build/utils.o -o bin/program

build/main.o: src/main.c include/utils.h | build/
	$(CC) $(CFLAGS) -c src/main.c -o build/main.o

build/utils.o: src/utils.c include/utils.h | build/
	$(CC) $(CFLAGS) -c src/utils.c -o build/utils.o

build/:
	mkdir -p build

bin/:
	mkdir -p bin

clean:
	rm -rf build bin

.PHONY: all clean