CC=gcc
CXX=g++
CFLAGS=-O3 -march=native -mtune=native -funroll-loops -ffast-math -ffinite-math-only -fno-signed-zeros -fassociative-math -freciprocal-math -funsafe-math-optimizations -Wall
LDFLAGS=-lm

SOURCES=main.c compute.c matrix.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=mathproject

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(EXECUTABLE)
	./run_tests.sh

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

.PHONY: all clean test