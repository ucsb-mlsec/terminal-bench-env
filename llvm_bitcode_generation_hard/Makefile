CC=gcc
CFLAGS=-Wall -O2
LDFLAGS=

OBJECTS=vector.o matrix.o stats.o
TARGET=test

.PHONY: all clean

all: $(TARGET)

vector.o: vector.c mathlib.h
	$(CC) $(CFLAGS) -c vector.c -o vector.o

matrix.o: matrix.c mathlib.h
	$(CC) $(CFLAGS) -c matrix.c -o matrix.o

stats.o: stats.c mathlib.h
	$(CC) $(CFLAGS) -c stats.c -o stats.o

$(TARGET): $(OBJECTS) test.c
	$(CC) $(CFLAGS) test.c $(OBJECTS) -o $(TARGET) $(LDFLAGS) -lm

clean:
	rm -f $(OBJECTS) $(TARGET)