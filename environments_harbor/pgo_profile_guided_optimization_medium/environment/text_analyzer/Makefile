CC=gcc
CFLAGS=-O2 -Wall -std=c99
SRCDIR=src
TARGET=analyzer

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCDIR)/analyzer.c
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCDIR)/analyzer.c

clean:
	rm -f $(TARGET) *.o *.gcda *.gcno