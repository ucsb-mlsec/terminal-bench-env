CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c99
SOURCES = filter.c equalizer.c compressor.c main.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = processor

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean