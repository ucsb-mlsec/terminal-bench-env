CC = gcc
CFLAGS = -fopenmp -Wall -O2
TARGET = histogram

all: $(TARGET)

$(TARGET): histogram.c
	$(CC) $(CFLAGS) -o $(TARGET) histogram.c -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean