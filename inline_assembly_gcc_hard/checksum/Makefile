CC = gcc
CFLAGS = -O2 -Wall -std=c99
TARGET = validator

all: $(TARGET)

$(TARGET): validator.c
	$(CC) $(CFLAGS) -o $(TARGET) validator.c

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all run clean