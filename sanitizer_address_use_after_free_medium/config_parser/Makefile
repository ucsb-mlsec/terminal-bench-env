CC = gcc
CFLAGS = -Wall -g
TARGET = parser

.PHONY: all clean

all: $(TARGET)

$(TARGET): parser.c
	$(CC) $(CFLAGS) -o $(TARGET) parser.c

clean:
	rm -f $(TARGET) *.o