CC=gcc
CFLAGS=-Wall -Wextra -O2
TARGET=calculator

all: $(TARGET)

$(TARGET): calculator.c
	$(CC) $(CFLAGS) -o $(TARGET) calculator.c

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean