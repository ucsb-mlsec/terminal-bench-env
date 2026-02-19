CC = gcc
CFLAGS = -Wall -fPIC
TARGET = libmath.so
SRC = mathlib.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -shared $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o