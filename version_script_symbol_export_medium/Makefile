CC = gcc
CFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared
TARGET = libcalculator.so
SRC = calculator.c
OBJ = calculator.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)