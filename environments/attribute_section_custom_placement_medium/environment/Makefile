CC = gcc
CFLAGS = -Wall -fPIC -O2
LDFLAGS = -shared
TARGET = libsensor.so
SRC = sensor.c
OBJ = sensor.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)