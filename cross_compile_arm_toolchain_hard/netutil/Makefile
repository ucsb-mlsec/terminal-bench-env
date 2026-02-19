CC=gcc
CFLAGS=-Wall -O2
TARGET=netutil
SRC=netutil.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o