CC=gcc
CFLAGS=-Wall -O2
TARGET=dbutil
SRC=main.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o test.db