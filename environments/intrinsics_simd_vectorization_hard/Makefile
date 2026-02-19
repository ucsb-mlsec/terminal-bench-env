CC=gcc
CFLAGS=-O0 -Wall
TARGET=compute

all: $(TARGET)

$(TARGET): compute.c
	$(CC) $(CFLAGS) -o $(TARGET) compute.c

clean:
	rm -f $(TARGET)