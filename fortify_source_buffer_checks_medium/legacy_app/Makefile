CC=gcc
CFLAGS=-O2 -D_FORTIFY_SOURCE=2 -Wall -Wextra
SOURCES=main.c utils.c
OBJECTS=main.o utils.o
TARGET=app

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean