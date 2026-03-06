CC=gcc
CFLAGS=-Wall -O2
SOURCES=main.c utils.c
OBJECTS=main.o utils.o
TARGET=cli-tool

all: $(TARGET)

$(TARGET): $(OBJECTS)
    $(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

main.o: main.c
    $(CC) $(CFLAGS) -c main.c

utils.o: utils.c
    $(CC) $(CFLAGS) -c utils.c

clean:
    rm -f $(TARGET) $(OBJECTS)