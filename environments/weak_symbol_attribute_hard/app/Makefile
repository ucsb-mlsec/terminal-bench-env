CC=gcc
CFLAGS=-Wall -Wextra
SOURCES=main.c audio.c network.c storage.c
OBJECTS=$(SOURCES:.c=.o)
TARGET=app

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)