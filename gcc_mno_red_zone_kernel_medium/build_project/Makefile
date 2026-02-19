CC=gcc
CFLAGS=-Wall -O2
SOURCES=main.c utils.c helpers.c
OBJECTS=$(SOURCES:.c=.o)
TARGET=program

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean