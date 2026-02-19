CC=clang
CFLAGS=-Wall -Wextra -O2
TARGET=data_processor
SOURCES=main.c utils.c
OBJECTS=main.o utils.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean