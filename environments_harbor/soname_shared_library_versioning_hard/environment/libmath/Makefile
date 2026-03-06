CC = gcc
CFLAGS = -fPIC -Wall
TARGET = libmath.so
SOURCES = math_utils.c
OBJECTS = math_utils.o

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -shared -o $(TARGET) $(OBJECTS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	ldconfig

clean:
	rm -f $(OBJECTS) $(TARGET)