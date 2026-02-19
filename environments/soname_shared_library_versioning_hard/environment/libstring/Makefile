CC = gcc
CFLAGS = -fPIC -Wall -I../libmath
LDFLAGS = -L../libmath -lmath
TARGET = libstring.so
SOURCES = string_utils.c
OBJECTS = string_utils.o

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -shared -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	cp $(TARGET) /usr/local/lib
	ldconfig

clean:
	rm -f $(OBJECTS) $(TARGET)