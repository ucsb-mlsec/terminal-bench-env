CC = gcc
CFLAGS = -Wall -Wextra -I/usr/include
LDFLAGS = -lGL -lm
TARGET = gl_test
SRC = gl_test.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean