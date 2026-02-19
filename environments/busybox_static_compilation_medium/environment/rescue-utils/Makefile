CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = 
TARGET = bin/rescue
SRC = rescue.c

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p bin
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(SRC)
	strip $(TARGET)

clean:
	rm -rf bin

.PHONY: all clean