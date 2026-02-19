CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -lm
TARGET = sensor_processor
SRC = sensor_processor.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean