CC=gcc
CFLAGS=-Wall -Wextra -O2
TARGET=sensor_logger

all: $(TARGET)

$(TARGET): data_writer.c sensor_data.h
	$(CC) $(CFLAGS) -o $(TARGET) data_writer.c

clean:
	rm -f $(TARGET) output.bin solution.json

.PHONY: all clean