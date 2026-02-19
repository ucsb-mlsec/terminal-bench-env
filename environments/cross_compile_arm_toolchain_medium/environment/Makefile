CC = gcc
CFLAGS = -O2 -Wall
TARGET = sensor_logger

all: $(TARGET)

$(TARGET): sensor_logger.c
	$(CC) $(CFLAGS) -o $(TARGET) sensor_logger.c

clean:
	rm -f $(TARGET) *.o