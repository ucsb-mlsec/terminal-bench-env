CC = mpicc
CFLAGS = -Wall -O2
TARGET = aggregator

.PHONY: all clean

all: $(TARGET)

$(TARGET): aggregator.c sensor_data.h
	$(CC) $(CFLAGS) -o $(TARGET) aggregator.c

clean:
	rm -f $(TARGET) results.txt