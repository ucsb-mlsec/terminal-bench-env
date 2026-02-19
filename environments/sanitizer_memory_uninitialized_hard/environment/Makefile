CC = gcc
CFLAGS = -Wall -g
TARGET = data_processor

all: $(TARGET)

$(TARGET): data_processor.c
	$(CC) $(CFLAGS) -o $(TARGET) data_processor.c

clean:
	rm -f $(TARGET) *.o