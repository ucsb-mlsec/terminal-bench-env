CC=gcc
CFLAGS=-std=c99 -Werror -Wall -Wextra -pedantic
TARGET=sensor_system
OBJS=main.o processor.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS)

%.o: %.c sensor.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)