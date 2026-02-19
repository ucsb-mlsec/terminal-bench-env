CC = gcc
CFLAGS = -Wall -Wextra -O2
TARGET = filestat
OBJS = main.o processor.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.c processor.h
	$(CC) $(CFLAGS) -c main.c

processor.o: processor.c processor.h
	$(CC) $(CFLAGS) -c processor.c

clean:
	rm -f $(OBJS) $(TARGET)