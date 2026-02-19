# Makefile for dataproc application
# WARNING: This Makefile contains circular dependencies that need to be fixed!

CC = gcc
CFLAGS = -Wall -g
TARGET = dataproc
OBJS = main.o processor.o utils.o

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Circular dependency issue: main.o depends on processor.o
main.o: main.c main.h processor.o
	$(CC) $(CFLAGS) -c main.c

# Circular dependency issue: processor.o depends on utils.o
processor.o: processor.c processor.h utils.o
	$(CC) $(CFLAGS) -c processor.c

# Circular dependency issue: utils.o depends on processor.o
utils.o: utils.c utils.h processor.o
	$(CC) $(CFLAGS) -c utils.c

# Additional header dependencies that create more circular issues
main.h: processor.h

processor.h: utils.h

utils.h: processor.h

# Clean target
clean:
	rm -f *.o $(TARGET)

# Phony targets
.PHONY: all clean