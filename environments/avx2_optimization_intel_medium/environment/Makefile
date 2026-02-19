CC = gcc
CFLAGS = -Wall -O2
LDFLAGS =

TARGET = test_suite
LIB = libmath.a
OBJS = matrix.o vector.o test_suite.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

matrix.o: matrix.c
	$(CC) $(CFLAGS) -c matrix.c

vector.o: vector.c
	$(CC) $(CFLAGS) -c vector.c

test_suite.o: test_suite.c
	$(CC) $(CFLAGS) -c test_suite.c

clean:
	rm -f *.o $(TARGET) $(LIB)

.PHONY: all clean