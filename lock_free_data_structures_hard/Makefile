CC=gcc
CFLAGS=-Wall -Wextra -O2 -pthread -std=c11 -g
LDFLAGS=-pthread

TARGET=test_harness
OBJS=test_harness.o lockfree_queue.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

test_harness.o: test_harness.c lockfree_queue.h
	$(CC) $(CFLAGS) -c test_harness.c

lockfree_queue.o: lockfree_queue.c lockfree_queue.h
	$(CC) $(CFLAGS) -c lockfree_queue.c

clean:
	rm -f $(TARGET) $(OBJS)