CC = gcc
CFLAGS = -Wall -O2

.PHONY: all clean

all: test_a test_b

test_a: performance_test.c implementation_a.c text_buffer.h
	$(CC) $(CFLAGS) -o test_a performance_test.c implementation_a.c

test_b: performance_test.c implementation_b.c text_buffer.h
	$(CC) $(CFLAGS) -o test_b performance_test.c implementation_b.c

clean:
	rm -f test_a test_b