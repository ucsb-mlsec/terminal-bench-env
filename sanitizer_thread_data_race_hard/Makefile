CC = gcc
CFLAGS = -pthread -Wall
SANITIZE_FLAGS = -fsanitize=thread -g

.PHONY: clean

all: counter

counter: counter.c
	$(CC) $(CFLAGS) counter.c -o counter

sanitize: counter.c
	$(CC) $(CFLAGS) $(SANITIZE_FLAGS) counter.c -o counter

tsan: counter.c
	$(CC) $(CFLAGS) $(SANITIZE_FLAGS) counter.c -o counter

clean:
	rm -f counter