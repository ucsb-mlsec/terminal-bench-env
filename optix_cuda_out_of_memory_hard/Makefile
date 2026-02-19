CC=gcc
CFLAGS=-O0 -w
TARGET=mem_profiler

all: $(TARGET)

$(TARGET): mem_profiler.c
	$(CC) $(CFLAGS) mem_profiler.c $(TARGET) -o

clean:
	rm -f $(TARGET)