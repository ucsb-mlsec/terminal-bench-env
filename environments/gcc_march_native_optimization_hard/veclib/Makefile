CC = gcc
CFLAGS = -march=x86-64 -O2

all: benchmark

compute.o: compute.c
	$(CC) $(CFLAGS) -c compute.c -o compute.o

benchmark.o: benchmark.c
	$(CC) $(CFLAGS) -c benchmark.c -o benchmark.o

benchmark: compute.o benchmark.o
	$(CC) $(CFLAGS) compute.o benchmark.o -o benchmark -lm

clean:
	rm -f *.o benchmark