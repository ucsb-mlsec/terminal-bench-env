CC = gcc
CFLAGS = -fopenmp -O3 -Wall -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed

all: compute

compute: compute.c
	$(CC) $(CFLAGS) -o compute compute.c -lm

clean:
	rm -f compute *.o