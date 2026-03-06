CC = gcc
CFLAGS = -fopenmp -O2 -Wall

.PHONY: all clean

all: histogram

histogram: histogram.c
	$(CC) $(CFLAGS) -o histogram histogram.c

histogram_fixed: histogram_fixed.c
	$(CC) $(CFLAGS) -o histogram_fixed histogram_fixed.c

clean:
	rm -f histogram histogram_fixed *.o