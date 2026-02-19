CC = gcc
CFLAGS = -Wall

all: matrix_bench

matrix_bench: matrix_bench.o matrix_ops.o matrix_utils.o
	$(CC) $(CFLAGS) -o matrix_bench matrix_bench.o matrix_ops.o matrix_utils.o -lm

%.o: %.c matrix_ops.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o matrix_bench