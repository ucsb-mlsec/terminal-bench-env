CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -lm
AR = ar
ARFLAGS = rcs

OBJS = matrix.o vector.o trigonometry.o statistics.o
BENCHMARK_OBJS = $(OBJS) benchmark.o

all: benchmark

benchmark: $(BENCHMARK_OBJS)
	$(CC) $(CFLAGS) -o benchmark $(BENCHMARK_OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

libmath.a: $(OBJS)
	$(AR) $(ARFLAGS) libmath.a $(OBJS)

clean:
	rm -f *.o benchmark libmath.a

.PHONY: all clean