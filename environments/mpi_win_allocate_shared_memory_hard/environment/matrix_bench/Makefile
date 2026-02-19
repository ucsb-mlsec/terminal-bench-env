CC = mpicc
CFLAGS = -O2 -Wall
LDFLAGS = -lm
TARGET = matrix_benchmark
SRCS = matrix_benchmark.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

run:
	mpirun -np 4 ./$(TARGET)