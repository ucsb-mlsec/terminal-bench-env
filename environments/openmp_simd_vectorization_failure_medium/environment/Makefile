CC = gcc
CFLAGS = -O3 -fopenmp -march=native -fopt-info-vec-optimized -fopt-info-vec-missed
LDFLAGS = -fopenmp -lm

TARGET = matrix_compute
SRC = matrix_compute.c
OBJ = matrix_compute.o

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) output.txt