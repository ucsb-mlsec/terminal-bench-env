CC = gcc
CFLAGS = -fopenmp -O3 -Wall
LDFLAGS = -lm
TARGET = wave_sim

.PHONY: all clean

all: $(TARGET)

$(TARGET): wave_sim.c
	$(CC) $(CFLAGS) -o $(TARGET) wave_sim.c $(LDFLAGS)

clean:
	rm -f $(TARGET)