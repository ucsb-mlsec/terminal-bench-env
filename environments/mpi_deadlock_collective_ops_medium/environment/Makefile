CC = mpicc
CFLAGS = -Wall -O2
TARGET = climate_compute
SRC = climate_compute.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean