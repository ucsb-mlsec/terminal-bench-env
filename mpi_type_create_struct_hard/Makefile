CC = mpicc
CFLAGS = -Wall -O2
TARGET = weather_processor

.PHONY: all clean run

all: $(TARGET)

$(TARGET): weather_processor.c
	$(CC) $(CFLAGS) -o $(TARGET) weather_processor.c

clean:
	rm -f $(TARGET) *.o

run: $(TARGET)
	mpirun -np 2 ./$(TARGET)