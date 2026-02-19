CC = gcc
CFLAGS = -O2 -Wall -Wextra -march=native
LDFLAGS = -lm
TARGET = matrix_ops
SRC = matrix_ops.c

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o output.txt

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET) && echo "Test completed successfully"