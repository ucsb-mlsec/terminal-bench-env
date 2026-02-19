# Makefile for Curve Rasterizer
# Compiles the Bézier curve rendering library and runs validation tests

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm
TARGET = rasterizer
SRC = rasterizer.c
OBJ = $(SRC:.c=.o)
OUTPUT_DIR = output
TEST_DATA = test_curves.json
VALIDATOR = validate.py

.PHONY: all clean test

# Default target - build the rasterizer
all: $(TARGET)

# Build the rasterizer executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Compile object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run test suite and validate output
test: $(TARGET)
	@mkdir -p $(OUTPUT_DIR)
	@echo "Running rasterizer on test curves..."
	./$(TARGET) $(TEST_DATA) $(OUTPUT_DIR)
	@echo "Validating output against references..."
	python3 $(VALIDATOR)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJ)
	rm -rf $(OUTPUT_DIR)/*.bmp