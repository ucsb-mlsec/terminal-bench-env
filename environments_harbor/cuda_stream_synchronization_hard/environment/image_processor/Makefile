CC=gcc
CFLAGS=-Wall -pthread -O2
LDFLAGS=-lpthread
TARGET=image_processor
SRC=image_processor.c
OBJ=$(SRC:.c=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
	rm -rf processed_images

test: $(TARGET)
	mkdir -p processed_images
	./$(TARGET) test_images processed_images
	@echo "Processing complete. Check processed_images directory for results."
	@echo "Running corruption detection..."
	@./$(TARGET) test_images processed_images
	@./$(TARGET) test_images processed_images
	@./$(TARGET) test_images processed_images
	@echo "Multiple test runs completed."

run: $(TARGET)
	mkdir -p processed_images
	./$(TARGET) test_images processed_images