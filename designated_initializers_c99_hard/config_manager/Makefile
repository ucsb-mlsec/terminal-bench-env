CC=gcc
CFLAGS=-std=c99 -Wall -Wextra -Werror
SOURCES=config_core.c config_loader.c device_manager.c protocol_handler.c system_init.c
OBJECTS=$(SOURCES:.c=.o)
TARGET=config_test

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET)
	@echo "Running test suite..."
	@./$(TARGET)
	@if [ $$? -eq 0 ]; then \
		echo "All tests passed!"; \
	else \
		echo "Tests failed!"; \
		exit 1; \
	fi

clean:
	rm -f $(OBJECTS) $(TARGET)