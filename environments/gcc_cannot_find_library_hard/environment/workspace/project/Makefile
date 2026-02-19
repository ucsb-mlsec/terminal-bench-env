CC=gcc
CFLAGS=-Wall -Wextra -I./include
LDFLAGS=-lmath -lutils
SRC_DIR=src
BUILD_DIR=build
TARGET=$(BUILD_DIR)/app

SOURCES=$(SRC_DIR)/main.c
OBJECTS=$(BUILD_DIR)/main.o

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)