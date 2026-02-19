CC = gcc
CFLAGS = -Wall
LDFLAGS =

BUILD_DIR = build
SOURCES = main.c utils.c data_tables.c
OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/data_tables.o
TARGET = $(BUILD_DIR)/app

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean