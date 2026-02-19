CC=gcc
CFLAGS=-Wall -Wextra -I./include
TARGET=program
SRC_DIR=src
UTILS_DIR=src/utils
CORE_DIR=src/core

# Object files - but incomplete!
MAIN_OBJ=$(SRC_DIR)/main.o
UTILS_OBJ=$(UTILS_DIR)/helpers.o
# Missing core objects!

# Wrong: only listing some objects
OBJS=$(MAIN_OBJ)

all: $(TARGET)

# Broken link command - missing required object files
$(TARGET): $(MAIN_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(MAIN_OBJ)

# Compile main
$(SRC_DIR)/main.o: $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) -c $(SRC_DIR)/main.c -o $(SRC_DIR)/main.o

# Compile utils but never link them
$(UTILS_DIR)/helpers.o: $(UTILS_DIR)/helpers.c
	$(CC) $(CFLAGS) -c $(UTILS_DIR)/helpers.c -o $(UTILS_DIR)/helpers.o

# Core files not even being compiled!
# Missing rules for src/core/engine.c
# Missing rules for src/core/processor.c

clean:
	rm -f $(SRC_DIR)/*.o $(UTILS_DIR)/*.o $(CORE_DIR)/*.o $(TARGET)

.PHONY: all clean