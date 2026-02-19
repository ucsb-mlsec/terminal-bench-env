CC = g++
CFLAGS = -Wall -Iinclude
AR = ar
ARFLAGS = rcs

SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

SOURCES = $(SRC_DIR)/addition.cpp $(SRC_DIR)/subtraction.cpp $(SRC_DIR)/multiplication.cpp $(SRC_DIR)/division.cpp
OBJECTS = $(BUILD_DIR)/addition.o $(BUILD_DIR)/subtraction.o $(BUILD_DIR)/multiplication.o $(BUILD_DIR)/division.o

LIBRARY = $(BUILD_DIR)/libcalculator.a
TEST_EXEC = test_calculator

.PHONY: all clean

all: $(TEST_EXEC)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBRARY): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(TEST_EXEC): test_main.cpp $(LIBRARY)
	$(CC) $(CFLAGS) test_main.cpp -L$(BUILD_DIR) -lcalculator -o $@

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TEST_EXEC)