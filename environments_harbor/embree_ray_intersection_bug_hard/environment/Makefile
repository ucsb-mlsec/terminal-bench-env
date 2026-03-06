CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
SRC_DIR = src
BIN_DIR = bin

.PHONY: all clean

all: $(BIN_DIR)/test_intersect

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/test_intersect: $(SRC_DIR)/intersect.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/intersect.cpp -o $(BIN_DIR)/test_intersect

clean:
	rm -rf $(BIN_DIR)