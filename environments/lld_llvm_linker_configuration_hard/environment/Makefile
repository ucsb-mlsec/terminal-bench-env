CXX=clang++
LD=ld
CXXFLAGS=-std=c++11 -Wall -fPIC
LDFLAGS=--no-as-needed -rpath-link=/usr/lib -z relro -z now --build-id
LIB_DIR=lib
SRC_DIR=src
BIN_DIR=bin
MATH_DIR=$(LIB_DIR)/math
UTILS_DIR=$(LIB_DIR)/utils

MATH_LIB=$(MATH_DIR)/liboperations.so
UTILS_LIB=$(UTILS_DIR)/libhelpers.so
EXECUTABLE=$(BIN_DIR)/calculator

all: $(EXECUTABLE)

$(MATH_DIR)/operations.o: $(MATH_DIR)/operations.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MATH_LIB): $(MATH_DIR)/operations.o
	$(CXX) -shared $(LDFLAGS) -o $@ $^

$(UTILS_DIR)/helpers.o: $(UTILS_DIR)/helpers.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(UTILS_LIB): $(UTILS_DIR)/helpers.o
	$(CXX) -shared $(LDFLAGS) -o $@ $^

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(SRC_DIR)/main.o $(MATH_LIB) $(UTILS_LIB)
	mkdir -p $(BIN_DIR)
	$(CXX) $(LDFLAGS) -o $@ $< -L$(MATH_DIR) -L$(UTILS_DIR) -loperations -lhelpers -Wl,-rpath=$(MATH_DIR) -Wl,-rpath=$(UTILS_DIR)

clean:
	rm -f $(MATH_DIR)/*.o $(MATH_DIR)/*.so
	rm -f $(UTILS_DIR)/*.o $(UTILS_DIR)/*.so
	rm -f $(SRC_DIR)/*.o
	rm -f $(EXECUTABLE)
	rm -rf $(BIN_DIR)

.PHONY: all clean