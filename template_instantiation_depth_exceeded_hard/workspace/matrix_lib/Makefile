CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
TARGET = test_matrix
SOURCES = test_matrix.cpp
HEADERS = matrix.hpp operations.hpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)