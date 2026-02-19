.PHONY: all clean

CXX = g++
CXXFLAGS = -std=c++11 -pthread -O2 -Wall
TARGET = processor
SRC = src/processor.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)