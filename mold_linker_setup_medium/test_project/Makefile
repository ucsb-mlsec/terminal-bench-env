CXX=g++
CXXFLAGS=-Wall -std=c++17
TARGET=test_app
SRC=main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean