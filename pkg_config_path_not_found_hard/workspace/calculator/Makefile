CXX = g++
CXXFLAGS = -Wall -std=c++11 $(shell pkg-config --cflags mathutils)
LDFLAGS = $(shell pkg-config --libs mathutils)
TARGET = calculator
SOURCES = main.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean