CXX = g++
CXXFLAGS = -Wall -std=c++11 -O2
SOURCES = main.cpp utils.cpp processing.cpp calculations.cpp output.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = program

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean