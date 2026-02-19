CXX=g++
CXXFLAGS=-Wall -std=c++11 -O2
SOURCES=main.cpp utils.cpp math_ops.cpp string_ops.cpp data_processor.cpp
OBJECTS=$(SOURCES:.cpp=.o)
TARGET=myapp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)