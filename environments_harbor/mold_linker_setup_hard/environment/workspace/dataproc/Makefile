CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
LDFLAGS = 
LIBS = -lpthread -lm

SOURCES = main.cpp parser.cpp filter.cpp aggregator.cpp transformer.cpp validator.cpp writer.cpp config.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = dataproc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)