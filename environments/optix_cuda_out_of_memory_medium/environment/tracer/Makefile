CXX = g++
CXXFLAGS = -std=c++11 -Wall -g
TARGET = tracer
SRCDIR = src
OBJDIR = src

SOURCES = $(SRCDIR)/main.cpp $(SRCDIR)/processor.cpp $(SRCDIR)/utils.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)