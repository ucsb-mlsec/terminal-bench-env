CXX = g++
CXXFLAGS = -std=c++17 -Wall
INCLUDES = -I.
TARGET = main
PCH_HEADER = pch.h
PCH_SOURCE = pch.cpp
PCH_OUTPUT = pch.h.gch

SOURCES = main.cpp
OBJECTS = main.o

.PHONY: all clean

all: $(TARGET)

# PCH generation rule - INCORRECT: missing include flag and dependencies
$(PCH_OUTPUT): $(PCH_SOURCE)
	$(CXX) $(CXXFLAGS) -c $(PCH_SOURCE) -o $(PCH_OUTPUT)

# Main object compilation - depends on PCH
main.o: main.cpp $(PCH_OUTPUT)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -include $(PCH_HEADER) -c main.cpp -o main.o

# Linking
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(PCH_OUTPUT) $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all (default) - Build the project"
	@echo "  clean         - Remove build artifacts"