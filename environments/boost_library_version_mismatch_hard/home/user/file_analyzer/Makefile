CXX = g++
CXXFLAGS = -Wall -std=c++11
LIBS = -lboost_filesystem -lboost_system
TARGET = analyzer
SOURCE = analyzer.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean