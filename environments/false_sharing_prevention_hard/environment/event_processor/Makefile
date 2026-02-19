CXX = g++
CXXFLAGS = -std=c++11 -pthread -O2 -Wall

all: processor

processor: processor.cpp
	$(CXX) $(CXXFLAGS) -o processor processor.cpp

clean:
	rm -f processor *.o

.PHONY: all clean