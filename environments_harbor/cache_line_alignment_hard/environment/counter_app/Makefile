CXX=g++
CXXFLAGS=-std=c++11 -O3 -pthread -Wall

.PHONY: all clean

all: counter_app

counter_app: main.cpp counter.h
	$(CXX) $(CXXFLAGS) main.cpp -o counter_app

clean:
	rm -f counter_app